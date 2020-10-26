#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import pdb
import json
import random
import pickle
import logging
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

logger = logging.getLogger('new_lib_logger')

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WordEmbeddingLoader(object):
    def __init__(self, config, vocab):
        self.embedding_path = config.embedding_path  # path of pre-trained word embedding
        self.embedding_dim = config.embedding_dim  # dimension of word embedding
        self.config = config
        if not isinstance(vocab, set):
            vocab = set(vocab)
        self.vocab = vocab
        self.whole_vocab_len = len(self.vocab)

    def load_word_vec(self):
        if os.path.exists(os.path.join(self.config.data_dir, 'word2id.pkl')) and \
            os.path.exists(os.path.join(self.config.data_dir, 'word_vec.pkl')):
            with open(os.path.join(self.config.data_dir, 'word2id.pkl'), 'rb') as fin:
                word2id = pickle.load(fin)
            logger.info(f"OOV: {self.whole_vocab_len - len(word2id)}/{self.whole_vocab_len} = \
                {(self.whole_vocab_len - len(word2id))/self.whole_vocab_len*100}%")
            with open(os.path.join(self.config.data_dir, 'word_vec.pkl'), 'rb') as fin:
                word_vec = pickle.load(fin)
            return word2id, word_vec
        word2id = {}  # word to wordID
        word_vec = []  # wordID to word embedding

        word2id['<PAD>'] = len(word2id)  # PAD character
        word2id['<UNK>'] = len(word2id)  # words, out of vocabulary

        with open(self.embedding_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.embedding_dim + 1:
                    continue
                if line[0] in self.vocab:
                    word2id[line[0]] = len(word2id)
                    word_vec.append(np.asarray(line[1: 1 + self.embedding_dim], dtype=np.float32))

        word_vec = np.stack(word_vec).reshape(-1, self.embedding_dim)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        extra_vec = np.random.normal(vec_mean, vec_std, size=(2, self.embedding_dim))

        word_vec = np.concatenate((extra_vec, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.embedding_dim)
        word_vec = torch.from_numpy(word_vec)
        with open(os.path.join(self.config.data_dir, 'word2id.pkl'), 'wb') as fin:
            pickle.dump(word2id, fin)
        with open(os.path.join(self.config.data_dir, 'word_vec.pkl'), 'wb') as fin:
            pickle.dump(word_vec, fin)
        logger.info(f"OOV: {self.whole_vocab_len - len(word2id)}/{self.whole_vocab_len} = {(self.whole_vocab_len - len(word2id))/self.whole_vocab_len*100}%")
        return word2id, word_vec


class NythDataset(Dataset):
    def __init__(self, path, config, word2id, rel2id, bag_label2id):
        with open(path, 'rt', encoding='utf-8') as fin:
            self.data = json.load(fin)
            
        self.word2id = word2id
        self.rel2id = rel2id
        self.config = config
        self.bag_label2id = bag_label2id

    def get_vocab(self):
        vocab = set()
        for ins in self.data:
            for word in ins['sentence'].split():
                vocab.add(word.lower())
        return vocab

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        token2ids =[self.word2id.get(x, self.word2id['<UNK>']) for x in self.data[index]['sentence'].lower().split()]
        if len(token2ids) >= self.config.max_len:
            token2ids = token2ids[:self.config.max_len]
        else:
            token2ids = token2ids + [self.word2id['<PAD>']]*(self.config.max_len - len(token2ids))
        ins = self.data[index]
        head_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['head']['word'].lower().split()]
        tail_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['tail']['word'].lower().split()]
        
        str_token2ids = " ".join([str(x) for x in token2ids])
        str_head_ids = " ".join([str(x) for x in head_word_ids])
        str_tail_ids = " ".join([str(x) for x in tail_word_ids])
        try:
            if len(head_word_ids) > len(tail_word_ids):
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
            else:
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
        except:
            # sentence is too long to capture the pos information
            head_pos = 0
            tail_pos = self.config.max_len

        if head_pos >= self.config.max_len:
            head_pos = 0
        if tail_pos >= self.config.max_len:
            tail_pos = self.config.max_len - 1

        mask = [0]*self.config.max_len        
        for i in range(0, self.config.max_len):
            if 0 <= i < min(head_pos, tail_pos):
                mask[i] = 1
            elif min(head_pos, tail_pos) <= i < max(head_pos, tail_pos):
                mask[i] = 2
            elif max(head_pos, tail_pos) <= i < min(self.config.max_len, len(self.data[index]['sentence'])):
                mask[i] = 3
            else:
                mask[i] = 0

        pos1 = np.array(list(range(head_pos, -1, -1)) + list(range(1, self.config.max_len - head_pos, 1)))
        pos2 = np.array(list(range(tail_pos, -1, -1)) + list(range(1, self.config.max_len - tail_pos, 1)))
        try:
            assert pos1.shape[0] == self.config.max_len
            assert pos2.shape[0] == self.config.max_len
        except:
            import pdb
            pdb.set_trace()
        return torch.tensor(token2ids), torch.tensor(pos1), torch.tensor(pos2), \
                torch.tensor(mask), torch.tensor(self.rel2id[ins['relation']]), \
                torch.tensor(self.bag_label2id[ins['bag_label']]), \
                ins['instance_id'], ins['bag_id']

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is None:
                step = 1
            else:
                step = index.step
            results = list()
            for i in range(index.start, index.stop, step):
                token2id, pos1, pos2, mask, rel2id, bag_label2id, instance_id, bag_id = self._get_item(i)
                results.append([token2id, pos1, pos2, mask, rel2id, bag_label2id, instance_id, bag_id])
            return results
        else:
            return self._get_item(index)


class SentDataProcessor(object):
    def __init__(self, config):
        self.config = config
    
        logger.info('--------------------------------------')
        logger.info('start to load data ...')
        with open(os.path.join(config.data_dir, 'rel2id.json'), 'rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        id2rel = {val: key for key, val in rel2id.items()}
        with open(os.path.join(config.data_dir, 'bag_label2id.json'), 'rt', encoding='utf-8') as fin:
            bag_label2id = json.load(fin)
        id2bag_label = {val: key for key, val in bag_label2id.items()}
        
        logger.info('build vocab')
        if os.path.exists(os.path.join(self.config.data_dir, 'vocab.pkl')):
            with open(os.path.join(self.config.data_dir, 'vocab.pkl'), 'rb') as fin:
                vocab = pickle.load(fin)
        else:
            train_vocab = set()
            with open(os.path.join(config.data_dir, 'train.json'), 'r') as fin:
                train = json.load(fin)
                for ins in train:
                    for word in ins['sentence'].lower().split():
                        train_vocab.add(word)

            dev_vocab = set()
            with open(os.path.join(config.data_dir, 'dev.json'), 'r') as fin:
                dev = json.load(fin)
                for ins in dev:
                    for word in ins['sentence'].lower().split():
                        dev_vocab.add(word)

            test_vocab = set()
            with open(os.path.join(config.data_dir, 'test.json'), 'r') as fin:
                test = json.load(fin)
                for ins in test:
                    for word in ins['sentence'].lower().split():
                        test_vocab.add(word)

            vocab = train_vocab | test_vocab | dev_vocab
            with open(os.path.join(self.config.data_dir, 'vocab.pkl'), 'wb') as fin:
                pickle.dump(vocab, fin)
        word2id, word_vec = WordEmbeddingLoader(config, vocab).load_word_vec()
        logger.info(len(word2id))
        logger.info(word_vec.shape)

        logger.info('test data ...')
        test_dataset = self._load_data('test.json', 'test', word2id, rel2id, bag_label2id)
        test_bags = set()
        for ins in test_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            test_bags.add(index)
        logger.info(f'len of test: {len(test_dataset)}, bag of test: {len(test_bags)}')

        logger.info('dev data ...')
        dev_dataset = self._load_data('dev.json', 'dev', word2id, rel2id, bag_label2id)
        dev_bags = set()
        for ins in dev_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            dev_bags.add(index)
        logger.info(f'len of dev: {len(dev_dataset)}, bag of dev: {len(dev_bags)}')

        logger.info('train data ...')
        train_dataset = self._load_data('train.json', 'train', word2id, rel2id, bag_label2id)
        train_bags = set()
        for ins in train_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            train_bags.add(index)
        logger.info(f'len of train: {len(train_dataset)}, bag of train: {len(train_bags)}')

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        self.word_vec = word_vec
        self.word2id = word2id
        self.id2word = {val: key for key, val in word2id.items()}
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.bag_label2id = bag_label2id
        self.id2bag_label = id2bag_label

        self.num_train_examples = len(train_dataset)
        self.num_dev_examples = len(dev_dataset)
        self.num_test_examples = len(test_dataset)
        self.class_num = len(rel2id)
        self.test_dataset = test_dataset

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        logger.info('finished !')
        logger.info('--------------------------------------')

    def _load_data(self, datafile, set_type, word2id, rel2id, bag_label2id):
        path = os.path.join(self.config.data_dir, datafile)
        if os.path.exists(os.path.join(self.config.data_dir, set_type+'.pkl')):
            with open(os.path.join(self.config.data_dir, set_type+'.pkl'), 'rb') as fin:
                dataset = pickle.load(fin)
        else:
            dataset = NythDataset(path, self.config, word2id, rel2id, bag_label2id)
            with open(os.path.join(self.config.data_dir, set_type+'.pkl'), 'wb') as fin:
                pickle.dump(dataset, fin)
        return dataset

    def get_data(self):
        return self.word_vec, self.train_loader, self.dev_loader, self.test_loader


class NythBagDataset(Dataset):
    def __init__(self, path, config, word2id, rel2id, bag_label2id):
        with open(path, 'rt', encoding='utf-8') as fin:
            self.data = json.load(fin)
            
        self.word2id = word2id
        self.rel2id = rel2id
        self.config = config
        self.bag_label2id = bag_label2id

        self.bag_scope = []
        self.bag2id = {}
        self.bag_ids = []
        
        for idx, ins in enumerate(self.data):
            if ins['bag_id'] not in self.bag2id:
                self.bag2id[ins['bag_id']] = len(self.bag2id)
                self.bag_scope.append([])
                self.bag_ids.append(ins['bag_id'])
            self.bag_scope[self.bag2id[ins['bag_id']]].append(idx)

    def get_vocab(self):
        vocab = set()
        for ins in self.data:
            for word in ins['sentence'].split():
                vocab.add(word.lower())
        return vocab

    def __len__(self):
        return len(self.bag_scope)

    def _process_ins(self, index, bag_label):
        token2ids =[self.word2id.get(x, self.word2id['<UNK>']) for x in self.data[index]['sentence'].lower().split()]
        if len(token2ids) >= self.config.max_len:
            token2ids = token2ids[:self.config.max_len]
        else:
            token2ids = token2ids + [self.word2id['<PAD>']]*(self.config.max_len - len(token2ids))
        ins = self.data[index]
        head_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['head']['word'].lower().split()]
        tail_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['tail']['word'].lower().split()]
        
        str_token2ids = " ".join([str(x) for x in token2ids])
        str_head_ids = " ".join([str(x) for x in head_word_ids])
        str_tail_ids = " ".join([str(x) for x in tail_word_ids])
        try:
            if len(head_word_ids) > len(tail_word_ids):
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
            else:
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
        except:
            # sentence is too long to capture the pos information
            head_pos = 0
            tail_pos = self.config.max_len

        if head_pos >= self.config.max_len:
            head_pos = 0
        if tail_pos >= self.config.max_len:
            tail_pos = self.config.max_len - 1

        mask = [0]*self.config.max_len
        for i in range(0, self.config.max_len):
            if 0 <= i < min(head_pos, tail_pos):
                mask[i] = 1
            elif min(head_pos, tail_pos) <= i < max(head_pos, tail_pos):
                mask[i] = 2
            elif max(head_pos, tail_pos) <= i < min(self.config.max_len, len(self.data[index]['sentence'].split())):
                mask[i] = 3
            else:
                mask[i] = 0
            
        pos1 = np.array(list(range(head_pos, -1, -1)) + list(range(1, self.config.max_len - head_pos, 1)))
        pos2 = np.array(list(range(tail_pos, -1, -1)) + list(range(1, self.config.max_len - tail_pos, 1)))
        assert pos1.shape[0] == self.config.max_len
        assert pos2.shape[0] == self.config.max_len

        return torch.tensor(token2ids), torch.tensor(pos1), torch.tensor(pos2), \
                torch.tensor(mask), torch.tensor(self.rel2id[ins['relation']]), \
                torch.tensor(self.bag_label2id[bag_label]), \
                ins['instance_id'], ins['bag_id']

    def _get_item(self, index):
        results = list()
        bag_label = 'no'
        for ind in self.bag_scope[index]:
            if self.data[ind]['bag_label'] == 'yes':
                bag_label = 'yes'
        for ind in self.bag_scope[index]:
            ins = self._process_ins(ind, bag_label)
            results.append(ins)
        return results

    def __getitem__(self, index):
        return self._get_item(index)
    
    def collate_func(self, data):
        token2id = list(); pos1 = list(); pos2 = list(); mask = list()
        rel2id = list(); bag_label2id = list(); instance_id = list(); bag_id = list()
        
        scope = []; scope_begin = 0
        # data is a batch, for every bag in the batch:
        for bag in data:
            scope_ = [scope_begin]
            # for every instance in the bag
            for ins in bag:
                token2id_, pos1_, pos2_, mask_, rel2id_, bag_label2id_, instance_id_, bag_id_ = ins
                token2id.append(token2id_)
                pos1.append(pos1_)
                pos2.append(pos2_)
                mask.append(mask_)
                instance_id.append(instance_id_)
                scope_begin += 1
            bag_id.append(bag_id_)
            bag_label2id.append(bag_label2id_)
            rel2id.append(rel2id_)
            pdb.set_trace()
            scope_.append(scope_begin)
            scope.append(scope_)
        token2id = torch.stack(token2id)
        pos1 = torch.stack(pos1)
        pos2 = torch.stack(pos2)
        mask = torch.stack(mask)
        rel2id = torch.stack(rel2id)
        bag_label2id = torch.stack(bag_label2id)
        remain_return = [token2id, pos1, pos2, mask, rel2id, bag_label2id, instance_id, bag_id, scope]
        return remain_return


class BagDataProcessor(object):
    def __init__(self, config):
        self.config = config
    
        logger.info('--------------------------------------')
        logger.info('start to load data ...')
        with open(os.path.join(config.data_dir, 'rel2id.json'), 'rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        id2rel = {val: key for key, val in rel2id.items()}
        with open(os.path.join(config.data_dir, 'bag_label2id.json'), 'rt', encoding='utf-8') as fin:
            bag_label2id = json.load(fin)
        id2bag_label = {val: key for key, val in bag_label2id.items()}
        
        logger.info('build vocab')
        if os.path.exists(os.path.join(self.config.data_dir, 'vocab_bag.pkl')):
            with open(os.path.join(self.config.data_dir, 'vocab_bag.pkl'), 'rb') as fin:
                vocab = pickle.load(fin)
        else:
            train_vocab = set()
            with open(os.path.join(config.data_dir, 'train.json'), 'r') as fin:
                train = json.load(fin)
                for ins in train:
                    for word in ins['sentence'].lower().split():
                        train_vocab.add(word)

            dev_vocab = set()
            with open(os.path.join(config.data_dir, 'dev.json'), 'r') as fin:
                dev = json.load(fin)
                for ins in dev:
                    for word in ins['sentence'].lower().split():
                        dev_vocab.add(word)

            test_vocab = set()
            with open(os.path.join(config.data_dir, 'test.json'), 'r') as fin:
                test = json.load(fin)
                for ins in test:
                    for word in ins['sentence'].lower().split():
                        test_vocab.add(word)

            vocab = train_vocab | test_vocab | dev_vocab
            with open(os.path.join(self.config.data_dir, 'vocab_bag.pkl'), 'wb') as fin:
                pickle.dump(vocab, fin)
        word2id, word_vec = WordEmbeddingLoader(config, vocab).load_word_vec()
        logger.info(len(word2id))
        logger.info(word_vec.shape)

        logger.info('test data ...')
        test_dataset = self._load_data('test.json', 'test', word2id, rel2id, bag_label2id)
        test_bags = set()
        for ins in test_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            test_bags.add(index)
        logger.info(f'len of test: {len(test_dataset)}, bag of test: {len(test_bags)}')

        logger.info('dev data ...')
        dev_dataset = self._load_data('dev.json', 'dev', word2id, rel2id, bag_label2id)
        dev_bags = set()
        for ins in dev_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            dev_bags.add(index)
        logger.info(f'len of dev: {len(dev_dataset)}, bag of dev: {len(dev_bags)}')

        logger.info('train data ...')
        train_dataset = self._load_data('train.json', 'train', word2id, rel2id, bag_label2id)
        train_bags = set()
        for ins in train_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            train_bags.add(index)
        logger.info(f'len of train: {len(train_dataset)}, bag of train: {len(train_bags)}')

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, \
                                  collate_fn=self.collate_func, num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, \
                                collate_fn=self.collate_func, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, \
                                 collate_fn=self.collate_func, num_workers=4)

        self.word_vec = word_vec
        self.word2id = word2id
        self.id2word = {val: key for key, val in word2id.items()}
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.bag_label2id = bag_label2id
        self.id2bag_label = id2bag_label

        self.num_train_examples = len(train_dataset)
        self.num_dev_examples = len(dev_dataset)
        self.num_test_examples = len(test_dataset)
        self.class_num = len(rel2id)
        self.test_dataset = test_dataset

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        train_counter = Counter([self.rel2id[x['relation']] for x in train_dataset.data])
        weight = 1 / (np.asarray([train_counter[i] for i in range(len(self.rel2id))])*0.05)
        self.weight = torch.tensor(weight, device=self.config.device, dtype=torch.float32)

        logger.info('finished !')
        logger.info('--------------------------------------')

    def _load_data(self, datafile, set_type, word2id, rel2id, bag_label2id):
        path = os.path.join(self.config.data_dir, datafile)
        if os.path.exists(os.path.join(self.config.data_dir, set_type+'_bag.pkl')):
            with open(os.path.join(self.config.data_dir, set_type+'_bag.pkl'), 'rb') as fin:
                dataset = pickle.load(fin)
        else:
            dataset = NythBagDataset(path, self.config, word2id, rel2id, bag_label2id)
            with open(os.path.join(self.config.data_dir, set_type+'_bag.pkl'), 'wb') as fin:
                pickle.dump(dataset, fin)

        return dataset

    def get_data(self):
        return self.word_vec, self.train_loader, self.dev_loader, self.test_loader
    
    def collate_func(self, data):
        token2id = list(); pos1 = list(); pos2 = list(); mask = list()
        rel2id = list(); bag_label2id = list(); instance_id = list(); bag_id = list()
        
        scope = []; scope_begin = 0
        # data is a batch, for every bag in the batch:
        for bag in data:
            scope_ = [scope_begin]
            # for every instance in the bag
            for ins in bag:
                token2id_, pos1_, pos2_, mask_, rel2id_, bag_label2id_, instance_id_, bag_id_ = ins
                token2id.append(token2id_)
                pos1.append(pos1_)
                pos2.append(pos2_)
                mask.append(mask_)
                instance_id.append(instance_id_)
                scope_begin += 1
            bag_id.append(bag_id_)
            bag_label2id.append(bag_label2id_)
            rel2id.append(rel2id_)
            scope_.append(scope_begin)
            scope.append(scope_)
        token2id = torch.stack(token2id)
        pos1 = torch.stack(pos1)
        pos2 = torch.stack(pos2)
        mask = torch.stack(mask)
        rel2id = torch.stack(rel2id)
        bag_label2id = torch.stack(bag_label2id)
        remain_return = [token2id, pos1, pos2, mask, rel2id, bag_label2id, instance_id, bag_id, scope]
        return remain_return


class NythBag2SentDataset(Dataset):
    def __init__(self, path, config, word2id, rel2id, bag_label2id):
        with open(path, 'rt', encoding='utf-8') as fin:
            self.data = json.load(fin)
            
        self.word2id = word2id
        self.rel2id = rel2id
        self.config = config
        self.bag_label2id = bag_label2id

        self.bag_scope = []
        self.bag2id = {}
        self.bag_ids = []
        
        for idx, ins in enumerate(self.data):
            if ins['instance_id'] not in self.bag2id:
                self.bag2id[ins['instance_id']] = len(self.bag2id)
                self.bag_scope.append([])
                self.bag_ids.append(ins['instance_id'])
            self.bag_scope[self.bag2id[ins['instance_id']]].append(idx)

    def get_vocab(self):
        vocab = set()
        for ins in self.data:
            for word in ins['sentence'].split():
                vocab.add(word.lower())
        return vocab

    def __len__(self):
        return len(self.bag_scope)

    def _process_ins(self, index):
        token2ids =[self.word2id.get(x, self.word2id['<UNK>']) for x in self.data[index]['sentence'].lower().split()]
        if len(token2ids) >= self.config.max_len:
            token2ids = token2ids[:self.config.max_len]
        else:
            token2ids = token2ids + [self.word2id['<PAD>']]*(self.config.max_len - len(token2ids))
        ins = self.data[index]
        head_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['head']['word'].lower().split()]
        tail_word_ids = [self.word2id.get(x, self.word2id['<UNK>']) for x in ins['tail']['word'].lower().split()]
        
        str_token2ids = " ".join([str(x) for x in token2ids])
        str_head_ids = " ".join([str(x) for x in head_word_ids])
        str_tail_ids = " ".join([str(x) for x in tail_word_ids])
        try:
            if len(head_word_ids) > len(tail_word_ids):
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
            else:
                tail_pos = len(str_token2ids[:str_token2ids.index(str_tail_ids)].split())
                head_pos = len(str_token2ids[:str_token2ids.index(str_head_ids)].split())
        except:
            # sentence is too long to capture the pos information
            head_pos = 0
            tail_pos = self.config.max_len

        if head_pos >= self.config.max_len:
            head_pos = 0
        if tail_pos >= self.config.max_len:
            tail_pos = self.config.max_len - 1

        mask = [0]*self.config.max_len
        for i in range(0, self.config.max_len):
            if 0 <= i < min(head_pos, tail_pos):
                mask[i] = 1
            elif min(head_pos, tail_pos) <= i < max(head_pos, tail_pos):
                mask[i] = 2
            elif max(head_pos, tail_pos) <= i < min(self.config.max_len, len(self.data[index]['sentence'].split())):
                mask[i] = 3
            else:
                mask[i] = 0
            
        pos1 = np.array(list(range(head_pos, -1, -1)) + list(range(1, self.config.max_len - head_pos, 1)))
        pos2 = np.array(list(range(tail_pos, -1, -1)) + list(range(1, self.config.max_len - tail_pos, 1)))
        assert pos1.shape[0] == self.config.max_len
        assert pos2.shape[0] == self.config.max_len

        return torch.tensor(token2ids), torch.tensor(pos1), torch.tensor(pos2), \
                torch.tensor(mask), torch.tensor(self.rel2id[ins['relation']]), \
                torch.tensor(self.bag_label2id[ins['bag_label']]), \
                ins['instance_id'], ins['bag_id']

    def _get_item(self, index):
        results = list()
        for ind in self.bag_scope[index]:
            ins = self._process_ins(ind)
            results.append(ins)
        return results

    def __getitem__(self, index):
        return self._get_item(index)


class Bag2SentDataProcessor(object):
    def __init__(self, config):
        self.config = config
    
        logger.info('--------------------------------------')
        logger.info('start to load data ...')
        with open(os.path.join(config.data_dir, 'rel2id.json'), 'rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        id2rel = {val: key for key, val in rel2id.items()}
        with open(os.path.join(config.data_dir, 'bag_label2id.json'), 'rt', encoding='utf-8') as fin:
            bag_label2id = json.load(fin)
        id2bag_label = {val: key for key, val in bag_label2id.items()}
        
        logger.info('build vocab')
        if os.path.exists(os.path.join(self.config.data_dir, 'vocab_bag2sent.pkl')):
            with open(os.path.join(self.config.data_dir, 'vocab_bag2sent.pkl'), 'rb') as fin:
                vocab = pickle.load(fin)
        else:
            train_vocab = set()
            with open(os.path.join(config.data_dir, 'train.json'), 'r') as fin:
                train = json.load(fin)
                for ins in train:
                    for word in ins['sentence'].lower().split():
                        train_vocab.add(word)

            dev_vocab = set()
            with open(os.path.join(config.data_dir, 'dev.json'), 'r') as fin:
                dev = json.load(fin)
                for ins in dev:
                    for word in ins['sentence'].lower().split():
                        dev_vocab.add(word)

            test_vocab = set()
            with open(os.path.join(config.data_dir, 'test.json'), 'r') as fin:
                test = json.load(fin)
                for ins in test:
                    for word in ins['sentence'].lower().split():
                        test_vocab.add(word)

            vocab = train_vocab | test_vocab | dev_vocab
            with open(os.path.join(self.config.data_dir, 'vocab_bag2sent.pkl'), 'wb') as fin:
                pickle.dump(vocab, fin)
        word2id, word_vec = WordEmbeddingLoader(config, vocab).load_word_vec()
        logger.info(len(word2id))
        logger.info(word_vec.shape)

        logger.info('test data ...')
        test_dataset = self._load_data('test.json', 'test', word2id, rel2id, bag_label2id)
        test_bags = set()
        for ins in test_dataset.data:
            index = "{}###{}".format(ins['head']['word'], ins['tail']['word']).lower()
            test_bags.add(index)
        logger.info(f'len of test: {len(test_dataset)}, bag of test: {len(test_bags)}')

        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, \
                                 collate_fn=self.collate_func, num_workers=4)

        self.word_vec = word_vec
        self.word2id = word2id
        self.id2word = {val: key for key, val in word2id.items()}
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.bag_label2id = bag_label2id
        self.id2bag_label = id2bag_label

        self.test_dataset = test_dataset
        self.num_test_examples = len(test_dataset)
        self.class_num = len(rel2id)

        self.test_loader = test_loader
        logger.info('finished !')
        logger.info('--------------------------------------')

    def _load_data(self, datafile, set_type, word2id, rel2id, bag_label2id):
        path = os.path.join(self.config.data_dir, datafile)
        if os.path.exists(os.path.join(self.config.data_dir, set_type+'_bag2sent.pkl')):
            with open(os.path.join(self.config.data_dir, set_type+'_bag2sent.pkl'), 'rb') as fin:
                dataset = pickle.load(fin)
        else:
            dataset = NythBag2SentDataset(path, self.config, word2id, rel2id, bag_label2id)
            with open(os.path.join(self.config.data_dir, set_type+'_bag2sent.pkl'), 'wb') as fin:
                pickle.dump(dataset, fin)
        return dataset

    def get_data(self):
        return self.word_vec, self.test_loader
    
    def collate_func(self, data):
        token2id = list(); pos1 = list(); pos2 = list(); mask = list()
        rel2id = list(); bag_label2id = list(); instance_id = list(); bag_id = list()
        
        scope = []; scope_begin = 0
        # data is a batch, for every bag in the batch:
        for bag in data:
            scope_ = [scope_begin]
            # for every instance in the bag
            for ins in bag:
                token2id_, pos1_, pos2_, mask_, rel2id_, bag_label2id_, instance_id_, bag_id_ = ins
                token2id.append(token2id_)
                pos1.append(pos1_)
                pos2.append(pos2_)
                mask.append(mask_)
                instance_id.append(instance_id_)
                scope_begin += 1
            bag_id.append(bag_id_)
            bag_label2id.append(bag_label2id_)
            rel2id.append(rel2id_)
            scope_.append(scope_begin)
            scope.append(scope_)
        token2id = torch.stack(token2id)
        pos1 = torch.stack(pos1)
        pos2 = torch.stack(pos2)
        mask = torch.stack(mask)
        rel2id = torch.stack(rel2id)
        bag_label2id = torch.stack(bag_label2id)
        remain_return = [token2id, pos1, pos2, mask, rel2id, bag_label2id, instance_id, bag_id, scope]
        return remain_return
