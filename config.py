#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import torch
import os


class Config(object):
    def __init__(self):
        self.args = self._get_config()
        for key in self.args.__dict__:
            setattr(self, key, self.args.__dict__[key])
        self.device = torch.device(self.main_device)
        self.device_ids = eval(self.device_ids)

    def _get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'
        parser.add_argument('--data_dir', type=str, default='data/train',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str, default='output_dir',
                            help='dir to save models')
        parser.add_argument('--eval_model', type=str, default='best', choices=('best', 'final'),
                            help='dir to save models')
        parser.add_argument('--overwrite_output_dir', default=False, action='store_true',
                            help='overwrite output directory?')
        parser.add_argument('--description', type=str, default='NYT-H',
                            help='description')
        parser.add_argument('--task_name', type=str, default='sent',
                            help='task name')
        parser.add_argument('--model_name', type=str, default='sent_cnn',
                            help='model name')

        parser.add_argument('--embedding_path', type=str, default='/home/tzhu/embeddings/glove.6B.50d.txt',
                            help='pre_trained word embedding')
        parser.add_argument('--embedding_dim', type=int, default=50,
                            help='dimension of word embedding')

        parser.add_argument('--seed', type=int, default=4227019,
                            help='random seed')
        parser.add_argument('--main_device', type=str, default='cuda:1',
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--device_ids', type=str, default='[1]',
                            help='device_ids when parallel computing')
        parser.add_argument('--epoch', type=int, default=5,
                            help='max epoches during training')

        parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
        parser.add_argument('--dropout_rate', type=float, default=0.5,
                            help='the possiblity of dropout')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
        parser.add_argument('--max_len', type=int, default=150,
                            help='max length of sentence')
        parser.add_argument('--pos_dis', type=int, default=50,
                            help='max distance of position embedding')
        parser.add_argument('--pos_dim', type=int, default=5,
                            help='dimension of position embedding')
        parser.add_argument('--conv_filter_num', type=int, default=230,
                            help='the number of filters in convolution')
        parser.add_argument('--conv_window', type=int, default=3,
                            help='the size of window in convolution')
        
        parser.add_argument('--do_train', 
                            default=False, action='store_true',
                            help='train?')
        parser.add_argument('--select_score', 
                            default='non_na_macro_f1', type=str,
                            help='which score to select best models')
        parser.add_argument('--do_eval', 
                            default=False, action='store_true',
                            help='evaluate?')
        parser.add_argument('--do_eval_while_train', 
                            default=False, action='store_true',
                            help='evaluate?')
        parser.add_argument('--save_best_model', 
                            default=False, action='store_true',
                            help='whether to save best model while training')
        parser.add_argument('--tb_logging_step', type=int, default=30,
                            help='how many steps does the tensorboard writer write an evaluation result')
        
        # CR-CNN
        parser.add_argument('--margin_positive', type=float, default=2.5,
                            help='positive margin in the CRCNN loss function')
        parser.add_argument('--margin_negative', type=float, default=0.5,
                            help='negative margin in the CRCNN loss function')
        parser.add_argument('--gamma', type=float, default=2.0,
                            help='scaling factor \'gamma\' in the CRCNN loss function')
        parser.add_argument('--beta', type=float, default=0.001,
                            help='L2 weight decay')
        
        # LSTM
        parser.add_argument('--lstm_embedding_dropout', type=float, default=0.3,
                            help='the possiblity of dropout in embedding layer')
        parser.add_argument('--lstm_dropout', type=float, default=0.3,
                            help='the possiblity of dropout in (Bi)LSTM layer')
        parser.add_argument('--lstm_liner_dropout', type=float, default=0.5,
                            help='the possiblity of dropout in liner layer')
        parser.add_argument('--lstm_hidden_size', type=int, default=100,
                            help='the dimension of hidden units in (Bi)LSTM layer')
        parser.add_argument('--lstm_L2_decay', type=float, default=1e-5,
                            help='L2 weight decay')

        args = parser.parse_args()
        return args

    def print_config(self):
        print(self.args)


if __name__ == "__main__":
    config = Config()
