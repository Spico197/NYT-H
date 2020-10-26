#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import pdb
import os
import sys
import json
import math
import random
import logging
import socket
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange

from config import Config
from helper import (set_seed, WordEmbeddingLoader, NythDataset,
                    SentDataProcessor, BagDataProcessor,
                    Bag2SentDataProcessor)
from model import (SentCNN, CRCNN, RankingLoss, PCNN, ATT_BLSTM,
                    CNN_ONE, PCNN_ONE, CNN_ATT, PCNN_ATT)
from evaluate import evaluate_nyth, evaluate_crcnn, evaluate_bag2sent


logger = logging.getLogger('new_lib_logger')


def get_logdir_suffix(comment=None):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    suffix = current_time + '_' + socket.gethostname()
    if comment:
        suffix += comment
    return suffix


def get_output_dirs_ready(config):
    r"""Get ready for output dirs"""
    if os.path.exists(config.output_dir) and os.listdir(config.output_dir) and config.do_train and not config.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(config.output_dir))
    if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

    if not os.path.exists(os.path.join(config.output_dir, "backup")):
        os.makedirs(os.path.join(config.output_dir, "backup"))
    if not os.path.exists(os.path.join(config.output_dir, "checkpoints")):
        os.makedirs(os.path.join(config.output_dir, "checkpoints"))
    if not os.path.exists(os.path.join(config.output_dir, "checkpoints", "best")):
        os.makedirs(os.path.join(config.output_dir, "checkpoints", "best"))
    checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
    if not os.path.exists(os.path.join(config.output_dir, "eval")):
        os.makedirs(os.path.join(config.output_dir, "eval"))
    if not os.path.exists(os.path.join(config.output_dir, "tb_log_dir")):
        os.makedirs(os.path.join(config.output_dir, "tb_log_dir"))

    current_file_path = os.path.abspath(__file__)
    current_base_path = os.path.abspath(os.path.dirname(current_file_path))
    with open(current_file_path, 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", str(__file__) + "_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'run.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "run.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'model.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "model.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'config.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "config.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'evaluate.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "evaluate.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'helper.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "helper.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)
    with open(os.path.join(current_base_path, 'plot_prc.py'), 'rt', encoding='utf-8') as fin:
        current_file = fin.read()
        with open(os.path.join(config.output_dir, "backup", "plot_prc.py_bak.py"), "wt", encoding='utf-8') as fout:
            fout.write(current_file)

    with open(os.path.join(config.output_dir, "backup", "config.json"), "wt", encoding='utf-8') as fout:
        json.dump(config.args.__dict__, fout, ensure_ascii=False)
    with open(os.path.join(config.output_dir, "readme.txt"), "wt", encoding='utf-8') as fout:
        fout.write(config.description)


def train(logger, config, model, processor):
    comment = f"_TASK-{config.task_name}_MODEL-{config.model_name}" + \
                f"_EPOCH-{config.epoch}_BATCH-{config.batch_size}_LR-{config.lr}"
    suffix = get_logdir_suffix(comment)
    logger.info(suffix)
    tb_log_dir = os.path.join(config.output_dir, 'tb_log_dir', suffix)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)

    if config.model_name == 'sent_crcnn':
        criterion = RankingLoss(processor.class_num, config)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {processor.num_train_examples}")
    logger.info(f"  Num Epochs = {config.epoch}")
    logger.info(f"  Train batch size = {config.batch_size}")

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(config.epoch, desc="Epoch")
    set_seed(config.seed)
    best_nonna_macro_f1 = 0.0
    epoch_num = 1
    for _ in train_iterator:
        train_loader = processor.train_loader
        epoch_iterator = tqdm(train_loader, desc="Iteration", ncols=60)
        for step, raw_batch in enumerate(epoch_iterator):
            model.train()
            if config.task_name == 'sent':
                batch = tuple(t.to(config.device) for t in raw_batch[:-2])
                rel_labels = batch[4]
                bag_labels = batch[5]
                instance_id = raw_batch[6]
                bag_id = raw_batch[7]
                inputs = {
                    "token2ids": batch[0],
                    "pos1s": batch[1],
                    "pos2s": batch[2],
                    "mask": batch[3],
                }
            elif config.task_name == 'bag':
                batch = tuple(t.to(config.device) for t in raw_batch[:-3])
                rel_labels = batch[4]
                bag_labels = batch[5]
                instance_id = raw_batch[6]
                bag_id = raw_batch[7]
                inputs = {
                    "token2ids": batch[0],
                    "pos1s": batch[1],
                    "pos2s": batch[2],
                    "mask": batch[3],
                    "scopes": raw_batch[8],
                    "is_training": True,
                    "rel_labels": rel_labels,
                }
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            out = model(**inputs)
            loss = criterion(out, rel_labels.to(config.device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
            if config.do_eval_while_train:
                if global_step % config.tb_logging_step == 0:
                    if config.model_name == 'sent_crcnn':
                        results, eval_loss, preds, labels, outs = evaluate_crcnn(model, criterion, logger, processor, config, "train", f"E-{epoch_num}_S-{step+1}")
                    else:
                        results, eval_loss, preds, labels, outs = evaluate_nyth(model, criterion, logger, processor, config, "train", f"E-{epoch_num}_S-{step+1}")

                    for key, val in results.items():
                        if 'report' in key:
                            continue
                        tb_writer.add_scalar(f"{key}/train", val, global_step)
                    tb_writer.add_scalar("loss/train", (train_loss - logging_loss)/config.tb_logging_step, global_step)
                    probs = torch.nn.functional.softmax(torch.tensor(outs), dim=1)
                    thresholds, indices = probs.max(dim=1)
                    tb_writer.add_pr_curve('pr_curve/train', labels==preds, thresholds, global_step=global_step, num_thresholds=len(preds))
                    logging_loss = train_loss
        
        if config.model_name == 'sent_crcnn':
            results, eval_loss, preds, labels, outs = evaluate_crcnn(model, criterion, logger, processor, config, "dev", f"E-{epoch_num}_S-{step+1}")
        else:
            results, eval_loss, preds, labels, outs = evaluate_nyth(model, criterion, logger, processor, config, "dev", f"E-{epoch_num}_S-{step+1}")
        for key, val in results.items():
            if 'report' in key:
                continue
            tb_writer.add_scalar(f"{key}/dev", val, global_step)

        probs = torch.nn.functional.softmax(torch.tensor(outs), dim=1)
        thresholds, indices = probs.max(dim=1)
        tb_writer.add_pr_curve('pr_curve/dev', labels==preds, thresholds, global_step=global_step, num_thresholds=len(preds))
        nonna_macro_f1 = results[config.select_score]
        if nonna_macro_f1 > best_nonna_macro_f1:
            best_nonna_macro_f1 = nonna_macro_f1
            logger.info(f"Epoch: {epoch_num}, *Best DEV {config.select_score}: {best_nonna_macro_f1}")
            if config.save_best_model:
                output_dir = os.path.join(config.output_dir, 'checkpoints', 'best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                logger.info(f"Epoch: {epoch_num}, Saving model to {output_dir}")
        else:
            logger.info(f"Epoch: {epoch_num}, DEV {config.select_score}: {nonna_macro_f1}")
        epoch_num += 1
    tb_writer.close()
    return global_step, train_loss/global_step


processors = {
    "sent": SentDataProcessor,
    "bag": BagDataProcessor,
    "bag2sent": Bag2SentDataProcessor,
}

model_classes = {
    "sent_cnn": SentCNN,
    "sent_crcnn": CRCNN,
    "sent_pcnn": PCNN,
    "sent_att_blstm": ATT_BLSTM,
    "bag_cnn_one": CNN_ONE,
    "bag_pcnn_one": PCNN_ONE,
    "bag_cnn_att": CNN_ATT,
    "bag_pcnn_att": PCNN_ATT
}


if __name__ == '__main__':
    config = Config()
    set_seed(config.seed)
    get_output_dirs_ready(config)

    """setup logging module"""
    logger.setLevel(logging.INFO)
    log_path = os.path.join(config.output_dir, 'log.log')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = "[%(asctime)-15s]-%(levelname)s-%(filename)s-%(lineno)d-%(process)d: %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.warning(f"device: {config.main_device}, n_gpu: {len(config.device_ids)}")
    logger.info(f'Configurations: {config.__dict__}')

    """get models ready"""
    config.task_name = config.task_name.lower()
    if config.task_name not in processors:
        raise ValueError("Task not found: %s" % (config.task_name))
    processor = processors[config.task_name](config)
    class_num = processor.class_num
    word_vec = processor.word_vec

    config.model_name = config.model_name.lower()
    if config.model_name not in model_classes:
        raise ValueError("Model not found: %s" % (config.model_name))

    Model = model_classes[config.model_name]
    model = Model(word_vec=word_vec, class_num=class_num, config=config)
    if len(config.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.device_ids)
    model = model.to(config.device)
    
    logger.info(model)
    logger.info('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info('%s :  %s' % (name, str(param.data.shape)))
    logger.info('--------------------------------------')
    logger.info('start to train the model ...')

    if config.do_train:
        global_step, train_loss = train(logger, config, model, processor)
        logger.info(f"global step = {global_step}, average loss = {train_loss}")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(config.output_dir, 'checkpoints', 'final_model.pth'))
    
    if config.do_eval:
        if config.eval_model == 'final':
            path = os.path.join(config.output_dir, 'checkpoints', 'final_model.pth')
            if not os.path.exists(path) or not os.path.isfile(path):
                raise ValueError(f"Path {path} not exists!")
        elif config.eval_model == 'best':
            path = os.path.join(config.output_dir, 'checkpoints', 'best', 'best_model.pth')
            if not os.path.exists(path) or not os.path.isfile(path):
                raise ValueError(f"Path {path} not exists!")
        else:
            raise ValueError(f"Wrong eval_model mode: {config.eval_model}, should choose from (best, final)")

        model.load_state_dict(torch.load(path, map_location=config.device))
        model = model.to(config.device)
        if config.task_name == "bag2sent" and config.model_name.startswith("bag_"):
            results, eval_loss, preds, labels, outs = evaluate_bag2sent(model, nn.CrossEntropyLoss(), logger, processor, config, "test")
        else:
            if config.model_name == 'sent_crcnn':
                criterion = RankingLoss(processor.class_num, config)
                criterion.to(config.device)
                results, eval_loss, preds, labels, outs = evaluate_crcnn(model, criterion, logger, processor, config, "test")
            else:
                results, eval_loss, preds, labels, outs = evaluate_nyth(model, nn.CrossEntropyLoss(), logger, processor, config, "test")
