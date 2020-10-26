#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6
import pdb
import math
from math import floor, ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentCNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(SentCNN, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.embedding_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.conv_filter_num = config.conv_filter_num
        self.conv_window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.embedding_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_filter_num,
            kernel_size=(self.conv_window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.conv_filter_num,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.conv_filter_num, self.max_len)
        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.conv_filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask):
        tokens = token2ids
        pos1 = pos1s
        pos2 = pos2s
        mask = mask
        x = self.input(tokens, pos1, pos2)
        x = self.convolution(x, mask)
        x = self.maxpool(x)
        x = x.view(-1, self.conv_filter_num)
        x = self.tanh(x)
        x = self.dropout(x)
        out = self.dense(x)
        return out


class CRCNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(CRCNN, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.filter_num = config.conv_filter_num
        self.window = config.conv_window
        self.keep_prob = config.dropout_rate

        self.dim = self.word_dim + 2 * self.pos_dim

        self.r = (6/(self.class_num + self.filter_num))**(0.5)
        self.relation_weight = nn.Parameter(2 * self.r * (torch.rand(self.filter_num, self.class_num) - 0.5))

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.keep_prob)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask):
        tokens = token2ids
        pos1 = pos1s
        pos2 = pos2s
        mask = mask
        x = self.input(tokens, pos1, pos2)
        x = self.convolution(x, mask)
        x = self.maxpool(x)
        x = x.view(-1, self.filter_num)
        out = torch.mm(x, self.relation_weight)
        return out


class RankingLoss(nn.Module):
    def __init__(self, class_num, config):
        super(RankingLoss, self).__init__()
        self.class_num = class_num
        self.margin_positive = config.margin_positive
        self.margin_negative = config.margin_negative
        self.gamma = config.gamma
        self.device = config.device

    def forward(self, scores, labels):
        labels = labels.view(-1, 1)
        positive_mask = (torch.ones([labels.shape[0], self.class_num], device=self.device)*float('inf')).scatter_(1, labels, 0.0)
        negative_mask = torch.zeros([labels.shape[0], self.class_num], device=self.device).scatter_(1, labels, float('inf'))
        positive_scores, _ = torch.max(scores-positive_mask, dim=1)
        negative_scores, _ = torch.max(scores-negative_mask, dim=1)
        positive_loss = torch.log1p(torch.exp(self.gamma*(self.margin_positive-positive_scores)))
        positive_loss[labels[:, 0] == 0] = 0.0
        negative_loss = torch.log1p(torch.exp(self.gamma*(self.margin_negative+negative_scores)))
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class PCNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(PCNN, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.filter_num = config.conv_filter_num
        self.window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.filter_num*3,
            out_features=self.class_num,
            bias=True
        )

        # mask operation for pcnn
        self.mask_embedding = nn.Embedding(4, 3)
        masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.mask_embedding.weight.data.copy_(masks)
        self.mask_embedding.weight.requires_grad = False

        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def piece_maxpool(self, x, mask):
        x = x.permute(0, 2, 1, 3)
        mask_embed = self.mask_embedding(mask)
        mask_embed = mask_embed.unsqueeze(dim=-2)
        x = x + mask_embed
        x = torch.max(x, dim=1)[0] - 100
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask):
        tokens = token2ids
        pos1 = pos1s
        pos2 = pos2s
        mask = mask
        x = self.input(tokens, pos1, pos2)
        x = self.convolution(x, mask)
        x = self.piece_maxpool(x, mask)
        x = self.tanh(x)
        x = self.dropout(x)
        out = self.dense(x)
        return out


class ATT_BLSTM(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.dim = self.word_dim + 2 * self.pos_dim

        self.embedding_dropout = config.lstm_embedding_dropout
        self.lstm_dropout = config.lstm_dropout
        self.liner_dropout = config.lstm_liner_dropout
        self.hidden_size = config.lstm_hidden_size

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.bi_lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.tanh = nn.Tanh()
        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout)
        self.attention_weight = nn.Parameter(torch.randn(self.hidden_size))
        self.liner_dropout = nn.Dropout(p=self.liner_dropout)
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, (_, _) = self.bi_lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=self.max_len)
        x = x.view(-1, self.max_len, 2, self.hidden_size)
        x = torch.sum(x, dim=2)
        return x

    def attention_layer(self, x, mask):
        att = self.attention_weight.view(1, -1, 1).expand(x.shape[0], -1, -1)  # B*C*1
        att_score = torch.bmm(self.tanh(x), att)  # B*L*C  *  B*C*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)
        att_score = att_score.masked_fill_(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = torch.bmm(x.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*C*L *  B*L*1 -> B*C*1 -> B*C
        x = self.tanh(reps)  # B*C
        return x

    def forward(self, token2ids, pos1s, pos2s, mask):
        tokens = token2ids
        pos1 = pos1s
        pos2 = pos2s
        mask = mask
        x = self.input(tokens, pos1, pos2)
        x = self.embedding_dropout(x)
        x = self.lstm_layer(x, mask)
        x = self.lstm_dropout(x)
        x = self.attention_layer(x, mask)
        x = self.liner_dropout(x)
        logits = self.dense(x)
        return logits


class CNN_ONE(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(CNN_ONE, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.embedding_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.conv_filter_num = config.conv_filter_num
        self.conv_window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.embedding_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_filter_num,
            kernel_size=(self.conv_window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.conv_filter_num,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.conv_filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.conv_filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask, scopes, rel_labels, is_training=False):
        # encoder
        x = self.input(token2ids, pos1s, pos2s)
        x = self.convolution(x, mask)
        x = self.maxpool(x)
        x = x.view(-1, self.conv_filter_num)
        x = self.tanh(x)
        x = self.dropout(x)
        # selector
        bag_rep = list()
        for ind, scope in enumerate(scopes):
            sent_rep = x[scope[0]:scope[1], :]
            out = self.dense(sent_rep)
            probs = torch.softmax(out, dim=-1)
            if is_training:
                _, j = torch.max(probs[:, rel_labels[ind]], dim=-1)
                bag_rep.append(out[j])
            else:
                row_prob, row_idx = torch.max(probs, dim=-1)
                if row_idx.sum() > 0:
                    mask = row_idx.view(-1, 1).expand(-1, probs.shape[-1])
                    probs = probs.masked_fill_(mask.eq(0), float('-inf'))
                    row_prob, _ = torch.max(probs[:, 1:], dim=-1)
                    _, row_idx = torch.max(row_prob, dim=0)
                else:
                    _, row_idx = torch.max(row_prob, dim=-1)
                bag_rep.append(out[row_idx])
                
        bag_rep = torch.stack(bag_rep)
        return bag_rep


class PCNN_ONE(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(PCNN_ONE, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.filter_num = config.conv_filter_num
        self.window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.filter_num*3,
            out_features=self.class_num,
            bias=True
        )

        # mask operation for pcnn
        self.mask_embedding = nn.Embedding(4, 3)
        masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.mask_embedding.weight.data.copy_(masks)
        self.mask_embedding.weight.requires_grad = False

        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def piece_maxpool(self, x, mask):
        x = x.permute(0, 2, 1, 3)
        mask_embed = self.mask_embedding(mask)
        mask_embed = mask_embed.unsqueeze(dim=-2)
        x = x + mask_embed
        x = torch.max(x, dim=1)[0] - 100
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask, scopes, rel_labels, is_training=False):
        x = self.input(token2ids, pos1s, pos2s)
        x = self.convolution(x, mask)
        x = self.piece_maxpool(x, mask)
        x = self.tanh(x)
        x = self.dropout(x)
        # selector
        bag_rep = list()
        for ind, scope in enumerate(scopes):
            sent_rep = x[scope[0]:scope[1], :]
            out = self.dense(sent_rep)
            probs = torch.softmax(out, dim=-1)
            if is_training:
                _, j = torch.max(probs[:, rel_labels[ind]], dim=-1)
                bag_rep.append(out[j])
            else:
                row_prob, row_idx = torch.max(probs, dim=-1)
                if row_idx.sum() > 0:
                    mask = row_idx.view(-1, 1).expand(-1, probs.shape[-1])
                    probs = probs.masked_fill_(mask.eq(0), float('-inf'))
                    row_prob, _ = torch.max(probs[:, 1:], dim=-1)
                    _, row_idx = torch.max(row_prob, dim=0)
                else:
                    _, row_idx = torch.max(row_prob, dim=-1)
                bag_rep.append(out[row_idx])

        bag_rep = torch.stack(bag_rep)
        return bag_rep


class CNN_ATT(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(CNN_ATT, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.embedding_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.conv_filter_num = config.conv_filter_num
        self.conv_window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.embedding_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_filter_num,
            kernel_size=(self.conv_window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.conv_filter_num,
            out_features=self.class_num,
            bias=True
        )
        self.attention = nn.Parameter(torch.rand(size=(1, self.conv_filter_num)))
        self.softmax = nn.Softmax(-1)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.conv_filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.conv_filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask, scopes, rel_labels, is_training=False):
        # encoder
        x = self.input(token2ids, pos1s, pos2s)
        x = self.convolution(x, mask)
        x = self.maxpool(x)
        x = x.view(-1, self.conv_filter_num)
        x = self.tanh(x)
        x = self.dropout(x)
        # selector
        if is_training:
            query = torch.zeros((x.size(0))).long()
            query.to(x.device)
            for i in range(len(scopes)):
                query[scopes[i][0]:scopes[i][1]] = rel_labels[i]
            att_mat = self.dense.weight[query]
            att_score = (x * att_mat).sum(-1)
            bag_rep = []
            for i in range(len(scopes)):
                bag_mat = x[scopes[i][0]:scopes[i][1]]
                softmax_att_score = F.softmax(att_score[scopes[i][0]:scopes[i][1]], dim=-1)
                bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0))
            bag_rep = torch.stack(bag_rep, 0)
            bag_rep = self.dropout(bag_rep)
            bag_logits = self.dense(bag_rep)
        else:
            bag_logits = []
            att_score = torch.matmul(x, self.dense.weight.transpose(0, 1))
            for i in range(len(scopes)):
                bag_mat = x[scopes[i][0]:scopes[i][1]]
                softmax_att_score = F.softmax(att_score[scopes[i][0]:scopes[i][1]].transpose(0, 1), dim=-1) # (N, (softmax)n) 
                rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                logit_for_each_rel = F.softmax(self.dense(rep_for_each_rel), dim=-1) # ((each rel)N, (logit)N)
                logit_for_each_rel = logit_for_each_rel.diag() # (N)
                bag_logits.append(logit_for_each_rel)
            bag_logits = torch.stack(bag_logits, 0) # after **softmax**
        return bag_logits


class PCNN_ATT(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super(PCNN_ATT, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.filter_num = config.conv_filter_num
        self.window = config.conv_window
        self.dropout_rate = config.dropout_rate

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.filter_num*3,
            out_features=self.class_num,
            bias=True
        )

        # mask operation for pcnn
        self.mask_embedding = nn.Embedding(4, 3)
        masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.mask_embedding.weight.data.copy_(masks)
        self.mask_embedding.weight.requires_grad = False

        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def piece_maxpool(self, x, mask):
        # fast piecewise pooling
        x = x.permute(0, 2, 1, 3)
        mask_embed = self.mask_embedding(mask)
        mask_embed = mask_embed.unsqueeze(dim=-2)
        x = x + mask_embed
        x = torch.max(x, dim=1)[0] - 100
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, token2ids, pos1s, pos2s, mask, scopes, rel_labels, is_training=False):
        x = self.input(token2ids, pos1s, pos2s)
        x = self.convolution(x, mask)
        x = self.piece_maxpool(x, mask)
        x = self.tanh(x)
        x = self.dropout(x)
        # selector
        if is_training:
            query = torch.zeros((x.size(0))).long()
            query.to(x.device)
            for i in range(len(scopes)):
                query[scopes[i][0]:scopes[i][1]] = rel_labels[i]
            att_mat = self.dense.weight[query]
            att_score = (x * att_mat).sum(-1)
            bag_rep = []
            for i in range(len(scopes)):
                bag_mat = x[scopes[i][0]:scopes[i][1]]
                softmax_att_score = F.softmax(att_score[scopes[i][0]:scopes[i][1]], dim=-1)
                bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0))
            bag_rep = torch.stack(bag_rep, 0)
            bag_rep = self.dropout(bag_rep)
            bag_logits = self.dense(bag_rep)
        else:
            bag_logits = []
            att_score = torch.matmul(x, self.dense.weight.transpose(0, 1))
            for i in range(len(scopes)):
                bag_mat = x[scopes[i][0]:scopes[i][1]]
                softmax_att_score = F.softmax(att_score[scopes[i][0]:scopes[i][1]].transpose(0, 1), dim=-1) # (N, (softmax)n) 
                rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                logit_for_each_rel = F.softmax(self.dense(rep_for_each_rel), dim=-1) # ((each rel)N, (logit)N)
                logit_for_each_rel = logit_for_each_rel.diag() # (N)
                bag_logits.append(logit_for_each_rel)
            bag_logits = torch.stack(bag_logits, 0) # after **softmax**
        return bag_logits
