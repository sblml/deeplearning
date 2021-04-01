# !/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import random
from functools import partial

import numpy as np
import pandas as pd
import os
import gc
import tokenizers
import string
import torch
import transformers
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm.autonotebook import tqdm
import re
from torch.optim import lr_scheduler
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW, get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import RobertaTokenizer
from toolz.itertoolz import random_sample


SEED = 42
ver = [('v1.0', 'max_len=96,bsz=64,lr=3e-5,hidden=1, kernel-style'),
       
       ]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


class config:
    LANG = 'fr'
    MAX_LEN = 120
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 256
    EPOCHS = 10
    LR = 3e-5
    # BERT_PATH = "../models/roberta-base-squad2/"
    # BERT_PATH = "../models/pretrain_output_sent140_all/"
    # BERT_PATH = "../models/DialoGPT-small/"
    # BERT_PATH = "../models/roberta-base/"
    BERT_PATH = "../input/robertabaseconf/"
    # BERT_PATH = '../models/deepset-roberta-base-squad2/'
    MODEL_PATH = "model.bin"
    # TRAINING_FILE = "../../input/train_5folds.csv"
    # TRAINING_FILE = '../../input/train_5folds_weight_0.717.csv'
    TRAINING_FILE = '../../input/train_5folds_weight_v1.20.1_v1.54.1.csv'
    # TRAINING_FILE = '../../input/train_5folds_weight_v1.48.csv'
    # TRAINING_FILE = '../../input/train_5folds_weight_v1.20.1_v1.54.1_v1.74.1.csv'
    # TRAINING_FILE = '../../input/train_5folds_weight_v1.20.1_v1.54.1_v1.74.1_modify.csv'

    # TRAINING_FILE = "../../input/train_5folds_fr_de_ru.csv"
    # TRAINING_FILE = "../../input/train_5folds_yinhao.csv"
    # TRAINING_FILE = "../../input/train_5folds_space_mod_yinhao.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{BERT_PATH}/vocab.json",
        merges_file=f"{BERT_PATH}/merges.txt",
        lowercase=True,
        add_prefix_space=True
    )
    gradient_accumulation_steps = 1
    # num_warmup_steps = 0
    num_warmup_steps = 100
    max_answer_length = 30
    # max_answer_length = 1200
    # PSEUDO_FILE = "../../input/v1.74.1_pseudo.csv"
    PSEUDO_FILE = "../../input/sent140_pseudo_from_train_top1_score.csv"
    # PSEUDO_FILE = None
    # PSEUDO_FILE = '../../input/sent140_pseudo_from_train_top1.csv'


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.best_start = None
        self.best_end = None

    def __call__(self, epoch_score, model, model_path, start_oof=None, end_oof=None):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.best_start = start_oof
            self.best_end = end_oof
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_start = start_oof
            self.best_end = end_oof
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            # torch.save(model.state_dict(), model_path)
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), model_path)
        self.val_score = epoch_score


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard_score(
        original_tweet,
        target_string,
        sentiment_val,
        idx_start,
        idx_end,
        offsets,
        verbose=False):
    if idx_end < idx_start:
        filtered_output = original_tweet
    else:
        enc = config.TOKENIZER.encode(original_tweet)
        filtered_output = config.TOKENIZER.decode(enc.ids[idx_start - 1:idx_end])

    # if idx_end < idx_start:
    #     filtered_output = original_tweet
    # else:
    #     filtered_output = ""
    #     for ix in range(idx_start, idx_end + 1):
    #         filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
    #         if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
    #             filtered_output += " "

    # todo: 不确定要不要删掉
    # if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
    # # if len(original_tweet.split()) < 2:
    #     filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output.strip()


def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            probas_char[offset[0]:offset[1]] = preds[i]    
    
    return probas_char


def process_data(tweet, selected_text, sentiment, tokenizer, max_len, mode, textID):
    # FIND OVERLAP
    if mode == 'train':
        text1 = " " + " ".join(tweet.split())
        text2 = " ".join(selected_text.split())
        # 随机去掉一部分
        # if random.random() < 0.5:
        #     slice = text1.split(text2)
        #     l = ' '.join(list(random_sample(0.7, slice[0].split(), SEED)))
        #     r = ' '.join(list(random_sample(0.7, slice[-1].split(), SEED)))
        #     text1 = l + ' ' + text2 + ' ' + r
        # if random.random() < 0.5:
        #     text1 = text2

        # # 随机翻转
        # if random.random() < 0.7:
        #     text1 = " " + " ".join(tweet.split())
        #     text2 = " ".join(selected_text.split())
        # else:
        #     text1 = " " + " ".join(tweet.split()[::-1])
        #     text2 = " ".join(selected_text.split()[::-1])
    else:
        text1 = " " + " ".join(tweet.split())
        text2 = " ".join(selected_text.split())

    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx + len(text2)] = 1
    if text1[idx - 1] == ' ': chars[idx - 1] = 1
    enc = tokenizer.encode(text1)

    # ID_OFFSETS
    tweet_offsets = [];
    idx = 0
    for t in enc.ids:
        w = tokenizer.decode([t])
        tweet_offsets.append((idx, idx + len(w)))
        idx += len(w)

    # START END TOKENS
    toks = []
    for i, (a, b) in enumerate(tweet_offsets):
        sm = np.sum(chars[a:b])
        if sm > 0: toks.append(i)
    if len(toks) == 0:
        toks = [0]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }

    s_tok = sentiment_id[sentiment]
    input_ids = np.ones(config.MAX_LEN, dtype='int32')
    input_ids[:len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
    mask = np.zeros(config.MAX_LEN, dtype='int32')
    mask[:len(enc.ids) + 5] = 1
    # targets_start = np.zeros(config.MAX_LEN, dtype='int32')
    # targets_end = np.zeros(config.MAX_LEN, dtype='int32')
    # if len(toks) > 0:
    #     targets_start[toks[0] + 1] = 1
    #     targets_end[toks[-1] + 1] = 1
    token_type_ids = np.zeros(config.MAX_LEN, dtype='int32')

    padding_length = max_len - len(tweet_offsets)
    if padding_length > 0:
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
#         tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)] * (padding_length-1)

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': toks[0] + 1,
        'targets_end': toks[-1] + 1,
        'orig_tweet': text1,
        'orig_selected': text2,
        'sentiment': sentiment,
        'offsets': tweet_offsets,
        'textID': textID
    }


# class TweetDataset:
#     def __init__(self, tweet, sentiment, selected_text, mode='valid', weight=None, shorter_tweet=None):
#         self.tweet = tweet
#         self.sentiment = sentiment
#         self.selected_text = selected_text
#         self.tokenizer = config.TOKENIZER
#         self.max_len = config.MAX_LEN
#         assert mode in ['valid', 'train']
#         self.mode = mode
#         self.weight = weight
#         self.shorter_tweet = shorter_tweet
#
#
#     def __len__(self):
#         return len(self.tweet)
#
#     def __getitem__(self, item):
#
#         data = process_data(
#             self.tweet[item],
#             self.selected_text[item],
#             self.sentiment[item],
#             self.tokenizer,
#             self.max_len,
#             self.mode
#         )
#         if random.random() < 0.5 and self.shorter_tweet is not None:
#             data = process_data(
#                 self.shorter_tweet[item],
#                 self.selected_text[item],
#                 self.sentiment[item],
#                 self.tokenizer,
#                 self.max_len,
#                 self.mode
#             )
#
#         if self.weight is not None:
#             return {
#                 'ids': torch.tensor(data["ids"], dtype=torch.long),
#                 'mask': torch.tensor(data["mask"], dtype=torch.long),
#                 'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
#                 'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
#                 'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
#                 'orig_tweet': data["orig_tweet"],
#                 'orig_selected': data["orig_selected"],
#                 'sentiment': data["sentiment"],
#                 'offsets': torch.tensor(data["offsets"], dtype=torch.long),
#                 "weight": torch.tensor(self.weight[item], dtype=torch.float)
#             }
#         else:
#             return {
#                 'ids': torch.tensor(data["ids"], dtype=torch.long),
#                 'mask': torch.tensor(data["mask"], dtype=torch.long),
#                 'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
#                 'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
#                 'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
#                 'orig_tweet': data["orig_tweet"],
#                 'orig_selected': data["orig_selected"],
#                 'sentiment': data["sentiment"],
#                 'offsets': torch.tensor(data["offsets"], dtype=torch.long),
#             }


class TweetDataset:
    def __init__(self, data_l, weight=None):
        self.data_l = data_l
        self.weight = weight


    def __len__(self):
        return len(self.data_l)

    def __getitem__(self, item):

        data = self.data_l[item]

        if self.weight is not None:
            return {
                'ids': torch.tensor(data["ids"], dtype=torch.long),
                'mask': torch.tensor(data["mask"], dtype=torch.long),
                'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
                'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
                'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
                'orig_tweet': data["orig_tweet"],
                'orig_selected': data["orig_selected"],
                'sentiment': data["sentiment"],
                'offsets': torch.tensor(data["offsets"], dtype=torch.long),
                "weight": torch.tensor(self.weight[item], dtype=torch.float),
                'textID': data["textID"]
            }
        else:
            return {
                'ids': torch.tensor(data["ids"], dtype=torch.long),
                'mask': torch.tensor(data["mask"], dtype=torch.long),
                'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
                'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
                'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
                'orig_tweet': data["orig_tweet"],
                'orig_selected': data["orig_selected"],
                'sentiment': data["sentiment"],
                'offsets': torch.tensor(data["offsets"], dtype=torch.long),
                'textID': data["textID"]
            }


class CNN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(CNN_Layer, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size)

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        return torch.nn.functional.tanh(self.cnn(x))


class TweetModel0(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel0, self).__init__(conf)
        self.bert = transformers.RobertaModel(config=conf)
        self.drop_out = nn.Dropout(0.1)
        # self.l0 = nn.Linear(conf.hidden_size*2, 2)
        # torch.nn.init.normal_(self.l0.weight, std=0.02)

        # self.l0 = nn.Sequential(
        #     nn.Linear(conf.hidden_size * 2, 100),
        #     nn.Dropout(0.4),
        #     nn.ReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(100, 2),
        # )

        self.l0 = nn.Linear(128, 1)
        self.weights_init(self.l0)

        self.l1 = nn.Linear(128, 1)
        self.weights_init(self.l1)

        self.cnn1 = CNN_Layer(conf.hidden_size * 3, 128, 1)
        self.cnn1.init_weights()
        self.cnn2 = CNN_Layer(conf.hidden_size * 3, 128, 1)
        self.cnn2.init_weights()

    def truncated_normal_(self, tensor, mean=0.0, std=1.0):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            self.truncated_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, ids, mask, token_type_ids):
        _, __, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = out[-1]
        out = torch.cat(out[-3:], dim=-1)

        out = self.drop_out(out)
        # print(out.shape)
        start_logits = self.cnn1(out.permute(0, 2, 1))
        start_logits = self.l0(start_logits.permute(0, 2, 1))
        end_logits = self.cnn2(out.permute(0, 2, 1))
        end_logits = self.l1(end_logits.permute(0, 2, 1))

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.RobertaModel(config=conf)
        self.drop_out = nn.Dropout(0.1)  # ori
        # self.drop_out = nn.Dropout(0.3)

        # self.l0 = nn.Linear(conf.hidden_size*2, 2)
        # torch.nn.init.normal_(self.l0.weight, std=0.02)

        # self.l0 = nn.Sequential(
        #     nn.Linear(conf.hidden_size * 2, 100),
        #     nn.Dropout(0.4),
        #     nn.ReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(100, 2),
        # )

        self.l0 = nn.Linear(128, 1)
        # self.l0 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.Dropout(0.3, inplace=True),
        #     nn.LeakyReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(64, 1),
        # )
        self.weights_init(self.l0)

        self.l1 = nn.Linear(128, 1)
        # self.l1 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.Dropout(0.3, inplace=True),
        #     nn.LeakyReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(64, 1),
        # )
        self.weights_init(self.l1)

        # self.lstm1 = nn.LSTM(conf.hidden_size * 3, 64, 1,
        #                      bidirectional=True, batch_first=True, dropout=0.2)
        # self.lstm2 = nn.LSTM(conf.hidden_size * 3, 64, 1,
        #                      bidirectional=True, batch_first=True, dropout=0.2)

        self.lstm1 = nn.GRU(conf.hidden_size * 3, 64, 1,
                            bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.GRU(conf.hidden_size * 3, 64, 1,
                            bidirectional=True, batch_first=True, dropout=0.2)

    def truncated_normal_(self, tensor, mean=0.0, std=1.0):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            self.truncated_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, ids, mask, token_type_ids):
        _, __, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = out[-1]
        out = torch.cat(out[-3:], dim=-1)
        # out = torch.cat(out[-6:], dim=-1)
        # out = torch.cat([out[-1], out[-3], out[-5]], dim=-1)

        out = self.drop_out(out)
        # print(out.shape)
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        start_logits, _ = self.lstm1(out)
        start_logits = nn.Tanh()(start_logits)
        start_logits = self.l0(start_logits)
        end_logits, _ = self.lstm2(out)
        end_logits = nn.Tanh()(end_logits)
        end_logits = self.l1(end_logits)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
    
    
class TweetModelBart(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModelBart, self).__init__(conf)
        self.bert = transformers.BartModel(config=conf)
        # self.bert = BARTModel.from_pretrained('../models/bart-large-fairseq/', checkpoint_file='model.pt')
        self.drop_out = nn.Dropout(0.1)  # ori
        # self.drop_out = nn.Dropout(0.3)

        # self.l0 = nn.Linear(conf.hidden_size*2, 2)
        # torch.nn.init.normal_(self.l0.weight, std=0.02)

        # self.l0 = nn.Sequential(
        #     nn.Linear(conf.hidden_size * 2, 100),
        #     nn.Dropout(0.4),
        #     nn.ReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(100, 2),
        # )

        self.l0 = nn.Linear(128, 1)
        # self.l0 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.Dropout(0.3, inplace=True),
        #     nn.LeakyReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(64, 1),
        # )
        self.weights_init(self.l0)

        self.l1 = nn.Linear(128, 1)
        # self.l1 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.Dropout(0.3, inplace=True),
        #     nn.LeakyReLU(inplace=True),
        #     SyncBatchNorm(config.MAX_LEN),
        #     nn.Linear(64, 1),
        # )
        self.weights_init(self.l1)

        # self.lstm1 = nn.LSTM(conf.hidden_size * 3, 64, 1,
        #                      bidirectional=True, batch_first=True, dropout=0.2)
        # self.lstm2 = nn.LSTM(conf.hidden_size * 3, 64, 1,
        #                      bidirectional=True, batch_first=True, dropout=0.2)

        self.lstm1 = nn.GRU(1024, 64, 1,
                            bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.GRU(1024, 64, 1,
                            bidirectional=True, batch_first=True, dropout=0.2)

    def truncated_normal_(self, tensor, mean=0.0, std=1.0):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            self.truncated_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, ids):
        # outputs = self.bert.extract_features(ids, return_all_hiddens=True)
        outputs = self.bert(
            ids,
        )

        # print(len(outputs))
        # print(len(outputs[0]), len(outputs[1]), len(outputs[2]), len(outputs[3]))
        # print(outputs[1][0].shape)
        out = outputs[1][-1]
        # out = torch.cat(out[-6:], dim=-1)
        # out = torch.cat([outputs[1], outputs[2]], dim=-1)

        out = self.drop_out(out)
        # print(out.shape)
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        start_logits, _ = self.lstm1(out)
        start_logits = nn.Tanh()(start_logits)
        start_logits = self.l0(start_logits)
        end_logits, _ = self.lstm2(out)
        end_logits = nn.Tanh()(end_logits)
        end_logits = self.l1(end_logits)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits



# class TweetModel(transformers.BertPreTrainedModel):
#     def __init__(self, conf):
#         super(TweetModel, self).__init__(conf)
#         self.bert = transformers.RobertaModel.from_pretrained(config.BERT_PATH, config=conf)
#         self.drop_out = nn.Dropout(0.1)
#         # self.l0 = nn.Linear(conf.hidden_size*2, 2)
#         # torch.nn.init.normal_(self.l0.weight, std=0.02)
#
#         self.l0 = nn.Sequential(
#             nn.Linear(conf.hidden_size * 3, 100),
#             nn.Dropout(0.1),
#             nn.ReLU(inplace=True),
#             nn.Linear(100, 2),
#         )
#         self.weights_init(self.l0)
#
#     def truncated_normal_(self, tensor, mean=0.0, std=1.0):
#         size = tensor.shape
#         tmp = tensor.new_empty(size + (4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.data.mul_(std).add_(mean)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             self.truncated_normal_(m.weight, std=0.02)
#             nn.init.zeros_(m.bias)
#
#     def freeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True
#
#     def forward(self, ids, mask, token_type_ids):
#         _, __, out = self.bert(
#             ids,
#             attention_mask=mask,
#             token_type_ids=token_type_ids
#         )
#
#         # out = torch.cat((out[-1], out[-2]), dim=-1)
#         out = torch.cat(out[-3:], dim=-1)
#         out = self.drop_out(out)
#         logits = self.l0(out)
#
#         start_logits, end_logits = logits.split(1, dim=-1)
#
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         return start_logits, end_logits


# class TweetModel(transformers.BertPreTrainedModel):
#     def __init__(self, conf):
#         super(TweetModel, self).__init__(conf)
#         self.bert = transformers.RobertaModel.from_pretrained(config.BERT_PATH, config=conf)
#         self.drop_out = nn.Dropout(0.1)
#         # self.l0 = nn.Linear(conf.hidden_size*2, 2)
#         # torch.nn.init.normal_(self.l0.weight, std=0.02)
#
#         # self.l0 = nn.Sequential(
#         #     nn.Linear(conf.hidden_size * 2, 100),
#         #     nn.Dropout(0.4),
#         #     nn.ReLU(inplace=True),
#         #     SyncBatchNorm(config.MAX_LEN),
#         #     nn.Linear(100, 2),
#         # )
#
#         self.l0 = nn.Linear(128, 1)
#         # self.l0 = nn.Sequential(
#         #     nn.Linear(128, 64),
#         #     nn.Dropout(0.3, inplace=True),
#         #     nn.LeakyReLU(inplace=True),
#         #     SyncBatchNorm(config.MAX_LEN),
#         #     nn.Linear(64, 1),
#         # )
#         self.weights_init(self.l0)
#
#         self.l1 = nn.Linear(128, 1)
#         # self.l1 = nn.Sequential(
#         #     nn.Linear(128, 64),
#         #     nn.Dropout(0.3, inplace=True),
#         #     nn.LeakyReLU(inplace=True),
#         #     SyncBatchNorm(config.MAX_LEN),
#         #     nn.Linear(64, 1),
#         # )
#         self.weights_init(self.l1)
#
#         self.lstm1 = nn.LSTM(conf.hidden_size * 3, 64, 1,
#                              bidirectional=True, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.LSTM(conf.hidden_size * 3, 64, 1,
#                              bidirectional=True, batch_first=True, dropout=0.2)
#
#         # self.lstm1 = nn.GRU(conf.hidden_size*3, 64, 1,
#         #                     bidirectional=True, batch_first=True, dropout=0.2)
#         # self.lstm2 = nn.GRU(conf.hidden_size*3, 64, 1,
#         #                     bidirectional=True, batch_first=True, dropout=0.2)
#
#     def truncated_normal_(self, tensor, mean=0.0, std=1.0):
#         size = tensor.shape
#         tmp = tensor.new_empty(size + (4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.data.mul_(std).add_(mean)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             self.truncated_normal_(m.weight, std=0.02)
#             nn.init.zeros_(m.bias)
#
#     def freeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True
#
#     def forward(self, ids, mask, token_type_ids):
#         _, __, out = self.bert(
#             ids,
#             attention_mask=mask,
#             token_type_ids=token_type_ids
#         )
#
#         # out = out[-1]
#         out = torch.cat(out[-3:], dim=-1)
#
#         out = self.drop_out(out)
#         # print(out.shape)
#         self.lstm1.flatten_parameters()
#         self.lstm2.flatten_parameters()
#         start_logits, _ = self.lstm1(out)
#         start_logits = nn.Tanh()(start_logits)
#         start_logits = self.l0(start_logits)
#         end_logits, _ = self.lstm2(out)
#         end_logits = nn.Tanh()(end_logits)
#         end_logits = self.l1(end_logits)
#
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         return start_logits, end_logits


# class TweetModel(transformers.BertPreTrainedModel):
#     def __init__(self, conf):
#         super(TweetModel, self).__init__(conf)
#         self.bert = transformers.RobertaModel.from_pretrained(config.BERT_PATH, config=conf)
#         self.drop_out = nn.Dropout(0.1)
#         # self.l0 = nn.Linear(conf.hidden_size*2, 2)
#         # torch.nn.init.normal_(self.l0.weight, std=0.02)
#
#         # self.l0 = nn.Sequential(
#         #     nn.Linear(conf.hidden_size * 2, 100),
#         #     nn.Dropout(0.4),
#         #     nn.ReLU(inplace=True),
#         #     SyncBatchNorm(config.MAX_LEN),
#         #     nn.Linear(100, 2),
#         # )
#
#         self.l0 = nn.Linear(128*2, 1)
#         self.weights_init(self.l0)
#
#         self.l1 = nn.Linear(128*2, 1)
#         self.weights_init(self.l1)
#
#         self.cnn1 = CNN_Layer(conf.hidden_size * 3, 128, 1)
#         self.cnn1.init_weights()
#         self.cnn2 = CNN_Layer(conf.hidden_size * 3, 128, 1)
#         self.cnn2.init_weights()
#
#         self.lstm1 = nn.GRU(conf.hidden_size * 3, 64, 1,
#                                     bidirectional=True, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.GRU(conf.hidden_size * 3, 64, 1,
#                             bidirectional=True, batch_first=True, dropout=0.2)
#
#     def truncated_normal_(self, tensor, mean=0.0, std=1.0):
#         size = tensor.shape
#         tmp = tensor.new_empty(size + (4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.data.mul_(std).add_(mean)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             self.truncated_normal_(m.weight, std=0.02)
#             nn.init.zeros_(m.bias)
#
#     def freeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True
#
#     def forward(self, ids, mask, token_type_ids):
#         _, __, out = self.bert(
#             ids,
#             attention_mask=mask,
#             token_type_ids=token_type_ids
#         )
#
#         # out = out[-1]
#         out = torch.cat(out[-3:], dim=-1)
#
#         out = self.drop_out(out)
#
#         self.lstm1.flatten_parameters()
#         self.lstm2.flatten_parameters()
#         lstm_start_logits, _ = self.lstm1(out)
#         lstm_start_logits = nn.Tanh()(lstm_start_logits)
#         lstm_end_logits, _ = self.lstm2(out)
#         lstm_end_logits = nn.Tanh()(lstm_end_logits)
#
#
#         cnn_start_logits = self.cnn1(out.permute(0, 2, 1)).permute(0, 2, 1)
#         cnn_end_logits = self.cnn2(out.permute(0, 2, 1)).permute(0, 2, 1)
#
#         start_logits = torch.cat([lstm_start_logits, cnn_start_logits], 2)
#         start_logits = self.l0(start_logits)
#         end_logits = torch.cat([lstm_end_logits, cnn_end_logits], 2)
#         end_logits = self.l0(end_logits)
#
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         return start_logits, end_logits


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def dist_between(start_logits, end_logits, device='cpu', max_seq_len=128):
    """get dist btw. pred & ground_truth"""

    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False, dtype=torch.float32)
    linear_func = linear_func.to(device)

    start_pos = (start_logits * linear_func).sum(axis=1)
    end_pos = (end_logits * linear_func).sum(axis=1)

    diff = end_pos - start_pos

    return diff.sum(axis=0) / diff.size(0)


def dist_loss(start_logits, end_logits, start_positions, end_positions, device='cpu', max_seq_len=128, scale=1):
    """calculate distance loss between prediction's length & GT's length

    Input
    - start_logits ; shape (batch, max_seq_len{128})
        - logits for start index
    - end_logits
        - logits for end index
    - start_positions ; shape (batch, 1)
        - start index for GT
    - end_positions
        - end index for GT
    """
    start_logits = torch.nn.Softmax(1)(start_logits)  # shape ; (batch, max_seq_len)
    end_logits = torch.nn.Softmax(1)(end_logits)

    start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_seq_len).to(device)
    end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_seq_len).to(device)

    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len)
    gt_dist = dist_between(start_one_hot, end_one_hot, device, max_seq_len)  # always positive
    diff = (gt_dist - pred_dist)

    rev_diff_squared = 1 - torch.sqrt(diff * diff)  # as diff is smaller, make it get closer to the one
    loss = -torch.log(
        rev_diff_squared)  # by using negative log function, if argument is near zero -> inifinite, near one -> zero

    return loss * scale


def ohem_loss(cls_pred, cls_target, rate=0.7):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def loss_fn(start_logits, end_logits, start_positions, end_positions, weight=None):
    if weight is not None:
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        start_loss = loss_fct(start_logits, start_positions)
        # print(start_loss.shape)
        # start_loss = start_loss*weight
        start_loss = (start_loss * weight / weight.sum()).sum()
        start_loss = start_loss.mean()
        end_loss = loss_fct(end_logits, end_positions)
        # end_loss = end_loss * weight
        end_loss = (end_loss * weight / weight.sum()).sum()
        end_loss = end_loss.mean()
    else:
        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = LabelSmoothingLoss(config.MAX_LEN, smoothing=0.1)
        # loss_fct = SmoothCrossEntropyLoss(smoothing=0.9)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        # start_loss = ohem_loss(start_logits, start_positions)
        # end_loss = ohem_loss(end_logits, end_positions)

    total_loss = (start_loss + end_loss)
    # total_loss = distloss
    # total_loss = (start_loss + end_loss)
    return total_loss


# start :position < target position的惩罚大
# end :position > target position的惩罚大
def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
    # neg_weight for when pred position < target position
    # pos_weight for when pred position > target position
    gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
    gap = gap.type(torch.float32)
    return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)


# def loss_fn(start_logits, end_logits, start_positions, end_positions):
#     loss_fct = nn.CrossEntropyLoss(reduce='none')  # do reduction later
#
#     start_loss = loss_fct(start_logits, start_positions) * pos_weight(start_logits, start_positions, 4, 1)
#     end_loss = loss_fct(end_logits, end_positions) * pos_weight(end_logits, end_positions, 1, 4)
#
#     start_loss = torch.mean(start_loss)
#     end_loss = torch.mean(end_loss)
#
#     total_loss = (start_loss + end_loss)
#     return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None, opt=None, max_ans_length=config.max_answer_length, epc=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    # fgm = FGM(model)
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]
        # weight = d['weight']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        # weight = weight.to(device, dtype=torch.float)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, None)
        loss = loss.mean()
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (bi + 1) % config.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                ans_pair = []
                # if tweet_sentiment == 'neutral':
                #     m_l = 1200
                # else:
                #     m_l = 30
                for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                    for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        # if length > max_ans_length:
                        #     continue
                        ans_pair.append((start_index, end_index))
                if len(ans_pair) == 0:
                    ans_pair.append((-1, 0))
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=ans_pair[0][0],
                    idx_end=ans_pair[0][1],
                    # idx_start=np.argmax(outputs_start[px, :]),
                    # idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
            # if epc > 7 and random.random() < 0.3:
            #     optimizer.update_swa()


def eval_fn(data_loader, model, device, max_ans_length=config.max_answer_length):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    start_array, end_array = [], []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            # start_array.append(outputs_start.cpu().detach().numpy())
            # end_array.append(outputs_end.cpu().detach().numpy())
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            start_array.append(outputs_start)
            end_array.append(outputs_end)

            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                # if tweet_sentiment == 'neutral':
                #     m_l = 1200
                # else:
                #     m_l = 30
                ans_pair = []
                # start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                # end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                    for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        # if length > max_ans_length:
                        #     continue
                        ans_pair.append((start_index, end_index))
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=ans_pair[0][0],
                    idx_end=ans_pair[0][1],
                    # idx_start=np.argmax(outputs_start[px, :]),
                    # idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    start_array = np.concatenate(start_array)
    end_array = np.concatenate(end_array)
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg, start_array, end_array


import math
def get_my_schedule_with_warmup(optimizer, num_warmup_steps, num_linear_steps, num_training_steps, num_cycles, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif num_warmup_steps < current_step <= num_linear_steps:
            return max(
                0.0, float(num_linear_steps - current_step) / float(max(1, num_linear_steps - num_warmup_steps))
            )
        else:
#             return 0.0001
            progress = float(current_step - num_linear_steps) / float(max(1, num_training_steps - num_linear_steps))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))) *0.1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def run(fold, dfx, tr, val, freeze_weight=None):
    df_train = dfx.iloc[tr]
    df_valid = dfx.iloc[val]
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    # # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=777)
    # dfx = pd.read_csv(config.TRAINING_FILE)
    #
    # for i, (tr, val) in enumerate(skf.split(dfx, dfx.sentiment.values)):
    #     if i == fold:
    #         df_train = dfx.iloc[tr]
    #         df_valid = dfx.iloc[val]

    # if config.LANG == 'fr':
    #     df_trans = pd.read_csv('../../input/tweet-train-for-pseudo-google-fr-check.csv')
    #     df_train = df_trans.iloc[tr]
    #     df_train = df_train[df_train['check']==True]
    # elif config.LANG == 'de':
    #     df_trans = pd.read_csv('../../input/tweet-train-for-pseudo-google-de-check.csv')
    #     df_train = df_trans.iloc[tr]
    #     df_train = df_train[df_train['check'] == True]
    # elif config.LANG == 'ru':
    #     df_trans = pd.read_csv('../../input/tweet-train-for-pseudo-google-ru-check.csv')
    #     df_train = df_trans.iloc[tr]
    #     df_train = df_train[df_train['check'] == True]

    if config.PSEUDO_FILE:
        dfx_pseudo = pd.read_csv(config.PSEUDO_FILE)
        for i, (tr, val) in enumerate(skf.split(dfx_pseudo, dfx_pseudo.sentiment.values)):
            if i == fold:
                dfx_pseudo_train = dfx_pseudo.iloc[tr]
        # dfx_pseudo_train.score += 0.3
        df_train = pd.concat([df_train, dfx_pseudo_train])
        # pseudo_train_dataset = TweetDataset(
        #     tweet=dfx_pseudo_train.text.values,
        #     sentiment=dfx_pseudo_train.sentiment.values,
        #     selected_text=dfx_pseudo_train.selected_text.values,
        #     mode='train',
        #     weight=np.zeros(len(dfx_pseudo_train))
        # )
        #
        # pseudo_train_data_loader = torch.utils.data.DataLoader(
        #     pseudo_train_dataset,
        #     sampler=RandomSampler(pseudo_train_dataset),
        #     batch_size=config.TRAIN_BATCH_SIZE,
        #     num_workers=4
        # )

    set_seed(SEED)

    tr_l = []
    for i in tqdm(range(len(df_train))):
        line = df_train.iloc[i]
        tr_l += [process_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train')]
    val_l = []
    for i in tqdm(range(len(df_valid))):
        line = df_valid.iloc[i]
        val_l += [
            process_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train')]

    # train_dataset = TweetDataset(
    #     tweet=df_train.text.values,
    #     sentiment=df_train.sentiment.values,
    #     selected_text=df_train.selected_text.values,
    #     mode='train',
    #     weight=df_train.score.values,
    #     # weight=df_train['v1.74.1_score'].values,
    #     # shorter_tweet = df_train.shorter_text.astype(str).values,
    # )

    train_dataset = TweetDataset(
        data_l=tr_l
    )

    # train_sampler = RandomSampler(train_dataset)
    weights = torch.FloatTensor(df_train.score.values)
    # weights = torch.FloatTensor(df_train['score_modify_v1'].values)
    train_sampler = WeightedRandomSampler(weights, num_samples=int(len(df_train) * 0.6), replacement=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    # valid_dataset = TweetDataset(
    #     tweet=df_valid.text.values,
    #     sentiment=df_valid.sentiment.values,
    #     selected_text=df_valid.selected_text.values,
    #     weight=df_train.score.values
    # )

    valid_dataset = TweetDataset(
        data_l=val_l
    )

    eval_sampler = SequentialSampler(valid_dataset)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=eval_sampler,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)
    if freeze_weight:
        model.load_state_dict(torch.load(freeze_weight))

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE // config.gradient_accumulation_steps * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LR)

    scheduler = None
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_steps // 20,
        # num_warmup_steps=50,
        num_training_steps=num_train_steps
    )
    # scheduler = get_my_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_train_steps // 20,
    #     num_linear_steps=int(len(df_train) / config.TRAIN_BATCH_SIZE // config.gradient_accumulation_steps * 7),
    #     num_training_steps=num_train_steps,
    #     num_cycles=6
    # )
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=num_train_steps // 20,
    #                                            num_training_steps=num_train_steps,
    #                                            num_cycles=0.5
    #                                            )

    # swa
    # optimizer = SWA(optimizer)
    # optimizer = SWA(optimizer, swa_start=num_train_steps // 20, swa_freq=5, swa_lr=1e-5)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=num_train_steps // 20,
    #                                            num_training_steps=num_train_steps,
    #                                            num_cycles=3.0
    #                                            )

    # freeze_opt = AdamW(optimizer_parameters, lr=0.001)
    # model.freeze()
    # for epoch in range(3):
    #     train_fn_freeze(train_data_loader, model, freeze_opt, device)
    # model.unfreeze()

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = torch.nn.DataParallel(model)

    es = EarlyStopping(patience=3, mode="max", delta=0.0005)
    print(f"Training is Starting for fold={fold}")

    set_seed(SEED)

    for epoch in range(config.EPOCHS):

        if epoch > 5:
            print('add bsz')
            # train_dataset = TweetDataset(
            #     tweet=df_train.text.values,
            #     # tweet=df_train.shorter_text.astype(str).values,
            #     sentiment=df_train.sentiment.values,
            #     selected_text=df_train.selected_text.values,
            #     mode='train',
            #     weight=df_train.score.values
            # )
            #
            # # train_sampler = RandomSampler(train_dataset)
            # # weights = torch.FloatTensor(df_train.score.values)
            # weights = torch.FloatTensor(df_train['v1.74.1_score'].values)

            train_sampler = WeightedRandomSampler(weights, num_samples=int(len(df_train) * 0.6), replacement=True)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=config.TRAIN_BATCH_SIZE + (epoch - 5) * 30,
                num_workers=4
            )

        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler, epc=epoch)

        # if epoch > 7:
        #     optimizer.swap_swa_sgd()
        jaccard, startoof, endoof = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"output/ckpt_{ver[-1][0]}_{fold}.bin", start_oof=startoof, end_oof=endoof)
        # if epoch > 7:
        #     optimizer.swap_swa_sgd()
        if es.early_stop:
            print("Early stopping")
            break
    del model, optimizer
    gc.collect()
    return es.best_score, len(valid_dataset), es.best_start, es.best_end


# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=777)
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# dfx = pd.read_csv(config.TRAINING_FILE)
# start_oof = np.zeros((len(dfx), config.MAX_LEN))
# end_oof = np.zeros((len(dfx), config.MAX_LEN))
# val_score = 0
# for i, (tr, val) in enumerate(skf.split(dfx, dfx.sentiment.values)):
# #     if i not in [4]:
# #         continue
#     score_fold, cnt, start_logits_fold, end_logits_fold = run(i, dfx, tr, val)
#     val_score += score_fold
#     start_oof[val, :] = start_logits_fold
#     end_oof[val, :] = end_logits_fold

# np.save(f"./oof/oof_start_logits_{ver[-1][0]}.npy", start_oof)
# np.save(f"./oof/oof_end_logits_{ver[-1][0]}.npy", end_oof)
# val_score = val_score / 5.
# print(f'5fold mean Jaccard Score = {val_score}')

# # 5fold infer
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
# dfx = pd.read_csv(config.TRAINING_FILE)
# # dfx = dfx[dfx.sentiment=='neutral']
# ans = []
# nbest = []
# char_pref_oof_start = []
# char_pref_oof_end = []
# char_pref_oof_ids = []
# for i, (tr, val) in enumerate(skf.split(dfx, dfx.sentiment.values)):
#     nbest_fold = []
#     set_seed(SEED)
#     device = torch.device("cuda")
#     config.BERT_PATH = "../../bart/models/bart-large/"
#     model_config = transformers.BartConfig.from_pretrained(config.BERT_PATH)
#     model_config.output_hidden_states = True
#     model1 = TweetModelBart(conf=model_config)
#     model1.to(device)
#     model1.load_state_dict(torch.load(f"../../bart/bart_large/output//ckpt_v1.1.1_{i}.bin"))
#     model1.eval()

#     oof = []
#     df_valid = dfx.iloc[val].copy()
#     # valid_dataset = TweetDataset(
#     #     tweet=df_valid.text.values,
#     #     sentiment=df_valid.sentiment.values,
#     #     selected_text=df_valid.selected_text.values
#     # )

#     val_l = []
#     for i in tqdm(range(len(df_valid))):
#         line = df_valid.iloc[i]
#         val_l += [
#             process_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train', line.textID)]
#     valid_dataset = TweetDataset(
#         data_l=val_l
#     )

#     eval_sampler = SequentialSampler(valid_dataset)
#     valid_data_loader = torch.utils.data.DataLoader(
#         valid_dataset,
#         sampler=eval_sampler,
#         batch_size=config.VALID_BATCH_SIZE*3,
#         num_workers=2
#     )

#     with torch.no_grad():
#         tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))
#         for bi, d in enumerate(tk0):
#             ids = d["ids"]
#             token_type_ids = d["token_type_ids"]
#             mask = d["mask"]
#             sentiment = d["sentiment"]
#             orig_selected = d["orig_selected"]
#             orig_tweet = d["orig_tweet"]
#             targets_start = d["targets_start"]
#             targets_end = d["targets_end"]
#             offsets = d["offsets"].numpy()
#             textIDs = d["textID"]

#             ids = ids.to(device, dtype=torch.long)
#             token_type_ids = token_type_ids.to(device, dtype=torch.long)
#             mask = mask.to(device, dtype=torch.long)
#             targets_start = targets_start.to(device, dtype=torch.long)
#             targets_end = targets_end.to(device, dtype=torch.long)

#             outputs_start1, outputs_end1 = model1(
#                 ids=ids,
# #                 mask=mask,
# #                 token_type_ids=token_type_ids
#             )

#             outputs_start = outputs_start1
#             outputs_end = outputs_end1

#             loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
#             outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
#             outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

#             jaccard_scores = []
#             for px, tweet in enumerate(orig_tweet):
#                 selected_tweet = orig_selected[px]
#                 tweet_sentiment = sentiment[px]
#                 tid = textIDs[px]
#                 # if tweet_sentiment == 'neutral':
#                 #     m_l = 1200
#                 # else:
#                 #     m_l = 30
#                 ans_pair = []
#                 top5 = []
#                 # start_indexes = _get_best_indexes(result.start_logits, n_best_size)
#                 # end_indexes = _get_best_indexes(result.end_logits, n_best_size)
#                 for start_index in np.argsort(outputs_start[px, :])[::-1][:5]:
#                     for end_index in np.argsort(outputs_end[px, :])[::-1][:5]:
#                         if end_index < start_index:
#                             continue
#                         length = end_index - start_index + 1
#                         # if tweet_sentiment!='neutral' and length > 15:
#                         #     continue
#                         ans_pair.append((start_index, end_index))
#                         ans_pair.append((start_index, end_index))
#                         score = outputs_start[px, :][start_index] + outputs_end[px, :][end_index]
#                         _, output_sentence = calculate_jaccard_score(
#                             original_tweet=tweet,
#                             target_string=selected_tweet,
#                             sentiment_val=tweet_sentiment,
#                             idx_start=start_index,
#                             idx_end=end_index,
#                             offsets=offsets[px]
#                         )

#                         top5.append((output_sentence, score))
#                 nbest_fold.append(top5)

#                 jaccard_score, _ = calculate_jaccard_score(
#                     original_tweet=tweet,
#                     target_string=selected_tweet,
#                     sentiment_val=tweet_sentiment,
#                     idx_start=ans_pair[0][0],
#                     idx_end=ans_pair[0][1],
#                     # idx_start=np.argmax(outputs_start[px, :]),
#                     # idx_end=np.argmax(outputs_end[px, :]),
#                     offsets=offsets[px],
#                     #                     logits_start=outputs_start[px, :],
#                     #                     logits_end=outputs_end[px, :]
#                 )
#                 oof.append(_)
#                 char_pref_oof_start.append(token_level_to_char_level(tweet, offsets[px], outputs_start[px, 1:]))
#                 char_pref_oof_end.append(token_level_to_char_level(tweet, offsets[px], outputs_end[px, 1:]))
#                 char_pref_oof_ids.append(textIDs[px])

#                 # topk_output = []
#                 # for i in range(2):
#                 #     jaccard_score, _ = calculate_jaccard_score(
#                 #         original_tweet=tweet,
#                 #         target_string=selected_tweet,
#                 #         sentiment_val=tweet_sentiment,
#                 #         idx_start=ans_pair[i][0],
#                 #         idx_end=ans_pair[i][1],
#                 #         # idx_start=np.argmax(outputs_start[px, :]),
#                 #         # idx_end=np.argmax(outputs_end[px, :]),
#                 #         offsets=offsets[px]
#                 #     )
#                 #     topk_output.append(_)
#                 # if tweet_sentiment!='neutral':
#                 #     oof.append(' '.join(topk_output))
#                 # else:
#                 #     oof.append(topk_output[0])
#                 jaccard_scores.append(jaccard_score)
#     for id, topk in zip(df_valid.textID.values, nbest_fold):
#         nbest.append({id: topk})
#     df_valid['pred'] = oof
#     ans.append(df_valid)
# df_oof = pd.concat(ans)
# df_oof.to_csv(f"oof_roberta_{ver[-1][0]}.csv", index=False)
# with open("oof_roberta-nbest-{}.pkl".format(ver[-1][0]), 'wb') as f:
#     pickle.dump(nbest, f)
# with open(f"bart-char_pref_oof_start.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_oof_start, fp)
    
# with open(f"bart-char_pref_oof_end.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_oof_end, fp)

# with open(f"bart-char_pref_oof_ids.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_oof_ids, fp)

    
char_pref_pred_start = []
char_pref_pred_end = []
char_pref_pred_ids = []
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
# df_test = pd.read_csv("../../input/preivate_sent140_pseudo.csv")
# df_test = pd.read_csv("../../input/Sentiment140_filter_v1.csv")
# 引号
# df_test['text'] = df_test['text'].apply(lambda x: x.replace('`', "'"))

df_test.loc[:, "selected_text"] = df_test.text.values

device = torch.device("cuda")
config.BERT_PATH = "../input/robertabaseconf/"
model_config = transformers.RobertaConfig.from_pretrained(config.BERT_PATH)
model_config.output_hidden_states = True
model1 = TweetModel0(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load(f"../input/hi-checkpoints/ckpt_v1.20.1_0.bin"))
model1.eval()

model2 = TweetModel0(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load(f"../input/hi-checkpoints/ckpt_v1.20.1_1.bin"))
model2.eval()

model3 = TweetModel0(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load(f"../input/hi-checkpoints/ckpt_v1.20.1_2.bin"))
model3.eval()

model4 = TweetModel0(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load(f"../input/hi-checkpoints/ckpt_v1.20.1_3.bin"))
model4.eval()

model5 = TweetModel0(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load(f"../input/hi-checkpoints/ckpt_v1.20.1_4.bin"))
model5.eval()


# model_l = []
# for ckpt in os.listdir('output'):
#     if '103' in ckpt or '74' in ckpt:
#         model5 = TweetModel(conf=model_config)
#         model5.to(device)
#         model5.load_state_dict(torch.load(f"output/{ckpt}"))
#         model5.eval()
#         model_l.append(model5)


final_output = []

val_l = []
for i in tqdm(range(len(df_test))):
    line = df_test.iloc[i]
    val_l += [
        process_data(line.text, line.selected_text, line.sentiment, config.TOKENIZER, config.MAX_LEN, 'train', line.textID)]
test_dataset = TweetDataset(
    data_l=val_l
)
# test_dataset = TweetDataset(
#     tweet=df_test.text.values,
#     sentiment=df_test.sentiment.values,
#     selected_text=df_test.selected_text.values
# )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=2
)

nbest = []
with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"].numpy()
        textIDs = d["textID"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )


        # simple average
#         outputs_start = (outputs_start1*2 + outputs_start2*2 + outputs_start3*2 + outputs_start4*2 + outputs_start5*2 +
#                         outputs_start6 + outputs_start7 + outputs_start8 + outputs_start9 + outputs_start10 +
#                         outputs_start11 + outputs_start12 + outputs_start13 + outputs_start14 + outputs_start15 +
#                         outputs_start16 + outputs_start17 + outputs_start18 + outputs_start19 + outputs_start20) / 25
#         outputs_end = (outputs_end1*2 + outputs_end2*2 + outputs_end3*2 + outputs_end4*2 + outputs_end5*2 +
#                       outputs_end6 + outputs_end7 + outputs_end8 + outputs_end9 + outputs_end10+
#                       outputs_end11 + outputs_end12 + outputs_end13 + outputs_end14 + outputs_end15+
#                       outputs_end16 + outputs_end17 + outputs_end18 + outputs_end19 + outputs_end20) / 25
        outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5
#         outputs_start = torch.zeros_like(ids, dtype=torch.float)
#         outputs_end = torch.zeros_like(ids, dtype=torch.float)
#         for m in model_l:
#             outputs_start5, outputs_end5 = model5(
#                 ids=ids,
#                 mask=mask,
#                 token_type_ids=token_type_ids
#             )
#             outputs_start += outputs_start5
#             outputs_end += outputs_end5
#         outputs_start /= len(model_l)
#         outputs_end /= len(model_l)
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]

            ans_pair = []
            top5 = []
            # start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            # end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in np.argsort(outputs_start[px, :])[::-1][:5]:
                for end_index in np.argsort(outputs_end[px, :])[::-1][:5]:
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    # if tweet_sentiment!='neutral' and length > 15:
                    #     continue
                    ans_pair.append((start_index, end_index))
                    score = outputs_start[px, :][start_index] + outputs_end[px, :][end_index]
                    _, output_sentence = calculate_jaccard_score(
                        original_tweet=tweet,
                        target_string=selected_tweet,
                        sentiment_val=tweet_sentiment,
                        idx_start=start_index,
                        idx_end=end_index,
                        offsets=offsets[px]
                    )

                    top5.append((output_sentence, score))
            nbest.append(top5)
            
            char_pref_pred_start.append(token_level_to_char_level(tweet, offsets[px], outputs_start[px, 1:]))
            char_pref_pred_end.append(token_level_to_char_level(tweet, offsets[px], outputs_end[px, 1:]))
            char_pref_pred_ids.append(textIDs[px])


            _, output_sentence = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                # idx_start=np.argmax(outputs_start[px, :]),
                # idx_end=np.argmax(outputs_end[px, :]),
                idx_start=ans_pair[0][0],
                idx_end=ans_pair[0][1],
                offsets=offsets[px]
            )
            final_output.append(output_sentence)

# with open("roberta-nbest-{}.pkl".format(ver[-1][0]), 'wb') as f:
#     pickle.dump(nbest, f)
    
# with open(f"roberta-char_pref_test_start.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_pred_start, fp)
    
# with open(f"roberta-char_pref_test_end.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_pred_end, fp)

# with open(f"roberta-char_pref_test_ids.pkl", "wb") as fp:   #Pickling
#     pickle.dump(char_pref_pred_ids, fp)



with open(f"hiki_roberta-char_pred_test_start.pkl", "wb") as fp:   #Pickling
    pickle.dump(char_pref_pred_start, fp)
    
with open(f"hiki_roberta-char_pred_test_end.pkl", "wb") as fp:   #Pickling
    pickle.dump(char_pref_pred_end, fp)