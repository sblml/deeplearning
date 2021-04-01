import re
import os
import pickle
import random
import sklearn
import warnings
import datetime
import tokenizers
import numpy as np
import pandas as pd
import transformers
import seaborn as sns
import matplotlib.pyplot as plt

from tokenizers import *
from datetime import date
from transformers import *
from itertools import product
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


SEED = 42
K = 5

DATA_PATH = '../input/tweet-sentiment-extraction/'

MODEL_PATHS = {
    'bert-base-uncased': '../input/bertconfigs/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/',
    'bert-large-uncased-whole-word-masking-finetuned-squad': '../input/bertconfigs/wwm_uncased_L-24_H-1024_A-16/wwm_uncased_L-24_H-1024_A-16/',
    'albert-large-v2': '../input/albert-configs/albert-large-v2/albert-large-v2/',
    'albert-base-v2': '../input/albert-configs/albert-base-v2/albert-base-v2/',
    'distilbert': '../input/albert-configs/distilbert/distilbert/',
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_weights(model, filename, verbose=1, cp_folder=""):
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model

def trim_tensors(tokens, input_ids, model_name='bert', min_len=10):
    pad_token = 1 if "roberta" in model_name else 0
    max_len = max(torch.max(torch.sum((tokens != pad_token), 1)), min_len)
    return tokens[:, :max_len], input_ids[:, :max_len]



import os
import sys
import sentencepiece as spm

sys.path.insert(0, "../input/sentencepiece-pb2/")
import sentencepiece_pb2


class EncodedText:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets

        
class SentencePieceTokenizer:
    def __init__(self, model_path, lowercase=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path))
        self.lowercase = lowercase
    
    def encode(self, sentence):
        if self.lowercase:
            sentence = sentence.lower()
            
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return EncodedText(tokens, offsets)


def create_tokenizer_and_tokens(config):
    if "albert" in config.selected_model:
        tokenizer = SentencePieceTokenizer(f'{MODEL_PATHS[config.selected_model]}/{config.selected_model}-spiece.model')
        
        tokens = {
            'cls': 2,
            'sep': 3,
            'pad': 0,
        }
        
    else:
        tokenizer = BertWordPieceTokenizer(
            MODEL_PATHS[config.selected_model] + 'vocab.txt',
            lowercase=config.lowercase,
#             add_special_tokens=False  # This doesn't work smh
        )

        tokens = {
            'cls': tokenizer.token_to_id('[CLS]'),
            'sep': tokenizer.token_to_id('[SEP]'),
            'pad': tokenizer.token_to_id('[PAD]'),
        }

    for sentiment in ['positive', 'negative', 'neutral']:
        ids = tokenizer.encode(sentiment).ids
        tokens[sentiment] = ids[0] if ids[0] != tokens['cls'] else ids[1]
    
    return tokenizer, tokens


def process_test_data(text, sentiment, tokenizer, tokens, max_len=100, model_name="bert", use_old_sentiment=False):
    text = " " + " ".join(str(text).split())
    
    tokenized = tokenizer.encode(text)
    input_ids_text = tokenized.ids
    text_offsets = tokenized.offsets
    
    if input_ids_text[0] == tokens["cls"]: # getting rid of special tokens
        input_ids_text = input_ids_text[1:-1] 
        text_offsets = text_offsets[1:-1]

    if use_old_sentiment:
        new_max_len = max_len - 5
        input_ids = (
            [tokens["cls"], tokens[sentiment], tokens["neutral"], tokens["sep"]]
            + input_ids_text[:new_max_len]
            + [tokens["sep"]]
        )
        token_type_ids = [0, 0, 0, 0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
        text_offsets = [(0, 0)] * 4 + text_offsets[:new_max_len] + [(0, 0)]

    else:
        new_max_len = max_len - 4
        input_ids = (
            [tokens["cls"], tokens[sentiment], tokens["sep"]]
            + input_ids_text[:new_max_len]
            + [tokens["sep"]]
        )
        token_type_ids = [0, 0, 0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
        text_offsets = [(0, 0)] * 3 + text_offsets[:new_max_len] + [(0, 0)]

    assert len(input_ids) == len(token_type_ids) and len(input_ids) == len(text_offsets), (len(input_ids), len(text_offsets))

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)

    return {
        "ids": input_ids,
        "token_type_ids": token_type_ids,
        "text": text,
        "sentiment": sentiment,
        "offsets": text_offsets,
    }


class TweetTestDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, max_len=200, model_name="bert", use_old_sentiment=False):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        self.use_old_sentiment = use_old_sentiment
        
        self.texts = df['text'].values
        self.sentiments = df['sentiment'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        data = process_test_data(self.texts[idx], self.sentiments[idx], self.tokenizer, self.tokens, 
                                 max_len=self.max_len, model_name=self.model_name, use_old_sentiment=self.use_old_sentiment)

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'text': data["text"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


TRANSFORMERS = {   
    'albert-base-v2': (AlbertModel, 'albert-base-v2', AlbertConfig),
    'albert-large-v2': (AlbertModel, 'albert-large-v2', AlbertConfig),
    "bert-base-uncased": (BertModel, "bert-base-uncased", BertConfig),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (BertModel, "bert-large-uncased-whole-word-masking-finetuned-squad", BertConfig),
    "distilbert": (DistilBertModel, "distilbert-base-uncased-distilled-squad", DistilBertConfig),
}


class TweetQATransformer(nn.Module):
    def __init__(self, model, nb_layers=1, nb_ft=None, sentiment_ft=0, multi_sample_dropout=False, use_squad_weights=True):
        super().__init__()
        self.name = model
        self.nb_layers = nb_layers
        self.multi_sample_dropout = multi_sample_dropout

        self.sentiment_ft = sentiment_ft

        self.pad_idx = 1 if "roberta" in self.name else 0
        
        model_class, _, config_class = TRANSFORMERS[model]
        
        try:
            config = config_class.from_json_file(MODEL_PATHS[model] + 'bert_config.json')
        except:
            config = config_class.from_json_file(MODEL_PATHS[model] + 'config.json')

        config.output_hidden_states = True

        self.transformer = model_class(config)

        if "distil" in self.name:
            self.nb_features = self.transformer.transformer.layer[-1].ffn.lin2.out_features
        elif "albert" in self.name:
            self.nb_features = self.transformer.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features
        else:
            self.nb_features = self.transformer.pooler.dense.out_features
        
        if nb_ft is None:
            nb_ft = self.nb_features

        self.logits = nn.Sequential(
            nn.Linear(self.nb_features * self.nb_layers, nb_ft),
            nn.Tanh(),
            nn.Linear(nb_ft, 2), 
        )

        self.high_dropout = nn.Dropout(p=0.5)
    
    def forward(self, tokens, token_type_ids, sentiment=0):
        if "distil" in self.name:
            hidden_states = self.transformer(
                tokens, 
                attention_mask=(tokens != self.pad_idx).long(),
            )[-1]
        else:
            hidden_states = self.transformer(
                tokens, 
                attention_mask=(tokens != self.pad_idx).long(),
                token_type_ids=token_type_ids,
            )[-1]

        hidden_states = hidden_states[::-1]

        features = torch.cat(hidden_states[:self.nb_layers], -1)
        
        if self.multi_sample_dropout and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits



def predict(model, dataset, batch_size=32, num_workers=1):
    model.eval()
    start_probas = []
    end_probas = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for data in loader:
            ids, token_type_ids = trim_tensors(
                data["ids"], data["token_type_ids"], model.name
            )
            
            start_logits, end_logits = model(
                ids.cuda(), token_type_ids.cuda()
            )

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for s, e in zip(start_probs, end_probs):
                start_probas.append(list(s))
                end_probas.append(list(e))

    return start_probas, end_probas


def k_fold_inference(config, test_dataset, seed=42):
    seed_everything(seed)
    pred_tests = [] 
    for weight in config.weights:         
        model = TweetQATransformer(
            config.selected_model, 
            nb_layers=config.nb_layers, 
            nb_ft=config.nb_ft,
            sentiment_ft=config.sentiment_ft,
            multi_sample_dropout=config.multi_sample_dropout,
        ).cuda()
        
        model = load_model_weights(model, weight, cp_folder=config.weights_path, verbose=1)
        model.zero_grad()
        
        pred_test = predict(model, test_dataset, batch_size=config.batch_size_val, num_workers=config.num_workers)
        pred_tests.append(pred_test)            
    
    return pred_tests


def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]: # remove padding and sentiment
            probas_char[offset[0]:offset[1]] = preds[i]
    
    return probas_char


def get_char_preds(pred_test, test_dataset):
    char_pred_test_start = []
    char_pred_test_end = []

    for idx in range(len(test_dataset)):
        d = test_dataset[idx]
        text = d['text']
        offsets = d['offsets']

        start_preds = np.mean([pred_test[i][0][idx] for i in range(len(pred_test))], 0)
        end_preds = np.mean([pred_test[i][1][idx] for i in range(len(pred_test))], 0)

        char_pred_test_start.append(token_level_to_char_level(text, offsets, start_preds))
        char_pred_test_end.append(token_level_to_char_level(text, offsets, end_preds))
        
    return char_pred_test_start, char_pred_test_end


class ConfigAlbert:
    # Architecture
    selected_model = "albert-large-v2"
    lowercase = True
    nb_layers = 8
    nb_ft = 128
    sentiment_ft = 0
    multi_sample_dropout = True
    use_old_sentiment = True

    # Inference
    batch_size_val = 32
    max_len_val = 100
    num_workers = 4
    

    
    weights_path = '../input/tweet-checkpoints-2/'
    weights = sorted([f for f in os.listdir(weights_path) if "albert" in f])
    name = 'albert-large-squad'


def main():
    preds = {}
    seed_everything(SEED)
    df_test = pd.read_csv(DATA_PATH + 'test.csv').fillna('')
    
    configs = [
        # ConfigDistil(),
        # ConfigBertBase(), 
        # ConfigBertWWM(),
        ConfigAlbert(),
    ]
    
    for config in configs:
        
        print(f'\n   -  Doing inference for {config.name}\n')
    
        tokenizer, tokens = create_tokenizer_and_tokens(config)
        test_dataset = TweetTestDataset(df_test, tokenizer, tokens, max_len=config.max_len_val, model_name=config.selected_model, use_old_sentiment=config.use_old_sentiment)
        pred_test = k_fold_inference(
            config,
            test_dataset,
            seed=SEED,
        )

        char_pred_test_start, char_pred_test_end = get_char_preds(pred_test, test_dataset)
        preds[config.name] = (char_pred_test_start, char_pred_test_end)

        with open('albert-large-v2-squad-char_pred_test_start.pkl', 'wb') as handle:
            pickle.dump(char_pred_test_start, handle)
        with open('albert-large-v2-squad-char_pred_test_end.pkl', 'wb') as handle:
            pickle.dump(char_pred_test_end, handle)
    
    return preds


if __name__ == '__main__':
    main()