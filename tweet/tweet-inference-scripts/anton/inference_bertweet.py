# !pip install /kaggle/input/bertweet-libs/sacrebleu-1.4.10-py3-none-any.whl
# !cp -R /kaggle/input/bertweet-libs/fairseq-0.9.0/fairseq-0.9.0 /kaggle/working
# !cp -R /kaggle/input/bertweet-libs/fastBPE-0.1.0/fastBPE-0.1.0/ /kaggle/working
# !pip install /kaggle/working/fairseq-0.9.0/
# !pip install /kaggle/working/fastBPE-0.1.0/


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import torch
import argparse
from sklearn import model_selection
import tokenizers

from transformers import RobertaConfig
from transformers import RobertaModel


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import random
import gc

from types import SimpleNamespace
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from fairseq import options  

import pickle

parser = options.get_preprocessing_parser()  
parser.add_argument('--bpe-codes', type=str,default="../input/bertweet-transformer-private/bpe.codes")


class BERTweetTokenizer():
    
    def __init__(self,pretrained_path = '../input/bertweet-transformer-private/', parser=parser):
        

        self.bpe = fastBPE(args=parser.parse_args(args=[]))
        self.vocab = Dictionary()
        self.vocab.add_from_file(pretrained_path + "dict.txt")
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.cls_token = '<s> '
        self.sep_token = ' </s>'
        
    def bpe_encode(self,text):
        return self.bpe.encode(text)
    
    def encode(self,text, add_special_tokens=False):
        subwords = self.cls_token + self.bpe.encode(text) + self.sep_token
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=True).long().tolist()
        return input_ids
    
    def tokenize(self,text):
        return self.bpe_encode(text).split()
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids

    
    def decode_id(self,id):
        return self.vocab.string(id, bpe_symbol = '@@')


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(2020)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr=6e-5

class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    EPOCHS = 3
    config = RobertaConfig.from_pretrained(
    "../input/bertweet-transformer-private/config.json")
    config.output_hidden_states = True

    BERTweet = RobertaModel.from_pretrained(
    "../input/bertweet-transformer-private/model.bin",
    config=config)
    BERTweetpath="../input/bertweet-transformer-private/"
    TRAINING_FILE = "../input/twe-myfolds/train_folds_20200425.csv"
    TOKENIZER = BERTweetTokenizer('../input/bertweet-transformer-private/',parser=parser)


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    
    tweet_orig = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    len_st = len(selected_text)


    idx = tweet_orig.find(selected_text)
    char_targets = np.zeros((len(tweet_orig)))
    char_targets[idx:idx+len(selected_text)]=1

    tok_tweet = config.TOKENIZER.encode(tweet_orig)

    # Convert into torch tensor
    all_input_ids = torch.tensor([tok_tweet], dtype=torch.long)

    tok_tweet=tok_tweet[1:-1]

    # ID_OFFSETS
    offsets = []; idx=0
    
    try:
        for t in tok_tweet:
            w = config.TOKENIZER.decode_id([t])
            if tweet[tweet.find(w,idx)-1] == " ":
                idx+=1
                offsets.append((idx,idx+len(w)))
                idx += len(w)
            else:
                offsets.append((idx,idx+len(w)))
                idx += len(w)
    except:
        
        print("***",tweet_orig)
        pass
    

    input_ids_orig = tok_tweet
    tweet_offsets = offsets


    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if np.sum(char_targets[offset1:offset2])> 0:
            target_idx.append(j)

    if  len(target_idx)>0:
        targets_start = target_idx[0]
        targets_end = target_idx[-1]

    else:
        targets_start = 0
        targets_end= len(char_targets)


    #print(targets_start,targets_end)    
    sentiment_id = {
        'positive': 1809,
        'negative': 3392,
        'neutral': 14058
    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet_orig,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }



def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            probas_char[offset[0]:offset[1]] = preds[i]    
    
    return probas_char


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, textID):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.textID = textID
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

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
            'textID': self.textID[item]
        }


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = RobertaModel.from_pretrained("/kaggle/input/bertweetweights20200606/bertweet_model_0.bin", config=conf)
        
        self.high_dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.classifier.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.stack(tuple(out[-i - 1] for i in range(10)), dim=0)
        
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        
        out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        

        return start_logits, end_logits

def main():
    # Data
    df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values

    df_test['text']=df_test['text'].apply(lambda x: x.strip())
    df_test['selected_text']=df_test['selected_text'].apply(lambda x: x.strip())
    
    # Models
    models_weights = ['model_0.bin', 'model_1.bin', 'model_2.bin', 'model_3.bin', 'model_4.bin']
    models = []

    for i in models_weights:
        model = TweetModel(config.config)
        model.to(device)
        model.load_state_dict(torch.load("/kaggle/input/bertweetweights20200606/bertweet_" + str(i)))
        model.eval()
        models.append(model)
    
    # Dataset
    test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        textID  = df_test.textID.values
    )
    
    # Data Loader
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # Inference
    char_pred_test_start = []
    char_pred_test_end = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            offsets = d["offsets"].numpy()


            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            st_outputs = np.zeros((targets_start.size(0), config.MAX_LEN))
            en_outputs = np.zeros((targets_start.size(0), config.MAX_LEN))

            for model in  models:
                outputs_start, outputs_end = model(ids=ids, mask=mask,token_type_ids=token_type_ids)

                outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
                outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

                st_outputs += outputs_start
                en_outputs += outputs_end

            outputs_start = st_outputs / len(models)
            outputs_end = en_outputs / len(models)

            for px, tweet in enumerate(orig_tweet):
                idx_start_1 = np.argmax(outputs_start[px, :])
                idx_end_1 = np.argmax(outputs_end[px, :])

                char_pred_test_start.append(token_level_to_char_level(tweet, offsets[px], outputs_start[px, :]))
                char_pred_test_end.append(token_level_to_char_level(tweet, offsets[px], outputs_end[px, :]))

    with open('bertweet-char_pred_test_start.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_start, handle)
    with open('bertweet-char_pred_test_end.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_end, handle)

    return {"bertweet" : (char_pred_test_start, char_pred_test_end)}

if __name__ == '__main__':
    main()