import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import *
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
# import utils
import random
import re

import pickle


SEL_MODEL = (RobertaForQuestionAnswering,    RobertaTokenizer,    'roberta-base', RobertaConfig)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(2020)

lr=3e-5

MODEL_PATH = "/kaggle/input/roberta/"
class config:
    TRAINING_FILE = "../input/twe-myfolds/train_folds_20200425.csv"
    TRAIN_BATCH_SIZE = 96
    VALID_BATCH_SIZE = 96
    MAX_LEN = 128
    EPOCHS = 6
    conf_file = '/kaggle/input/all-weights/config.json'
    MODEL_PATH = "/kaggle/input/all-weights/"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{MODEL_PATH}/vocab.json", 
    merges_file=f"/kaggle/input/tweeter-offline-eval/merges-1.txt", 
    lowercase=True,
    add_prefix_space=True)
#     TOKENIZER = tokenizer


def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            probas_char[offset[0]:offset[1]] = preds[i]
    
    return probas_char

hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def process_text(text):
#     text = hashtags.sub(' hashtag', text)
#     text = mentions.sub(' entity', text)
    text = urls.sub(' url', text)
    return text
  
def match_expr(pattern, string):
    return not pattern.search(string) == None


def process_data(tweet, selected_text, sentiment, tokenizer, max_len,  addrep = False):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())
    
    orig_tweet = tweet
    orig_selected_text = selected_text
    
    if addrep:
        tweet = process_text(tweet)
        selected_text = process_text(selected_text)

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
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
        
    targets = np.zeros((max_len))
    targets[targets_start: (targets_end+1)] = 1.
    
    target_length = targets_end - targets_start + 1
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': orig_tweet,
        'orig_selected': orig_selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets,
        'targets': targets,
        'targets_len': target_length
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
#         print('item ', item)
        
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
            'targets': torch.tensor(data["targets"], dtype=torch.float),
            'targets_length': torch.tensor(data["targets_len"], dtype=torch.long)
        }


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = SEL_MODEL[0].from_pretrained(config.MODEL_PATH, config=conf).base_model
        self.nb_layers = 10
        self.logit = nn.Sequential(nn.Linear(768*self.nb_layers, 128),
            nn.Tanh(),
            nn.Linear(128, 2), 
        )

        # self.logit.apply(self._init_weights)
        self.high_dropout = nn.Dropout(p=0.4)
    
    def forward(self, ids, mask, token_type_ids):
        x, _, hidden_states = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        hidden_states = hidden_states[::-1]
        features = torch.cat(hidden_states[:self.nb_layers], -1)

        if self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logit(self.high_dropout(features)) for _ in range(4)],
                    dim=0,),
                dim=0)
        else:
            logits =  self.logit(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


def main():
    # Data
    df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values
    
    # Models
    device = torch.device("cuda")
    models_weights = ['model_0.bin', 'model_1.bin', 'model_2.bin', 'model_3.bin', 'model_4.bin']
    models = []
    
    model_config = RobertaConfig.from_pretrained(config.MODEL_PATH)
    model_config.output_hidden_states = True

    for i in models_weights:
        model = TweetModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load("/kaggle/input/all-weights/" + str(i)))
        model.eval()
        models.append(model)
    
    # Dataset
    test_dataset = TweetDataset(
            tweet=df_test.text.values,
            sentiment=df_test.sentiment.values,
            selected_text=df_test.selected_text.values
        )
    
    # Dataloader
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
    
    with open('roberta_anton-char_pred_test_start.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_start, handle)
    with open('roberta_anton-char_pred_test_end.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_end, handle)

    return {"roberta-anton" : (char_pred_test_start, char_pred_test_end)}


if __name__ == '__main__':
    main()