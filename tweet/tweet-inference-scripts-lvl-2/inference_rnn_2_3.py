import re
import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torch.nn import functional as F
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences


SEED = 2020
add_spaces_to = ["bert_", 'xlnet_', 'electra_', 'bertweet-']
VOCAB = {'UNK': 1, ' ': 2, 'e': 3, 't': 4, 'o': 5, 'a': 6, 'i': 7, 'n': 8, 's': 9, 'h': 10, 'r': 11, 'l': 12, 'd': 13, 'm': 14, 'y': 15, 'u': 16, 'g': 17, 'w': 18, '.': 19, 'c': 20, 'p': 21, 'f': 22, 'b': 23, 'k': 24, '!': 25, 'v': 26, '`': 27, ',': 28, '*': 29, 'j': 30, '/': 31, '?': 32, 'x': 33, '-': 34, ':': 35, 'z': 36, '2': 37, '0': 38, '1': 39, '_': 40, '3': 41, "'": 42, '4': 43, 'q': 44, ')': 45, '5': 46, '&': 47, '(': 48, '6': 49, '#': 50, '8': 51, '7': 52, '9': 53, ';': 54, '<': 55, '@': 56, '=': 57, 'ï': 58, '¿': 59, '½': 60, '~': 61, '$': 62, '+': 63, '>': 64, ']': 65, '%': 66, '[': 67, '^': 68, '|': 69, '\\': 70, '{': 71, '}': 72, 'â': 73, '\xa0': 74, '\t': 75, '´': 76}


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
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


def create_input_data(models):
    char_pred_test_starts = []
    char_pred_test_ends = []

    for model, _ in models:
        with open(model + 'char_pred_test_start.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)  

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_starts.append(probas)

        with open(model + 'char_pred_test_end.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_ends.append(probas)
            
    char_pred_test_start = [np.concatenate([char_pred_test_starts[m][i][:, np.newaxis] for m in range(len(models))], 
                                           1) for i in range(len(char_pred_test_starts[0]))]

    char_pred_test_end = [np.concatenate([char_pred_test_ends[m][i][:, np.newaxis] for m in range(len(models))], 
                                         1) for i in range(len(char_pred_test_starts[0]))]
    
    return char_pred_test_start, char_pred_test_end


class TweetCharDataset(Dataset):
    def __init__(self, df, X, start_probas, end_probas, n_models=1, max_len=150, train=True):
        self.max_len = max_len

        self.X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
        
        self.start_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(start_probas):
            len_ = min(len(p), max_len)
            self.start_probas[i, :len_] = p[:len_]

        self.end_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(end_probas):
            len_ = min(len(p), max_len)
            self.end_probas[i, :len_] = p[:len_]
            
        self.sentiments_list = ['positive', 'neutral', 'negative']
        
        self.texts = df['text'].values
        self.selected_texts = df['selected_text'].values if train else [''] * len(df)
        self.sentiments = df['sentiment'].values
        self.sentiments_input = [self.sentiments_list.index(s) for s in self.sentiments]
        
        # Targets
        self.seg_label = np.zeros((len(df), max_len))
        
        if train:
            self.start_idx = []
            self.end_idx = []
            for i, (text, sel_text) in enumerate(zip(df['text'].values, df['selected_text'].values)):
                start, end = get_start_end_string(text, sel_text.strip())
                self.start_idx.append(start)
                self.end_idx.append(end)
                self.seg_label[i, start:end] = 1
        else:
            self.start_idx = [0] * len(df)
            self.end_idx = [0] * len(df)
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'ids': torch.tensor(self.X[idx], dtype=torch.long),
            'probas_start': torch.tensor(self.start_probas[idx]).float(),
            'probas_end': torch.tensor(self.end_probas[idx]).float(),
            'target_start': torch.tensor(self.start_idx[idx], dtype=torch.long),
            'target_end': torch.tensor(self.end_idx[idx], dtype=torch.long),
            'text': self.texts[idx],
            'selected_text': self.selected_texts[idx],
            'sentiment': self.sentiments[idx],
            'sentiment_input': torch.tensor(self.sentiments_input[idx]),
            'seg_label': torch.tensor(self.seg_label[idx])
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )
                
    def forward(self, x):
        return self.conv(x)

class Waveblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1], padding="same"):
        super().__init__()
        self.n = len(dilations)
        
        if padding == "same":
            padding = kernel_size // 2
            
        self.init_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        self.convs_tanh = nn.ModuleList([])
        self.convs_sigm = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        
        for dilation in dilations:
            self.convs_tanh.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding*dilation, dilation=dilation),
                    nn.Tanh(),
                )
            )
            self.convs_sigm.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding*dilation, dilation=dilation),
                    nn.Sigmoid(),
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, 1))
        
    def forward(self, x):
        x = self.init_conv(x)
        res_x = x
        
        for i in range(self.n):
            x_tanh = self.convs_tanh[i](x)
            x_sigm = self.convs_sigm[i](x)
            x = x_tanh * x_sigm
            x = self.convs[i](x)
            res_x = res_x + x
        
        return res_x
    
    
class TweetCharModel(nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 embed_dim=64, lstm_dim=64, char_embed_dim=32, sent_embed_dim=32, ft_lstm_dim=32, n_models=1):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.proba_lstm = nn.LSTM(n_models * 2, ft_lstm_dim, batch_first=True, bidirectional=True)
        
        self.lstm = nn.LSTM(char_embed_dim + ft_lstm_dim * 2 + sent_embed_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim *  4, lstm_dim),
            nn.ReLU(),
            nn.Linear(lstm_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
    
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1)
        probas_fts, _ = self.proba_lstm(probas)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        features = torch.cat([char_fts, sentiment_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)
        
        features = torch.cat([features, features2], -1)
        
        if self.use_msd and self.training:
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
    

class ConvNet(nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 cnn_dim=64, char_embed_dim=32, sent_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3, use_bn=False):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)
         
        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size, use_bn=use_bn),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 8, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
        
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1) # [Bs x T x nb_ft]
    
        if self.use_msd and self.training:
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
    
    
class WaveNet(nn.Module):
    def __init__(self, len_voc, use_msd=True, dilations=[1], 
                 cnn_dim=64, char_embed_dim=32, sent_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3, use_bn=True):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)
         
        self.cnn = nn.Sequential(
            Waveblock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim),
            Waveblock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 2),
            Waveblock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 4),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 4, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
        
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1) # [Bs x T x nb_ft]
    
        if self.use_msd and self.training:
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
            start_logits, end_logits = model(
                data["ids"].cuda(), 
                data['sentiment_input'].cuda(), 
                data['probas_start'].cuda(), 
                data['probas_end'].cuda()
            )

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for s, e in zip(start_probs, end_probs):
                start_probas.append(list(s))
                end_probas.append(list(e))

    return start_probas, end_probas


def k_fold_inference(config, test_dataset, len_voc, seed=42):
    seed_everything(seed)
    pred_tests = [] 
    
    for weight in config.weights:         
        if config.model == 'rnn':
            model = TweetCharModel(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,   
                lstm_dim=config.lstm_dim,
                ft_lstm_dim=config.ft_lstm_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
            ).cuda()
            
        elif config.model == 'cnn':
            model = ConvNet(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,  
                use_bn=config.use_bn,
                cnn_dim=config.cnn_dim,
                proba_cnn_dim=config.proba_cnn_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
                kernel_size=config.kernel_size,
            ).cuda()
            
        else:
            model = WaveNet(
                len_voc,
                use_msd=config.use_msd, 
                n_models=config.n_models,  
                use_bn=config.use_bn,
                cnn_dim=config.cnn_dim,
                proba_cnn_dim=config.proba_cnn_dim,
                char_embed_dim=config.char_embed_dim,
                sent_embed_dim=config.sent_embed_dim,
                kernel_size=config.kernel_size,
                dilations=config.dilations, 
            ).cuda()
        
        model = load_model_weights(model, weight, cp_folder=config.weights_path, verbose=1)
        model.zero_grad()
        
        pred_test = predict(model, test_dataset, batch_size=config.batch_size_val, num_workers=config.num_workers)
        pred_tests.append(pred_test)    
        
    return pred_tests


class Config:
    model = 'rnn'
    model_name = 'rnn_2_3'
    
    models = [
        ('distilbert-base-uncased-distilled-squad-', 'theo'),
        ('bert-base-uncased-', 'theo'),
        ('bert-wwm-neutral-', 'theo'),
        ('albert-large-v2-squad-', 'theo'),
        ('bertweet-', 'anton'),
        ('roberta_anton-', 'anton'),
        ("roberta-", 'hk'),
        ("distil_", 'hk'),
        ("large_", 'hk'),
        ("xlnet_", 'hk'),
    ]        
    
    n_models = len(models)
    
    # Architecture
    sent_embed_dim = 16 # 32 works as well
    char_embed_dim = 8
    ft_lstm_dim = 16
    lstm_dim = 64
    use_msd = True

    # Inference
    batch_size_val = 512
    num_workers = 4
    max_len_val = 150
    
    weights_path = '../input/tweet-cp-lvl-2/'
    weights = sorted([f for f in os.listdir(weights_path) if 'rnn_2_3' in f])
#     print(weights)


def main():
    config = Config()
    
    char_pred_test_start, char_pred_test_end = create_input_data(config.models)
    
    df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')
    df_test['selected_text'] = ''
    
    
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
    tokenizer.word_index = VOCAB

    len_voc = len(tokenizer.word_index) + 1

    X_test = tokenizer.texts_to_sequences(df_test['text'].values)
    
    
    test_dataset = TweetCharDataset(
        df_test, X_test, char_pred_test_start, char_pred_test_end, 
        max_len=config.max_len_val, train=False, n_models=config.n_models
    )
    
    pred_tests = k_fold_inference(config, test_dataset, len_voc, seed=42)

    np.save(f"preds_char_test_{config.model_name}.npy", np.array(pred_tests))


if __name__ == '__main__':
    main()