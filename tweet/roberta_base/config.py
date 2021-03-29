import tokenizers


# Paths
TOKENIZER_PATH = '../input/roberta-tokenizer'
TEST_FILE = '../input/tweet-sentiment-extraction/test.csv'
TRAINED_MODEL_PATH = '../input/roberta-base'

# Model config
MODEL_CONFIG = '../input/configs/roberta_base.json'

# Model params
SEED = 25
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 96  # actually = 86
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f'{TOKENIZER_PATH}/vocab.json',
    merges_file=f'{TOKENIZER_PATH}/merges.txt',
    lowercase=True,
    add_prefix_space=True)
HIDDEN_SIZE = 768
N_LAST_HIDDEN = 12
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 0.6
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
