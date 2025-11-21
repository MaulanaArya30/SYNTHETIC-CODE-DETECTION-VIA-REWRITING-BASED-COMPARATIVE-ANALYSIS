import torch

#model rewriter
REWRITER_MODEL_NAME = 'Salesforce/codet5-base'

#CodeT5
CODET5_MODEL_NAME = 'Salesforce/codet5-base'
CODET5_SIMCSE_PATH = './models/codet5-simcse'

#graphcodeBERT
GCB_MODEL_NAME = 'microsoft/graphcodebert-base'
GCB_SIMCSE_PATH = './models/graphcodebert-simcse'

#dataset
VARIANT_FILES = [
    'data/test_df.csv'
]
CODE_SEARCH_NET = 'Nan-Do/code-search-net-python'

#naming
HUMAN_COLUMN_NAME = 'Python Code'
AI_COLUMN_NAME = 'GPT Answer'

#training config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIMCSE_EPOCHS = 5
SIMCSE_BATCH_SIZE = 4#16
SIMCSE_LR = 1e-4

#rewrite num
NUM_REWRITES = [2, 4, 8]

