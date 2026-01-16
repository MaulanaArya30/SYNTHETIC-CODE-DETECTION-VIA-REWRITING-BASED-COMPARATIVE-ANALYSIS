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
# VARIANT_FILES = [
#     'data/variant_1_full_processed.csv',
#     'data/variant_4_full_processed.csv',
#     'data/variant_5_full_processed.csv',
#     'data/variant_6_full_processed.csv',
#     'data/variant_7_full_processed.csv',
#     'data/variant_8_full_processed.csv',
#     'data/variant_9_full_processed.csv',
#     'data/variant_10_full_processed.csv',
#     'data/variant_11_full_processed.csv',
#     'data/variant_12_full_processed.csv',
#     'data/variant_13_full_processed.csv',
# ]
VARIANT_FILES = [
    'data/test_df.csv',
]
CODE_SEARCH_NET = 'Nan-Do/code-search-net-python'

#naming
HUMAN_COLUMN_NAME = 'Python Code'
AI_COLUMN_NAME = 'GPT Answer'

#training config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIMCSE_EPOCHS = 15#5
SIMCSE_BATCH_SIZE = 16#16
SIMCSE_LR = 3e-5#1e-4

#rewrite num
NUM_REWRITES = [2, 4, 8]

#batch size for evaluation
EVAL_BATCH_SIZE = 64
