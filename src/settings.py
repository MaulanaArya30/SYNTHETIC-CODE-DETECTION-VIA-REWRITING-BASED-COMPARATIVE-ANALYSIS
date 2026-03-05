import torch

#model rewriter
# REWRITER_MODEL_NAME = 'Salesforce/codet5-base' #not used anymore because of corrupt data
REWRITER_MODEL_NAME = 'deepseek-ai/deepseek-coder-6.7b-instruct'
#REWRITER_MODEL_NAME = 'deepseek-ai/deepseek-coder-1.3b-instruct'

#CodeT5
CODET5_MODEL_NAME = 'Salesforce/codet5-base'
CODET5_SIMCSE_PATH = './models/codet5-simcse'

#graphcodeBERT
GCB_MODEL_NAME = 'microsoft/graphcodebert-base'
GCB_SIMCSE_PATH = './models/graphcodebert-simcse'
#dataset
VARIANT_FILES = [
    'data/variant_1_full_processed.csv',
    'data/variant_4_full_processed.csv',
    'data/variant_5_full_processed.csv',
    'data/variant_6_full_processed.csv',
    'data/variant_7_full_processed.csv',
    'data/variant_8_full_processed.csv',
    'data/variant_9_full_processed.csv',
    'data/variant_10_full_processed.csv',
    'data/variant_11_full_processed.csv',
    'data/variant_12_full_processed.csv',
    'data/variant_13_full_processed.csv',
]
# VARIANT_FILES = [
#     'data/train_data/data_test_df.csv',
# ]
CODE_SEARCH_NET = 'Nan-Do/code-search-net-python'

#naming
HUMAN_COLUMN_NAME = 'Python Code'
AI_COLUMN_NAME = 'GPT Answer'

#training config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIMCSE_EPOCHS = 20#5
SIMCSE_BATCH_SIZE = 8#16
GRADIENT_ACCUMULATION_STEPS =  16#because GPU is limited
SIMCSE_LR = 1e-4#3e-5
SIMCSE_TEMP = 0.1#0.07

#rewrite num
NUM_REWRITES = [2, 4, 8]

#batch size for evaluation
EVAL_BATCH_SIZE = 64
