import pandas as pd
import settings
import glob

HUMAN_COLUMN_NAME = settings.HUMAN_COLUMN_NAME
AI_COLUMN_NAME = settings.AI_COLUMN_NAME    

#paired data list
def load_paired_dataset(file_path):
    
    try:
        df = pd.read_csv(file_path)

        if AI_COLUMN_NAME not in df.columns or HUMAN_COLUMN_NAME not in df.columns:
            print(f'___' * 10)
            print(f'ERROR: {file_path} is missing coulumn {HUMAN_COLUMN_NAME} and {AI_COLUMN_NAME}.')
            print(f'___' * 10)
            return []
        
        paired_data = []
        df = df.dropna(subset=[AI_COLUMN_NAME, HUMAN_COLUMN_NAME])

        for _, row in df.iterrows():
            human_code = row[HUMAN_COLUMN_NAME]
            ai_code = row[AI_COLUMN_NAME]
            paired_data.append((human_code, ai_code))

        return paired_data

    except Exception as e:
        print(f'ERROR processing file {file_path}. Details: {e}')
        return []
    

#simcse training data list
def load_simcse_training_data():
    #one time library import
    from datasets import load_dataset
    print("Loading 'code_search_net' dataset for SimCSE training...")

    dataset = load_dataset(settings.CODE_SEARCH_NET, split='train')

    print("Available columns:", dataset.column_names)
    code_samples = [sample['code'] for sample in dataset]
    print(f'Loaded {len(code_samples)} samples for SimCSE training')
    return code_samples
