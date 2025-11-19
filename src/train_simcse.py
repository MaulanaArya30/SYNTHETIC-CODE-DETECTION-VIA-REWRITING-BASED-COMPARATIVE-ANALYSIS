import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from torch.optim import AdamW
from tqdm import tqdm
import argparse

import settings
from data_loader import load_simcse_training_data


#pytroch dataset
class CodeDataset(Dataset):
    def __init__(self, code_samples, tokenizer, max_length=512):
        self.code_samples = code_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        code = self.code_samples[idx]   
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


#constrastive loss simCSE
def constrastive_loss(embeddings_a, embeddings_b, temperature=0.07):
    batch_size = embeddings_a.shape[0]

    #cosine similarity for simCSE
    sim_matrix = F.cosine_similarity(embeddings_a.unsqueeze(1), embeddings_b.unsqueeze(0), dim=-2)
    lables = torch.arange(batch_size).long().to(settings.DEVICE)

    loss = F.cross_entropy(sim_matrix / temperature, lables)
    return loss

#main training
def train(model_type):
    print(f'---Starting SimCSE Training for {model_type}---')

    #CodeT5
    if model_type == 'codet5':
        tokenizer = RobertaTokenizer.from_pretrained(settings.CODET5_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(settings.CODET5_MODEL_NAME)

        encoder_model = model.encoder
        save_path = settings.CODET5_SIMCSE_PATH


    elif model_type == 'graphcodebert':
        tokenizer = RobertaTokenizer.from_pretrained(settings.GCB_MODEL_NAME)
        model = RobertaModel.from_pretrained(settings.GCB_MODEL_NAME)

        encoder_model = model
        save_path = settings.GCB_SIMCSE_PATH

    else:
        raise ValueError('Invalid model_type. Choose codet5 or graphcodebert')
    
    encoder_model = encoder_model.to(settings.DEVICE)
    encoder_model.train() #with dropout by default

    #load data
    code_samples = load_simcse_training_data()
    train_dataset = CodeDataset(code_samples, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    optimizer = AdamW(encoder_model.parameters(), lr=settings.SIMCSE_LR)

    for epoch in range(settings.SIMCSE_EPOCHS):
        print(f'--- Epoch {epoch+1}/{settings.SIMCSE_EPOCHS} ---')

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(settings.DEVICE)
            attention_mask = batch['attention_mask'].to(settings.DEVICE)

            #SimCSE -> passed twice
            #first pass
            outputs_a = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            #second pass
            outputs_b = encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            if model_type == 'codet5':
                emb_a = outputs_a.last_hidden_state.mean(dim=1)
                emb_b = outputs_b.last_hidden_state.mean(dim=1)
            else:
                emb_a = outputs_a.last_hidden_state[:, 0]
                emb_b = outputs_b.last_hidden_state[:, 0]

            loss = constrastive_loss(emb_a, emb_b)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1} Loss: {loss.item()}')

    #save encoder
    print(f'Training complete. Saving to {save_path}...')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimCSE for CodeT5 or GraphCodeBert")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['codet5', 'graphcodebert'],
        help="The model architecture to train (as per Table 3.1)"
    )
    args = parser.parse_args()
    
    train(args.model)