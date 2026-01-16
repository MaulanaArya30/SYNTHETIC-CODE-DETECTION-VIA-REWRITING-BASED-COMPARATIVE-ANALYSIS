import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os
import random
import re

import settings
from data_loader import load_simcse_training_data


class CodeDataset(Dataset):
    def __init__(self, code_samples, tokenizer, max_length=512, augment=True):
        self.code_samples = code_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.code_samples)
    
    def augment_code(self, code):
        """
        Apply simple code augmentation to create harder positive pairs.
        These augmentations preserve semantics while changing surface form.
        """
        if not self.augment or random.random() > 0.3:  # 30% augmentation rate
            return code
        
        augmented = code
        
        # 1. Whitespace variations (most common)
        if random.random() < 0.5:
            # Toggle between spaces and tabs
            if '    ' in augmented:
                augmented = augmented.replace('    ', '\t')
            elif '\t' in augmented:
                augmented = augmented.replace('\t', '    ')
        
        # 2. Line reordering (for independent statements)
        if random.random() < 0.2:
            lines = augmented.split('\n')
            # Only reorder import statements (safe to reorder)
            import_lines = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
            other_lines = [l for l in lines if l not in import_lines]
            
            if len(import_lines) > 1:
                random.shuffle(import_lines)
                augmented = '\n'.join(import_lines + other_lines)
        
        # 3. Add/remove trailing whitespace
        if random.random() < 0.3:
            lines = augmented.split('\n')
            augmented = '\n'.join([l.rstrip() + (' ' if random.random() < 0.5 else '') for l in lines])
        
        # 4. Quote style variation (single vs double)
        if random.random() < 0.2:
            # Only change string literals, not docstrings
            augmented = re.sub(r"'([^']*)'", r'"\1"', augmented)
        
        return augmented
    
    def __getitem__(self, idx):
        code = self.code_samples[idx]
        
        # Apply augmentation
        code = self.augment_code(code)
        
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


def simcse_loss(z1, z2, temperature=0.07):
    batch_size = z1.shape[0]
    
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    sim_11 = torch.matmul(z1, z1.T) / temperature
    sim_22 = torch.matmul(z2, z2.T) / temperature
    sim_12 = torch.matmul(z1, z2.T) / temperature
    
    mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
    sim_11 = sim_11.masked_fill(mask, -9e15)
    sim_22 = sim_22.masked_fill(mask, -9e15)
    
    logits = torch.cat([sim_12, sim_11], dim=1)
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss


def get_embeddings(encoder_model, model_type, outputs):
    if model_type == 'codet5':
        return outputs.last_hidden_state.mean(dim=1)
    else:
        return outputs.last_hidden_state[:, 0]


@torch.no_grad()
def evaluate_on_validation(encoder_model, model_type, val_loader):
    encoder_model.eval()
    total_loss = 0.0
    steps = 0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(settings.DEVICE)
        attention_mask = batch['attention_mask'].to(settings.DEVICE)
        
        outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        
        emb_1 = get_embeddings(encoder_model, model_type, outputs_1)
        emb_2 = get_embeddings(encoder_model, model_type, outputs_2)
        
        loss = simcse_loss(emb_1, emb_2)
        total_loss += loss.item()
        steps += 1
    
    return total_loss / steps if steps > 0 else float('inf')


def train(model_type):
    print(f'\n{"="*60}')
    print(f'SimCSE Training with Augmentation: {model_type.upper()}')
    print(f'{"="*60}')

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
        raise ValueError('Invalid model_type')
    
    os.makedirs(save_path, exist_ok=True)
    encoder_model = encoder_model.to(settings.DEVICE)

    print("Loading training data...")
    code_samples = load_simcse_training_data()
    
    # Train/val split
    split_idx = int(len(code_samples) * 0.95)
    train_samples = code_samples[:split_idx]
    val_samples = code_samples[split_idx:]
    
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    # IMPORTANT: Augmentation only for training, not validation
    train_dataset = CodeDataset(train_samples, tokenizer, augment=True)
    val_dataset = CodeDataset(val_samples, tokenizer, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    optimizer = AdamW(
        encoder_model.parameters(), 
        lr=settings.SIMCSE_LR,
        weight_decay=0.01
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_steps = len(train_loader) * settings.SIMCSE_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"Configuration:")
    print(f"  Epochs: {settings.SIMCSE_EPOCHS}")
    print(f"  Batch size: {settings.SIMCSE_BATCH_SIZE}")
    print(f"  Learning rate: {settings.SIMCSE_LR}")
    print(f"  Weight decay: 0.01")
    print(f"  Data augmentation: ON (30% probability)")
    print(f"  Early stopping: {patience} epochs patience\n")

    for epoch in range(settings.SIMCSE_EPOCHS):
        print(f'{"─"*60}')
        print(f'Epoch {epoch+1}/{settings.SIMCSE_EPOCHS}')
        print(f'{"─"*60}')
        
        encoder_model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(settings.DEVICE)
            attention_mask = batch['attention_mask'].to(settings.DEVICE)

            outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            emb_1 = get_embeddings(encoder_model, model_type, outputs_1)
            emb_2 = get_embeddings(encoder_model, model_type, outputs_2)

            loss = simcse_loss(emb_1, emb_2)
            
            if not torch.isfinite(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = running_loss / steps
        train_losses.append(avg_train_loss)
        
        val_loss = evaluate_on_validation(encoder_model, model_type, val_loader)
        val_losses.append(val_loss)
        
        print(f"  Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ★ Best! Saving...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  Early stopping!")
                break

    loss_file = os.path.join(save_path, "loss_log.txt")
    with open(loss_file, "w") as f:
        f.write(f"Training: {model_type}\n")
        f.write(f"Best Val: {best_val_loss:.6f}\n")
        f.write(f"Stopped: Epoch {epoch+1}\n")
        f.write(f"Augmentation: ON\n\n")
        for i, (t, v) in enumerate(zip(train_losses, val_losses)):
            f.write(f"Epoch {i+1}: Train={t:.6f}, Val={v:.6f}\n")

    print(f'\n{"="*60}')
    print(f'Completed: {model_type.upper()}')
    print(f'  Best Val: {best_val_loss:.4f}')
    print(f'  Epochs: {epoch+1}/{settings.SIMCSE_EPOCHS}')
    print(f'{"="*60}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                       choices=['codet5', 'graphcodebert', 'all'])
    args = parser.parse_args()
    
    if args.model == "all":
        for m in ["codet5", "graphcodebert"]:
            train(m)
    else:
        train(args.model)