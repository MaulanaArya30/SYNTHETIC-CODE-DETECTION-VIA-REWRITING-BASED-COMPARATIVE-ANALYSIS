import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os

import settings
from data_loader import load_simcse_training_data


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


def simcse_loss(z1, z2, temperature=0.07):
    """
    SimCSE contrastive loss (NT-Xent).
    Same as Ye et al. (2025).
    """
    batch_size = z1.shape[0]
    
    # Normalize to unit sphere
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    # Compute similarity matrices
    sim_11 = torch.matmul(z1, z1.T) / temperature
    sim_22 = torch.matmul(z2, z2.T) / temperature
    sim_12 = torch.matmul(z1, z2.T) / temperature
    
    # Mask self-similarities
    mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
    sim_11 = sim_11.masked_fill(mask, -9e15)
    sim_22 = sim_22.masked_fill(mask, -9e15)
    
    # Concatenate and compute loss
    logits = torch.cat([sim_12, sim_11], dim=1)
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss


def get_embeddings(encoder_model, model_type, outputs):
    """Extract embeddings with appropriate pooling."""
    if model_type == 'codet5':
        # Mean pooling for CodeT5
        return outputs.last_hidden_state.mean(dim=1)
    else:
        # CLS token for GraphCodeBERT
        return outputs.last_hidden_state[:, 0]


@torch.no_grad()
def evaluate_on_validation(encoder_model, model_type, val_loader):
    """Evaluate model on validation set."""
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
    print(f'SimCSE Training (Ye et al. 2025): {model_type.upper()}')
    print(f'{"="*60}')

    # Load model
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

    # Load data
    print("Loading training data...")
    code_samples = load_simcse_training_data()
    
    # Train/val split (95/5)
    split_idx = int(len(code_samples) * 0.95)
    train_samples = code_samples[:split_idx]
    val_samples = code_samples[split_idx:]
    
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Val:   {len(val_samples):,} samples")
    
    # Create datasets
    train_dataset = CodeDataset(train_samples, tokenizer)
    val_dataset = CodeDataset(val_samples, tokenizer)
    
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

    # Optimizer (no weight decay - matching Ye et al.)
    optimizer = AdamW(
        encoder_model.parameters(), 
        lr=settings.SIMCSE_LR
    )
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_steps = len(train_loader) * settings.SIMCSE_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Gradient accumulation settings
    gradient_accumulation_steps = getattr(settings, 'GRADIENT_ACCUMULATION_STEPS', 1)
    effective_batch_size = settings.SIMCSE_BATCH_SIZE * gradient_accumulation_steps
    
    # Early stopping
    best_val_loss = float('inf')
    patience = getattr(settings, 'EARLY_STOPPING_PATIENCE', 8)
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    print(f"\nConfiguration:")
    print(f"  Epochs: {settings.SIMCSE_EPOCHS}")
    print(f"  Physical batch size: {settings.SIMCSE_BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {settings.SIMCSE_LR}")
    print(f"  Temperature: 0.07")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {settings.DEVICE}\n")

    for epoch in range(settings.SIMCSE_EPOCHS):
        print(f'{"─"*60}')
        print(f'Epoch {epoch+1}/{settings.SIMCSE_EPOCHS}')
        print(f'{"─"*60}')
        
        encoder_model.train()
        running_loss = 0.0
        accumulated_loss = 0.0
        steps = 0
        
        # Zero gradients at start of epoch
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(settings.DEVICE)
            attention_mask = batch['attention_mask'].to(settings.DEVICE)

            # Two forward passes with different dropout masks
            outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract embeddings
            emb_1 = get_embeddings(encoder_model, model_type, outputs_1)
            emb_2 = get_embeddings(encoder_model, model_type, outputs_2)

            # Compute loss
            loss = simcse_loss(emb_1, emb_2, temperature=0.07)
            
            if not torch.isfinite(loss):
                print(f"\n⚠️  Non-finite loss! Skipping batch...")
                continue
            
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Accumulate loss for monitoring
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Track loss (multiply back by accumulation_steps for true loss)
                running_loss += accumulated_loss * gradient_accumulation_steps
                steps += 1
                accumulated_loss = 0.0
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        # Handle remaining gradients if batch doesn't divide evenly
        if accumulated_loss > 0:
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += accumulated_loss * gradient_accumulation_steps
            steps += 1

        avg_train_loss = running_loss / steps if steps > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = evaluate_on_validation(encoder_model, model_type, val_loader)
        val_losses.append(val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ★ Best validation loss! Saving checkpoint...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  ⚠️  Early stopping triggered!")
                break

    # Save training log
    loss_file = os.path.join(save_path, "loss_log.txt")
    with open(loss_file, "w") as f:
        f.write(f"Training: {model_type}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Final Train Loss: {avg_train_loss:.6f}\n")
        f.write(f"Stopped at Epoch: {epoch+1}/{settings.SIMCSE_EPOCHS}\n")
        f.write(f"Effective Batch Size: {effective_batch_size}\n\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"Epoch {i+1}: Train={t_loss:.6f}, Val={v_loss:.6f}\n")

    print(f'\n{"="*60}')
    print(f'Training Completed: {model_type.upper()}')
    print(f'{"="*60}')
    print(f'  Best Val Loss: {best_val_loss:.4f}')
    print(f'  Final Train Loss: {avg_train_loss:.4f}')
    print(f'  Epochs Trained: {epoch+1}/{settings.SIMCSE_EPOCHS}')
    print(f'  Effective Batch Size: {effective_batch_size}')
    print(f'  Saved to: {save_path}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SimCSE for code embeddings (Ye et al. 2025 method)"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['codet5', 'graphcodebert', 'all'],
        help="Model architecture to train"
    )
    args = parser.parse_args()
    
    if args.model == "all":
        for m in ["codet5", "graphcodebert"]:
            train(m)
    else:
        train(args.model)