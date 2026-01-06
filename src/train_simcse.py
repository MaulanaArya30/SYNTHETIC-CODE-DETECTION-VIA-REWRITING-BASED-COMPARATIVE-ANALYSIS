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


def simcse_loss(z1, z2, temperature=0.07, debug=False):
    batch_size = z1.shape[0]
    
    if debug:
        print(f"\n[PRE-NORM] z1 norm: {z1.norm(dim=1).mean():.4f}, z2 norm: {z2.norm(dim=1).mean():.4f}")
    
    # Normalize to unit sphere
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    if debug:
        print(f"[POST-NORM] z1 norm: {z1.norm(dim=1).mean():.4f}, z2 norm: {z2.norm(dim=1).mean():.4f}")
    
    # Compute similarities (will be between -1 and 1 after normalization)
    sim_11 = torch.matmul(z1, z1.T)
    sim_22 = torch.matmul(z2, z2.T)
    sim_12 = torch.matmul(z1, z2.T)
    
    if debug:
        print(f"[PRE-TEMP] sim_12 range: [{sim_12.min():.3f}, {sim_12.max():.3f}]")
        print(f"[PRE-TEMP] sim_12 diag mean: {sim_12.diag().mean():.3f}")
    
    # Apply temperature scaling
    sim_11 = sim_11 / temperature
    sim_22 = sim_22 / temperature
    sim_12 = sim_12 / temperature
    
    if debug:
        print(f"[POST-TEMP] sim_12 range: [{sim_12.min():.3f}, {sim_12.max():.3f}]")
    
    # Mask self-similarities
    mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
    sim_11 = sim_11.masked_fill(mask, -9e15)
    sim_22 = sim_22.masked_fill(mask, -9e15)
    
    # Concatenate and compute loss
    logits = torch.cat([sim_12, sim_11], dim=1)
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    
    if debug:
        print(f"[LOSS] {loss.item():.4f}\n")
    
    return loss


def get_embeddings(encoder_model, model_type, outputs):
    """
    Extract embeddings with proper pooling.
    FIXED: Correct parameter name (was 'ecoder_model')
    """
    if model_type == 'codet5':
        # Mean pooling for CodeT5
        emb = outputs.last_hidden_state.mean(dim=1)
    else:
        # CLS token for GraphCodeBERT
        emb = outputs.last_hidden_state[:, 0]
    
    return emb


def train(model_type):
    print(f'\n{"="*60}')
    print(f'SimCSE Training: {model_type.upper()}')
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
    train_dataset = CodeDataset(code_samples, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    optimizer = AdamW(encoder_model.parameters(), lr=settings.SIMCSE_LR)
    
    # Add learning rate scheduler to prevent overconfidence
    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_steps = len(train_loader) * settings.SIMCSE_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    epoch_losses = []
    best_loss = float('inf')

    print(f"Configuration:")
    print(f"  Epochs: {settings.SIMCSE_EPOCHS}")
    print(f"  Batch size: {settings.SIMCSE_BATCH_SIZE}")
    print(f"  Learning rate: {settings.SIMCSE_LR}")
    print(f"  Samples: {len(code_samples)}")
    print(f"  Device: {settings.DEVICE}")

    for epoch in range(settings.SIMCSE_EPOCHS):
        print(f'{"─"*60}')
        print(f'Epoch {epoch+1}/{settings.SIMCSE_EPOCHS}')
        print(f'{"─"*60}')
        
        encoder_model.train()  # Enable dropout
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc="Training", unit="it")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(settings.DEVICE)
            attention_mask = batch['attention_mask'].to(settings.DEVICE)

            # Two forward passes with different dropout
            outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract embeddings
            emb_1 = get_embeddings(encoder_model, model_type, outputs_1)
            emb_2 = get_embeddings(encoder_model, model_type, outputs_2)

            # First batch diagnostics
            if batch_idx == 0:
                debug_mode = True
                with torch.no_grad():
                    diff = (emb_1 - emb_2).abs().mean().item()
                    sim = F.cosine_similarity(emb_1, emb_2, dim=1).mean().item()
                    print(f"[First Batch Check]")
                    print(f"  Embedding difference: {diff:.6f} (should be > 0.001)")
                    print(f"  Cosine similarity: {sim:.4f} (should be < 0.99)")
                    if diff < 0.0001:
                        print(f"  ⚠️  Dropout may not be active!")
                    else:
                        print(f"  ✓ Dropout active")
            else:
                debug_mode = False

            # Compute loss (using temperature 0.07 to prevent overconfidence)
            loss = simcse_loss(emb_1, emb_2, temperature=0.07, debug=debug_mode)
            
            # Sanity checks
            if not torch.isfinite(loss):
                print(f"\n⚠️  Non-finite loss! Skipping batch...")
                continue
            
            if batch_idx == 0:
                print(f"  First loss: {loss.item():.4f}")
                if loss.item() < 0.1:
                    print(f"  ⚠️  Loss too low! Check implementation!")
                    print(f"  Expected: 2-4, Got: {loss.item():.4f}")
                elif loss.item() > 10:
                    print(f"  ⚠️  Loss too high!")
                else:
                    print(f"  ✓ Loss looks good")
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate

            running_loss += loss.item()
            steps += 1
            
            # Show current learning rate occasionally
            if batch_idx % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / steps
        epoch_losses.append(avg_loss)
        
        print(f"  Average Loss: {avg_loss:.4f}")
        
        if avg_loss < 0.1:
            print(f"  ⚠️  CRITICAL: Loss too low! Implementation bug likely!")
        elif avg_loss > 5.0:
            print(f"  ⚠️  Loss high, may need more epochs")
        else:
            print(f"  ✓ Loss in expected range")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ★ Best loss! Saving...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    # Save log
    loss_file = os.path.join(save_path, "loss_log.txt")
    with open(loss_file, "w") as f:
        f.write(f"Training: {model_type}\n")
        f.write(f"Best: {best_loss:.6f}\n")
        f.write(f"Final: {avg_loss:.6f}\n\n")
        for i, l in enumerate(epoch_losses):
            f.write(f"Epoch {i+1}: {l:.6f}\n")

    print(f'\n{"="*60}')
    print(f'Completed: {model_type.upper()}')
    print(f'{"="*60}')
    print(f'  Final: {avg_loss:.4f}')
    print(f'  Best: {best_loss:.4f}')
    print(f'  Saved: {save_path}')
    
    if best_loss > 2.0:
        print(f'\n  ⚠️  High loss - consider more epochs')
    elif best_loss < 0.1:
        print(f'\n  ⚠️  Very low loss - check for bugs!')
    else:
        print(f'\n  ✓ Training successful!?')
    print()


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