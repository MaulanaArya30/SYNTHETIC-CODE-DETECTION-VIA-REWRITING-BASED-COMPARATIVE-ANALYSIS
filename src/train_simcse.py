import os
os.makedirs("/dev/shm/.hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/dev/shm/.hf_cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import argparse

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


# ── NEW: MLP Projection Head ──────────────────────────────────────────────────
# Added to match Ye et al. (2025) Section "Similarity Measurement":
#   "During the unsupervised training stage, we introduce an MLP layer to
#    obtain the final representation h of x."
# The head is used ONLY during training; it is discarded at save time so that
# the saved encoder produces clean CLS embeddings at inference (matching the
# paper's inference procedure).
class MLPProjectionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# ─────────────────────────────────────────────────────────────────────────────


def simcse_loss(z1, z2, temperature=settings.SIMCSE_TEMP):
    """
    SimCSE contrastive loss (NT-Xent) — Ye et al. (2025) Equation 1.

    Temperature is set to 0.1 to match the paper (previously 0.07).
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


def get_cls_embedding(model_type, outputs):
    """
    Extract CLS / mean-pool embedding BEFORE the projection head.
    This is what gets saved and used at inference.
    """
    if model_type == 'codet5':
        return outputs.last_hidden_state.mean(dim=1)
    else:
        # CLS token — matches the paper exactly
        return outputs.last_hidden_state[:, 0]


@torch.no_grad()
def evaluate_on_validation(encoder_model, projection_head, model_type, val_loader):
    """Evaluate SimCSE loss on validation set (with projection head)."""
    encoder_model.eval()
    projection_head.eval()
    total_loss = 0.0
    steps = 0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(settings.DEVICE)
        attention_mask = batch['attention_mask'].to(settings.DEVICE)
        
        outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Project for loss — matches training exactly
        emb_1 = projection_head(get_cls_embedding(model_type, outputs_1))
        emb_2 = projection_head(get_cls_embedding(model_type, outputs_2))
        
        loss = simcse_loss(emb_1, emb_2)
        total_loss += loss.item()
        steps += 1
    
    return total_loss / steps if steps > 0 else float('inf')


def train(model_type):
    print(f'\n{"="*60}')
    print(f'SimCSE Training (Ye et al. 2025): {model_type.upper()}')
    print(f'{"="*60}')

    # ── Load encoder ──────────────────────────────────────────────────────────
    if model_type == 'codet5':
        tokenizer = RobertaTokenizer.from_pretrained(settings.CODET5_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(settings.CODET5_MODEL_NAME)
        encoder_model = model.encoder
        hidden_size = model.config.d_model
        save_path = settings.CODET5_SIMCSE_PATH

    elif model_type == 'graphcodebert':
        tokenizer = RobertaTokenizer.from_pretrained(settings.GCB_MODEL_NAME)
        model = RobertaModel.from_pretrained(settings.GCB_MODEL_NAME)
        encoder_model = model
        hidden_size = model.config.hidden_size
        save_path = settings.GCB_SIMCSE_PATH

    else:
        raise ValueError('Invalid model_type')

    os.makedirs(save_path, exist_ok=True)
    encoder_model = encoder_model.to(settings.DEVICE)

    # ── NEW: Instantiate projection head ──────────────────────────────────────
    # hidden_size is read from the model config so it is always correct
    # (768 for GraphCodeBERT, 768 for CodeT5-base).
    projection_head = MLPProjectionHead(hidden_size).to(settings.DEVICE)
    print(f"  MLP projection head: {hidden_size} → ReLU → {hidden_size}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data...")
    code_samples = load_simcse_training_data()
    
    split_idx = int(len(code_samples) * 0.95)
    train_samples = code_samples[:split_idx]
    val_samples = code_samples[split_idx:]
    
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Val:   {len(val_samples):,} samples")
    
    train_dataset = CodeDataset(train_samples, tokenizer)
    val_dataset = CodeDataset(val_samples, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.SIMCSE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── Optimizer — includes projection head parameters ───────────────────────
    # Paper uses Adam (we use AdamW which is standard); LR 1e-4 matching paper.
    optimizer = AdamW(
        list(encoder_model.parameters()) + list(projection_head.parameters()),
        lr=settings.SIMCSE_LR,   # set to 1e-4 in settings.py to match paper
    )

    # ── NEW: Linear decay scheduler (replaces CosineAnnealingLR) ─────────────
    # Ye et al.: "Adam optimizer … maximum learning rate 1e-4 with linear decay"
    gradient_accumulation_steps = getattr(settings, 'GRADIENT_ACCUMULATION_STEPS', 1)
    total_optimizer_steps = (
        len(train_loader) // gradient_accumulation_steps
    ) * settings.SIMCSE_EPOCHS
    warmup_steps = int(total_optimizer_steps * 0.06)   # ~6 % warm-up (common default)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    # ─────────────────────────────────────────────────────────────────────────

    effective_batch_size = settings.SIMCSE_BATCH_SIZE * gradient_accumulation_steps

    # Early stopping
    best_val_loss = float('inf')
    patience = getattr(settings, 'EARLY_STOPPING_PATIENCE', 8)
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"\nConfiguration (Ye et al. 2025):")
    print(f"  Epochs:                   {settings.SIMCSE_EPOCHS}")
    print(f"  Physical batch size:       {settings.SIMCSE_BATCH_SIZE}")
    print(f"  Gradient accum steps:      {gradient_accumulation_steps}")
    print(f"  Effective batch size:      {effective_batch_size}")
    print(f"  Learning rate:             {settings.SIMCSE_LR}  (paper: 1e-4)")
    print(f"  LR schedule:               Linear decay with {warmup_steps} warmup steps")
    print(f"  Temperature:               {settings.SIMCSE_TEMP}  (paper value)")
    print(f"  MLP projection head:       YES (discarded at save)")
    print(f"  Early stopping patience:   {patience}")
    print(f"  Device:                    {settings.DEVICE}\n")

    for epoch in range(settings.SIMCSE_EPOCHS):
        print(f'{"─"*60}')
        print(f'Epoch {epoch+1}/{settings.SIMCSE_EPOCHS}')
        print(f'{"─"*60}')

        encoder_model.train()
        projection_head.train()   # keep head in train mode too
        running_loss = 0.0
        accumulated_loss = 0.0
        steps = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(settings.DEVICE)
            attention_mask = batch['attention_mask'].to(settings.DEVICE)

            # Two forward passes → different dropout masks
            outputs_1 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_2 = encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            # ── NEW: pass through projection head before loss ──────────────
            # Raw CLS/mean-pool embeddings first, then projected
            cls_1 = get_cls_embedding(model_type, outputs_1)
            cls_2 = get_cls_embedding(model_type, outputs_2)
            proj_1 = projection_head(cls_1)
            proj_2 = projection_head(cls_2)
            # ─────────────────────────────────────────────────────────────

            # Temperature 0.1 — matches the paper
            loss = simcse_loss(proj_1, proj_2, temperature=0.1)

            if not torch.isfinite(loss):
                print(f"\n⚠️  Non-finite loss! Skipping batch...")
                continue

            loss = loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients for both encoder AND projection head
                torch.nn.utils.clip_grad_norm_(
                    list(encoder_model.parameters()) + list(projection_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += accumulated_loss * gradient_accumulation_steps
                steps += 1
                accumulated_loss = 0.0

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}',
                })

        # Handle leftover accumulated gradients
        if accumulated_loss > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder_model.parameters()) + list(projection_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += accumulated_loss * gradient_accumulation_steps
            steps += 1

        avg_train_loss = running_loss / steps if steps > 0 else 0.0
        train_losses.append(avg_train_loss)

        val_loss = evaluate_on_validation(
            encoder_model, projection_head, model_type, val_loader
        )
        val_losses.append(val_loss)

        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ★ Best validation loss! Saving checkpoint...")
            # ── IMPORTANT: save the base model only, NOT the projection head ──
            # This matches the paper: "the MLP layer is removed, and we use only
            # the last-layer hidden states of the [CLS] token" at inference.
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            # projection_head is intentionally NOT saved
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
        f.write(f"MLP head + linear decay + temp={settings.SIMCSE_TEMP}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Final Train Loss: {avg_train_loss:.6f}\n")
        f.write(f"Stopped at Epoch: {epoch+1}/{settings.SIMCSE_EPOCHS}\n")
        f.write(f"Effective Batch Size: {effective_batch_size}\n\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"Epoch {i+1}: Train={t_loss:.6f}, Val={v_loss:.6f}\n")

    print(f'\n{"="*60}')
    print(f'Training Completed: {model_type.upper()}')
    print(f'{"="*60}')
    print(f'  Best Val Loss:      {best_val_loss:.4f}')
    print(f'  Final Train Loss:   {avg_train_loss:.4f}')
    print(f'  Epochs Trained:     {epoch+1}/{settings.SIMCSE_EPOCHS}')
    print(f'  Effective Batch:    {effective_batch_size}')
    print(f'  Saved to:           {save_path}')
    print(f'  NOTE: MLP head was discarded — saved encoder is inference-ready.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SimCSE for code embeddings (Ye et al. 2025 method)"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['codet5', 'graphcodebert', 'all'],
        help="Model architecture to train",
    )
    args = parser.parse_args()

    if args.model == "all":
        for m in ["codet5", "graphcodebert"]:
            train(m)
    else:
        train(args.model)