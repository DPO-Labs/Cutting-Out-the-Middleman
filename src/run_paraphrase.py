import sys
import os
import copy

# ─── Auto-detect project root and add src/ to path ───
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, script_dir)

# ─── Disable HF hub to prevent hangs on import ───
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ─── Import from our project modules ───
from model import GPT2Wrapper
from trainer import Trainer
from dpo_utils import compute_dpo_loss

# ─── Device ───
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ===== Cell =====

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===== Cell =====

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
print('transformers ok')
from datasets import load_dataset
print('datasets ok')


# ===== Cell =====

# ══════════════════════════════════════════════════
#  HYPERPARAMETERS — change these to run ablations
# ══════════════════════════════════════════════════

MODEL_NAME      = 'gpt2'      # 'gpt2' (124M) or 'gpt2-large' (774M)
MAX_LENGTH      = 128         # max tokenized sequence length
BATCH_SIZE        = 8           # batch size (DPO needs 2x model, so smaller)
PREDICT_BATCH_SIZE = 64         # inference batch (no gradients for prediction)
SFT_LR            = 5e-6        # Lower learning rate prevents loss spikes
DPO_LR            = 1e-6        # Lower DPO learning rate for stable alignment
SFT_EPOCHS        = 1           # 1 epoch quick test
DPO_EPOCHS        = 1           # 1 epoch DPO quick test
BETA            = 0.1         # DPO beta (paper swept {0.01, 0.1}, best was 0.1)
TRAIN_SUBSET    = 5000        # tiny subset to verify everything works

# ─── Critical token IDs (GPT-2 vocabulary) ───
# These are the natural cloze-style answer tokens for yes/no
# Token 3919 = "no"   (no leading space; follows newline naturally)
# Token 8505 = "yes"  (no leading space; follows newline naturally)
NO_TOKEN_ID  = 3919
YES_TOKEN_ID = 8505

print(f"Config: model={MODEL_NAME}, sft_lr={SFT_LR}, dpo_lr={DPO_LR}, beta={BETA}")

# ===== Cell =====

def make_prompt(q1: str, q2: str) -> str:
    """
    Format a question pair as a cloze-style prompt.
    The model will predict the NEXT token after this prompt,
    which should be ' yes' (8505) or ' no' (3919).
    """
    return (
        f'Question 1: "{q1}"\n'
        f'Question 2: "{q2}"\n'
        f'Are these questions asking the same thing?\n'
    )

# Quick sanity check — verify token IDs are correct
tokenizer_check = AutoTokenizer.from_pretrained('gpt2')
assert tokenizer_check.encode('no')[0]  == NO_TOKEN_ID,  f"NO_TOKEN_ID mismatch! Got {tokenizer_check.encode('no')[0]}, expected {NO_TOKEN_ID}"
assert tokenizer_check.encode('yes')[0] == YES_TOKEN_ID, f"YES_TOKEN_ID mismatch! Got {tokenizer_check.encode('yes')[0]}, expected {YES_TOKEN_ID}"
print(f"Token IDs confirmed:  'no' = {NO_TOKEN_ID},  'yes' = {YES_TOKEN_ID}")

# ===== Cell =====

class QQPDataset(Dataset):
    """
    QQP dataset formatted for SFT cloze-style training.

    Each item returns:
        input_ids      — tokenized prompt (left-padded to MAX_LENGTH)
        attention_mask — 1 for real tokens, 0 for padding
        labels         — same as input_ids, but -100 everywhere except
                         the LAST position, which holds the correct token
                         (YES_TOKEN_ID or NO_TOKEN_ID)

    Why left-padding?
        GPT-2 is causal — we want the last real token position to predict
        the answer. Left-padding ensures the answer is always at position [-1].
    """

    def __init__(self, questions1, questions2, labels, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.prompts    = [make_prompt(q1, q2) for q1, q2 in zip(questions1, questions2)]
        # Convert dataset labels (0/1) to vocabulary token IDs
        self.answer_ids = [YES_TOKEN_ID if lbl == 1 else NO_TOKEN_ID for lbl in labels]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt    = self.prompts[idx]
        answer_id = self.answer_ids[idx]

        # Tokenize the prompt (no answer appended — we predict it)
        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids      = enc['input_ids'].squeeze(0)       # (max_length,)
        attention_mask = enc['attention_mask'].squeeze(0)  # (max_length,)

        # Labels: -100 everywhere (ignored by cross-entropy)
        # except at the last REAL token position → correct answer token
        labels = torch.full_like(input_ids, fill_value=-100)
        last_real_pos = attention_mask.sum().item() - 1  # last non-padding position
        labels[last_real_pos] = answer_id

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'answer_id':      torch.tensor(answer_id),  # for evaluation
        }

# ===== Cell =====

class QQPDPODataset(Dataset):
    """
    QQP dataset formatted for DPO training.

    For each example, winner and loser are just different token labels
    on the SAME prompt. The prompt input_ids are identical for both.

    winner_labels: -100 everywhere, correct token at last real position
    loser_labels:  -100 everywhere, incorrect token at last real position

    This is the key insight from the paper:
    We get free preference pairs from labeled QQP data — no annotation needed.
    """

    def __init__(self, questions1, questions2, labels, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.prompts    = [make_prompt(q1, q2) for q1, q2 in zip(questions1, questions2)]
        self.labels     = labels  # original 0/1 labels

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        lbl    = self.labels[idx]

        # Correct / incorrect token IDs for this example
        winner_token = YES_TOKEN_ID if lbl == 1 else NO_TOKEN_ID
        loser_token  = NO_TOKEN_ID  if lbl == 1 else YES_TOKEN_ID

        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        # Both winner and loser share the same prompt input
        last_real_pos = attention_mask.sum().item() - 1

        winner_labels = torch.full_like(input_ids, fill_value=-100)
        winner_labels[last_real_pos] = winner_token

        loser_labels = torch.full_like(input_ids, fill_value=-100)
        loser_labels[last_real_pos] = loser_token

        return {
            # Same prompt for both — DPO only differs in labels
            'winner_input_ids':       input_ids,
            'winner_attention_mask':  attention_mask,
            'winner_labels':          winner_labels,
            'loser_input_ids':        input_ids,          # identical to winner
            'loser_attention_mask':   attention_mask,
            'loser_labels':           loser_labels,
        }

# ===== Cell =====

# ─── Load tokenizer ───
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 has no pad token — use EOS as pad (standard approach)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'   # left-pad so last real token is always at position [-1]
print(f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

# ===== Cell =====

# ─── Download QQP training CSV if missing (subprocess, won't hang kernel) ───
import os, subprocess
TRAIN_CSV = 'src/train-00000-of-00001.csv'
if not os.path.exists(TRAIN_CSV):
    print("Downloading QQP training data (one-time)...")
    subprocess.run([sys.executable, '-c', 'from datasets import load_dataset; ds = load_dataset("glue", "qqp", split="train"); ds.to_csv("src/train-00000-of-00001.csv", index=False)'], check=True)
    print("Train CSV saved to", TRAIN_CSV)
else:
    print("Train CSV already cached.")

# ─── Load training data from local CSV ───
train_df = pd.read_csv(TRAIN_CSV)

if TRAIN_SUBSET:
    train_df = train_df.iloc[:TRAIN_SUBSET]

train_q1  = train_df['question1'].tolist()
train_q2  = train_df['question2'].tolist()
train_lbl = train_df['label'].tolist()
print(f"Train examples: {len(train_q1):,}")
print(f"Label distribution — paraphrase: {sum(train_lbl):,}, non-paraphrase: {len(train_lbl)-sum(train_lbl):,}")

# ===== Cell =====

# ─── Load dev and test from local CSVs ───
DEV_CSV  = 'src/validation-00000-of-00001.csv'
TEST_CSV = 'src/test-00000-of-00001 (1).csv'

dev_df  = pd.read_csv(DEV_CSV)
test_df = pd.read_csv(TEST_CSV)

dev_q1  = dev_df['question1'].tolist()
dev_q2  = dev_df['question2'].tolist()
dev_lbl = dev_df['label'].tolist()

test_q1 = test_df['question1'].tolist()
test_q2 = test_df['question2'].tolist()

print(f"Dev examples:  {len(dev_q1):,}")
print(f"Test examples: {len(test_q1):,} (no labels)")

# ===== Cell =====

# ─── Build PyTorch Datasets & DataLoaders ───

# SFT datasets
train_sft_ds = QQPDataset(train_q1, train_q2, train_lbl, tokenizer, MAX_LENGTH)
dev_sft_ds   = QQPDataset(dev_q1,   dev_q2,   dev_lbl,   tokenizer, MAX_LENGTH)

train_sft_loader = DataLoader(train_sft_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
dev_sft_loader   = DataLoader(dev_sft_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# DPO datasets (same data, different format)
train_dpo_ds  = QQPDPODataset(train_q1, train_q2, train_lbl, tokenizer, MAX_LENGTH)
train_dpo_loader = DataLoader(train_dpo_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

print(f"SFT train batches: {len(train_sft_loader):,}")
print(f"SFT dev batches:   {len(dev_sft_loader):,}")

# Quick sample check
sample = train_sft_ds[0]
print(f"\nSample prompt (decoded):\n{tokenizer.decode(sample['input_ids'], skip_special_tokens=True)[:200]}")
print(f"Answer token id: {sample['answer_id'].item()}  →  '{tokenizer.decode([sample['answer_id'].item()])}'")

# ===== Cell =====

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate paraphrase detection accuracy.

    For each example:
    1. Run forward pass → get logits at last real token position
    2. Compare logits[NO_TOKEN_ID] vs logits[YES_TOKEN_ID]
    3. Predict 'yes' if YES logit is higher, 'no' otherwise

    Returns:
        accuracy (float): fraction of correct predictions
    """
    model.eval()
    correct = 0
    total   = 0

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        input_ids      = batch['input_ids'].to(device)       # (B, L)
        attention_mask = batch['attention_mask'].to(device)  # (B, L)
        answer_ids     = batch['answer_id'].to(device)        # (B,)

        # Forward pass — shape: (B, L, vocab_size)
        with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
            logits = model(input_ids, attention_mask)

        # Get the last real token position for each example in the batch
        last_positions = attention_mask.sum(dim=1) - 1        # (B,)

        # Extract logits at last position for each example
        # logits[b, last_positions[b], :] → shape (B, vocab_size)
        batch_indices  = torch.arange(len(input_ids), device=device)
        last_logits    = logits[batch_indices, last_positions, :]  # (B, vocab_size)

        # Compare " yes" vs " no" logits — pick the higher one
        yes_logits = last_logits[:, YES_TOKEN_ID]  # (B,)
        no_logits  = last_logits[:, NO_TOKEN_ID]   # (B,)

        # Predicted token: YES_TOKEN_ID if yes > no, else NO_TOKEN_ID
        predicted = torch.where(yes_logits > no_logits,
                                torch.tensor(YES_TOKEN_ID, device=device),
                                torch.tensor(NO_TOKEN_ID,  device=device))

        correct += (predicted == answer_ids).sum().item()
        total   += len(answer_ids)

    accuracy = correct / total
    model.train()
    return accuracy


@torch.no_grad()
def generate_predictions(model, questions1, questions2, tokenizer, max_length, device, batch_size=64):
    """
    Generate predictions for test data (no labels).
    Returns list of token IDs (3919 or 8505) for each example.
    """
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(questions1), batch_size), desc='Predicting test'):
        batch_q1 = questions1[i:i+batch_size]
        batch_q2 = questions2[i:i+batch_size]
        prompts  = [make_prompt(q1, q2) for q1, q2 in zip(batch_q1, batch_q2)]

        enc = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
            logits = model(input_ids, attention_mask)
        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices  = torch.arange(len(input_ids), device=device)
        last_logits    = logits[batch_indices, last_positions, :]

        yes_logits = last_logits[:, YES_TOKEN_ID]
        no_logits  = last_logits[:, NO_TOKEN_ID]
        predicted  = torch.where(yes_logits > no_logits,
                                 torch.tensor(YES_TOKEN_ID, device=device),
                                 torch.tensor(NO_TOKEN_ID,  device=device))

        predictions.extend(predicted.cpu().tolist())

    model.train()
    return predictions

print("Evaluation helpers defined.")

# ===== Cell =====

# ─── Initialise model ───
print(f"Loading {MODEL_NAME}...")
policy_model = GPT2Wrapper(model_name=MODEL_NAME, device=DEVICE)
print(f"Model loaded: {MODEL_NAME} on {DEVICE}")

# Check baseline accuracy BEFORE any training
print("\nBaseline accuracy (untrained GPT-2):")
baseline_acc = evaluate(policy_model, dev_sft_loader, DEVICE)
print(f"  Dev accuracy: {baseline_acc:.4f}")

# ===== Cell =====

# ─── Initialise Trainer for SFT ───
# ref_model is not needed for SFT — only required for DPO
sft_trainer = Trainer(
    policy_model=policy_model,
    ref_model=None,           # not needed for SFT
    learning_rate=SFT_LR,
    device=DEVICE,
    beta=BETA
)
print("SFT Trainer initialised.")

# ===== Cell =====

# ─── SFT Training Loop with per-epoch dev evaluation ───
# We add dev accuracy evaluation after each epoch manually,
# since the trainer tracks loss but not our cloze-style accuracy metric.

sft_dev_accuracies = []
best_sft_accuracy  = 0.0
best_sft_state     = None

print("Starting SFT training...")
print(f"Staff baseline to beat: 0.882 dev accuracy\n")

for epoch in range(SFT_EPOCHS):
    # ── one epoch of SFT ──
    policy_model.train()
    epoch_losses = []

    pbar = tqdm(train_sft_loader, desc=f"SFT Epoch {epoch+1}/{SFT_EPOCHS}")
    for batch in pbar:
        loss, metrics = sft_trainer.sft_train_step(batch)
        sft_trainer.scaler.scale(loss).backward()
        sft_trainer.scaler.step(sft_trainer.optimizer)
        sft_trainer.scaler.update()
        sft_trainer.optimizer.zero_grad()

        epoch_losses.append(metrics['sft_loss'])
        pbar.set_postfix({'loss': f"{metrics['sft_loss']:.4f}"})

    avg_loss = sum(epoch_losses) / len(epoch_losses)

    # ── evaluate on dev set after each epoch ──
    dev_acc = evaluate(policy_model, dev_sft_loader, DEVICE)
    sft_dev_accuracies.append(dev_acc)

    print(f"Epoch {epoch+1}/{SFT_EPOCHS} | Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}", end="")

    # ── save best checkpoint ──
    if dev_acc > best_sft_accuracy:
        best_sft_accuracy = dev_acc
        best_sft_state = copy.deepcopy(policy_model.model.state_dict())
        print("  ← best so far", end="")

    # Flag if we beat the staff baseline
    if dev_acc > 0.882:
        print("  ✓ BEATS staff baseline", end="")
    print()

print(f"\nSFT Complete. Best dev accuracy: {best_sft_accuracy:.4f}")

# ===== Cell =====

# ─── Load best SFT checkpoint before DPO ───
policy_model.model.load_state_dict(best_sft_state)
print(f"Loaded best SFT checkpoint (dev acc: {best_sft_accuracy:.4f})")

# Save the SFT checkpoint to disk
torch.save(best_sft_state, 'sft_best_checkpoint.pt')
print("SFT checkpoint saved to sft_best_checkpoint.pt")

# Free stale SFT optimizer (~1 GB) and move state dict to CPU before DPO
del sft_trainer
best_sft_state = {k: v.cpu() for k, v in best_sft_state.items()}
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

# ===== Cell =====

# ─── Create frozen reference model from the best SFT checkpoint ───
# get_reference_model() deep-copies the model and sets requires_grad=False
ref_model = policy_model.get_reference_model()
ref_model.eval()

# Verify reference model is frozen
frozen = sum(1 for p in ref_model.model.parameters() if not p.requires_grad)
total  = sum(1 for p in ref_model.model.parameters())
print(f"Reference model: {frozen}/{total} parameters frozen (all should be frozen)")

# ===== Cell =====

# ─── DPO Optimizer (separate from SFT optimizer) ───
dpo_optimizer = AdamW(
    policy_model.model.parameters(),
    lr=DPO_LR,
    weight_decay=0.0   # no weight decay for DPO (following paper)
)

# ===== Cell =====

def dpo_step_paraphrase(batch, policy_model, ref_model, beta, device, padding_id=-100):
    """
    Single DPO training step for paraphrase detection.

    For paraphrase detection, winner and loser use the SAME input prompt.
    The difference is only in the target token (yes vs no).

    We call compute_dpo_loss ONCE with both winner_labels and loser_labels,
    passing the same logits for both (since the prompt is identical).

    DPO loss = -log σ(β * (log π_θ(y_win|x)/π_ref(y_win|x)
                          - log π_θ(y_los|x)/π_ref(y_los|x)))

    Args:
        batch: dict with winner_input_ids, winner_attention_mask,
                        winner_labels, loser_labels
        policy_model: GPT2Wrapper being trained
        ref_model:    GPT2Wrapper frozen reference (from SFT)
        beta:         DPO temperature (0.1 from paper)
        device:       torch device

    Returns:
        loss:    scalar DPO loss
        metrics: dict with dpo_loss, margin, accuracy
    """
    # Both winner and loser share the same prompt input
    input_ids      = batch['winner_input_ids'].to(device)
    attention_mask = batch['winner_attention_mask'].to(device)
    winner_labels  = batch['winner_labels'].to(device)
    loser_labels   = batch['loser_labels'].to(device)

    # ── Policy forward pass (gradients enabled) ──
    # Shape: (B, L, vocab_size)
    policy_logits = policy_model(input_ids, attention_mask)

    # ── Reference forward pass (no gradients) ──
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask)

    # ── Compute DPO loss using dpo_utils.compute_dpo_loss ──
    # We pass the same logits for both winner and loser sequences
    # (because the prompt is identical — only labels differ)
    loss, metrics = compute_dpo_loss(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        winner_labels=winner_labels,
        loser_labels=loser_labels,
        beta=beta,
        padding_token_id=padding_id
    )

    return loss, metrics

print("DPO step function defined.")

# ===== Cell =====

# ─── DPO Training Loop ───
dpo_dev_accuracies = []
best_dpo_accuracy  = 0.0
best_dpo_state     = None

print("Starting DPO training...")
print(f"SFT best dev accuracy: {best_sft_accuracy:.4f}  ← DPO must exceed this")
print(f"Beta = {BETA}\n")

for epoch in range(DPO_EPOCHS):
    policy_model.train()
    epoch_losses    = []
    epoch_dpo_accs  = []  # DPO's internal accuracy metric (winner preferred over loser)
    epoch_margins   = []

    pbar = tqdm(train_dpo_loader, desc=f"DPO Epoch {epoch+1}/{DPO_EPOCHS}")
    dpo_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    for batch in pbar:
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
            loss, metrics = dpo_step_paraphrase(
                batch, policy_model, ref_model, BETA, DEVICE
            )

        dpo_scaler.scale(loss).backward()
        dpo_scaler.step(dpo_optimizer)
        dpo_scaler.update()
        dpo_optimizer.zero_grad()

        epoch_losses.append(loss.item())
        epoch_dpo_accs.append(metrics['accuracy'])
        epoch_margins.append(metrics['margin'])

        pbar.set_postfix({
            'loss':   f"{loss.item():.4f}",
            'margin': f"{metrics['margin']:.4f}"
        })

    avg_loss   = sum(epoch_losses)   / len(epoch_losses)
    avg_margin = sum(epoch_margins)  / len(epoch_margins)

    # ── evaluate cloze accuracy on dev set ──
    dev_acc = evaluate(policy_model, dev_sft_loader, DEVICE)
    dpo_dev_accuracies.append(dev_acc)

    print(f"DPO Epoch {epoch+1}/{DPO_EPOCHS} | Loss: {avg_loss:.4f} | Margin: {avg_margin:.4f} | Dev Acc: {dev_acc:.4f}", end="")

    if dev_acc > best_dpo_accuracy:
        best_dpo_accuracy = dev_acc
        best_dpo_state    = copy.deepcopy(policy_model.model.state_dict())
        print("  ← best DPO", end="")

    if dev_acc > best_sft_accuracy:
        print("  ✓ DPO BEATS SFT", end="")

    print()

print(f"\nDPO Complete. Best dev accuracy: {best_dpo_accuracy:.4f}")

# ===== Cell =====

# ─── Load best DPO model and do final evaluation ───
policy_model.model.load_state_dict(best_dpo_state)
final_dev_acc = evaluate(policy_model, dev_sft_loader, DEVICE)

print("\n" + "="*55)
print(" RESULTS SUMMARY")
print("="*55)
print(f"  Staff baseline (paper)         : 0.882 dev")
print(f"  SFT best                       : {best_sft_accuracy:.4f} dev")
print(f"  SFT + DPO best                 : {best_dpo_accuracy:.4f} dev")
print(f"  SFT ∆ vs baseline              : {best_sft_accuracy - 0.882:+.4f}")
print(f"  DPO ∆ vs SFT                   : {best_dpo_accuracy - best_sft_accuracy:+.4f}")
print("="*55)

print("\nSFT dev accuracy per epoch:")
for i, acc in enumerate(sft_dev_accuracies, 1):
    print(f"  Epoch {i}: {acc:.4f}")

print("\nDPO dev accuracy per epoch:")
for i, acc in enumerate(dpo_dev_accuracies, 1):
    print(f"  Epoch {i}: {acc:.4f}")

# ===== Cell =====

# ─── Generate test predictions (no labels — token IDs expected by autograder) ───
# The autograder expects token IDs 3919 (no) or 8505 (yes) in the prediction column.
# Only runs on full dataset (TRAIN_SUBSET=None); skipped for validation runs.

if TRAIN_SUBSET is None:
    print("Generating test predictions...")
    test_predictions = generate_predictions(
        policy_model, test_q1, test_q2, tokenizer, MAX_LENGTH, DEVICE,
        batch_size=PREDICT_BATCH_SIZE
    )

    pred_df = pd.DataFrame({
        'idx':        test_df['idx'],
        'prediction': test_predictions
    })
    pred_df.to_csv('paraphrase_predictions.csv', index=False)
    print(f"Saved {len(pred_df):,} predictions to paraphrase_predictions.csv")
    pred_df.head(5)
else:
    print(f"Skipping test predictions (TRAIN_SUBSET={TRAIN_SUBSET}, not a full run)")

# ===== Cell =====

# ─── β ablation (short — 1 epoch each), error analysis, and model save ───
# Only on full run; skipped during validation.
# Also: clean up old optimizers & move state dicts to CPU to prevent OOM.

if TRAIN_SUBSET is None:
    # ── Free memory from old optimizers before starting beta ablation ──
    del sft_trainer, dpo_optimizer
    # Move saved state dicts to CPU to free CUDA memory
    best_sft_state = {k: v.cpu() for k, v in best_sft_state.items()}
    best_dpo_state = {k: v.cpu() for k, v in best_dpo_state.items()}
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    beta_results = {}

    for beta_val in [0.01, 0.1]:
        print(f"\nTesting β = {beta_val}...")

        policy_model.model.load_state_dict(
            {k: v.to(DEVICE) for k, v in best_sft_state.items()}
        )
        beta_optimizer = AdamW(policy_model.model.parameters(), lr=DPO_LR)

        policy_model.train()
        pbar = tqdm(train_dpo_loader, desc=f"β={beta_val} DPO (1 epoch)")
        beta_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
        for batch in pbar:
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                loss, metrics = dpo_step_paraphrase(
                    batch, policy_model, ref_model, beta_val, DEVICE
                )
            beta_scaler.scale(loss).backward()
            beta_scaler.step(beta_optimizer)
            beta_scaler.update()
            beta_optimizer.zero_grad()

        acc = evaluate(policy_model, dev_sft_loader, DEVICE)
        beta_results[beta_val] = acc
        print(f"  β={beta_val} → dev accuracy: {acc:.4f}")

        # Free memory before next beta iteration
        del beta_optimizer, beta_scaler
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print("\nβ Ablation Summary:")
    print(f"  β=0.01: {beta_results.get(0.01, 'N/A'):.4f}")
    print(f"  β=0.10: {beta_results.get(0.1,  'N/A'):.4f}")
    print(f"  Paper finding: β=0.1 gives better accuracy (0.894 vs 0.881)")

    # ===== Cell =====

    # ─── Error analysis ───
    policy_model.model.load_state_dict(
        {k: v.to(DEVICE) for k, v in best_dpo_state.items()}
    )
    policy_model.eval()

    errors = []

    with torch.no_grad():
        for i in tqdm(range(min(2000, len(dev_q1))), desc='Error analysis'):
            prompt = make_prompt(dev_q1[i], dev_q2[i])
            enc = tokenizer(prompt, max_length=MAX_LENGTH, truncation=True,
                            padding='max_length', return_tensors='pt')

            input_ids = enc['input_ids'].to(DEVICE)
            attention_mask = enc['attention_mask'].to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                logits = policy_model(input_ids, attention_mask)
            last_pos = attention_mask.sum() - 1
            last_logits = logits[0, last_pos, :]

            yes_logit = last_logits[YES_TOKEN_ID].item()
            no_logit = last_logits[NO_TOKEN_ID].item()
            predicted = YES_TOKEN_ID if yes_logit > no_logit else NO_TOKEN_ID
            true_token = YES_TOKEN_ID if dev_lbl[i] == 1 else NO_TOKEN_ID

            if predicted != true_token:
                errors.append({
                    'idx': i,
                    'q1': dev_q1[i],
                    'q2': dev_q2[i],
                    'true': 'yes' if dev_lbl[i] == 1 else 'no',
                    'predicted': 'yes' if predicted == YES_TOKEN_ID else 'no',
                    'yes_logit': round(yes_logit, 3),
                    'no_logit': round(no_logit, 3),
                    'confidence': round(abs(yes_logit - no_logit), 3)
                })

    print(f"Errors found: {len(errors)} / 2000 ({100 * len(errors) / 2000:.1f}%)")
    print(f"\nSample errors (low confidence = model was unsure):")
    error_df = pd.DataFrame(errors)
    if len(error_df) > 0:
        print(error_df.sort_values('confidence').head(5)[['q1', 'q2', 'true', 'predicted', 'confidence']].to_string())

    # ===== Cell =====

    # ─── Save best DPO model ───
    torch.save(best_dpo_state, 'dpo_best_checkpoint.pt')
    print("DPO checkpoint saved to dpo_best_checkpoint.pt")

    print("\nAll done! Summary:")
    print(f"  Best SFT dev accuracy : {best_sft_accuracy:.4f}")
    print(f"  Best DPO dev accuracy : {best_dpo_accuracy:.4f}")
    print(f"  Staff baseline        : 0.882")

else:
    print(f"\nSkipping β ablation, error analysis, and model save (TRAIN_SUBSET={TRAIN_SUBSET})")
    print("\nAll done! Summary:")
    print(f"  Best SFT dev accuracy : {best_sft_accuracy:.4f}")
    print(f"  Best DPO dev accuracy : {best_dpo_accuracy:.4f}")
    print(f"  Staff baseline        : 0.882")