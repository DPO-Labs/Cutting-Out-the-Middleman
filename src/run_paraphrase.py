import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, script_dir)

from dpo_utils import compute_dpo_loss
from model import GPT2Wrapper


NO_TOKEN_ID = 3919
YES_TOKEN_ID = 8505


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass
class Config:
    profile: str
    model_name: str
    max_length: int
    micro_batch_size: int
    eval_batch_size: int
    predict_batch_size: int
    grad_accum_steps: int
    sft_lr: float
    dpo_lr: float
    sft_epochs: int
    dpo_epochs: int
    beta: float
    train_subset: Optional[int]
    trainable_last_n_layers: int
    use_gradient_checkpointing: bool
    local_files_only: bool
    run_test_predictions: bool
    run_beta_ablation: bool
    run_error_analysis: bool
    checkpoint_dir: str
    device: str


def build_config() -> Config:
    profile = os.getenv("PARAPHRASE_PROFILE", "smoke").strip().lower()

    defaults = {
        "smoke": {
            "max_length": 96,
            "micro_batch_size": 1,
            "eval_batch_size": 8,
            "predict_batch_size": 8,
            "grad_accum_steps": 4,
            "sft_lr": 5e-6,
            "dpo_lr": 1e-6,
            "sft_epochs": 1,
            "dpo_epochs": 1,
            "beta": 0.1,
            "train_subset": 256,
            "trainable_last_n_layers": 2,
            "use_gradient_checkpointing": True,
        },
        "rtx3050": {
            "max_length": 96,
            "micro_batch_size": 1,
            "eval_batch_size": 8,
            "predict_batch_size": 8,
            "grad_accum_steps": 16,
            "sft_lr": 5e-6,
            "dpo_lr": 1e-6,
            "sft_epochs": 1,
            "dpo_epochs": 1,
            "beta": 0.1,
            "train_subset": 30000,
            "trainable_last_n_layers": 4,
            "use_gradient_checkpointing": True,
        },
        "full": {
            "max_length": 128,
            "micro_batch_size": 2,
            "eval_batch_size": 16,
            "predict_batch_size": 16,
            "grad_accum_steps": 16,
            "sft_lr": 5e-6,
            "dpo_lr": 1e-6,
            "sft_epochs": 2,
            "dpo_epochs": 1,
            "beta": 0.1,
            "train_subset": None,
            "trainable_last_n_layers": 4,
            "use_gradient_checkpointing": True,
        },
    }

    if profile not in defaults:
        raise ValueError(f"Unknown PARAPHRASE_PROFILE='{profile}'. Use smoke, rtx3050, or full.")

    selected = defaults[profile]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_ckpt_dir = str(Path(project_root) / "checkpoints" / f"paraphrase_{profile}")

    subset_raw = os.getenv("TRAIN_SUBSET")
    train_subset = selected["train_subset"] if subset_raw is None else (None if subset_raw.lower() == "none" else int(subset_raw))

    return Config(
        profile=profile,
        model_name=os.getenv("MODEL_NAME", "gpt2"),
        max_length=env_int("MAX_LENGTH", selected["max_length"]),
        micro_batch_size=env_int("MICRO_BATCH_SIZE", selected["micro_batch_size"]),
        eval_batch_size=env_int("EVAL_BATCH_SIZE", selected["eval_batch_size"]),
        predict_batch_size=env_int("PREDICT_BATCH_SIZE", selected["predict_batch_size"]),
        grad_accum_steps=env_int("GRAD_ACCUM_STEPS", selected["grad_accum_steps"]),
        sft_lr=env_float("SFT_LR", selected["sft_lr"]),
        dpo_lr=env_float("DPO_LR", selected["dpo_lr"]),
        sft_epochs=env_int("SFT_EPOCHS", selected["sft_epochs"]),
        dpo_epochs=env_int("DPO_EPOCHS", selected["dpo_epochs"]),
        beta=env_float("BETA", selected["beta"]),
        train_subset=train_subset,
        trainable_last_n_layers=env_int("TRAINABLE_LAST_N_LAYERS", selected["trainable_last_n_layers"]),
        use_gradient_checkpointing=env_bool("USE_GRADIENT_CHECKPOINTING", selected["use_gradient_checkpointing"]),
        local_files_only=env_bool("LOCAL_FILES_ONLY", False),
        run_test_predictions=env_bool("RUN_TEST_PREDICTIONS", False),
        run_beta_ablation=env_bool("RUN_BETA_ABLATION", False),
        run_error_analysis=env_bool("RUN_ERROR_ANALYSIS", False),
        checkpoint_dir=os.getenv("CHECKPOINT_DIR", default_ckpt_dir),
        device=device,
    )


def configure_hf_access(local_files_only: bool) -> None:
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def make_prompt(q1: str, q2: str) -> str:
    return (
        f'Question 1: "{q1}"\n'
        f'Question 2: "{q2}"\n'
        f"Are these questions asking the same thing?\n"
    )


def trainable_parameters(model: torch.nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def apply_low_vram_tuning(wrapper: GPT2Wrapper, config: Config) -> None:
    backbone = wrapper.model
    if config.use_gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

    if not hasattr(backbone, "transformer") or not hasattr(backbone.transformer, "h"):
        return

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    trainable_last_n_layers = max(1, min(config.trainable_last_n_layers, len(backbone.transformer.h)))
    for block in backbone.transformer.h[-trainable_last_n_layers:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in backbone.transformer.ln_f.parameters():
        parameter.requires_grad = True


class QQPDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = [make_prompt(q1, q2) for q1, q2 in zip(questions1, questions2)]
        self.answer_ids = [YES_TOKEN_ID if lbl == 1 else NO_TOKEN_ID for lbl in labels]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        last_real_pos = attention_mask.sum().item() - 1
        labels[last_real_pos] = self.answer_ids[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_id": torch.tensor(self.answer_ids[idx]),
        }


class QQPDPODataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = [make_prompt(q1, q2) for q1, q2 in zip(questions1, questions2)]
        self.labels = labels

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        winner_token = YES_TOKEN_ID if label == 1 else NO_TOKEN_ID
        loser_token = NO_TOKEN_ID if label == 1 else YES_TOKEN_ID

        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        last_real_pos = attention_mask.sum().item() - 1

        winner_labels = torch.full_like(input_ids, fill_value=-100)
        winner_labels[last_real_pos] = winner_token
        loser_labels = torch.full_like(input_ids, fill_value=-100)
        loser_labels[last_real_pos] = loser_token

        return {
            "winner_input_ids": input_ids,
            "winner_attention_mask": attention_mask,
            "winner_labels": winner_labels,
            "loser_input_ids": input_ids,
            "loser_attention_mask": attention_mask,
            "loser_labels": loser_labels,
        }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer_ids = batch["answer_id"].to(device)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(input_ids, attention_mask)

        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(input_ids), device=device)
        last_logits = logits[batch_indices, last_positions, :]
        yes_logits = last_logits[:, YES_TOKEN_ID]
        no_logits = last_logits[:, NO_TOKEN_ID]
        predicted = torch.where(
            yes_logits > no_logits,
            torch.tensor(YES_TOKEN_ID, device=device),
            torch.tensor(NO_TOKEN_ID, device=device),
        )

        correct += (predicted == answer_ids).sum().item()
        total += len(answer_ids)

    model.train()
    return correct / total


@torch.no_grad()
def generate_predictions(model, questions1, questions2, tokenizer, max_length, device, batch_size):
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(questions1), batch_size), desc="Predicting test"):
        prompts = [make_prompt(q1, q2) for q1, q2 in zip(questions1[i:i + batch_size], questions2[i:i + batch_size])]
        enc = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(input_ids, attention_mask)

        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(input_ids), device=device)
        last_logits = logits[batch_indices, last_positions, :]
        yes_logits = last_logits[:, YES_TOKEN_ID]
        no_logits = last_logits[:, NO_TOKEN_ID]
        predicted = torch.where(
            yes_logits > no_logits,
            torch.tensor(YES_TOKEN_ID, device=device),
            torch.tensor(NO_TOKEN_ID, device=device),
        )
        predictions.extend(predicted.cpu().tolist())

    model.train()
    return predictions


def sft_train_step(policy_model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
        logits = policy_model(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )

    metrics = {
        "sft_loss": loss.item(),
        "perplexity": torch.exp(loss.detach()).item(),
    }
    return loss, metrics


def dpo_step_paraphrase(batch, policy_model, ref_model, beta, device):
    input_ids = batch["winner_input_ids"].to(device)
    attention_mask = batch["winner_attention_mask"].to(device)
    winner_labels = batch["winner_labels"].to(device)
    loser_labels = batch["loser_labels"].to(device)

    with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
        policy_logits = policy_model(input_ids, attention_mask)
        with torch.no_grad():
            ref_logits = ref_model(input_ids, attention_mask)
        loss, metrics = compute_dpo_loss(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            winner_labels=winner_labels,
            loser_labels=loser_labels,
            beta=beta,
            padding_token_id=-100,
        )

    return loss, metrics


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_state(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_state(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def save_sft_checkpoint(
    checkpoint_dir: Path,
    epoch_index: int,
    policy_model: GPT2Wrapper,
    optimizer: AdamW,
    scaler,
    best_sft_accuracy: float,
    best_sft_state: Dict,
    sft_dev_accuracies,
    config: Config,
) -> None:
    payload = {
        "epoch_index": epoch_index,
        "policy_state_dict": policy_model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_sft_accuracy": best_sft_accuracy,
        "best_sft_state_dict": best_sft_state,
        "sft_dev_accuracies": sft_dev_accuracies,
        "config": asdict(config),
    }
    save_state(checkpoint_dir / "sft_last.pt", payload)
    save_json(
        checkpoint_dir / "status.json",
        {
            "stage": "sft",
            "completed_sft_epochs": epoch_index + 1,
            "best_sft_accuracy": best_sft_accuracy,
            "config": asdict(config),
        },
    )


def save_dpo_checkpoint(
    checkpoint_dir: Path,
    epoch_index: int,
    policy_model: GPT2Wrapper,
    ref_model: GPT2Wrapper,
    optimizer: AdamW,
    scaler,
    best_sft_accuracy: float,
    best_sft_state: Dict,
    best_dpo_accuracy: float,
    best_dpo_state: Dict,
    sft_dev_accuracies,
    dpo_dev_accuracies,
    config: Config,
) -> None:
    payload = {
        "epoch_index": epoch_index,
        "policy_state_dict": policy_model.model.state_dict(),
        "reference_state_dict": ref_model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_sft_accuracy": best_sft_accuracy,
        "best_sft_state_dict": best_sft_state,
        "best_dpo_accuracy": best_dpo_accuracy,
        "best_dpo_state_dict": best_dpo_state,
        "sft_dev_accuracies": sft_dev_accuracies,
        "dpo_dev_accuracies": dpo_dev_accuracies,
        "config": asdict(config),
    }
    save_state(checkpoint_dir / "dpo_last.pt", payload)
    save_json(
        checkpoint_dir / "status.json",
        {
            "stage": "dpo",
            "completed_dpo_epochs": epoch_index + 1,
            "best_sft_accuracy": best_sft_accuracy,
            "best_dpo_accuracy": best_dpo_accuracy,
            "config": asdict(config),
        },
    )


def run_beta_ablation(
    policy_model: GPT2Wrapper,
    ref_model: GPT2Wrapper,
    train_dpo_loader,
    dev_sft_loader,
    best_sft_state: Dict,
    config: Config,
) -> Dict[float, float]:
    beta_results = {}

    for beta_value in [0.01, 0.1]:
        policy_model.model.load_state_dict(best_sft_state)
        beta_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.dpo_lr, weight_decay=0.0)
        beta_scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))
        policy_model.train()

        pbar = tqdm(train_dpo_loader, desc=f"β={beta_value} DPO")
        beta_optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            loss, metrics = dpo_step_paraphrase(batch, policy_model, ref_model, beta_value, config.device)
            loss = loss / config.grad_accum_steps
            beta_scaler.scale(loss).backward()

            if step % config.grad_accum_steps == 0 or step == len(train_dpo_loader):
                beta_scaler.step(beta_optimizer)
                beta_scaler.update()
                beta_optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{loss.item() * config.grad_accum_steps:.4f}", margin=f"{metrics['margin']:.4f}")

        beta_results[beta_value] = evaluate(policy_model, dev_sft_loader, config.device)
        del beta_optimizer, beta_scaler
        if config.device == "cuda":
            torch.cuda.empty_cache()

    return beta_results


def run_error_analysis(policy_model: GPT2Wrapper, tokenizer, dev_q1, dev_q2, dev_lbl, config: Config):
    policy_model.eval()
    errors = []

    with torch.no_grad():
        for i in tqdm(range(min(2000, len(dev_q1))), desc="Error analysis"):
            prompt = make_prompt(dev_q1[i], dev_q2[i])
            enc = tokenizer(prompt, max_length=config.max_length, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(config.device)
            attention_mask = enc["attention_mask"].to(config.device)

            with torch.amp.autocast(device_type=config.device, enabled=(config.device == "cuda")):
                logits = policy_model(input_ids, attention_mask)

            last_pos = attention_mask.sum() - 1
            last_logits = logits[0, last_pos, :]
            yes_logit = last_logits[YES_TOKEN_ID].item()
            no_logit = last_logits[NO_TOKEN_ID].item()
            predicted = YES_TOKEN_ID if yes_logit > no_logit else NO_TOKEN_ID
            true_token = YES_TOKEN_ID if dev_lbl[i] == 1 else NO_TOKEN_ID

            if predicted != true_token:
                errors.append(
                    {
                        "idx": i,
                        "q1": dev_q1[i],
                        "q2": dev_q2[i],
                        "true": "yes" if dev_lbl[i] == 1 else "no",
                        "predicted": "yes" if predicted == YES_TOKEN_ID else "no",
                        "confidence": round(abs(yes_logit - no_logit), 3),
                    }
                )

    return errors


def main():
    config = build_config()
    configure_hf_access(config.local_files_only)

    print(f"Using device: {config.device}")
    print(f"Profile: {config.profile}")
    if config.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(
        f"Config: model={config.model_name}, max_length={config.max_length}, "
        f"micro_batch={config.micro_batch_size}, grad_accum={config.grad_accum_steps}, "
        f"sft_epochs={config.sft_epochs}, dpo_epochs={config.dpo_epochs}, subset={config.train_subset}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=config.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer_check = AutoTokenizer.from_pretrained("gpt2", local_files_only=config.local_files_only)
    assert tokenizer_check.encode("no")[0] == NO_TOKEN_ID
    assert tokenizer_check.encode("yes")[0] == YES_TOKEN_ID
    print(f"Token IDs confirmed: 'no'={NO_TOKEN_ID}, 'yes'={YES_TOKEN_ID}")

    train_csv = Path(project_root) / "src" / "train-00000-of-00001.csv"
    dev_csv = Path(project_root) / "src" / "validation-00000-of-00001.csv"
    test_csv = Path(project_root) / "src" / "test-00000-of-00001 (1).csv"

    train_df = pd.read_csv(train_csv)
    if config.train_subset is not None:
        train_df = train_df.iloc[:config.train_subset]

    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)

    train_q1 = train_df["question1"].tolist()
    train_q2 = train_df["question2"].tolist()
    train_lbl = train_df["label"].tolist()
    dev_q1 = dev_df["question1"].tolist()
    dev_q2 = dev_df["question2"].tolist()
    dev_lbl = dev_df["label"].tolist()
    test_q1 = test_df["question1"].tolist()
    test_q2 = test_df["question2"].tolist()

    print(f"Train examples: {len(train_q1):,}")
    print(f"Dev examples: {len(dev_q1):,}")
    print(f"Test examples: {len(test_q1):,}")

    pin_memory = config.device == "cuda"
    train_sft_ds = QQPDataset(train_q1, train_q2, train_lbl, tokenizer, config.max_length)
    dev_sft_ds = QQPDataset(dev_q1, dev_q2, dev_lbl, tokenizer, config.max_length)
    train_dpo_ds = QQPDPODataset(train_q1, train_q2, train_lbl, tokenizer, config.max_length)

    train_sft_loader = DataLoader(train_sft_ds, batch_size=config.micro_batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    dev_sft_loader = DataLoader(dev_sft_ds, batch_size=config.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    train_dpo_loader = DataLoader(train_dpo_ds, batch_size=config.micro_batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    sample = train_sft_ds[0]
    print(f"Sample answer token: {sample['answer_id'].item()} -> '{tokenizer.decode([sample['answer_id'].item()])}'")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(checkpoint_dir / "run_config.json", asdict(config))

    policy_model = GPT2Wrapper(model_name=config.model_name, device=config.device, local_files_only=config.local_files_only)
    apply_low_vram_tuning(policy_model, config)
    trainable_count, total_count = count_parameters(policy_model.model)
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    baseline_acc = evaluate(policy_model, dev_sft_loader, config.device)
    print(f"Baseline dev accuracy: {baseline_acc:.4f}")

    sft_scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))
    sft_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.sft_lr, weight_decay=0.0)
    sft_dev_accuracies = []
    dpo_dev_accuracies = []
    best_sft_accuracy = 0.0
    best_dpo_accuracy = 0.0
    best_sft_state = None
    best_dpo_state = None
    start_sft_epoch = 0
    start_dpo_epoch = 0
    resume_stage = "sft"

    sft_last_path = checkpoint_dir / "sft_last.pt"
    dpo_last_path = checkpoint_dir / "dpo_last.pt"

    if env_bool("RESUME", False):
        if dpo_last_path.exists():
            state = load_state(dpo_last_path)
            policy_model.model.load_state_dict(state["policy_state_dict"])
            apply_low_vram_tuning(policy_model, config)
            sft_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.sft_lr, weight_decay=0.0)
            dpo_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.dpo_lr, weight_decay=0.0)
            dpo_optimizer.load_state_dict(state["optimizer_state_dict"])
            dpo_scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))
            dpo_scaler.load_state_dict(state["scaler_state_dict"])
            ref_model = policy_model.get_reference_model()
            ref_model.model.load_state_dict(state["reference_state_dict"])
            ref_model.eval()
            best_sft_accuracy = state["best_sft_accuracy"]
            best_sft_state = state["best_sft_state_dict"]
            best_dpo_accuracy = state["best_dpo_accuracy"]
            best_dpo_state = state["best_dpo_state_dict"]
            sft_dev_accuracies = state.get("sft_dev_accuracies", [])
            dpo_dev_accuracies = state.get("dpo_dev_accuracies", [])
            start_dpo_epoch = state["epoch_index"] + 1
            resume_stage = "dpo"
            print(f"Resuming DPO from epoch {start_dpo_epoch + 1}")
        elif sft_last_path.exists():
            state = load_state(sft_last_path)
            policy_model.model.load_state_dict(state["policy_state_dict"])
            apply_low_vram_tuning(policy_model, config)
            sft_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.sft_lr, weight_decay=0.0)
            sft_optimizer.load_state_dict(state["optimizer_state_dict"])
            sft_scaler.load_state_dict(state["scaler_state_dict"])
            best_sft_accuracy = state["best_sft_accuracy"]
            best_sft_state = state["best_sft_state_dict"]
            sft_dev_accuracies = state.get("sft_dev_accuracies", [])
            start_sft_epoch = state["epoch_index"] + 1
            print(f"Resuming SFT from epoch {start_sft_epoch + 1}")

    if resume_stage == "sft":
        print("Starting SFT training...")
        sft_optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_sft_epoch, config.sft_epochs):
            policy_model.train()
            epoch_losses = []
            pbar = tqdm(train_sft_loader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")

            for step, batch in enumerate(pbar, start=1):
                loss, metrics = sft_train_step(policy_model, batch, config.device)
                loss = loss / config.grad_accum_steps
                sft_scaler.scale(loss).backward()

                if step % config.grad_accum_steps == 0 or step == len(train_sft_loader):
                    sft_scaler.step(sft_optimizer)
                    sft_scaler.update()
                    sft_optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(metrics["sft_loss"])
                pbar.set_postfix(loss=f"{metrics['sft_loss']:.4f}")

            dev_acc = evaluate(policy_model, dev_sft_loader, config.device)
            sft_dev_accuracies.append(dev_acc)
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"SFT Epoch {epoch + 1}/{config.sft_epochs} | Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}")

            if dev_acc > best_sft_accuracy:
                best_sft_accuracy = dev_acc
                best_sft_state = copy.deepcopy(policy_model.model.state_dict())
                save_state(
                    checkpoint_dir / "sft_best.pt",
                    {"best_sft_accuracy": best_sft_accuracy, "best_sft_state_dict": best_sft_state, "config": asdict(config)},
                )

            save_sft_checkpoint(
                checkpoint_dir,
                epoch,
                policy_model,
                sft_optimizer,
                sft_scaler,
                best_sft_accuracy,
                best_sft_state,
                sft_dev_accuracies,
                config,
            )

    if best_sft_state is None:
        best_state = load_state(checkpoint_dir / "sft_best.pt")
        best_sft_accuracy = best_state["best_sft_accuracy"]
        best_sft_state = best_state["best_sft_state_dict"]

    policy_model.model.load_state_dict(best_sft_state)
    if config.device == "cuda":
        torch.cuda.empty_cache()

    ref_model = policy_model.get_reference_model()
    ref_model.eval()
    dpo_optimizer = AdamW(trainable_parameters(policy_model.model), lr=config.dpo_lr, weight_decay=0.0)
    dpo_scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))

    if env_bool("RESUME", False) and resume_stage == "dpo" and dpo_last_path.exists():
        state = load_state(dpo_last_path)
        policy_model.model.load_state_dict(state["policy_state_dict"])
        ref_model.model.load_state_dict(state["reference_state_dict"])
        dpo_optimizer.load_state_dict(state["optimizer_state_dict"])
        dpo_scaler.load_state_dict(state["scaler_state_dict"])

    print("Starting DPO training...")
    dpo_optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_dpo_epoch, config.dpo_epochs):
        policy_model.train()
        epoch_losses = []
        epoch_margins = []
        pbar = tqdm(train_dpo_loader, desc=f"DPO Epoch {epoch + 1}/{config.dpo_epochs}")

        for step, batch in enumerate(pbar, start=1):
            loss, metrics = dpo_step_paraphrase(batch, policy_model, ref_model, config.beta, config.device)
            loss = loss / config.grad_accum_steps
            dpo_scaler.scale(loss).backward()

            if step % config.grad_accum_steps == 0 or step == len(train_dpo_loader):
                dpo_scaler.step(dpo_optimizer)
                dpo_scaler.update()
                dpo_optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(loss.item() * config.grad_accum_steps)
            epoch_margins.append(metrics["margin"])
            pbar.set_postfix(loss=f"{loss.item() * config.grad_accum_steps:.4f}", margin=f"{metrics['margin']:.4f}")

        dev_acc = evaluate(policy_model, dev_sft_loader, config.device)
        dpo_dev_accuracies.append(dev_acc)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_margin = sum(epoch_margins) / len(epoch_margins)
        print(f"DPO Epoch {epoch + 1}/{config.dpo_epochs} | Loss: {avg_loss:.4f} | Margin: {avg_margin:.4f} | Dev Acc: {dev_acc:.4f}")

        if dev_acc > best_dpo_accuracy:
            best_dpo_accuracy = dev_acc
            best_dpo_state = copy.deepcopy(policy_model.model.state_dict())
            save_state(
                checkpoint_dir / "dpo_best.pt",
                {"best_dpo_accuracy": best_dpo_accuracy, "best_dpo_state_dict": best_dpo_state, "config": asdict(config)},
            )

        save_dpo_checkpoint(
            checkpoint_dir,
            epoch,
            policy_model,
            ref_model,
            dpo_optimizer,
            dpo_scaler,
            best_sft_accuracy,
            best_sft_state,
            best_dpo_accuracy,
            best_dpo_state,
            sft_dev_accuracies,
            dpo_dev_accuracies,
            config,
        )

    if best_dpo_state is None and (checkpoint_dir / "dpo_best.pt").exists():
        best_state = load_state(checkpoint_dir / "dpo_best.pt")
        best_dpo_accuracy = best_state["best_dpo_accuracy"]
        best_dpo_state = best_state["best_dpo_state_dict"]

    final_state = best_dpo_state if best_dpo_state is not None else best_sft_state
    policy_model.model.load_state_dict(final_state)
    final_dev_acc = evaluate(policy_model, dev_sft_loader, config.device)

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"Baseline dev accuracy       : {baseline_acc:.4f}")
    print(f"Best SFT dev accuracy       : {best_sft_accuracy:.4f}")
    print(f"Best DPO dev accuracy       : {best_dpo_accuracy:.4f}")
    print(f"Final loaded dev accuracy   : {final_dev_acc:.4f}")
    print(f"Checkpoint directory        : {checkpoint_dir}")
    print("=" * 55)

    if config.run_test_predictions and config.train_subset is None:
        predictions = generate_predictions(
            policy_model,
            test_q1,
            test_q2,
            tokenizer,
            config.max_length,
            config.device,
            config.predict_batch_size,
        )
        prediction_path = checkpoint_dir / "paraphrase_predictions.csv"
        pd.DataFrame({"idx": test_df["idx"], "prediction": predictions}).to_csv(prediction_path, index=False)
        print(f"Saved test predictions to {prediction_path}")

    if config.run_beta_ablation:
        beta_results = run_beta_ablation(policy_model, ref_model, train_dpo_loader, dev_sft_loader, best_sft_state, config)
        save_json(checkpoint_dir / "beta_ablation.json", {str(k): v for k, v in beta_results.items()})
        print(f"Beta ablation: {beta_results}")

    if config.run_error_analysis:
        errors = run_error_analysis(policy_model, tokenizer, dev_q1, dev_q2, dev_lbl, config)
        error_path = checkpoint_dir / "error_analysis.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"Saved error analysis to {error_path}")


if __name__ == "__main__":
    main()
