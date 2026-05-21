from pathlib import Path
from json import dump, load
from typing import Literal
from dataclasses import asdict
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from land2vec.config import Config
from land2vec.tokenizer import Tokenizer
from land2vec.model import GPTDecoder


def get_target_folder(model_name: str):
    if Path.cwd() == (Path("/context") / "land2vec"):
        target = Path("models") / model_name
    else:
        target = Path("..") / "models" / model_name
    return target


# Config saving and loading
def save_config(config: Config, target_folder: Path):
    with open(target_folder / "config.json", "w") as f:
        dump(asdict(config), f, indent=2)
    print(f"Config saved to {target_folder / "config.json"}")


def load_config(target_folder: Path):
    with open(target_folder / "config.json", "r") as f:
        data = load(f)
    return Config(**data)


# Model saving and loading
def save_model(model: torch.nn.Module, target_folder: Path):
    torch.save(model.state_dict(), target_folder / "model.pt")
    print(f"Model saved to {target_folder / "model.pt"}")


def load_model(config: Config, target_folder: Path):
    model = GPTDecoder(
        vocab_size=len(Tokenizer.VOCAB),
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
    )
    model_state = torch.load(target_folder / "model.pt")
    model.load_state_dict(model_state)
    model = model.to(config.device)
    model.eval()
    return model


# Training data saving and loading
def save_train_results(results: dict[str, list], target_folder: Path):
    pd.DataFrame(results).to_csv(target_folder / "train_data.csv", index=False)
    print(f"Train data saved to {target_folder / "train_data.csv"}")


def load_train_results(target_folder: Path):
    return pd.read_csv(target_folder / "train_data.csv")


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: Literal["cpu", "cuda"],
    weights: torch.Tensor
):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=Tokenizer.VOCAB["[PAD]"],
            weight=weights,
            label_smoothing=0.1
        )
        total_loss += loss.detach()

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    preds = all_logits.reshape(-1, all_logits.size(-1)).argmax(dim=-1)
    targets = all_targets.reshape(-1)
    return preds.numpy(), targets.numpy(), total_loss / len(loader)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
