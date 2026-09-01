from pathlib import Path
from json import dump, load
from typing import Literal
from dataclasses import asdict
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from land2vec.config import Config
from land2vec.tokenizer import Tokenizer
from land2vec.model import GPTDecoder, TrajectoryAutoencoder


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
    print(f"Config saved to {target_folder / 'config.json'}")


def load_config(target_folder: Path):
    with open(target_folder / "config.json", "r") as f:
        data = load(f)
    return Config(**data)


# Model saving and loading
def save_model(model: torch.nn.Module, target_folder: Path):
    torch.save(model.state_dict(), target_folder / "model.pt")
    print(f"Model saved to {target_folder / 'model.pt'}")


def load_model(config: Config, target_folder: Path, arch: str | None = None):
    # arch permite forzar la arquitectura sin tocar config.json; por defecto se
    # usa config.arch (los config.json de antes de la v2 no tienen ese campo,
    # así que Config lo completa con su default "gpt_decoder").
    arch = arch or config.arch
    if arch == "gpt_decoder":
        model: nn.Module = GPTDecoder(
            vocab_size=len(Tokenizer.VOCAB),
            block_size=config.block_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
        )
    elif arch == "seq_autoencoder":
        if config.embed_dim is None:
            raise ValueError("config.embed_dim es obligatorio para arch='seq_autoencoder'")
        # block_size se reusa como seq_len: el largo fijo de secuencia que
        # espera el autoencoder (no hay ventaneo, a diferencia de GPTDecoder).
        model = TrajectoryAutoencoder(
            vocab_size=len(Tokenizer.VOCAB),
            seq_len=config.block_size,
            embed_dim=config.embed_dim,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            pooling=config.pooling,
        )
    else:
        raise ValueError(f"arch desconocida: {arch!r}")

    model_state = torch.load(target_folder / "model.pt", map_location=config.device)
    model.load_state_dict(model_state)
    model = model.to(config.device)
    model.eval()
    return model


# Training data saving and loading
def save_train_results(results: dict[str, list], target_folder: Path):
    pd.DataFrame(results).to_csv(target_folder / "train_data.csv", index=False)
    print(f"Train data saved to {target_folder / 'train_data.csv'}")


def load_train_results(target_folder: Path):
    return pd.read_csv(target_folder / "train_data.csv")


# Metrics
@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    weights: torch.Tensor,
    max_batches: int | None = 100,
):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    use_amp = device == "cuda"

    n_batches = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=Tokenizer.VOCAB["[UNK]"],
                weight=weights,
            )

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.reshape(-1).cpu().numpy())

        all_targets.extend(y.reshape(-1).cpu().numpy())

    preds = np.asarray(all_preds)
    targets = np.asarray(all_targets)

    avg_loss = total_loss / n_batches if n_batches else float("nan")

    return preds, targets, avg_loss


def compute_metrics(y_true, y_pred, labels: list[int] | None = None):
    # Por defecto fija las clases del vocabulario (sin [UNK]) en vez de dejar que
    # f1_score promedie solo sobre las clases presentes en y_true/y_pred: si no,
    # el macro F1 no es comparable entre evaluaciones con distinta composición de
    # clases (ver notebooks/test_2.ipynb, sección de la clase "B").
    if labels is None:
        labels = [i for i in Tokenizer.VOCAB.values() if i != Tokenizer.VOCAB["[UNK]"]]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }
