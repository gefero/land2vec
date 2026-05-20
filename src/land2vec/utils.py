from pathlib import Path
from json import dump, load
from dataclasses import asdict
import torch
import pandas as pd

from land2vec.config import Config
from land2vec.tokenizer import Tokenizer
from land2vec.model import GPTDecoder


def get_target_folder(model_name: str):
    if (Path("context") / "land2vec").exists():
        target = Path("/context") / "land2vec" / "models" / model_name
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
