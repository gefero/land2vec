from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class Config:
    # model
    block_size: int = 12
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    # training
    epochs: int = 25
    lr: float = 1e-3
    patience: int = 5
    min_lr: float = 1e-6
    weight_decay: float = 1e-2

    # dataset
    num_workers: int = 2
    batch_size: int = 256
    
    # Globals
    device: Literal["cuda", "cpu"] = "cuda"
    seed: int = 42