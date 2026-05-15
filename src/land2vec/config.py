from dataclasses import dataclass


@dataclass
class Config:
    # model
    block_size: int = 8
    n_embd: int = 32
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.1
    # training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
