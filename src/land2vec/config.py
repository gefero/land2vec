from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    # model
    block_size: int = 12
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    
    # training
    epochs: int = 25
    validate_every: int = 2
    patience: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 5
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    
    seed: int = 42