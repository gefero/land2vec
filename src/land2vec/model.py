import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torch.optim import Optimizer

from land2vec.tokenizer import Tokenizer


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool(),
        )

        self.register_buffer("position_ids", torch.arange(block_size), persistent=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        positions = self.position_ids[:T]  # type: ignore
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        mask = self.causal_mask[:T, :T]  # type: ignore
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=Tokenizer.VOCAB["[PAD]"],
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            next_token = torch.distributions.Categorical(logits=logits).sample()
            idx = torch.cat((idx, next_token.unsqueeze(1)), dim=1)
        return idx


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer | None = None,
    device: str = "cuda",
    scaler: torch.amp.GradScaler | None = None,  # type: ignore
    use_amp: bool = True,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.enable_grad() if training else torch.inference_mode():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=use_amp
            ):
                _, loss = model(x, y)

            if training:
                optimizer.zero_grad()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.detach()

    total_loss = total_loss / len(data_loader)

    return total_loss.item()  # type: ignore
