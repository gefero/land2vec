import torch
import torch.nn as nn
import torch.nn.functional as F

from src.land2vec.tokenizer import Tokenizer


class GPT(nn.Module):
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

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

        # final normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):

        B, T = x.shape

        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        # positions
        positions = torch.arange(T, device=x.device)

        # embeddings
        tok_emb = self.token_embedding(x)  # [B, T, C]
        pos_emb = self.position_embedding(positions)  # [T, C]

        x = tok_emb + pos_emb

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # transformer
        x = self.transformer(x, mask=mask)

        # final norm
        x = self.ln_f(x)

        # logits
        logits = self.lm_head(x)  # [B, T, vocab]

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=Tokenizer.VOCAB["[PAD]"],
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            # crop context
            idx_cond = idx[:, -self.block_size :]

            # forward
            logits, _ = self(idx_cond)

            # take last token
            logits = logits[:, -1, :]  # [B, vocab]

            # probabilities
            probs = F.softmax(logits, dim=-1)

            # sample
            next_token = torch.multinomial(probs, num_samples=1)

            # append
            idx = torch.cat((idx, next_token), dim=1)

        return idx
