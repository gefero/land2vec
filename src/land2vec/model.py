import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer

from land2vec.tokenizer import Tokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, is_causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.is_causal = is_causal

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, is_causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, is_causal=is_causal)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTDecoder(nn.Module):
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
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("position_ids", torch.arange(block_size), persistent=False)

    def hidden_states(self, x) -> torch.Tensor:
        "Estado contextual de n_embd dims por posición, antes de lm_head -- útil para extraer un embedding de la v1 (p. ej. promediando sobre posiciones)."
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        positions = self.position_ids[:T]  # type: ignore
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        return self.ln_f(x)

    def forward(self, x) -> torch.Tensor:
        logits = self.lm_head(self.hidden_states(x))
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = -float("inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class TrajectoryAutoencoder(nn.Module):
    """Autoencoder no autorregresivo: comprime una secuencia completa a un vector
    z de embed_dim y la reconstruye a partir de ese único vector.

    A diferencia de GPTDecoder, tanto el encoder como el decoder usan atención
    bidireccional (Block con is_causal=False): el objetivo no es predecir el
    próximo token, sino forzar que toda la señal de reconstrucción pase por el
    cuello de botella z. El decoder recibe únicamente z (difundido a las
    seq_len posiciones + position embedding), sin ver los tokens de entrada, así
    que no puede "copiar" la secuencia usando contexto local.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in ("mean", "query"):
            raise ValueError(f"pooling debe ser 'mean' o 'query', recibido: {pooling!r}")
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pooling = pooling

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(seq_len, n_embd)
        self.register_buffer("position_ids", torch.arange(seq_len), persistent=False)

        self.encoder_blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout, is_causal=False) for _ in range(n_layer)]
        )
        self.encoder_ln = nn.LayerNorm(n_embd)
        if pooling == "query":
            self.pool_query = nn.Parameter(torch.randn(n_embd) * n_embd**-0.5)
        self.to_latent = nn.Linear(n_embd, embed_dim)

        self.from_latent = nn.Linear(embed_dim, n_embd)
        self.decoder_blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout, is_causal=False) for _ in range(n_layer)]
        )
        self.decoder_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        "h: (B, T, C) -> (B, C)"
        if self.pooling == "mean":
            return h.mean(dim=1)
        # pooling == "query": atención de una sola cabeza con un query aprendido,
        # en vez de promediar todas las posiciones con el mismo peso.
        scores = (h @ self.pool_query) / (h.size(-1) ** 0.5)  # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (h * weights).sum(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        "x: (B, seq_len) de ids de token -> z: (B, embed_dim)"
        B, T = x.shape
        if T != self.seq_len:
            raise ValueError(f"encode() espera secuencias de largo {self.seq_len}, recibió {T}")

        h = self.token_embedding(x) + self.position_embedding(self.position_ids)  # type: ignore
        h = self.encoder_blocks(h)
        h = self.encoder_ln(h)
        return self.to_latent(self._pool(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        "z: (B, embed_dim) -> logits: (B, seq_len, vocab_size)"
        B = z.shape[0]
        h = self.from_latent(z).unsqueeze(1).expand(B, self.seq_len, -1)
        h = h + self.position_embedding(self.position_ids).unsqueeze(0)  # type: ignore
        h = self.decoder_blocks(h)
        h = self.decoder_ln(h)
        return self.lm_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    weights: torch.Tensor,
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

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                weight=weights.to(logits.dtype),
                ignore_index=Tokenizer.VOCAB["[UNK]"]
            )

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
