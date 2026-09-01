"""Entrena TrajectoryAutoencoder (land2vec v2): comprime una trayectoria de 23
años de uso de suelo a un vector de embed_dim y la reconstruye.

Combina data/id_seqs_text_2000_2022_chaco_santiago_frontier.zip (submuestreado,
igual que las zonas nuevas) con las 7 zonas de entrenamiento diversas construidas
por `build_eval_zones.py --zone-set train` -- nunca con las 7 zonas de
`--zone-set eval`, que son el benchmark held-out (ver notebooks/eval_ood_zones.ipynb).

Uso:
    # una corrida suelta
    python scripts/train_autoencoder.py --embed-dim 8 --out models/autoencoder_v2

    # barrido primario: dimensión del embedding, d en {4,8,12,16,32}
    python scripts/train_autoencoder.py --sweep dim --out-dir models/sweep_dim

    # barrido secundario (lr, n_layer, pooling, pesos de clase) al mejor d
    python scripts/train_autoencoder.py --sweep secondary --embed-dim 8 --out-dir models/sweep_secondary
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from land2vec.config import Config
from land2vec.dataset import SequenceDatasetAutoencoder
from land2vec.extract import subsample_constant_sequences
from land2vec.model import TrajectoryAutoencoder, run_epoch
from land2vec.tokenizer import Tokenizer
from land2vec.utils import collect_predictions, compute_metrics, save_config, save_model, save_train_results

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CHACO_FILE = DATA_DIR / "id_seqs_text_2000_2022_chaco_santiago_frontier.zip"
TRAIN_ZONE_NAMES = [
    "puna_salta_catamarca",
    "patagonia_santacruz",
    "periurbano_gba",
    "corrientes_humedal",
    "delta_oeste",
    "pampa_deprimida",
    "yungas",
]
TRAIN_ZONE_FILES = [DATA_DIR / f"id_seqs_text_2000_2022_{name}.zip" for name in TRAIN_ZONE_NAMES]

VALID_LABELS = [i for i in Tokenizer.VOCAB.values() if i != Tokenizer.VOCAB["[UNK]"]]

# Barrido secundario: baseline + variantes de a un factor por vez, al mejor d
# del barrido primario. No es un grid completo (2^4=16 corridas) -- son las
# combinaciones que importan para decidir cada hiperparámetro por separado.
SECONDARY_SWEEP = [
    {"name": "baseline", "lr": 1e-3, "n_layer": 4, "pooling": "mean", "weighted": True},
    {"name": "lr_bajo", "lr": 3e-4, "n_layer": 4, "pooling": "mean", "weighted": True},
    {"name": "n_layer_2", "lr": 1e-3, "n_layer": 2, "pooling": "mean", "weighted": True},
    {"name": "pooling_query", "lr": 1e-3, "n_layer": 4, "pooling": "query", "weighted": True},
    {"name": "sin_pesos", "lr": 1e-3, "n_layer": 4, "pooling": "mean", "weighted": False},
    {"name": "lr_bajo_query", "lr": 3e-4, "n_layer": 4, "pooling": "query", "weighted": True},
    {"name": "n_layer_2_query", "lr": 1e-3, "n_layer": 2, "pooling": "query", "weighted": True},
    {"name": "lr_bajo_n_layer_2", "lr": 3e-4, "n_layer": 2, "pooling": "mean", "weighted": True},
]


def load_training_sequences(max_constant_fraction: float, max_rows: int | None = None) -> pd.Series:
    "Chaco-Santiago (submuestreado acá) + las 7 zonas de train (ya submuestreadas al construirlas)."
    chaco = pd.read_csv(CHACO_FILE, usecols=["seqs"])
    chaco = subsample_constant_sequences(chaco, max_fraction=max_constant_fraction)

    zones = [pd.read_csv(f, usecols=["seqs"]) for f in TRAIN_ZONE_FILES]
    combined = pd.concat([chaco] + zones, ignore_index=True)["seqs"]

    if max_rows is not None and len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return combined


def compute_class_weights(sequences: pd.Series) -> torch.Tensor:
    "Peso inverso a la frecuencia de cada clase (normalizado a media 1 entre las presentes)."
    freq = sequences.str.split("-").explode().value_counts()
    weights = torch.zeros(len(Tokenizer.VOCAB))
    for token, idx in Tokenizer.VOCAB.items():
        weights[idx] = freq.get(token, 0)
    valid = weights > 0
    inv = torch.zeros_like(weights)
    inv[valid] = 1.0 / weights[valid]
    inv[valid] = inv[valid] / inv[valid].mean()
    return inv


def train_one(
    sequences: pd.Series,
    config: Config,
    out_dir: Path,
    weighted: bool = True,
    verbose: bool = True,
) -> dict:
    "Entrena una corrida, guarda config.json/model.pt/train_data.csv en out_dir, devuelve las métricas finales."
    torch.manual_seed(config.seed)

    dataset = SequenceDatasetAutoencoder(sequences)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    weights = compute_class_weights(sequences) if weighted else torch.ones(len(Tokenizer.VOCAB))
    weights[Tokenizer.VOCAB["[UNK]"]] = 0.0
    weights = weights.to(config.device)

    model = TrajectoryAutoencoder(
        vocab_size=len(Tokenizer.VOCAB),
        seq_len=config.block_size,
        embed_dim=config.embed_dim,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        pooling=config.pooling,
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    use_amp = config.device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None  # type: ignore

    history: dict[str, list] = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, weights, optimizer=optimizer, device=config.device, scaler=scaler, use_amp=use_amp)

        preds, targets, val_loss = collect_predictions(model, val_loader, config.device, weights, max_batches=None)
        metrics = compute_metrics(targets, preds, labels=VALID_LABELS)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_macro_f1"].append(metrics["macro_f1"])

        if verbose:
            print(
                f"  epoch {epoch:>2d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f} ({time.time() - t0:.0f}s)"
            )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if verbose:
                    print(f"  early stopping en epoch {epoch} (patience={config.patience})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir)
    save_model(model, out_dir)
    save_train_results(history, out_dir)

    return {"embed_dim": config.embed_dim, "best_val_macro_f1": best_macro_f1, "n_epochs": len(history["epoch"])}


def build_config(args, embed_dim: int, lr: float, n_layer: int, pooling: str) -> Config:
    return Config(
        block_size=23,  # 2000-2022 inclusive
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=n_layer,
        dropout=args.dropout,
        arch="seq_autoencoder",
        embed_dim=embed_dim,
        pooling=pooling,
        epochs=args.epochs,
        lr=lr,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=["none", "dim", "secondary"], default="none")
    parser.add_argument("--embed-dim", type=int, default=8, help="d; para --sweep secondary, el d fijo elegido en el barrido primario")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 8, 12, 16, 32], help="valores de d para --sweep dim")
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--pooling", choices=["mean", "query"], default="mean")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-constant-fraction", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=None, help="debug: recorta el dataset combinado a esta cantidad de filas")
    parser.add_argument("--out", type=Path, default=None, help="carpeta de salida para --sweep none")
    parser.add_argument("--out-dir", type=Path, default=None, help="carpeta base para --sweep dim/secondary (una subcarpeta por corrida)")
    args = parser.parse_args()

    print("Cargando y combinando secuencias de entrenamiento...")
    sequences = load_training_sequences(args.max_constant_fraction, max_rows=args.max_rows)
    print(f"  {len(sequences):,} secuencias (Chaco-Santiago submuestreado + {len(TRAIN_ZONE_NAMES)} zonas nuevas)")

    if args.sweep == "none":
        out_dir = args.out or Path("models/autoencoder_v2")
        config = build_config(args, args.embed_dim, args.lr, args.n_layer, args.pooling)
        print(f"Entrenando embed_dim={config.embed_dim} n_layer={config.n_layer} pooling={config.pooling} lr={config.lr} -> {out_dir}")
        result = train_one(sequences, config, out_dir)
        print(result)

    elif args.sweep == "dim":
        out_dir = args.out_dir or Path("models/sweep_dim")
        results = []
        for d in args.dims:
            run_dir = out_dir / f"d{d}"
            config = build_config(args, d, args.lr, args.n_layer, args.pooling)
            print(f"\n=== d={d} -> {run_dir} ===")
            results.append(train_one(sequences, config, run_dir))
        summary = pd.DataFrame(results)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print("\nResumen barrido dim:")
        print(summary)

    elif args.sweep == "secondary":
        out_dir = args.out_dir or Path("models/sweep_secondary")
        results = []
        for run in SECONDARY_SWEEP:
            run_dir = out_dir / run["name"]
            config = build_config(args, args.embed_dim, run["lr"], run["n_layer"], run["pooling"])
            print(f"\n=== {run['name']} (lr={run['lr']} n_layer={run['n_layer']} pooling={run['pooling']} weighted={run['weighted']}) -> {run_dir} ===")
            result = train_one(sequences, config, run_dir, weighted=run["weighted"])
            result["name"] = run["name"]
            results.append(result)
        summary = pd.DataFrame(results)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print("\nResumen barrido secundario:")
        print(summary)


if __name__ == "__main__":
    main()
