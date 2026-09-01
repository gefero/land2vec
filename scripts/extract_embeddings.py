"""Extrae embeddings (vector z de embed_dim) de land2vec v2 (TrajectoryAutoencoder)
para una zona ya construida con build_eval_zones.py (o cualquier CSV/zip con
columnas ID,seqs en el mismo formato).

Uso:
    python scripts/extract_embeddings.py --model models/autoencoder_v2 --zone ibera
    python scripts/extract_embeddings.py --model models/autoencoder_v2 --seqs-file data/id_seqs_text_2000_2022_ibera.zip --out data/embeddings_ibera.zip
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from land2vec.dataset import SequenceDatasetAutoencoder
from land2vec.utils import load_config, load_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@torch.inference_mode()
def extract_embeddings(model, seqs_file: Path, device: str, batch_size: int = 1024) -> pd.DataFrame:
    "Corre model.encode() sobre todas las filas de seqs_file, en el mismo orden del CSV."
    df = pd.read_csv(seqs_file)
    dataset = SequenceDatasetAutoencoder(df["seqs"])
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    model.eval()
    chunks = []
    for x, _ in loader:
        chunks.append(model.encode(x.to(device)).cpu())
    z = torch.cat(chunks).numpy()

    out = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
    out.insert(0, "ID", df["ID"].values)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="carpeta del modelo (config.json + model.pt)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zone", help="nombre de zona en data/ (id_seqs_text_2000_2022_<zona>.zip)")
    group.add_argument("--seqs-file", type=Path, help="ruta directa a un CSV/zip con columnas ID,seqs")
    parser.add_argument("--out", type=Path, default=None, help="por defecto data/embeddings_<zona>.zip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    seqs_file = args.seqs_file or DATA_DIR / f"id_seqs_text_2000_2022_{args.zone}.zip"
    zone_name = args.zone or seqs_file.stem.replace("id_seqs_text_2000_2022_", "")
    out_path = args.out or DATA_DIR / f"embeddings_{zone_name}.zip"

    config = load_config(args.model)
    if config.arch != "seq_autoencoder":
        raise ValueError(f"{args.model} tiene arch={config.arch!r}; se esperaba un modelo 'seq_autoencoder'")
    config = replace(config, device=args.device)
    model = load_model(config, args.model)

    print(f"Extrayendo embeddings ({config.embed_dim} dims) de {seqs_file} con {args.model} ...")
    out = extract_embeddings(model, seqs_file, args.device, batch_size=args.batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, compression="zip")
    print(f"Guardado: {out_path} ({len(out):,} filas, {config.embed_dim} dims)")


if __name__ == "__main__":
    main()
