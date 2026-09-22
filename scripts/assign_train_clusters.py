"""Asigna cluster a las 8 zonas de entrenamiento (`land2vec.zones.ZONES_BY_GROUP
["train"]`) contra las 6 configuraciones ya elegidas por
`scripts/tune_clustering.py --select` (`models/cluster_v2/chosen{suffix}.json`).

No reajusta nada: replica exactamente el paso de etiquetado del pool que hace
`refit_and_save()` en `scripts/tune_clustering.py` (transform + centroide más
cercano + corte "sin tipificar" con `untyped_dist_threshold`), pero sobre el
pool de entrenamiento en vez del de evaluación. El resultado es **in-sample
para el encoder** (vio estas secuencias al entrenar el autoencoder) aunque
**out-of-sample para el clustering** (que solo se ajustó sobre las 7 zonas de
evaluación) -- no es evidencia de generalización, ver
`docs/v2_autoencoder_training.md` §4.1 y `viz/clusters/README.md`.

Prerrequisitos:
    data/embeddings_<zona>.zip           por cada zona de entrenamiento, ver
                                          `scripts/extract_embeddings.py --zone <zona>`
    models/cluster_v2/chosen{suffix}.json  (los deja `tune_clustering.py --select`)

Escribe (mismo formato que `clusters_pooled_subsampled{suffix}.zip`, ID/zone/cluster):
    data/clusters_train_pooled{suffix}.zip

Uso:
    python scripts/assign_train_clusters.py
    python scripts/assign_train_clusters.py --only _medium
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from land2vec import cluster as C  # noqa: E402
from land2vec.zones import ZONES_BY_GROUP  # noqa: E402

DATA_DIR = ROOT / "data"
CLUSTER_DIR = ROOT / "models" / "cluster_v2"

# mismo criterio de suffixes que scripts/build_cluster_map.py:SUFFIXES / la
# matriz level_specs x family_specs de scripts/tune_clustering.py --select.
SUFFIXES = ["", "_parametric", "_medium", "_medium_parametric", "_coarse", "_coarse_parametric"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=CLUSTER_DIR)
    parser.add_argument("--only", choices=SUFFIXES, default=None,
                        help="procesar solo esta granularidad/familia")
    parser.add_argument("--max-constant-fraction", type=float, default=0.15,
                        help="tope de secuencias constantes en el pool (mismo default que tune_clustering.py --select)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_zones = ZONES_BY_GROUP["train"]
    missing_emb = [z for z in train_zones if not (args.data_dir / f"embeddings_{z}.zip").exists()]
    if missing_emb:
        sys.exit(
            "faltan embeddings de " + ", ".join(missing_emb) + " -- corré antes:\n" +
            "\n".join(f"  python scripts/extract_embeddings.py --model models/autoencoder_v2 --zone {z}"
                      for z in missing_emb)
        )

    print(f"Cargando el pool de entrenamiento ({len(train_zones)} zonas, "
          f"constantes al {args.max_constant_fraction:.0%})...")
    pool = C.load_pool_subsampled(
        train_zones, args.data_dir, max_fraction=args.max_constant_fraction, seed=args.seed,
    )
    print(f"  {len(pool.ids):,} filas")

    suffixes = [args.only] if args.only else SUFFIXES
    for suffix in suffixes:
        label = suffix.strip("_") or "fina"
        chosen_path = args.cluster_dir / f"chosen{suffix}.json"
        if not chosen_path.exists():
            print(f"[{label}] falta {chosen_path.name}, se omite")
            continue
        cfg = json.loads(chosen_path.read_text())
        transform = C.SpaceTransform.from_jsonable(cfg["transform"])
        centers = np.array(cfg["centers"])
        idx, dist = C.assign_pool(pool.z, transform, centers, return_dist=True)
        labels = np.where(dist > cfg["untyped_dist_threshold"], -1, idx)

        n_untyped = int((labels == -1).sum())
        print(f"[{label}] sin tipificar: {n_untyped:,} / {len(labels):,} ({n_untyped / len(labels):.1%})")
        for zone in train_zones:
            m = pool.zone == zone
            nz = int(m.sum())
            if nz == 0:
                continue
            nu = int((labels[m] == -1).sum())
            print(f"    {zone:26s} n={nz:>8,}  sin tipificar={nu / nz:.1%}")

        out = pd.DataFrame({"ID": pool.ids, "zone": pool.zone, "cluster": labels})
        out_path = args.data_dir / f"clusters_train_pooled{suffix}.zip"
        out.to_csv(out_path, index=False, compression="zip")
        print(f"  guardado: {out_path.relative_to(ROOT)} ({len(out):,} filas)")


if __name__ == "__main__":
    main()
