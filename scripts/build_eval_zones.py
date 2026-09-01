"""Construye datasets de zona a partir de data/landcover_timeseries_2000-2022.nc.

Dos conjuntos de zonas, seleccionables con --zone-set:

- "eval" (default): las 7 zonas out-of-domain usadas como benchmark held-out en
  notebooks/eval_ood_zones.ipynb. Sus bboxes no deben tocarse -- cambiar los
  datos invalidaría los resultados ya reportados en el README.
- "train": zonas nuevas para la v2 (autoencoder), en las mismas ecorregiones que
  las de "eval" pero con bboxes distintos y disjuntos de ellas, para sumar
  diversidad de uso de suelo al dataset de entrenamiento sin quemar el
  benchmark de evaluación.

En ambos casos, antes de guardar cada zona se verifica que su bounding box no
se solape con el área de entrenamiento original (Chaco-Santiago), el test set
actual, ni (para "train") con ninguna de las 7 zonas de "eval" -- evita repetir
el leakage documentado en notebooks/test_2.ipynb y evita filtrar el benchmark
de evaluación hacia el entrenamiento. También reporta la distribución de clases
resultante por zona.

Uso:
    python scripts/build_eval_zones.py
    python scripts/build_eval_zones.py --zone-set train --max-constant-fraction 0.15
    python scripts/build_eval_zones.py --zone-set eval --zones ibera puna_noa
"""

import argparse
from pathlib import Path

import numpy as np

from land2vec.extract import (
    LCCS_CODE_TO_TOKEN,
    crop_bbox,
    extract_zone,
    load_landcover_dataset,
    save_zone_csvs,
    subsample_constant_sequences,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# bbox = (minx, miny, maxx, maxy) en lon/lat
TRAIN_BBOX = (-63.44994621163554, -28.12819902009702, -59.37401847726054, -25.431378332142593)
TEST_BBOX = (-63.583374, -29.623609, -59.067993, -25.372568)

EVAL_ZONES: dict[str, tuple[float, float, float, float]] = {
    "puna_noa": (-67.0, -23.5, -65.0, -22.0),
    "patagonia_estepa": (-70.0, -43.0, -66.0, -40.0),
    "periurbano_cordoba": (-64.4, -31.6, -64.0, -31.2),
    "ibera": (-58.0, -29.0, -56.5, -27.5),
    "delta_parana": (-59.6, -33.8, -58.6, -32.4),
    "pampa_nucleo": (-62.0, -35.0, -60.0, -33.0),
    "misiones_selva": (-55.5, -27.5, -54.0, -25.5),
}

# Mismas ecorregiones que EVAL_ZONES (ver la tabla en el plan de la v2), bboxes
# distintos y disjuntos de ellas.
TRAIN_ZONES: dict[str, tuple[float, float, float, float]] = {
    "puna_salta_catamarca": (-68.5, -26.5, -66.5, -24.5),
    "patagonia_santacruz": (-71.0, -49.0, -67.0, -46.0),
    "periurbano_gba": (-58.8, -34.9, -58.3, -34.4),
    "corrientes_humedal": (-58.9, -27.4, -58.1, -26.3),
    "delta_oeste": (-59.95, -33.8, -59.65, -32.6),
    "pampa_deprimida": (-60.0, -37.5, -58.0, -36.0),
    "yungas": (-64.8, -25.5, -64.0, -24.0),
}

ZONE_SETS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "eval": EVAL_ZONES,
    "train": TRAIN_ZONES,
}


def bbox_overlap(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2
    return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)


def forbidden_bboxes(zone_set: str) -> dict[str, tuple[float, float, float, float]]:
    "bboxes contra los que ninguna zona de zone_set puede solaparse."
    forbidden = {"train_chaco_santiago": TRAIN_BBOX, "test_actual": TEST_BBOX}
    if zone_set == "train":
        # las zonas de entrenamiento de la v2 tampoco pueden pisar el benchmark de eval
        forbidden.update(EVAL_ZONES)
    return forbidden


def class_distribution(ds, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    ds_cropped = crop_bbox(ds, *bbox)
    codes = np.nan_to_num(ds_cropped["lccs_class"].values, nan=0).astype(int)
    vals, counts = np.unique(codes, return_counts=True)
    total = counts.sum()
    return {LCCS_CODE_TO_TOKEN.get(int(v), f"code{v}"): c / total * 100 for v, c in zip(vals, counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nc-path", type=Path, default=DATA_DIR / "landcover_timeseries_2000-2022.nc")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--zone-set", choices=list(ZONE_SETS), default="eval")
    parser.add_argument("--zones", nargs="+", default=None, help="subconjunto de nombres; por defecto, todas las del --zone-set")
    parser.add_argument(
        "--max-constant-fraction",
        type=float,
        default=None,
        help="submuestrea secuencias constantes a lo sumo a esta fracción del dataset "
        "(recomendado ~0.15 para --zone-set train; por defecto no se aplica)",
    )
    args = parser.parse_args()

    zones = ZONE_SETS[args.zone_set]
    names = args.zones or list(zones)
    unknown = sorted(set(names) - set(zones))
    if unknown:
        raise ValueError(f"zonas desconocidas para --zone-set {args.zone_set}: {unknown}")

    ds = load_landcover_dataset(args.nc_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    forbidden = forbidden_bboxes(args.zone_set)

    for name in names:
        bbox = zones[name]
        clashes = [label for label, other in forbidden.items() if bbox_overlap(bbox, other)]
        if clashes:
            raise ValueError(f"La zona '{name}' se solapa con {clashes} — revisar el bbox {bbox}")

        dist = class_distribution(ds, bbox)
        top = ", ".join(f"{k}={v:.1f}%" for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v > 0.05)
        print(f"[{name}] bbox={bbox} -> {top}")

        lat_long_df, seqs_df = extract_zone(ds, bbox)

        if args.max_constant_fraction is not None:
            before = len(seqs_df)
            seqs_df = subsample_constant_sequences(seqs_df, max_fraction=args.max_constant_fraction)
            lat_long_df = lat_long_df[lat_long_df["ID"].isin(seqs_df["ID"])]
            print(f"  submuestreo constantes: {before:,} -> {len(seqs_df):,} filas (max_fraction={args.max_constant_fraction})")

        seqs_path, lat_long_path = save_zone_csvs(lat_long_df, seqs_df, args.out_dir, name)
        print(f"  guardado: {seqs_path.name} ({len(seqs_df):,} filas), {lat_long_path.name}")


if __name__ == "__main__":
    main()
