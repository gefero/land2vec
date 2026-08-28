"""Construye los datasets de evaluación out-of-domain (fuera de Chaco-Santiago).

Recorta data/landcover_timeseries_2000-2022.nc en varias zonas geográficamente
disjuntas del área de entrenamiento, cada una dominada por una modalidad de uso
de suelo distinta, y guarda los CSV/zip en el mismo formato que usa
land2vec.dataset.load_data() (data/id_seqs_text_2000_2022_<zona>.zip,
data/lat_long_df_<zona>.zip).

Antes de guardar cada zona verifica que su bounding box no se solape con el de
entrenamiento ni con el del test set actual (evita repetir el leakage
documentado en notebooks/test_2.ipynb) y reporta la distribución de clases
resultante.

Uso:
    python scripts/build_eval_zones.py
    python scripts/build_eval_zones.py --out-dir data --zones ibera puna_noa
"""

import argparse
from pathlib import Path

import numpy as np

from land2vec.extract import LCCS_CODE_TO_TOKEN, crop_bbox, extract_zone, load_landcover_dataset, save_zone_csvs

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


def bbox_overlap(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2
    return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)


def class_distribution(ds, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    ds_cropped = crop_bbox(ds, *bbox)
    codes = np.nan_to_num(ds_cropped["lccs_class"].values, nan=0).astype(int)
    vals, counts = np.unique(codes, return_counts=True)
    total = counts.sum()
    return {LCCS_CODE_TO_TOKEN.get(int(v), f"code{v}"): c / total * 100 for v, c in zip(vals, counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-path", type=Path, default=DATA_DIR / "landcover_timeseries_2000-2022.nc")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--zones", nargs="+", choices=list(EVAL_ZONES), default=list(EVAL_ZONES))
    args = parser.parse_args()

    ds = load_landcover_dataset(args.nc_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.zones:
        bbox = EVAL_ZONES[name]
        if bbox_overlap(bbox, TRAIN_BBOX) or bbox_overlap(bbox, TEST_BBOX):
            raise ValueError(f"La zona '{name}' se solapa con train/test — revisar el bbox {bbox}")

        dist = class_distribution(ds, bbox)
        top = ", ".join(f"{k}={v:.1f}%" for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v > 0.05)
        print(f"[{name}] bbox={bbox} -> {top}")

        lat_long_df, seqs_df = extract_zone(ds, bbox)
        seqs_path, lat_long_path = save_zone_csvs(lat_long_df, seqs_df, args.out_dir, name)
        print(f"  guardado: {seqs_path.name} ({len(seqs_df):,} filas), {lat_long_path.name}")


if __name__ == "__main__":
    main()
