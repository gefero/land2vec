"""Genera un mapa con la huella espacial de las zonas de entrenamiento y de test.

Usa las tablas de lat/long por parcela (data/lat_long_df_*.zip) para
mostrar, en coordenadas geográficas, qué área cubrió el dataset de
entrenamiento (Chaco/Santiago del Estero/frontera) y cuál el set de test
held-out usado en notebooks/test_1.ipynb.

No hay basemap ni límites administrativos: son las coordenadas crudas de
las parcelas, agregadas en una grilla para poder graficar millones de
puntos sin que el plot sea ilegible o pesado.

Uso:
    python scripts/plot_train_test_zones.py
    python scripts/plot_train_test_zones.py --cell-size 0.02 --out imgs/train_test_zones.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_COLOR = "#2F6844"  # canopy (paleta del logo)
TEST_COLOR = "#2B6E8C"  # aquifer
OVERLAP_COLOR = "#A47B3F"  # loam


def load_coords(zip_path: Path) -> pd.DataFrame:
    return pd.read_csv(zip_path, usecols=["latitude", "longitude"])


def occupancy_grid(coords: pd.DataFrame, cell_size: float, lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(coords["latitude"], coords["longitude"], bins=[lat_edges, lon_edges])
    return hist > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "lat_long_df_chaco_santiago_frontier.zip")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "lat_long_df_test_set.zip")
    parser.add_argument("--cell-size", type=float, default=0.02, help="tamaño de celda en grados (~2.2 km)")
    parser.add_argument("--out", type=Path, default=Path("imgs/train_test_zones.png"))
    args = parser.parse_args()

    print(f"Cargando coordenadas de entrenamiento desde {args.train} ...")
    train = load_coords(args.train)
    print(f"Cargando coordenadas de test desde {args.test} ...")
    test = load_coords(args.test)

    lat_min = min(train.latitude.min(), test.latitude.min()) - args.cell_size
    lat_max = max(train.latitude.max(), test.latitude.max()) + args.cell_size
    lon_min = min(train.longitude.min(), test.longitude.min()) - args.cell_size
    lon_max = max(train.longitude.max(), test.longitude.max()) + args.cell_size

    lat_edges = np.arange(lat_min, lat_max + args.cell_size, args.cell_size)
    lon_edges = np.arange(lon_min, lon_max + args.cell_size, args.cell_size)

    print("Agregando puntos en grilla ...")
    train_occ = occupancy_grid(train, args.cell_size, lat_edges, lon_edges)
    test_occ = occupancy_grid(test, args.cell_size, lat_edges, lon_edges)

    # 0 = vacío, 1 = solo train, 2 = solo test, 3 = train y test (overlap)
    zone_map = np.zeros(train_occ.shape, dtype=np.uint8)
    zone_map[train_occ] = 1
    zone_map[test_occ] = np.where(train_occ[test_occ], 3, 2)

    cmap = ListedColormap(["none", TRAIN_COLOR, TEST_COLOR, OVERLAP_COLOR])

    mean_lat_rad = np.radians((lat_min + lat_max) / 2)
    aspect = 1 / np.cos(mean_lat_rad)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    ax.set_facecolor("white")
    ax.pcolormesh(
        lon_edges, lat_edges, zone_map,
        cmap=cmap, vmin=0, vmax=3, shading="flat",
    )
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("land2vec — zonas de entrenamiento y de test")

    legend_items = [
        Patch(facecolor=TRAIN_COLOR, label=f"Entrenamiento ({len(train):,} parcelas)"),
        Patch(facecolor=TEST_COLOR, label=f"Test ({len(test):,} parcelas)"),
        Patch(facecolor=OVERLAP_COLOR, label="Superposición"),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=9)

    train_keys = set(zip(train.latitude.round(5), train.longitude.round(5)))
    test_keys = set(zip(test.latitude.round(5), test.longitude.round(5)))
    overlap_pct_of_train = len(train_keys & test_keys) / len(train_keys) * 100
    ax.text(
        0.02, 0.98,
        f"{overlap_pct_of_train:.0f}% de las parcelas de entrenamiento\nreaparecen en el set de test",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=OVERLAP_COLOR),
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Mapa guardado en {args.out}")


if __name__ == "__main__":
    main()
