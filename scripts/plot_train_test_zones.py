"""Genera un mapa con la huella espacial de las zonas de entrenamiento y de test.

Usa las tablas de lat/long por parcela (data/lat_long_df_*.zip) para mostrar
qué área cubrió el dataset de entrenamiento (Chaco/Santiago del Estero/
frontera) y cuál el set de test held-out usado en notebooks/test_1.ipynb,
en el contexto del mapa de Argentina (data/geo/*.geojson, Natural Earth
1:10m/1:50m cultural vectors, dominio público).

Genera dos paneles: (1) Argentina completa con el recuadro del área de
estudio, (2) el detalle de esa área con las parcelas de train/test
agregadas en una grilla (son demasiados puntos para graficar uno por uno).

Uso:
    python scripts/plot_train_test_zones.py
    python scripts/plot_train_test_zones.py --cell-size 0.02 --out imgs/train_test_zones.png
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEO_DIR = DATA_DIR / "geo"

TRAIN_COLOR = "#2F6844"  # canopy (paleta del logo)
TEST_COLOR = "#2B6E8C"  # aquifer
OVERLAP_COLOR = "#A47B3F"  # loam
LAND_COLOR = "#F3EFE6"  # mist
BORDER_COLOR = "#9AA39C"
HIGHLIGHT_COLOR = "#5c665f"


def load_coords(zip_path: Path) -> pd.DataFrame:
    return pd.read_csv(zip_path, usecols=["latitude", "longitude"])


def occupancy_grid(coords: pd.DataFrame, lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
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

    pad = args.cell_size
    lat_min = min(train.latitude.min(), test.latitude.min()) - pad
    lat_max = max(train.latitude.max(), test.latitude.max()) + pad
    lon_min = min(train.longitude.min(), test.longitude.min()) - pad
    lon_max = max(train.longitude.max(), test.longitude.max()) + pad

    lat_edges = np.arange(lat_min, lat_max + args.cell_size, args.cell_size)
    lon_edges = np.arange(lon_min, lon_max + args.cell_size, args.cell_size)

    print("Agregando puntos en grilla ...")
    train_occ = occupancy_grid(train, lat_edges, lon_edges)
    test_occ = occupancy_grid(test, lat_edges, lon_edges)

    # 0 = vacio, 1 = solo train, 2 = solo test, 3 = train y test (overlap)
    zone_map = np.zeros(train_occ.shape, dtype=np.uint8)
    zone_map[train_occ] = 1
    zone_map[test_occ] = np.where(train_occ[test_occ], 3, 2)
    zone_cmap = ListedColormap(["none", TRAIN_COLOR, TEST_COLOR, OVERLAP_COLOR])

    print("Cargando límites geográficos ...")
    sudamerica = gpd.read_file(GEO_DIR / "sudamerica.geojson")
    ar_provinces = gpd.read_file(GEO_DIR / "ar_provinces.geojson")
    argentina = sudamerica[sudamerica["ADMIN"] == "Argentina"]

    fig, (ax_overview, ax_detail) = plt.subplots(1, 2, figsize=(15, 8), facecolor="white")

    # --- Panel 1: Argentina completa ---
    sudamerica.plot(ax=ax_overview, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.6)
    argentina.plot(ax=ax_overview, color=LAND_COLOR, edgecolor="#5c665f", linewidth=1.1)
    ar_provinces.plot(ax=ax_overview, color="none", edgecolor=BORDER_COLOR, linewidth=0.4)

    study_box = Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=1.6, edgecolor=OVERLAP_COLOR, facecolor=OVERLAP_COLOR, alpha=0.85,
    )
    ax_overview.add_patch(study_box)

    ar_bounds = argentina.total_bounds  # minx, miny, maxx, maxy
    ax_overview.set_xlim(ar_bounds[0] - 2, ar_bounds[2] + 2)
    ax_overview.set_ylim(ar_bounds[1] - 2, ar_bounds[3] + 2)
    ax_overview.set_aspect(1 / np.cos(np.radians(-38)))
    ax_overview.set_xlabel("Longitud")
    ax_overview.set_ylabel("Latitud")
    ax_overview.set_title("Argentina")
    ax_overview.annotate(
        "Área de estudio",
        xy=(lon_max, (lat_min + lat_max) / 2), xycoords="data",
        xytext=(lon_max + 3, (lat_min + lat_max) / 2 + 3), textcoords="data",
        fontsize=9, color=HIGHLIGHT_COLOR, ha="left", va="center", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_COLOR, linewidth=0.8),
    )

    # --- Panel 2: detalle del área de estudio ---
    ax_detail.pcolormesh(lon_edges, lat_edges, zone_map, cmap=zone_cmap, vmin=0, vmax=3, shading="flat")
    ar_provinces.plot(ax=ax_detail, color="none", edgecolor=BORDER_COLOR, linewidth=1.0)

    label_halo = [pe.withStroke(linewidth=3, foreground="white")]
    for prov_name in ["Chaco", "Santiago del Estero"]:
        prov = ar_provinces[ar_provinces["name"] == prov_name]
        if not prov.empty:
            c = prov.geometry.centroid.iloc[0]
            if lon_min <= c.x <= lon_max and lat_min <= c.y <= lat_max:
                ax_detail.text(c.x, c.y, prov_name, fontsize=10, color="#1B2321",
                                ha="center", va="center", fontweight="bold",
                                path_effects=label_halo)

    ax_detail.set_xlim(lon_min, lon_max)
    ax_detail.set_ylim(lat_min, lat_max)
    ax_detail.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax_detail.set_xlabel("Longitud")
    ax_detail.set_ylabel("Latitud")
    ax_detail.set_title("Zonas de entrenamiento y de test")
    for spine in ax_detail.spines.values():
        spine.set_edgecolor(OVERLAP_COLOR)
        spine.set_linewidth(1.6)

    train_keys = set(zip(train.latitude.round(5), train.longitude.round(5)))
    test_keys = set(zip(test.latitude.round(5), test.longitude.round(5)))
    overlap_pct_of_train = len(train_keys & test_keys) / len(train_keys) * 100

    legend_items = [
        Patch(facecolor=TRAIN_COLOR, label=f"Entrenamiento ({len(train):,} parcelas)"),
        Patch(facecolor=TEST_COLOR, label=f"Test ({len(test):,} parcelas)"),
        Patch(facecolor=OVERLAP_COLOR, label=f"Superposición ({overlap_pct_of_train:.0f}% del train)"),
    ]
    ax_detail.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=8.5)

    fig.suptitle("land2vec — zonas de entrenamiento y de test en Argentina", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Mapa guardado en {args.out}")


if __name__ == "__main__":
    main()
