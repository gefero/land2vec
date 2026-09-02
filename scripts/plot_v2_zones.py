"""Mapa de las zonas geográficas usadas para entrenar y evaluar land2vec v2.

Muestra, sobre el mapa de Argentina, tres categorías de zonas:

- Chaco-Santiago (base de train, heredada de la v1, submuestreada).
- Las 7 zonas nuevas sumadas al entrenamiento de v2 (TRAIN_ZONES en
  scripts/build_eval_zones.py), en las mismas ecorregiones que las de eval
  pero con bboxes disjuntos.
- Las 7 zonas de evaluación out-of-domain (EVAL_ZONES), el benchmark
  held-out que nunca se usa para entrenar.

Usa las coordenadas reales por píxel (data/lat_long_df_*.zip) agregadas en
grilla, igual que scripts/plot_train_test_zones.py, para que el área
dibujada sea la huella real de cada zona, no solo el bbox nominal.

Uso:
    python scripts/plot_v2_zones.py
    python scripts/plot_v2_zones.py --out imgs/v2_train_eval_zones.png
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEO_DIR = DATA_DIR / "geo"

CHACO_COLOR = "#5c665f"       # gris-verde oscuro (base heredada de la v1)
TRAIN_NEW_COLOR = "#2F6844"   # canopy (paleta del logo) -- zonas nuevas de train v2
EVAL_COLOR = "#2B6E8C"        # aquifer -- benchmark held-out, sin tocar
LAND_COLOR = "#F3EFE6"        # mist
BORDER_COLOR = "#9AA39C"

CHACO_ZONE = {
    "chaco_santiago_frontier": "Chaco-Santiago (base train, heredada v1)",
}

TRAIN_ZONES = {
    "puna_salta_catamarca": "Puna (Salta/Catamarca)",
    "patagonia_santacruz": "Patagonia (Santa Cruz)",
    "periurbano_gba": "Periurbano (GBA)",
    "corrientes_humedal": "Humedal (Corrientes)",
    "delta_oeste": "Delta oeste (Entre Ríos)",
    "pampa_deprimida": "Pampa deprimida",
    "yungas": "Yungas",
}

EVAL_ZONES = {
    "puna_noa": "Puna (Jujuy)",
    "patagonia_estepa": "Estepa patagónica",
    "periurbano_cordoba": "Periurbano (Córdoba)",
    "ibera": "Iberá",
    "delta_parana": "Delta del Paraná",
    "pampa_nucleo": "Pampa núcleo",
    "misiones_selva": "Selva misionera",
}


def load_coords(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"lat_long_df_{name}.zip", usecols=["latitude", "longitude"])


def occupancy_grid(coords: pd.DataFrame, lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(coords["latitude"], coords["longitude"], bins=[lat_edges, lon_edges])
    return hist > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-size", type=float, default=0.05, help="tamaño de celda en grados (~5.5 km)")
    parser.add_argument("--out", type=Path, default=Path("imgs/v2_train_eval_zones.png"))
    args = parser.parse_args()

    all_zones = {**CHACO_ZONE, **TRAIN_ZONES, **EVAL_ZONES}
    coords_by_zone = {name: load_coords(name) for name in all_zones}

    print("Cargando límites geográficos ...")
    sudamerica = gpd.read_file(GEO_DIR / "sudamerica.geojson")
    ar_provinces = gpd.read_file(GEO_DIR / "ar_provinces.geojson")
    argentina = sudamerica[sudamerica["ADMIN"] == "Argentina"]

    lat_min = min(df["latitude"].min() for df in coords_by_zone.values()) - 1
    lat_max = max(df["latitude"].max() for df in coords_by_zone.values()) + 1
    lon_min = min(df["longitude"].min() for df in coords_by_zone.values()) - 1
    lon_max = max(df["longitude"].max() for df in coords_by_zone.values()) + 1
    lat_edges = np.arange(lat_min, lat_max + args.cell_size, args.cell_size)
    lon_edges = np.arange(lon_min, lon_max + args.cell_size, args.cell_size)

    fig, ax = plt.subplots(figsize=(9, 12), facecolor="white")
    sudamerica.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.6)
    argentina.plot(ax=ax, color=LAND_COLOR, edgecolor="#5c665f", linewidth=1.1)
    ar_provinces.plot(ax=ax, color="none", edgecolor=BORDER_COLOR, linewidth=0.4)

    label_halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    # Overrides manuales para separar etiquetas en dos clusters densos (NE, y
    # Mesopotamia/GBA) donde el offset por defecto (3, 3) se superpone.
    LABEL_OFFSETS = {
        "chaco_santiago_frontier": (-6, -14, "right"),
        "corrientes_humedal": (6, 10, "left"),
        "misiones_selva": (6, -2, "left"),
        "ibera": (6, -12, "left"),
        "delta_oeste": (6, -12, "left"),
        "delta_parana": (6, 8, "left"),
        "periurbano_gba": (6, -12, "left"),
    }

    def draw_group(names: dict, color: str):
        for name, label in names.items():
            coords = coords_by_zone[name]
            occ = occupancy_grid(coords, lat_edges, lon_edges)
            zmap = np.where(occ, 1, np.nan)
            ax.pcolormesh(lon_edges, lat_edges, zmap, cmap=plt.matplotlib.colors.ListedColormap([color]),
                          shading="flat", zorder=3)
            cy, cx = coords["latitude"].mean(), coords["longitude"].mean()
            dx, dy, ha = LABEL_OFFSETS.get(name, (3, 3, "left"))
            ax.annotate(label, xy=(cx, cy), fontsize=7, color="#1B2321", ha=ha, va="bottom",
                        path_effects=label_halo, zorder=4,
                        xytext=(dx, dy), textcoords="offset points")

    draw_group(CHACO_ZONE, CHACO_COLOR)
    draw_group(TRAIN_ZONES, TRAIN_NEW_COLOR)
    draw_group(EVAL_ZONES, EVAL_COLOR)

    ar_bounds = argentina.total_bounds
    ax.set_xlim(min(ar_bounds[0], lon_min) - 1, max(ar_bounds[2], lon_max) + 1)
    ax.set_ylim(min(ar_bounds[1], lat_min) - 1, max(ar_bounds[3], lat_max) + 1)
    ax.set_aspect(1 / np.cos(np.radians(-38)))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("land2vec v2 -- zonas de entrenamiento y de evaluación out-of-domain")

    legend_items = [
        Patch(facecolor=CHACO_COLOR, label="Chaco-Santiago (base train, heredada v1)"),
        Patch(facecolor=TRAIN_NEW_COLOR, label="7 zonas nuevas de train (v2)"),
        Patch(facecolor=EVAL_COLOR, label="7 zonas de evaluación (held-out, sin tocar)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Mapa guardado en {args.out}")


if __name__ == "__main__":
    main()
