"""Mapas estáticos de las trayectorias de pérdida de cobertura sobre las 7 zonas OOD.

Un PNG por clusterización (las 6 corridas de `tune_clustering.py --select`),
**faceteado por región**, con los píxeles de **deforestación**, **degradación
forestal** y **urbanización** coloreados por proceso. Reusa la clasificación de
proceso de `scripts/build_cluster_map.py` (derivada de
`viz/typology/typology_browser.json`), así que no re-corre el modelo.

Prerrequisitos:
    viz/typology/typology_browser.json      (scripts/describe_clusters.py)
    data/clusters_{dynamic,pooled_subsampled}{,_medium,_coarse}{,_parametric}.zip
    data/lat_long_df_{zona}.zip
    data/geo/{sudamerica,ar_provinces}.geojson

Uso:
    python scripts/plot_process_maps.py                       # 6 PNG en imgs/
    python scripts/plot_process_maps.py --set pooled_subsampled
    python scripts/plot_process_maps.py --procesos deforestacion urbanizacion
    python scripts/plot_process_maps.py --only _medium        # una sola corrida
    python scripts/plot_process_maps.py --dpi 240
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from build_cluster_map import (
    PROCESSES, SUFFIXES, ZONES, ZONE_LABELS, load_typology_labels, process_color,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
OUT_DIR = ROOT / "imgs"

# los 3 procesos de "pérdida de cobertura" que pidió el análisis
DEFAULT_PROCS = ["deforestacion", "degradacion_forestal", "urbanizacion"]
LAND_COLOR = "#F3EFE6"
BORDER_COLOR = "#9AA39C"

_ACCENTS = str.maketrans("áéíóúñ", "aeioun")


def _slug(s: str) -> str:
    return s.lower().translate(_ACCENTS).replace(" / ", "-").replace(" ", "-")


def load_run_points(set_name: str, suffix: str, procs: list[str],
                    labels: dict[int, dict]) -> pd.DataFrame | None:
    """lat, lon, zone, proceso de los clusters cuyo proceso está en `procs`."""
    path = DATA_DIR / f"clusters_{set_name}{suffix}.zip"
    if not path.exists():
        return None
    clu = pd.read_csv(path)                                    # ID, zone, cluster
    clu["proceso"] = clu["cluster"].map(
        lambda c: (labels.get(int(c)) or {}).get("proceso"))
    clu = clu[clu["proceso"].isin(procs)]
    if clu.empty:
        return pd.DataFrame(columns=["lat", "lon", "zone", "proceso"])
    parts = []
    for zone, g in clu.groupby("zone"):
        ll = pd.read_csv(DATA_DIR / f"lat_long_df_{zone}.zip")  # ID, latitude, longitude
        m = g.merge(ll, on="ID", how="inner")
        parts.append(pd.DataFrame({"lat": m["latitude"], "lon": m["longitude"],
                                   "zone": zone, "proceso": m["proceso"]}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="dynamic",
                    choices=["dynamic", "pooled_subsampled"],
                    help="qué set de etiquetas usar (default: dynamic, las que cambian)")
    ap.add_argument("--procesos", nargs="+", default=DEFAULT_PROCS,
                    choices=list(PROCESSES), metavar="PROCESO")
    ap.add_argument("--only", choices=list(SUFFIXES), default=None,
                    help="procesar solo esta granularidad/familia")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--markersize", type=float, default=3.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    typ = load_typology_labels()
    if not typ:
        raise SystemExit(
            "falta viz/typology/typology_browser.json -- corré scripts/describe_clusters.py")

    print("Cargando límites geográficos ...")
    sudamerica = gpd.read_file(GEO_DIR / "sudamerica.geojson")
    ar_prov = gpd.read_file(GEO_DIR / "ar_provinces.geojson")

    pcolors = {k: process_color(k, 0.72) for k in args.procesos}
    suffixes = [args.only] if args.only else list(SUFFIXES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ncol = 3
    nrow = int(np.ceil(len(ZONES) / ncol))
    # extensión de cada zona, cacheada (se usa en las 6 figuras)
    extent = {}
    for zone in ZONES:
        zll = pd.read_csv(DATA_DIR / f"lat_long_df_{zone}.zip")
        extent[zone] = (zll["longitude"].min(), zll["longitude"].max(),
                        zll["latitude"].min(), zll["latitude"].max())

    for suffix in suffixes:
        df = load_run_points(args.set, suffix, args.procesos, typ.get(suffix, {}))
        if df is None:
            print(f"  clusters_{args.set}{suffix}.zip: no existe, se omite")
            continue

        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.1 * nrow),
                                 facecolor="white")
        axes = axes.ravel()
        for ax in axes[len(ZONES):]:
            ax.axis("off")

        for ax, zone in zip(axes, ZONES):
            x0, x1, y0, y1 = extent[zone]
            mx, my = (x1 - x0) * 0.06 or 0.05, (y1 - y0) * 0.06 or 0.05
            near = (slice(x0 - 1, x1 + 1), slice(y0 - 1, y1 + 1))
            sudamerica.cx[near].plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR,
                                     linewidth=0.5, zorder=0)
            ar_prov.cx[near].plot(ax=ax, color="none", edgecolor=BORDER_COLOR,
                                  linewidth=0.4, zorder=1)
            d = df[df["zone"] == zone]
            for k in args.procesos:
                dk = d[d["proceso"] == k]
                if len(dk):
                    ax.scatter(dk["lon"], dk["lat"], s=args.markersize, c=pcolors[k],
                               linewidths=0, alpha=0.8, zorder=3)
            ax.set_xlim(x0 - mx, x1 + mx)
            ax.set_ylim(y0 - my, y1 + my)
            ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
            ax.set_title(f"{ZONE_LABELS.get(zone, zone)}  (n={len(d):,})", fontsize=9)
            ax.tick_params(labelsize=6)

        handles = [Patch(facecolor=pcolors[k], label=PROCESSES[k]["label"])
                   for k in args.procesos]
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(args.procesos)),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle(
            f"{SUFFIXES[suffix]}  ·  set {args.set}  ·  trayectorias de pérdida de cobertura",
            fontsize=13)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])

        out = args.out_dir / f"process_maps_{_slug(SUFFIXES[suffix])}_{args.set}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  {out.relative_to(ROOT)}  ({len(df):,} puntos)")


if __name__ == "__main__":
    main()
