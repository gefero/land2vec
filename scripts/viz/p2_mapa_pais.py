"""Mapas de las tipologías de P2 para todo el país (plan autoencoder_v3 §4.2, producto).

Lee la serie reconstruida (data/autoencoder_v3/landcover_timeseries_2000-2022_rebuild.nc) por franjas,
busca la trayectoria de cada píxel de Argentina en el universo del censo y le asigna su tipo de
p2_labels.csv (universo completo, peso por superficie). Las constantes quedan como clases propias
(valor 50 + id de token) y no se pintan como cambio; fuera de Argentina = 255.

Escribe:
    data/autoencoder_v3/p2/mapas/raster_<espacio>_k<k>.npz   etiquetas uint8 de toda la grilla (+ lat, lon)
    docs/autoencoder_v3/mapas/<espacio>_k<k>_pais.png         mapa nacional (la superficie dinámica se agrega por bloques)
    docs/autoencoder_v3/mapas/<espacio>_k<k>_chaco.png        recorte del Chaco a resolución completa

Uso: python scripts/viz/p2_mapa_pais.py [--spaces om ae_d8] [--k 12]
"""
import argparse
import importlib.util
from pathlib import Path
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec.extract import LCCS_CODE_TO_TOKEN  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_spec = importlib.util.spec_from_file_location("censo", ROOT / "scripts" / "datos" / "censo_trayectorias.py")
CENSO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CENSO)

OUT = P.DATA / "autoencoder_v3" / "p2"
IMG = P.DOCS / "autoencoder_v3" / "mapas"
OUTSIDE, CONST0 = 255, 50
TOKEN_TO_CODE = {t: c for c, t in LCCS_CODE_TO_TOKEN.items()}
MULT = np.uint64(0x9E3779B97F4A7C15)


def keyed(hi, lo):
    return hi * MULT + lo


def build_lookup(uni: pd.DataFrame):
    codes = np.array([[TOKEN_TO_CODE[t] for t in s.split("-")] for s in uni.seqs], dtype=np.uint8)
    hi, lo = CENSO.encode(codes)
    key = keyed(hi, lo)
    assert len(np.unique(key)) == len(key), "colisión de claves"
    order = np.argsort(key)
    return key[order], hi[order], lo[order], order


def class_per_traj(uni: pd.DataFrame, lab: pd.DataFrame) -> np.ndarray:
    "uint8 por trayectoria del universo: cluster dinámico, o CONST0 + id de token para las constantes."
    cls = np.full(len(uni), OUTSIDE, np.uint8)
    cls[lab.traj_id.values] = lab.etiqueta.values.astype(np.uint8)
    cst = uni.constante.values
    cls[cst] = [CONST0 + Tokenizer.VOCAB[s.split("-")[0]] for s in uni.seqs[cst]]
    return cls


def rasterize_labels(var, mask, lookup, cls_by_space: dict[str, np.ndarray]):
    key_s, hi_s, lo_s, order = lookup
    T, NY, NX = var.shape
    out = {sp: np.full((NY, NX), OUTSIDE, np.uint8) for sp in cls_by_space}
    for y0 in range(0, NY, CENSO.BAND):
        y1 = min(NY, y0 + CENSO.BAND)
        m = mask[y0:y1, :]
        if not m.any():
            continue
        a = var[:, y0:y1, :].values.reshape(T, -1).T[m.ravel()]
        hi, lo = CENSO.encode(a)
        k = keyed(hi, lo)
        pos = np.searchsorted(key_s, k)
        pos = np.minimum(pos, len(key_s) - 1)
        assert ((hi_s[pos] == hi) & (lo_s[pos] == lo)).all(), "píxeles cuya trayectoria no está en el universo"
        idx = order[pos]
        for sp, cls in cls_by_space.items():
            band = out[sp][y0:y1, :]
            band[m] = cls[idx]
        print(f"  filas {y0}-{y1}", flush=True)
    return out


def block_mode(R: np.ndarray, f: int, k: int) -> np.ndarray:
    "Agrega por bloques f x f: el cluster dinámico más frecuente del bloque; si no hay, la clase constante dominante o fuera."
    NY, NX = R.shape
    ny, nx = NY // f, NX // f
    B = R[:ny * f, :nx * f].reshape(ny, f, nx, f)
    best = np.full((ny, nx), -1, np.int16)
    bestc = np.zeros((ny, nx), np.int32)
    for g in range(k):
        c = (B == g).sum((1, 3))
        upd = c > bestc
        best[upd], bestc[upd] = g, c[upd]
    out = np.where(best >= 0, best, OUTSIDE).astype(np.uint8)
    nodyn = best < 0
    inside = (B != OUTSIDE).any((1, 3))
    out[nodyn & inside] = CONST0
    return out


def palette(k: int):
    base = plt.get_cmap("tab20")(np.arange(20))
    order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    cols = [base[i] for i in order[:k]]
    return cols, np.array([0.86, 0.86, 0.86, 1.0])


def to_rgba(A: np.ndarray, k: int):
    cols, const = palette(k)
    rgba = np.ones(A.shape + (4,))
    rgba[A == OUTSIDE] = (1, 1, 1, 0)
    rgba[(A >= CONST0) & (A != OUTSIDE)] = const
    for g in range(k):
        rgba[A == g] = cols[g]
    return rgba


def draw(A, extent, title, legend_tab, k, path, figsize, prov, legend_loc="lower left"):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(to_rgba(A, k), extent=extent, interpolation="nearest", aspect="equal")
    prov.boundary.plot(ax=ax, color="#444", linewidth=0.3)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    cols, const = palette(k)
    handles = [Patch(color=cols[int(r.cluster)], label=f"{int(r.cluster)}: {r.etiqueta.split(' · ')[0]} ~{int(r.anio_1er_cambio_mediana)} ({r.share_px:.0%})") for r in legend_tab.itertuples()]
    handles.append(Patch(color=const, label="trayectoria constante"))
    ax.legend(handles=handles, loc=legend_loc, fontsize=6, frameon=True, framealpha=0.9, title="cluster: secuencia ~año 1.er cambio (% sup. dinámica)", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    import geopandas as gpd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spaces", nargs="+", default=["om", "ae_d8"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--block", type=int, default=6, help="factor de agregación del mapa nacional")
    args = ap.parse_args()
    (OUT / "mapas").mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    uni = pd.read_csv(P.DATA / "autoencoder_v3" / "universo_argentina.csv")
    labels = pd.read_csv(OUT / "p2_labels.csv")
    labels = labels[(labels.universo == "completo") & (labels.peso == "px") & (labels.k == args.k)]
    cls = {sp: class_per_traj(uni, labels[labels.espacio == sp]) for sp in args.spaces}

    ds = xr.open_dataset(P.NC_V3, mask_and_scale=False)
    var = ds["lccs_class"]
    lat, lon = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
    mask = CENSO.argentina_mask(lat, lon)
    print(f"grilla {var.shape}; Argentina: {mask.sum():,} px", flush=True)
    rasters = rasterize_labels(var, mask, build_lookup(uni), cls)

    dy, dx = abs(lat[0] - lat[-1]) / (len(lat) - 1), abs(lon[-1] - lon[0]) / (len(lon) - 1)
    ext = (lon.min() - dx / 2, lon.max() + dx / 2, lat.min() - dy / 2, lat.max() + dy / 2)
    prov = gpd.read_file(P.GEO / "ar_provinces.geojson")
    for sp, R in rasters.items():
        np.savez_compressed(OUT / "mapas" / f"raster_{sp}_k{args.k}.npz", etiquetas=R, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
        tab = pd.read_csv(OUT / f"p2_clusters_{sp}_k{args.k}.csv").sort_values("cluster")
        title = f"Tipología P2 ({sp}, k={args.k}) · Argentina 2000-2022"
        draw(block_mode(R, args.block, args.k), ext, title, tab, args.k, IMG / f"{sp}_k{args.k}_pais.png", (7, 11), prov, legend_loc="lower right")
        # recorte del Chaco (resolución completa)
        x0, x1, y0, y1 = -64.5, -57.0, -30.0, -22.0
        j0, j1 = np.searchsorted(lon, x0), np.searchsorted(lon, x1)
        i0, i1 = np.searchsorted(-lat, -y1), np.searchsorted(-lat, -y0)
        draw(R[i0:i1, j0:j1], (lon[j0] - dx / 2, lon[j1 - 1] + dx / 2, lat[i1 - 1] - dy / 2, lat[i0] + dy / 2),
             f"Tipología P2 ({sp}, k={args.k}) · Chaco", tab, args.k, IMG / f"{sp}_k{args.k}_chaco.png", (9, 9.5), prov)
        print(f"{sp}: listo", flush=True)


if __name__ == "__main__":
    main()
