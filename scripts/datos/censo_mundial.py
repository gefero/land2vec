"""Censo de trayectorias ESA CCI / C3S 2000-2022 a nivel mundial, directo desde los crudos (autoencoder_v3).

Recorre los 23 mapas anuales globales de data/ESA_data/raw_unzipped/ por bloques de 2025 x 2025 píxeles
(el chunk interno de los netCDF), agrupa los códigos LCCS con `preprocess.LCCS_IPCC` (la misma regla del .nc de
Argentina), codifica cada trayectoria en 92 bits (4 bits por año, igual que censo_trayectorias.py) y cuenta por
trayectoria distinta los píxeles **y el área real en km²** (a escala global el píxel mide distinto según la
latitud, así que el área es el peso correcto). No se guarda ningún ráster intermedio.

Máscara de tierra: sólo se cuentan los píxeles dentro del polígono de continentes (`--mascara`, por defecto
data/geo/World_Continents_*.geojson) dilatado `--buffer-px` píxeles (16 ≈ 5 km: sin dilatar se pierde ~0,5 % de la tierra del
producto por la costa generalizada del polígono; con 8 px, ~0,15 %; con 16 px, ~0,09 %. Un colchón de más sólo cuela agua de mar, que
queda como la trayectoria Wa constante y se filtra después; uno de menos pierde tierra costera, que es donde más cambia la cobertura). Los bloques que no tocan ningún polígono ni se leen. La superficie
descartada se registra aparte (descartado.json). El polígono incluye lagos grandes, la Antártida y Groenlandia: se filtran después.

Modos (desde la raíz del repo):
    python scripts/datos/censo_mundial.py --bloques 16 --workers 1     # medición de tiempos con 16 bloques al azar
    python scripts/datos/censo_mundial.py --completo --workers 6       # censo completo (con checkpoints; se puede retomar)

El modo --completo escribe en --out-dir (default data/autoencoder_v3/mundo/) universo_mundo.csv.gz con las columnas
hi, lo (uint64 de la codificación), n_px, area_km2, n_cambios, descartado.json y el checkpoint ckpt_<años>.pkl.
Período: --years (default 1992-2022 = P.V3_YEARS); el nombre de cada archivo de raw_unzipped debe empezar con el año.
"""
import argparse
import glob
import json
import pickle
import warnings
import time
from multiprocessing import Pool
from pathlib import Path
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec import geo as G  # noqa: E402
from land2vec import preprocess as pp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import geopandas as gpd  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

TILE = 2025
NY, NX = 64800, 129600
Y0, Y1 = P.V3_YEARS          # período de la serie (default 1992-2022); se cambia con --years antes de crear el Pool
T = Y1 - Y0 + 1
MULT = np.uint64(0x9E3779B97F4A7C15)
MAX_TASKS = 64              # bloques por proceso worker antes de reciclarlo (por si queda alguna fuga lenta)
CHUNK_ROWS = 256            # filas que se codifican juntas (acota la memoria: 256 x 2025 px)
OUT_DIR = P.DATA / "autoencoder_v3" / "mundo"   # --out-dir lo cambia
DEG = 1.0 / 360.0
MASK = {"path": None, "buffer_px": 16, "geoms": None, "sindex": None}

_tokens = pp.token_table(pp.LCCS_IPCC)
_LUT = np.full(256, 255, np.uint8)
for _c in range(255):
    _t = pp.LCCS_IPCC.get(_c, pp.LCCS_IPCC.get(_c // 10 * 10))
    if _t is not None:
        _LUT[_c] = _tokens.index(_t)

def set_years(y0: int, y1: int):
    global Y0, Y1, T
    Y0, Y1, T = y0, y1, y1 - y0 + 1


def _files():
    "Un archivo por año de Y0..Y1 (el año está en los 4 primeros caracteres del nombre)."
    files = sorted((f for f in P.ESA_RAW.glob("*.nc") if Y0 <= int(f.name[:4]) <= Y1), key=lambda p: int(p.name[:4]))
    assert [int(f.name[:4]) for f in files] == list(range(Y0, Y1 + 1)), \
        f"faltan o sobran años en {P.rel(P.ESA_RAW)}: se esperaba {Y0}-{Y1}, hay {[int(f.name[:4]) for f in files]}"
    return files


def split_years() -> int:
    "Años que van en `hi` (el resto en `lo`): 12 hasta T = 23 (compatible con el censo 2000-2022) y 16 hasta T = 31."
    assert T <= 31, "4 bits por año: más de 31 años no caben en dos uint64"
    return min(T, 12 if T <= 23 else 16)


def _mask_geoms():
    "Polígonos de tierra (partes sueltas, dilatadas buffer_px píxeles) con su índice espacial; se cargan una vez por proceso."
    if MASK["geoms"] is None:
        path = MASK["path"] or glob.glob(str(P.GEO / "World_Continents*.geojson"))[0]
        g = gpd.read_file(path).explode(index_parts=False).reset_index(drop=True)
        if MASK["buffer_px"]:
            with warnings.catch_warnings():            # el buffer en grados es intencional (el píxel mide 1/360°)
                warnings.simplefilter("ignore", UserWarning)
                g = g.set_geometry(g.geometry.buffer(MASK["buffer_px"] * DEG))
        MASK["geoms"], MASK["sindex"] = g.geometry.values, g.sindex
    return MASK["geoms"], MASK["sindex"]


def land_mask(y0, x0, h, w):
    "(h, w) bool de los píxeles del bloque dentro del polígono de tierra; None si el bloque no toca ningún polígono."
    geoms, sidx = _mask_geoms()
    lat_n, lon_w = 90.0 - y0 * DEG, -180.0 + x0 * DEG
    hits = sidx.query(box(lon_w, lat_n - h * DEG, lon_w + w * DEG, lat_n), predicate="intersects")
    if len(hits) == 0:
        return None
    m = rasterize(((geoms[i], 1) for i in hits), out_shape=(h, w), transform=from_origin(lon_w, lat_n, DEG, DEG),
                  fill=0, dtype="uint8").astype(bool)
    return m if m.any() else None


def encode(a: np.ndarray):
    "a: (n, T) uint8 con tokens 0..9 -> (hi, lo) uint64; misma codificación que censo_trayectorias.encode."
    k = split_years()
    hi = np.zeros(len(a), np.uint64)
    lo = np.zeros(len(a), np.uint64)
    for t in range(k):
        hi = (hi << np.uint64(4)) | a[:, t].astype(np.uint64)
    for t in range(k, T):
        lo = (lo << np.uint64(4)) | a[:, t].astype(np.uint64)
    return hi, lo


def n_changes(hi, lo):
    "n.º de cambios de cada trayectoria a partir de (hi, lo)."
    k = split_years()
    cols = [((hi >> np.uint64(4 * (k - 1 - i))) & np.uint64(15)) for i in range(k)] + \
           [((lo >> np.uint64(4 * (T - k - 1 - i))) & np.uint64(15)) for i in range(T - k)]
    return sum((cols[i] != cols[i + 1]).astype(np.int64) for i in range(T - 1))


def process_block(bid: int):
    """Devuelve (DataFrame hi, lo, n_px, area_km2 de las trayectorias distintas dentro de la máscara de tierra, segundos,
    bloque constante, id, n_px descartados, km² descartados)."""
    t0 = time.time()
    by, bx = divmod(bid, NX // TILE)
    y0, x0 = by * TILE, bx * TILE
    h, w = min(TILE, NY - y0), min(TILE, NX - x0)
    area = G.pixel_area_ha(np.arange(y0, y0 + h)) / 100.0                              # km² por píxel de cada fila
    empty = pd.DataFrame({"hi": np.array([], np.uint64), "lo": np.array([], np.uint64),
                          "n_px": np.array([], np.int64), "area_km2": np.array([], float)})
    mask = land_mask(y0, x0, h, w)
    if mask is None:                                                                   # océano puro: ni se lee
        return empty, time.time() - t0, True, bid, h * w, float(area.sum() * w)
    kept_px, kept_km2 = int(mask.sum()), float((mask.sum(1) * area).sum())
    disc = (h * w - kept_px, float(area.sum() * w) - kept_km2)
    stack = np.empty((T, h, w), np.uint8)               # un solo buffer: cada año se agrupa al leerlo (sin copias del tile entero)
    const = True
    for t, f in enumerate(_files()):
        # los archivos se abren y cierran en cada bloque: con handles persistentes la memoria de cada worker
        # crece ~100 MB por bloque (caché de chunks de HDF5) hasta agotar la RAM con varios procesos
        with xr.open_dataset(f, mask_and_scale=False) as ds:
            raw = ds["lccs_class"][0, y0:y0 + h, x0:x0 + w].values
        const &= raw.min() == raw.max()                 # ¿el año es uniforme?
        stack[t] = _LUT[raw]
        del raw
    if (stack == 255).any():
        raise ValueError(f"bloque {bid}: códigos LCCS sin token")
    if const and (stack == stack[:1]).all():            # bloque constante en espacio y tiempo (hielo, desierto, ...)
        hi, lo = encode(stack[:, :1, 0].T.copy())
        df = pd.DataFrame({"hi": hi, "lo": lo, "n_px": [kept_px], "area_km2": [kept_km2]})
        return df, time.time() - t0, True, bid, *disc
    parts = []
    for r in range(0, h, CHUNK_ROWS):
        sel = mask[r:r + CHUNK_ROWS].ravel()
        if not sel.any():
            continue
        a = stack[:, r:r + CHUNK_ROWS, :].reshape(T, -1).T[sel]                         # (n, 23), sólo píxeles de tierra
        hi, lo = encode(a)
        ar = np.repeat(area[r:r + CHUNK_ROWS], w)[sel]
        key = hi * MULT + lo
        uk, idx, inv = np.unique(key, return_index=True, return_inverse=True)
        assert (hi[idx][inv] == hi).all() and (lo[idx][inv] == lo).all(), "colisión de hash"
        parts.append(pd.DataFrame({"hi": hi[idx], "lo": lo[idx], "n_px": np.bincount(inv),
                                   "area_km2": np.bincount(inv, weights=ar)}))
    df = pd.concat(parts).groupby(["hi", "lo"], as_index=False).sum()
    del stack
    return df, time.time() - t0, False, bid, *disc


def merge(parts):
    parts = [p for p in parts if len(p)]
    if not parts:                                                                      # todos los bloques eran océano
        return pd.DataFrame({"hi": np.array([], np.uint64), "lo": np.array([], np.uint64),
                             "n_px": np.array([], np.int64), "area_km2": np.array([], float)})
    out = pd.concat(parts).groupby(["hi", "lo"], as_index=False).sum()
    return out.astype({"hi": np.uint64, "lo": np.uint64, "n_px": np.int64})            # una tabla vacía deja las columnas como float/object


def benchmark(n_blocks: int, workers: int, seed: int):
    ids = np.random.default_rng(seed).choice((NY // TILE) * (NX // TILE), n_blocks, replace=False)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=MAX_TASKS) as pool:
        res = pool.map(process_block, ids.tolist(), chunksize=1)
    wall = time.time() - t0
    secs = np.array([r[1] for r in res])
    skipped = sum(1 for r in res if r[2] and len(r[0]) == 0)
    tot = merge([r[0] for r in res])
    cum, acc = [], []
    for r in res:
        acc.append(r[0])
        cum.append(len(merge(acc)))
    nb = (NY // TILE) * (NX // TILE)
    print(f"{n_blocks} bloques con {workers} proceso(s): {wall:.0f}s de reloj; {secs.mean():.1f}s por bloque (mín {secs.min():.1f}, máx {secs.max():.1f}); "
          f"{skipped} bloques sin tierra (no se leen)")
    print(f"extrapolación a los {nb} bloques: {wall / n_blocks * nb / 3600:.2f} h con este número de procesos")
    print(f"trayectorias distintas acumuladas por bloque: {cum}")
    print(f"dentro de la máscara: {len(tot):,} trayectorias, {tot.n_px.sum():,} px, {tot.area_km2.sum():,.0f} km²; "
          f"descartado: {sum(r[4] for r in res):,} px, {sum(r[5] for r in res):,.0f} km²")
    tot["n_cambios"] = n_changes(tot.hi.values, tot.lo.values)
    print("área por n.º de cambios (km²):", tot.groupby("n_cambios").area_km2.sum().round(0).to_dict())
    print("trayectorias por n.º de cambios:", tot.n_cambios.value_counts().sort_index().to_dict())


def full(workers: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"ckpt_{Y0}-{Y1}.pkl"
    nb = (NY // TILE) * (NX // TILE)
    done, acc, disc = set(), pd.DataFrame({"hi": np.array([], np.uint64), "lo": np.array([], np.uint64),
                                           "n_px": np.array([], np.int64), "area_km2": np.array([], float)}), [0, 0.0]
    if ckpt.exists():
        done, acc, disc = pickle.load(open(ckpt, "rb"))
        acc = acc.astype({"hi": np.uint64, "lo": np.uint64, "n_px": np.int64, "area_km2": float})
        print(f"retomando: {len(done)}/{nb} bloques hechos, {len(acc):,} trayectorias", flush=True)
    todo = [b for b in range(nb) if b not in done]
    pending, t0 = [], time.time()
    with Pool(workers, maxtasksperchild=MAX_TASKS) as pool:
        for k, (df, secs, _, bid, dpx, dkm2) in enumerate(pool.imap_unordered(process_block, todo, chunksize=1), 1):
            pending.append(df)
            done.add(bid)
            disc[0] += dpx
            disc[1] += dkm2
            if k % 32 == 0 or k == len(todo):
                acc = merge([acc] + pending)
                pending = []
                pickle.dump((done, acc, disc), open(ckpt.with_suffix(".tmp"), "wb"))
                ckpt.with_suffix(".tmp").replace(ckpt)
            if k % 64 == 0:
                print(f"  {k}/{len(todo)} bloques ({(time.time() - t0) / 3600:.2f} h); {len(acc):,} trayectorias", flush=True)
    assert int(acc.n_px.sum()) + disc[0] == nb * TILE * TILE, "los píxeles dentro y fuera de la máscara no suman el planeta"
    acc["n_cambios"] = n_changes(acc.hi.values, acc.lo.values)
    out = out_dir / "universo_mundo.csv.gz"
    acc.sort_values("area_km2", ascending=False).to_csv(out, index=False)
    json.dump({"px": int(disc[0]), "km2": disc[1], "mascara": str(MASK["path"]), "buffer_px": MASK["buffer_px"],
               "dentro_px": int(acc.n_px.sum()), "dentro_km2": float(acc.area_km2.sum())}, open(out_dir / "descartado.json", "w"), indent=1)
    print(f"{len(acc):,} trayectorias, {acc.area_km2.sum():,.0f} km² dentro de la máscara; descartado {disc[1]:,.0f} km² -> {P.rel(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bloques", type=int, default=0, help="medición con N bloques al azar")
    ap.add_argument("--completo", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", default=f"{P.V3_YEARS[0]}-{P.V3_YEARS[1]}", help="período (default 1992-2022); la regresión con la serie vieja es --years 2000-2022")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--mascara", type=Path, default=None, help="polígono de tierra (default: data/geo/World_Continents_*.geojson)")
    ap.add_argument("--buffer-px", type=int, default=16, help="dilatación del polígono, en píxeles de 300 m")
    a = ap.parse_args()
    MASK["path"], MASK["buffer_px"] = a.mascara, a.buffer_px
    set_years(*(int(x) for x in a.years.split("-")))
    if a.completo:
        full(a.workers, a.out_dir)
    else:
        benchmark(a.bloques or 16, a.workers, a.seed)
