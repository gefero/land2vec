"""Censo de trayectorias ESA CCI 2000-2022 en Argentina (autoencoder_v3, plan §2 y §4.0).

Recorre el netCDF completo por franjas de latitud, codifica cada trayectoria de 23 años
en 92 bits (4 bits/año) y cuenta píxeles por trayectoria distinta, dentro de Argentina
(data/geo/ar_provinces.geojson) y en el rectángulo completo del archivo.

Escribe:
    data/autoencoder_v3/universo_argentina.csv   traj_id, seqs, n_px, n_cambios, constante, visto_en_train
    data/autoencoder_v3/universo_rectangulo.csv  idem para el rectángulo completo del netCDF
Imprime: resumen del universo, concentración, cambios por año (para detectar costuras
como la de 2014-2016) y, con --crecimiento, cómo crece el n.º de trayectorias con el
largo de la serie.

`visto_en_train` = la trayectoria aparece en alguna zona de entrenamiento de v2
(chaco_santiago_frontier + TRAIN_ZONES, archivos data/zonas/ completos).

Uso, desde la raíz del repo:
    python scripts/datos/censo_trayectorias.py [--crecimiento]
"""
import argparse
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402
from land2vec import preprocess as pp  # noqa: E402
from land2vec.extract import LCCS_CODE_TO_TOKEN  # noqa: E402
from land2vec.zones import CHACO_ZONE, TRAIN_ZONES  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

BAND = 2100  # alineado con los chunks del netCDF (3, 2100, 1320)
FILL = 15    # código 255 (_FillValue) -> 15 en la codificación de 4 bits
OUT_DIR = P.DATA / "autoencoder_v3"


def argentina_mask(lat, lon):
    return pp.vector_mask(lat, lon, P.GEO / "ar_provinces.geojson")


def encode(a):
    "a: (n_px, T) uint8 -> (hi, lo) uint64 con 4 bits por año (T <= 23)."
    a = np.where(a == 255, FILL, a).astype(np.uint64)
    T = a.shape[1]
    k = min(T, 12)
    hi = (a[:, :k] << (np.arange(k)[::-1].astype(np.uint64) * np.uint64(4))).sum(1, dtype=np.uint64)
    lo = np.zeros(len(a), np.uint64)
    if T > 12:
        lo = (a[:, 12:] << (np.arange(T - 12)[::-1].astype(np.uint64) * np.uint64(4))).sum(1, dtype=np.uint64)
    return hi, lo


def decode(hi, lo, T=23):
    k = min(T, 12)
    return [(int(hi) >> (4 * (k - 1 - i))) & 15 for i in range(k)] + \
           [(int(lo) >> (4 * (T - 13 - i))) & 15 for i in range(T - 12)]


def count(var, mask, windows=None):
    "Cuenta trayectorias por franjas. windows: lista de (inicio, largo) para el modo crecimiento."
    T, NY, _ = var.shape
    windows = windows or [(0, T)]
    acc = {w: [] for w in windows}
    for y0 in range(0, NY, BAND):
        a = var[:, y0:y0 + BAND, :].values.reshape(T, -1).T
        if mask is not None:
            a = a[mask[y0:y0 + BAND, :].ravel()]
        for (s, L) in windows:
            hi, lo = encode(a[:, s:s + L])
            acc[(s, L)].append(pd.DataFrame({"hi": hi, "lo": lo}).value_counts().rename("n").reset_index())
        print(f"  filas {y0}-{min(NY, y0 + BAND)}", flush=True)
    return {w: pd.concat(p).groupby(["hi", "lo"], as_index=False)["n"].sum() for w, p in acc.items()}


def seen_in_train():
    seen = set()
    for zone in list(CHACO_ZONE) + list(TRAIN_ZONES):
        seen |= set(pd.read_csv(P.seqs_file(zone), usecols=["seqs"])["seqs"].unique())
    return seen


def universe_table(df, seen):
    tok = {**LCCS_CODE_TO_TOKEN, FILL: "FILL"}
    codes = [decode(h, l) for h, l in zip(df.hi, df.lo)]
    out = pd.DataFrame({"seqs": ["-".join(tok[c] for c in t) for t in codes], "n_px": df.n.values})
    out["n_cambios"] = [sum(a != b for a, b in zip(t, t[1:])) for t in codes]
    out["constante"] = out.n_cambios == 0
    out["visto_en_train"] = out.seqs.isin(seen)
    out = out.sort_values("n_px", ascending=False).reset_index(drop=True)
    out.insert(0, "traj_id", range(len(out)))
    return out


def summary(u, name):
    dyn = u[~u.constante]
    cs = dyn.n_px.cumsum() / dyn.n_px.sum()
    print(f"\n=== {name} ===")
    print(f"píxeles: {u.n_px.sum():,}; trayectorias distintas: {len(u):,}")
    print(f"constantes: {u.constante.sum()} tipos, {u.n_px[u.constante].sum() / u.n_px.sum():.1%} de los px")
    print(f"dinámicas: {len(dyn):,} tipos, {dyn.n_px.sum():,} px; máx. cambios: {u.n_cambios.max()}")
    print("tipos por n.º de cambios:", dyn.n_cambios.value_counts().sort_index().to_dict())
    print(f"tipos para cubrir 50/90/99 % de px dinámicos: "
          f"{[int(np.searchsorted(cs.values, q) + 1) for q in (0.5, 0.9, 0.99)]}; singletons: {(dyn.n_px == 1).sum():,}")
    print(f"dinámicas vistas en train: {dyn.visto_en_train.sum():,} tipos "
          f"({dyn.n_px[dyn.visto_en_train].sum() / dyn.n_px.sum():.1%} de los px dinámicos)")
    yrs = u.seqs.str.split("-")
    ch = np.array([[a != b for a, b in zip(t, t[1:])] for t in yrs])
    px = (ch * u.n_px.values[:, None]).sum(0)
    print("px que cambian por año:", {f"{2000 + i}->{2001 + i}": int(v) for i, v in enumerate(px)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc-path", type=Path, default=P.NC_V3)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--crecimiento", action="store_true",
                    help="además, n.º de trayectorias en Argentina según el largo de la serie")
    args = ap.parse_args()

    ds = xr.open_dataset(args.nc_path, mask_and_scale=False)
    var = ds["lccs_class"]
    lat, lon = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
    mask = argentina_mask(lat, lon)
    print(f"grilla {var.shape[1]}x{var.shape[2]}; dentro de Argentina: {mask.sum():,} px", flush=True)

    seen = seen_in_train()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, m in (("argentina", mask), ("rectangulo", None)):
        u = universe_table(count(var, m)[(0, var.shape[0])], seen)
        path = args.out_dir / f"universo_{name}.csv"
        u.to_csv(path, index=False)
        summary(u, name)
        print(f"guardado: {P.rel(path)}")

    if args.crecimiento:
        T = var.shape[0]
        wins = [(0, L) for L in (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, T)]
        res = count(var, mask, wins)
        rows = []
        for (s, L), df in res.items():
            codes = np.array([decode(h, l, L) for h, l in zip(df.hi, df.lo)])
            nch = (codes[:, 1:] != codes[:, :-1]).sum(1)
            rows.append({"periodo": f"{2000 + s}-{2000 + s + L - 1}", "años": L, "distintas": len(df),
                         "dinamicas": int((nch > 0).sum()),
                         "px_dinamicos_%": round(100 * df.n[nch > 0].sum() / df.n.sum(), 2),
                         "max_cambios": int(nch.max())})
        print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
