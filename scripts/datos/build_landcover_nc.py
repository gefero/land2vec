"""Reconstruye la serie de v3 (default 1992-2022) a partir de los mapas anuales crudos (CLI sobre land2vec.preprocess).

Insumo: los mapas anuales globales de ESA CCI / C3S Land Cover tal como los entrega el
Climate Data Store (un archivo por año, `lccs_class` con la leyenda LCCS completa):
    ESACCI-LC-L4-LCCS-Map-300m-P1Y-<año>-v2.0.7cds.nc   (hasta 2015)
    C3S-LC-L4-LCCS-Map-300m-P1Y-<año>-v2.1.1.nc         (desde 2016)
Se buscan recursivamente en --raw-dir (default data/ESA_data/raw_unzipped/); los .zip / .tar.gz
del CDS se descomprimen solos en <raw-dir>/_extraidos/. El año y la versión se leen de los
metadatos de cada archivo, no del nombre, y se descartan los que no cubren el vector.

Por cada año hace `preprocess.process_year` (agrupar con --mapping, recortar con --vector,
guardar `lccs_<año>.nc` en --anual-dir) y al final `preprocess.stack_years`.

    --vector   ruta a un vector (geojson, shp, ...) o "minx,miny,maxx,maxy". Default: el
               rectángulo del netCDF original, -75,-55,-53,-20. Un rectángulo que empieza con
               signo menos va con "=": --vector=-75,-55,-53,-20 (con espacio, argparse lo toma
               por otra opción).
    --mapping  JSON {"código LCCS": "token"}. Default: preprocess.LCCS_IPCC, que reproduce
               el netCDF original.

El default de --out es la serie de v3 (P.NC_V3), data/autoencoder_v3/landcover_timeseries_1992-2022_rebuild.nc.
Con --years 2000-2022 --out <otro> se rehace la serie vieja (la regresión del censo de v3).
No toca el original de v1/v2, data/landcover_timeseries_2000-2022.nc.

Con --compare <nc> contrasta la salida con otro netCDF (el original): grilla, coincidencia
por año, matriz de coincidencia año contra año (para detectar años corridos o duplicados)
y pares de estados en los píxeles que difieren.

Uso, desde la raíz del repo:
    python scripts/datos/build_landcover_nc.py
    python scripts/datos/build_landcover_nc.py --vector data/geo/ar_provinces.geojson --out data/autoencoder_v3/argentina.nc
    python scripts/datos/build_landcover_nc.py --compare data/landcover_timeseries_2000-2022.nc
"""
import argparse
import json
import tarfile
import zipfile
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402
from land2vec import preprocess as pp  # noqa: E402
from land2vec.extract import LCCS_CODE_TO_TOKEN  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

OUT_FILE = P.NC_V3
DEFAULT_BBOX = (-75.0, -55.0, -53.0, -20.0)   # ventana del netCDF original
RES = 1 / 360
CHUNKS = (3, 2100, 1320)


def parse_years(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def parse_vector(s):
    "Ruta o 'minx,miny,maxx,maxy'."
    parts = s.split(",")
    return tuple(float(x) for x in parts) if len(parts) == 4 else Path(s)


def unpack_archives(raw_dir):
    "Descomprime los .zip / .tar.gz que traigan .nc en <raw_dir>/_extraidos/ (una sola vez)."
    out = raw_dir / "_extraidos"
    for f in sorted(raw_dir.rglob("*")):
        if out in f.parents or not f.is_file():
            continue
        if f.suffix == ".zip":
            with zipfile.ZipFile(f) as z:
                for m in z.namelist():
                    if m.endswith(".nc") and not (out / Path(m).name).exists():
                        out.mkdir(exist_ok=True)
                        (out / Path(m).name).write_bytes(z.read(m))
        elif f.name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(f) as t:
                for m in t.getmembers():
                    if m.name.endswith(".nc") and not (out / Path(m.name).name).exists():
                        out.mkdir(exist_ok=True)
                        (out / Path(m.name).name).write_bytes(t.extractfile(m).read())


def find_sources(raw_dir, years, vector):
    """Un archivo que cubra el vector por año; error si falta alguno.

    Si hay varios para el mismo año (p. ej. un recorte y el global), se comparan sobre el vector:
    si son idénticos se usa el más chico; si difieren, error con el n.º de px distintos."""
    unpack_archives(raw_dir)
    by_year = {}
    for f in sorted(raw_dir.rglob("*.nc")):
        try:
            da = pp.read_raw(f)
        except Exception as e:  # noqa: BLE001 -- p. ej. un archivo a medio copiar o sin lccs_class
            print(f"  no se pudo abrir ({type(e).__name__}), se ignora: {P.rel(f)}")
            continue
        y = da.attrs["year"]
        if y not in years:
            continue
        if pp.covers(da, vector):
            by_year.setdefault(y, []).append(f)
        else:
            print(f"  descartado (no cubre el vector): {P.rel(f)}")
    missing = [y for y in years if y not in by_year]
    if missing:
        raise SystemExit(f"faltan años: {missing}")
    chosen = {}
    for y in years:
        cands = sorted(by_year[y], key=lambda f: f.stat().st_size)
        ref = pp.clip_to_vector(pp.read_raw(cands[0]), vector).values
        for c in cands[1:]:
            n = int((pp.clip_to_vector(pp.read_raw(c), vector).values != ref).sum())
            if n:
                raise SystemExit(f"{y}: {P.rel(cands[0])} y {P.rel(c)} difieren en {n:,} px "
                                 "dentro del vector; dejar uno solo en --raw-dir")
            print(f"  {y}: {P.rel(c)} es idéntico sobre el vector a {P.rel(cands[0])}")
        chosen[y] = cands[0]
    return chosen


def build(raw_dir, out, years, vector, mapping, anual_dir):
    sources = find_sources(raw_dir, years, vector)
    files = []
    for y in years:
        f = pp.process_year(sources[y], vector, anual_dir, mapping)
        files.append(f)
        print(f"{y}  {P.rel(sources[y])} -> {P.rel(f)}  ({f.stat().st_size / 1e6:.1f} MB)")
    pp.stack_years(files, out)
    print(f"\nescrito {P.rel(out)}  ({Path(out).stat().st_size / 1e6:.0f} MB)")


def compare(new_path, old_path):
    a = xr.open_dataset(old_path, mask_and_scale=False)
    b = xr.open_dataset(new_path, mask_and_scale=False)
    print(f"viejo: {P.rel(old_path)}\nnuevo: {P.rel(new_path)}\n")
    for c in ("lat", "lon"):
        same = a[c].shape == b[c].shape and np.allclose(a[c], b[c], atol=RES / 10)
        print(f"grilla {c}: {'igual' if same else 'DISTINTA'}  "
              f"viejo {a[c].values[0]:.6f}..{a[c].values[-1]:.6f} ({a.sizes[c]})  "
              f"nuevo {b[c].values[0]:.6f}..{b[c].values[-1]:.6f} ({b.sizes[c]})")
    ya, yb = a.time.dt.year.values.tolist(), b.time.dt.year.values.tolist()
    print(f"años viejo {ya[0]}-{ya[-1]} ({len(ya)}), nuevo {yb[0]}-{yb[-1]} ({len(yb)})")
    if "procedencia_por_anio" in b.attrs:
        proc = json.loads(b.attrs["procedencia_por_anio"])
    else:
        proc = {}

    # Matriz año viejo x año nuevo sobre una submuestra 1/10 x 1/10: delata años corridos o repetidos.
    sa = a.lccs_class[:, ::10, ::10].values.reshape(len(ya), -1)
    sb = b.lccs_class[:, ::10, ::10].values.reshape(len(yb), -1)
    M = pd.DataFrame([[100 * (sa[i] == sb[j]).mean() for j in range(len(yb))] for i in range(len(ya))],
                     index=ya, columns=yb)
    print("\ncoincidencia (%) año viejo (filas) vs año nuevo (columnas), submuestra 1/100 px:")
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(M.round(2).to_string())

    print("\npor año, resolución completa:")
    for y in ya:
        if y not in yb:
            print(f"  {y}: no está en el nuevo")
            continue
        i, j = ya.index(y), yb.index(y)
        eq = n = 0
        pairs = {}
        for r in range(0, a.sizes["lat"], CHUNKS[1]):
            x = a.lccs_class[i, r:r + CHUNKS[1]].values
            z = b.lccs_class[j, r:r + CHUNKS[1]].values
            d = x != z
            eq += (~d).sum()
            n += d.size
            if d.any():
                k, c = np.unique(x[d].astype(np.int32) * 256 + z[d], return_counts=True)
                for kk, cc in zip(k, c):
                    pairs[kk] = pairs.get(kk, 0) + int(cc)
        # ¿El año nuevo y se parece más a otro año viejo (o al revés)? Margen para no marcar empates.
        flags = []
        col, row = M[y].drop(y), M.loc[y].drop(y)
        if len(col) and col.max() > M.loc[y, y] + 0.01:
            flags.append(f"el {y} nuevo se parece más al {col.idxmax()} viejo")
        if len(row) and row.max() > M.loc[y, y] + 0.01:
            flags.append(f"el {y} viejo se parece más al {row.idxmax()} nuevo")
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:4]
        tok = lambda s: LCCS_CODE_TO_TOKEN.get(s, str(s))  # noqa: E731
        print(f"  {y}  v{proc.get(str(y), {}).get('product_version', '?'):9s} iguales {100 * eq / n:8.4f}%"
              f"  difieren {n - eq:>10,} px"
              + "".join(f"  [{fl}]" for fl in flags)
              + ("  " + ", ".join(f"{tok(k // 256)}→{tok(k % 256)} {c:,}" for k, c in top) if top else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=P.ESA_RAW)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--anual-dir", type=Path, default=P.ANUAL, help="dónde dejar los lccs_<año>.nc")
    ap.add_argument("--years", default=f"{P.V3_YEARS[0]}-{P.V3_YEARS[1]}", help="rango, p. ej. 1992-2022 (default, P.V3_YEARS), 2000-2022 o 2015")
    ap.add_argument("--vector", type=parse_vector, default=DEFAULT_BBOX,
                    help="ruta o minx,miny,maxx,maxy; si empieza con '-', escribir --vector=-75,-55,-53,-20")
    ap.add_argument("--mapping", type=Path, help="JSON {código LCCS: token}; default preprocess.LCCS_IPCC")
    ap.add_argument("--compare", type=Path, metavar="NC",
                    help="en vez de construir, compara --out contra este netCDF")
    args = ap.parse_args()
    if args.compare:
        compare(args.out, args.compare)
        return
    mapping = pp.LCCS_IPCC if args.mapping is None else {int(k): v for k, v in json.loads(args.mapping.read_text()).items()}
    build(args.raw_dir, args.out, parse_years(args.years), args.vector, mapping, args.anual_dir)


if __name__ == "__main__":
    main()
