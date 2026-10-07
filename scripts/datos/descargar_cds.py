"""Descarga mapas anuales globales ESA CCI / C3S Land Cover desde el Climate Data Store (CDS) a data/ESA_data/raw_unzipped/.

Requisitos: `pip install "cdsapi>=0.7.7"`, ~/.cdsapirc con la URL y el token personal, y haber aceptado los términos del
dataset `satellite-land-cover` en https://cds.climate.copernicus.eu/datasets/satellite-land-cover?tab=download.

Un pedido por año (planeta entero, variable `all`). La versión sale del año: v2_0_7cds hasta 2015, v2_1_1 desde 2016.
Cada archivo se guarda con el nombre de los demás (`<año>_ESACCI-LC-L4-LCCS-Map-300m-P1Yv2.0.7cds.nc` /
`<año>_C3S-LC-L4-LCCS-Map-300m-P1Yv2.1.1.nc`), se verifica (abre, año correcto, grilla global) y recién ahí se borra el
archivo comprimido. Si el año ya está en --out-dir, se saltea.

Uso, desde la raíz del repo:
    python scripts/datos/descargar_cds.py 1992 1993 1994 1995
"""
import argparse
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402

import xarray as xr  # noqa: E402

DATASET = "satellite-land-cover"


def version_of(year: int):
    "(versión del pedido, nombre del archivo final) según el año."
    if year <= 2015:
        return "v2_0_7cds", f"{year}_ESACCI-LC-L4-LCCS-Map-300m-P1Yv2.0.7cds.nc"
    return "v2_1_1", f"{year}_C3S-LC-L4-LCCS-Map-300m-P1Yv2.1.1.nc"


def extract_nc(archive: Path, dest: Path) -> Path:
    "Saca el único .nc de un zip / tar / tar.gz (o acepta que ya sea un .nc)."
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            assert len(names) == 1, f"se esperaba un .nc en el zip, hay {names}"
            with z.open(names[0]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if m.name.endswith(".nc")]
            assert len(members) == 1, f"se esperaba un .nc en el tar, hay {[m.name for m in members]}"
            with t.extractfile(members[0]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    else:
        shutil.move(str(archive), dest)
    return dest


def verify(path: Path, year: int):
    "Abre el archivo y comprueba año y grilla global; devuelve un resumen."
    ds = xr.open_dataset(path, mask_and_scale=False)
    start = str(ds.attrs.get("time_coverage_start", ""))
    assert start.startswith(str(year)), f"{path.name}: time_coverage_start={start!r}, se esperaba {year}"
    v = ds["lccs_class"]
    assert v.shape == (1, 64800, 129600), f"{path.name}: forma {v.shape}"
    assert int(v[0, 30000:30010, 60000:60010].max()) > 0, f"{path.name}: sin datos en una muestra"
    return f"versión {ds.attrs.get('product_version')}, {path.stat().st_size / 1e9:.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("years", type=int, nargs="+")
    ap.add_argument("--out-dir", type=Path, default=P.ESA_RAW)
    ap.add_argument("--tmp-dir", type=Path, default=P.ESA_RAW.parent / "_cds_tmp")
    args = ap.parse_args()
    import cdsapi
    client = cdsapi.Client()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    for y in args.years:
        version, name = version_of(y)
        final = args.out_dir / name
        if final.exists():
            print(f"{y}: ya existe {P.rel(final)}, se saltea", flush=True)
            continue
        t0 = time.time()
        archive = args.tmp_dir / f"{y}.download"
        print(f"{y}: pidiendo {DATASET} {version} ...", flush=True)
        client.retrieve(DATASET, {"variable": "all", "year": [str(y)], "version": version}, str(archive))
        print(f"{y}: descargado en {(time.time() - t0) / 60:.1f} min ({archive.stat().st_size / 1e9:.2f} GB); extrayendo", flush=True)
        part = final.with_suffix(".nc.part")
        extract_nc(archive, part)
        try:
            info = verify(part, y)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        part.rename(final)
        archive.unlink(missing_ok=True)
        print(f"{y}: OK -> {P.rel(final)} ({info})", flush=True)


if __name__ == "__main__":
    main()
