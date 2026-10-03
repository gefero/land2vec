"""Preprocesamiento de los mapas anuales de ESA CCI / C3S Land Cover.

De los mapas crudos (un .nc global por año, leyenda LCCS completa) a la serie 2000-2022
agrupada y recortada, y de ahí a la tabla de secuencias `"F-F-…-A"`.

Por archivo anual (cada paso se puede usar solo):
    read_raw(path)                     1. abre el crudo, sin cargarlo en memoria
    clip_to_vector(da, vector)         3. recorta al bbox del vector; fuera del polígono -> 255
    group_codes(da, mapping)           2. agrupa los códigos LCCS con un dict código -> token
    save_year(da, out_dir)             4. guarda `lccs_<año>.nc` (sólo la variable agrupada)
    process_year(path, vector, out_dir, mapping)   los cuatro, encadenados

Sobre la serie:
    stack_years(paths, out)            .nc 3D (time, lat, lon) a partir de los anuales
    sequences_table(ds) / write_sequences(ds, path)   tabla ID, latitude, longitude, seqs

Convenciones: el valor entero de cada estado es su posición en la lista de tokens
(`token_table`); 255 es el relleno (fuera del vector) y no es un estado. `lat` baja y `lon`
sube, como en los productos. El dict de agrupación y la lista de tokens quedan en los
atributos `lccs_a_estado` y `estados` de cada archivo, y se verifican al apilar.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

from . import paths as P
from .extract import LCCS_CODE_TO_TOKEN

FILL = 255
_UNMAPPED = 254

# Estados por defecto, en el orden del netCDF original (Nd = 0).
DEFAULT_TOKENS: list[str] = [LCCS_CODE_TO_TOKEN[i] for i in sorted(LCCS_CODE_TO_TOKEN)]

# Agrupación por defecto, código LCCS -> token. Es la regla con la que se construyó el netCDF
# original (relevada contra los mapas crudos): la agregación IPCC estándar. Los subniveles
# (11, 12, 61, 62, 151-153, 201, 202, ...) heredan el token de su clase padre (code // 10 * 10).
# Ojo: difiere de la tabla de docs/v2/paper_metodologia.md §1.2 en 110 (-> G) y 160/170 (-> F).
LCCS_IPCC: dict[int, str] = {
    0: "Nd",                                          # no data
    10: "A", 20: "A", 30: "A", 40: "A",               # cultivos y mosaicos con cultivo
    50: "F", 60: "F", 70: "F", 80: "F", 90: "F",      # tree cover
    100: "F",                                         # mosaico árbol-arbusto (>50 %) / herbáceo
    160: "F", 170: "F",                               # tree cover flooded (agua dulce / salina)
    110: "G", 130: "G",                               # mosaico herbáceo (>50 %) / árbol-arbusto; grassland
    180: "Wt",                                        # shrub or herbaceous cover, flooded
    190: "U",                                         # urban
    120: "Sh",                                        # shrubland
    140: "Sp", 150: "Sp",                             # líquenes y musgos; vegetación esparsa
    200: "B",                                         # bare areas
    210: "Wa", 220: "Wa",                             # agua, nieve y hielo
}

_NAME_RE = re.compile(r"P1Y-(\d{4})-v(\d[\w.]*?)\.nc$")


# --------------------------------------------------------------------------- tokens
def token_table(mapping: dict[int, str], tokens: Sequence[str] | None = None) -> list[str]:
    """Lista de tokens; el código entero de cada estado es su posición.

    Con `tokens=None`: el orden del netCDF original si el dict usa exactamente esos tokens
    (`DEFAULT_TOKENS`); si no, `Nd` primero (si está) y el resto por orden de código LCCS.
    """
    used = list(dict.fromkeys(mapping[k] for k in sorted(mapping)))
    if tokens is None:
        tokens = DEFAULT_TOKENS if set(used) == set(DEFAULT_TOKENS) else sorted(used, key=lambda t: (t != "Nd", used.index(t)))
    tokens = list(tokens)
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"tokens repetidos: {tokens}")
    if len(tokens) >= _UNMAPPED:
        raise ValueError("demasiados tokens para uint8")
    missing = set(used) - set(tokens)
    if missing:
        raise ValueError(f"el dict usa tokens que no están en la lista: {sorted(missing)}")
    return tokens


# --------------------------------------------------------------------------- vector
def as_geoseries(vector) -> gpd.GeoSeries:
    """Geometrías en EPSG:4326 a partir de una ruta, un GeoDataFrame/GeoSeries, una geometría
    shapely o una tupla (minx, miny, maxx, maxy). Sin CRS se asume EPSG:4326."""
    if isinstance(vector, (str, Path)):
        vector = gpd.read_file(vector)
    elif isinstance(vector, (tuple, list)) and len(vector) == 4:
        vector = gpd.GeoSeries([box(*vector)], crs=4326)
    elif isinstance(vector, shapely.geometry.base.BaseGeometry):
        vector = gpd.GeoSeries([vector], crs=4326)
    geoms = vector.geometry if isinstance(vector, gpd.GeoDataFrame) else vector
    if not isinstance(geoms, gpd.GeoSeries):
        raise TypeError(f"vector no reconocido: {type(vector).__name__}")
    if geoms.crs is None:
        return geoms.set_crs(4326)
    return geoms if geoms.crs.to_epsg() == 4326 else geoms.to_crs(4326)


def _describe_vector(vector) -> str:
    if isinstance(vector, (str, Path)):
        return P.rel(vector)
    if isinstance(vector, (tuple, list)):
        return "bbox" + str(tuple(float(x) for x in vector))
    return type(vector).__name__


def _check_axes(lat: np.ndarray, lon: np.ndarray) -> None:
    if len(lat) < 2 or len(lon) < 2 or lat[0] < lat[-1] or lon[0] > lon[-1]:
        raise ValueError("se espera lat descendente y lon ascendente, con al menos 2 valores cada una")


def vector_mask(lat: np.ndarray, lon: np.ndarray, vector) -> np.ndarray:
    """(len(lat), len(lon)) bool: True si el centro del píxel cae dentro del vector.

    La grilla se toma regular (paso = rango / (n - 1)), así que no depende del redondeo de las
    coordenadas guardadas en el archivo."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    _check_axes(lat, lon)
    geoms = as_geoseries(vector)
    geoms = geoms[~(geoms.isna() | geoms.is_empty)]
    if len(geoms) == 1 and geoms.iloc[0].equals(geoms.iloc[0].envelope):   # caja: exacto y sin rasterizar
        minx, miny, maxx, maxy = geoms.total_bounds
        return np.outer((lat >= miny) & (lat <= maxy), (lon >= minx) & (lon <= maxx))
    dy, dx = (lat[0] - lat[-1]) / (len(lat) - 1), (lon[-1] - lon[0]) / (len(lon) - 1)
    return rasterize(((g, 1) for g in geoms), out_shape=(len(lat), len(lon)),
                     transform=from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy),
                     fill=0, dtype="uint8").astype(bool)


def covers(da: xr.DataArray, vector) -> bool:
    "True si la grilla de `da` abarca el bbox del vector (con un píxel de tolerancia)."
    minx, miny, maxx, maxy = as_geoseries(vector).total_bounds
    lat, lon = da["lat"].values.astype(np.float64), da["lon"].values.astype(np.float64)
    dy, dx = abs(lat[0] - lat[1]), abs(lon[1] - lon[0])
    return bool(lat.max() >= maxy - dy and lat.min() <= miny + dy and lon.max() >= maxx - dx and lon.min() <= minx + dx)


def _as_u8(values: np.ndarray) -> np.ndarray:
    "Los .nc del CDS traen los códigos como byte con _Unsigned = 'true' (int8 sin decodificar)."
    return values.view(np.uint8) if values.dtype == np.int8 else values.astype(np.uint8)


# --------------------------------------------------------------------------- pasos por archivo
def read_raw(path: Path | str) -> xr.DataArray:
    """Paso 1. `lccs_class` 2D (lat, lon) de un mapa anual crudo, sin cargar el archivo.

    Año, versión e id salen de los metadatos del archivo (`time_coverage_start`, `product_version`,
    `id`) y quedan en `attrs`; el nombre del archivo sólo es respaldo."""
    path = Path(path)
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=False)
    da = ds["lccs_class"]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    ident = str(ds.attrs.get("id", ""))
    m = _NAME_RE.search(ident + ".nc") or _NAME_RE.search(path.name)
    start = str(ds.attrs.get("time_coverage_start", ""))[:4]
    year = int(start) if start.isdigit() else (int(m.group(1)) if m else None)
    if year is None:
        raise ValueError(f"{path.name}: no se pudo determinar el año")
    da.attrs = {"year": year, "product_version": str(ds.attrs.get("product_version", "") or (m.group(2) if m else "?")),
                "id": ident or path.stem, "source_file": path.name}
    return da


def clip_to_vector(da: xr.DataArray, vector, strict: bool = True) -> xr.DataArray:
    """Paso 3. Recorta al bbox del vector (píxeles cuyo centro cae en el bbox) y pone 255 en los
    que quedan fuera del polígono. Recorta antes de cargar, así un mapa global no entra en memoria.
    Con `strict`, error si la grilla no abarca el vector."""
    geoms = as_geoseries(vector)
    if strict and not covers(da, geoms):
        raise ValueError(f"la grilla ({da.attrs.get('source_file', '?')}) no abarca el vector {tuple(geoms.total_bounds)}")
    minx, miny, maxx, maxy = geoms.total_bounds
    lat, lon = da["lat"].values.astype(np.float64), da["lon"].values.astype(np.float64)
    ii = np.flatnonzero((lat >= miny) & (lat <= maxy))
    jj = np.flatnonzero((lon >= minx) & (lon <= maxx))
    if not len(ii) or not len(jj):
        raise ValueError("el vector no se superpone con la grilla")
    win = da.isel(lat=slice(ii[0], ii[-1] + 1), lon=slice(jj[0], jj[-1] + 1)).load()
    mask = vector_mask(win["lat"].values, win["lon"].values, geoms)
    out = np.where(mask, _as_u8(win.values), FILL).astype(np.uint8)
    return win.copy(data=out)


def group_codes(da: xr.DataArray, mapping: dict[int, str] = LCCS_IPCC, tokens: Sequence[str] | None = None) -> xr.DataArray:
    """Paso 2. Agrupa los códigos LCCS con `mapping` (código -> token) y devuelve el código entero
    de cada token (`token_table`). Un código que no está en el dict hereda el de su clase padre
    (`code // 10 * 10`); si no tiene ninguno y aparece en los datos, error con los conteos. El 255
    se conserva como relleno."""
    tokens = token_table(mapping, tokens)
    index = {t: i for i, t in enumerate(tokens)}
    lut = np.full(256, _UNMAPPED, np.uint8)
    for code in range(FILL):
        tok = mapping.get(code, mapping.get(code // 10 * 10))
        if tok is not None:
            lut[code] = index[tok]
    lut[FILL] = FILL
    codes = _as_u8(da.values)
    out = lut[codes]
    if (out == _UNMAPPED).any():
        vals, cnt = np.unique(codes[out == _UNMAPPED], return_counts=True)
        raise ValueError(f"códigos LCCS sin token en el dict: {dict(zip(vals.tolist(), cnt.tolist()))}")
    attrs = {**da.attrs,
             "estados": json.dumps({str(i): t for i, t in enumerate(tokens)}),
             "lccs_a_estado": json.dumps({str(k): index[v] for k, v in sorted(mapping.items())})}
    return xr.DataArray(out, coords=da.coords, dims=da.dims, attrs=attrs)


def save_year(da: xr.DataArray, out_dir: Path | str) -> Path:
    """Paso 4. Guarda `<out_dir>/lccs_<año>.nc`: sólo la variable agrupada (uint8, relleno 255,
    comprimida), con la procedencia, el dict y los tokens como atributos globales."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lccs_{da.attrs['year']}.nc"
    ds = da.to_dataset(name="lccs_class")
    ds["lccs_class"].attrs = {}
    ds = ds.assign_coords(lat=ds["lat"].astype("float32"), lon=ds["lon"].astype("float32"))
    ds.attrs = dict(da.attrs)
    ny, nx = da.shape
    enc = {"lccs_class": dict(dtype="uint8", zlib=True, complevel=4, shuffle=True, _FillValue=np.uint8(FILL),
                              chunksizes=(min(2100, ny), min(1320, nx)))}
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)
    return path


def process_year(path: Path | str, vector, out_dir: Path | str, mapping: dict[int, str] = LCCS_IPCC,
                 tokens: Sequence[str] | None = None) -> Path:
    "Los cuatro pasos sobre un crudo: leer, recortar, agrupar y guardar. Devuelve la ruta escrita."
    da = clip_to_vector(read_raw(path), vector)
    da = group_codes(da, mapping, tokens)
    da.attrs.update(vector=_describe_vector(vector), history=f"{dt.datetime.now():%Y-%m-%d %H:%M} land2vec.preprocess.process_year")
    return save_year(da, out_dir)


# --------------------------------------------------------------------------- serie
def stack_years(paths: Sequence[Path | str], out: Path | str) -> Path:
    """Apila los anuales de `save_year` en un .nc 3D (time, lat, lon) con el mismo formato del
    netCDF original. Verifica años sin repetir ni saltear, misma grilla y mismo dict/tokens."""
    import h5netcdf.legacyapi as nc

    def _year(p):
        with xr.open_dataset(p, mask_and_scale=False, decode_times=False) as ds:
            return int(ds.attrs["year"])

    items = sorted((_year(p), Path(p)) for p in paths)
    years = [y for y, _ in items]
    if len(set(years)) != len(years):
        raise ValueError(f"años repetidos: {sorted({y for y in years if years.count(y) > 1})}")
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"faltan años entre {years[0]} y {years[-1]}: {sorted(set(range(years[0], years[-1] + 1)) - set(years))}")

    with xr.open_dataset(items[0][1], mask_and_scale=False, decode_times=False) as ds0:
        lat, lon, ref_attrs = ds0["lat"].values, ds0["lon"].values, dict(ds0.attrs)
    procedencia = {}
    for y, p in items:
        with xr.open_dataset(p, mask_and_scale=False, decode_times=False) as ds:
            if not (np.array_equal(ds["lat"].values, lat) and np.array_equal(ds["lon"].values, lon)):
                raise ValueError(f"{p.name}: la grilla no coincide con la de {items[0][1].name}")
            for key in ("estados", "lccs_a_estado"):
                if ds.attrs[key] != ref_attrs[key]:
                    raise ValueError(f"{p.name}: '{key}' distinto del de {items[0][1].name}")
            procedencia[y] = {"archivo": ds.attrs.get("source_file"), "product_version": ds.attrs.get("product_version"), "id": ds.attrs.get("id")}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out, "w") as f:
        f.createDimension("time", len(years))
        f.createDimension("lat", len(lat))
        f.createDimension("lon", len(lon))
        t = f.createVariable("time", "i4", ("time",))
        t.setncattr("units", "days since 2000-01-01 00:00:00")
        t.setncattr("calendar", "proleptic_gregorian")
        t[:] = [(dt.date(y, 1, 1) - dt.date(2000, 1, 1)).days for y in years]
        f.createVariable("lat", "f4", ("lat",))[:] = lat.astype(np.float32)
        f.createVariable("lon", "f4", ("lon",))[:] = lon.astype(np.float32)
        v = f.createVariable("lccs_class", "u1", ("time", "lat", "lon"),
                             chunksizes=(min(3, len(years)), min(2100, len(lat)), min(1320, len(lon))),
                             zlib=True, complevel=4, shuffle=True, fill_value=FILL)
        for k, (y, p) in enumerate(items):
            with xr.open_dataset(p, mask_and_scale=False, decode_times=False) as ds:
                v[k] = ds["lccs_class"].values
        attrs = {"title": "ESA CCI / C3S Land Cover 300 m agrupado (land2vec.preprocess)",
                 "estados": ref_attrs["estados"], "lccs_a_estado": ref_attrs["lccs_a_estado"],
                 "vector": ref_attrs.get("vector", ""),
                 "procedencia_por_anio": json.dumps(procedencia),
                 "history": f"{dt.datetime.now():%Y-%m-%d %H:%M} land2vec.preprocess.stack_years"}
        for k, val in attrs.items():
            f.setncattr(k, val)
    return out


# --------------------------------------------------------------------------- secuencias
def _tokens_of(ds: xr.Dataset, tokens: Sequence[str] | None) -> list[str]:
    if tokens is not None:
        return list(tokens)
    if "estados" in ds.attrs:
        e = json.loads(ds.attrs["estados"])
        return [e[str(i)] for i in range(len(e))]
    return DEFAULT_TOKENS   # netCDF original, sin atributos


def iter_sequences(ds: xr.Dataset, tokens: Sequence[str] | None = None, drop_fill: bool = True,
                   fill_token: str = "Nd", lat_rows: int = 500) -> Iterator[pd.DataFrame]:
    """Tablas ID, latitude, longitude, seqs por franjas de `lat_rows` filas de la grilla.

    ID = posición del píxel en la grilla de `ds` (lon varía más rápido), igual que
    `extract.build_lat_long_df`; con `drop_fill` quedan huecos en el ID. Los píxeles con relleno
    en algún año se descartan, o con `drop_fill=False` el relleno se escribe como `fill_token`.
    Cada trayectoria distinta se arma una sola vez, así que cuesta lo que las trayectorias
    distintas y no los píxeles."""
    toks = _tokens_of(ds, tokens)
    var = ds["lccs_class"]
    T, n_lat, n_lon = var.sizes["time"], var.sizes["lat"], var.sizes["lon"]
    lat, lon = ds["lat"].values, ds["lon"].values
    fill_code = toks.index(fill_token) if not drop_fill else None
    for r0 in range(0, n_lat, lat_rows):
        h = min(lat_rows, n_lat - r0)
        blk = np.nan_to_num(var.isel(lat=slice(r0, r0 + h)).values.astype(np.float32), nan=FILL).astype(np.uint8)
        a = np.ascontiguousarray(blk.transpose(1, 2, 0).reshape(-1, T))
        is_fill = a == FILL
        keep = ~is_fill.any(1) if drop_fill else np.ones(len(a), bool)
        if not drop_fill:
            a[is_fill] = fill_code
        a = a[keep]
        if len(a) == 0:
            continue
        if a.max() >= len(toks):
            raise ValueError(f"códigos fuera de la lista de tokens: {sorted(set(np.unique(a).tolist()) - set(range(len(toks))))}")
        void = a.view(np.dtype((np.void, T))).ravel()
        uniq, inv = np.unique(void, return_inverse=True)
        strings = np.array(["-".join(toks[c] for c in np.frombuffer(u.tobytes(), np.uint8)) for u in uniq], dtype=object)
        yield pd.DataFrame({"ID": np.arange(r0 * n_lon, (r0 + h) * n_lon)[keep],
                            "latitude": np.repeat(lat[r0:r0 + h], n_lon)[keep],
                            "longitude": np.tile(lon, h)[keep],
                            "seqs": strings[inv.ravel()]})


def sequences_table(ds: xr.Dataset, **kwargs) -> pd.DataFrame:
    "Tabla ID, latitude, longitude, seqs de `ds` completa (ver `iter_sequences`). Para zonas; para el país usar `write_sequences`."
    parts = list(iter_sequences(ds, **kwargs))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ID", "latitude", "longitude", "seqs"])


def write_sequences(ds: xr.Dataset, path: Path | str, **kwargs) -> Path:
    "Escribe la tabla de secuencias por franjas (sin tenerla entera en memoria) como .zip (CSV) o .parquet."
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = iter_sequences(ds, **kwargs)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        for df in parts:
            tab = pa.Table.from_pandas(df, preserve_index=False)
            writer = writer or pq.ParquetWriter(path, tab.schema)
            writer.write_table(tab)
        if writer:
            writer.close()
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z, z.open(path.stem + ".csv", "w", force_zip64=True) as f:
            for k, df in enumerate(parts):
                f.write(df.to_csv(index=False, header=(k == 0)).encode())
    else:
        raise ValueError("extensión no soportada (usar .zip o .parquet)")
    return path
