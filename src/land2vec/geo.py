"""Geometría para cruzar los polígonos de desmonte de Colección 13.0 (Chaco Seco,
monitoreodesmonte.com.ar, en `data/geo/data_validacion_chaco_Coleccion_13.0.rar`)
contra la grilla de píxeles ESA CCI de 300 m -- ver `scripts/datos/build_desmonte_labels.py`
(productor) y `scripts/validacion/eval_desmonte.py` (consumidor de la validación externa),
`docs/v2/paper_metodologia.md` §5.8 para el detalle metodológico.

El archivo `.rar` es en realidad un ZIP (firma `PK\x03\x04`). **No se lee vía
GDAL `/vsizip/`**: ese driver decide dónde termina el path del zip y empieza
el path interno mirando la *extensión* del archivo, y con `.rar` (aunque el
contenido sea un ZIP real) esa heurística falla en varias versiones de GDAL
("does not exist in the file system, and is not recognized as a supported
dataset name"). En cambio, `ensure_shapefile_extracted()` extrae el
shapefile una sola vez a una carpeta caché junto al `.rar`
(`data/geo/.<nombre>_extracted/`, ~15-30 s la primera vez, instantáneo
después) y se lee el `.shp` real de ahí -- ninguna otra función de este
módulo asume `/vsizip/`. El shapefile está en WGS84 geográfico (EPSG:4326),
el mismo CRS implícito que usa `land2vec.extract` para la grilla -- no hace
falta reproyectar para el cruce en sí (sí para verificar áreas, ver
`pixel_area_ha`).

La grilla de píxeles nunca se guarda como polígonos: cada `lat_long_df_<zona>.zip`
trae solo el centro de cada píxel. `pixel_bounds` reconstruye el cuadrado a
partir del centro y la resolución conocida (1/360 grado); recién ese cuadrado
se puede intersectar con un polígono de desmonte. `ZoneGrid` invierte esa
relación (de índice de grilla a `ID` de píxel) para una zona dada, incluyendo
las zonas con huecos (p. ej. `yungas`, extraída con submuestreo de constantes).

`pixel_poly_fractions` es la salida primaria: una fila por (polígono, píxel)
tocado, sin agregar -- necesaria para la validación por polígono, no solo por
píxel (un polígono puede quedar repartido en varios píxeles con distinta
fracción). `pixel_fractions` es su agregado por (píxel, época).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path


import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# Resolución de la grilla ESA CCI: 1/360 grado (~300 m). Los centros de píxel de
# `lat_long_df_<zona>.zip` caen en `-180 + (j+0.5)/360`, `90 - (i+0.5)/360` sobre
# la grilla global de 129600x64800 -- ver `land2vec.extract`.
PIXEL_DEG = 1.0 / 360.0

SHP_IN_ZIP = "Coleccion_13.0_Argentina_1976-2024/Coleccion13_WGS_84.shp"
DESM_COLS = ["FECHA_DESM", "PROVINCIA", "DEPARTAMEN", "SUPERF_ha"]

# Radio autálico WGS84 (esfera de igual área que el elipsoide) -- para el área
# geodésica del píxel en `pixel_area_ha`, calibrada contra `pyproj.Geod` en la
# verificación de `scripts/datos/build_desmonte_labels.py`.
_R_EARTH_M = 6_371_007.1809


# ---------------------------------------------------------------------------
# Grilla: índices globales <-> lat/lon, y la grilla local de una zona
# ---------------------------------------------------------------------------

def grid_index(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "Índices globales (i, j) de la grilla ESA CCI para cada centro de píxel (lat, lon)."
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    j = np.round((lon + 180.0) * 360.0 - 0.5).astype(np.int64)
    i = np.round((90.0 - lat) * 360.0 - 0.5).astype(np.int64)
    return i, j


def pixel_bounds(i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    "Bordes (lon_o, lat_s, lon_e, lat_n) del cuadrado del píxel (i, j), reconstruidos desde el índice."
    i = np.asarray(i, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    lon_o = -180.0 + j * PIXEL_DEG
    lon_e = lon_o + PIXEL_DEG
    lat_n = 90.0 - i * PIXEL_DEG
    lat_s = lat_n - PIXEL_DEG
    return lon_o, lat_s, lon_e, lat_n


def pixel_area_ha(i: np.ndarray) -> np.ndarray:
    "Área geodésica exacta (fórmula esférica) de los píxeles de la fila de latitud i, en hectáreas."
    i = np.atleast_1d(np.asarray(i, dtype=np.float64))
    _, lat_s, _, lat_n = pixel_bounds(i, np.zeros_like(i))
    phi_n = np.radians(lat_n)
    phi_s = np.radians(lat_s)
    dlambda = np.radians(PIXEL_DEG)
    area_m2 = (_R_EARTH_M ** 2) * dlambda * (np.sin(phi_n) - np.sin(phi_s))
    return area_m2 / 10_000.0


@dataclass(slots=True)
class ZoneGrid:
    "Grilla densa i,j -> ID de una zona, reconstruida desde `lat_long_df_<zona>.zip`."

    zone: str
    i0: int
    j0: int
    n_i: int
    n_j: int
    ids: np.ndarray  # (n_i, n_j) int64, -1 = sin píxel (hueco -- p. ej. yungas submuestreada)

    @classmethod
    def from_coords(cls, coords_df: pd.DataFrame, zone: str) -> "ZoneGrid":
        "coords_df: columnas ID, latitude, longitude (formato de lat_long_df_<zona>.zip)."
        lat = coords_df["latitude"].to_numpy()
        lon = coords_df["longitude"].to_numpy()
        i, j = grid_index(lat, lon)
        i0, j0 = int(i.min()), int(j.min())
        n_i, n_j = int(i.max() - i0 + 1), int(j.max() - j0 + 1)
        ids = np.full((n_i, n_j), -1, dtype=np.int64)
        ids[i - i0, j - j0] = coords_df["ID"].to_numpy()
        return cls(zone=zone, i0=i0, j0=j0, n_i=n_i, n_j=n_j, ids=ids)

    def lookup(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        "ID del píxel en (i, j), o -1 si cae fuera de la ventana de la zona o en un hueco."
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        ii, jj = i - self.i0, j - self.j0
        out = np.full(ii.shape, -1, dtype=np.int64)
        valid = (ii >= 0) & (ii < self.n_i) & (jj >= 0) & (jj < self.n_j)
        out[valid] = self.ids[ii[valid], jj[valid]]
        return out

    @property
    def complete(self) -> bool:
        "True si la grilla local no tiene huecos (rectángulo completo, p. ej. chaco_santiago_frontier)."
        return bool((self.ids != -1).all())


# ---------------------------------------------------------------------------
# Lectura y limpieza del shapefile de desmonte
# ---------------------------------------------------------------------------

def epoch_of(fecha_desm: np.ndarray) -> np.ndarray:
    "<=2000 -> -1 (línea de base acumulada 1976/1986/1996/2000) ; 2001..2022 -> el año ; >=2023 -> 9999 (posterior)."
    fecha = np.asarray(fecha_desm, dtype=np.int64)
    epoca = fecha.copy()
    epoca[fecha <= 2000] = -1
    epoca[fecha >= 2023] = 9999
    return epoca


def _cache_dir_for(rar: Path) -> Path:
    return rar.parent / f".{rar.stem}_extracted"


def ensure_shapefile_extracted(rar: Path) -> Path:
    """Extrae el shapefile de desmonte a una carpeta caché junto al `.rar` (una sola vez;
    llamadas siguientes son instantáneas) y devuelve la ruta al `.shp` real.

    No se usa GDAL `/vsizip/` acá: ese driver decide dónde termina el path del archivo
    comprimido y empieza el path interno mirando la *extensión* del archivo (reconoce
    `.zip`, `.kmz`, etc.), y con `.rar` -- aunque el contenido sea un ZIP real, firma
    `PK\\x03\\x04` -- esa heurística falla en varias versiones de GDAL/pyogrio con
    "does not exist in the file system, and is not recognized as a supported dataset
    name". Extraer una vez con `zipfile` (stdlib) evita depender de esa heurística por
    completo. Se extraen solo el `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg` (no los `.xlsx` de
    documentación, no hacen falta y son livianos igual)."""
    rar = Path(rar).resolve()
    cache_dir = _cache_dir_for(rar)
    shp_path = cache_dir / SHP_IN_ZIP
    if shp_path.exists():
        return shp_path

    print(f"  (primera vez: extrayendo el shapefile a {cache_dir} -- ~15-30s, después se reusa)")
    prefix = SHP_IN_ZIP.rsplit("/", 1)[0] + "/"
    with zipfile.ZipFile(rar) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and not m.lower().endswith(".xlsx")]
        zf.extractall(cache_dir, members=members)

    if not shp_path.exists():
        raise FileNotFoundError(f"no se encontró {SHP_IN_ZIP} tras extraer {rar} a {cache_dir}")
    return shp_path


def read_desmonte(
    rar: Path,
    bbox: tuple[float, float, float, float] | None = None,
    columns: list[str] = DESM_COLS,
) -> gpd.GeoDataFrame:
    """Lee el shapefile de desmonte (ver `ensure_shapefile_extracted` para por qué no se lee
    vía GDAL `/vsizip/` directo sobre el `.rar`). `bbox` (minx, miny, maxx, maxy) lo aplica
    GDAL contra el índice del shapefile antes de traer geometrías a memoria -- imprescindible
    para no cargar los 216 mil polígonos de toda Argentina cuando solo hace falta una zona.
    Agrega la columna `epoca` (`epoch_of` sobre `FECHA_DESM`), que el resto del módulo asume
    presente."""
    read_cols = list(columns)
    if "FECHA_DESM" not in read_cols:
        read_cols.append("FECHA_DESM")
    shp_path = ensure_shapefile_extracted(rar)
    gdf = gpd.read_file(str(shp_path), engine="pyogrio", bbox=bbox, columns=read_cols, force_2d=True)
    gdf["epoca"] = epoch_of(gdf["FECHA_DESM"].to_numpy())
    return gdf.reset_index(drop=True)


def clean_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Repara geometrías inválidas (frecuentes en digitalización manual: autointersecciones tipo
    "moño") con `shapely.make_valid(method="structure")` -- no `buffer(0)`, que puede borrar un
    lóbulo entero en vez de repararlo -- y descarta lo que no quede poligonal. Devuelve el
    GeoDataFrame limpio y un diccionario de conteos para loguear."""
    geoms = gdf.geometry.to_numpy()
    invalid = ~shapely.is_valid(geoms)
    n_invalid = int(invalid.sum())
    if n_invalid:
        geoms = geoms.copy()
        geoms[invalid] = shapely.make_valid(geoms[invalid], method="structure", keep_collapsed=False)

    kept: list = []
    n_partes_no_poligonales = 0
    for g in geoms:
        if g is None or g.is_empty:
            kept.append(None)
            continue
        if g.geom_type in ("Polygon", "MultiPolygon"):
            kept.append(g)
            continue
        parts = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
        if not parts:
            kept.append(None)
            continue
        n_partes_no_poligonales += 1
        kept.append(shapely.unary_union(parts) if len(parts) > 1 else parts[0])

    out = gdf.copy()
    out["geometry"] = kept
    n_total = len(out)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].reset_index(drop=True)
    stats = {
        "n_total": n_total,
        "n_invalidas_reparadas": n_invalid,
        "n_con_partes_no_poligonales": n_partes_no_poligonales,
        "n_descartadas": n_total - len(out),
    }
    return out, stats


# ---------------------------------------------------------------------------
# El cruce: fracción de área compartida entre cada píxel y cada polígono
# ---------------------------------------------------------------------------

def pixel_poly_fractions(gdf: gpd.GeoDataFrame, grid: ZoneGrid, *, min_frac: float = 1e-4) -> pd.DataFrame:
    """Fracción del PÍXEL cubierta por cada polígono que lo toca, sin agregar -- columnas
    `poly_id, ID, epoca, frac`. `poly_id` es el índice de fila de `gdf` (estable dentro de esta
    corrida, no entre zonas -- mismo criterio que el `ID` de píxel). Es la salida primaria:
    `pixel_fractions` es simplemente su agregado por (ID, epoca).

    Algoritmo por polígono: acota candidatos por el bbox del polígono (índices de grilla
    enteros), separa las celdas totalmente interiores (`contains_properly`, frac=1 sin calcular
    intersección) de las de borde (`intersects & ~contains_properly`, donde sí se calcula el área
    exacta de la intersección). Ver `docs/v2/paper_metodologia.md` §5.8."""
    poly_ids = gdf.index.to_numpy()
    geoms = gdf.geometry.to_numpy()
    epocas = gdf["epoca"].to_numpy()

    cell_area_deg2 = PIXEL_DEG * PIXEL_DEG
    j_min, j_max = grid.j0, grid.j0 + grid.n_j - 1
    i_min, i_max = grid.i0, grid.i0 + grid.n_i - 1

    ids_out, poly_out, epoca_out, frac_out = [], [], [], []

    for poly_id, poly, epoca in zip(poly_ids, geoms, epocas):
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        j_lo = max(j_min, int(np.floor((minx + 180.0) * 360.0)) - 1)
        j_hi = min(j_max, int(np.floor((maxx + 180.0) * 360.0)) + 1)
        i_lo = max(i_min, int(np.floor((90.0 - maxy) * 360.0)) - 1)
        i_hi = min(i_max, int(np.floor((90.0 - miny) * 360.0)) + 1)
        if j_lo > j_hi or i_lo > i_hi:
            continue  # el bbox del polígono no toca la ventana de esta zona

        ii, jj = np.meshgrid(np.arange(i_lo, i_hi + 1), np.arange(j_lo, j_hi + 1), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        lon_o, lat_s, lon_e, lat_n = pixel_bounds(ii, jj)
        cells = shapely.box(lon_o, lat_s, lon_e, lat_n)

        shapely.prepare(poly)
        contains = shapely.contains_properly(poly, cells)
        touches = shapely.intersects(poly, cells) & ~contains

        if contains.any():
            pix_ids = grid.lookup(ii[contains], jj[contains])
            valid = pix_ids != -1
            n = int(valid.sum())
            if n:
                ids_out.append(pix_ids[valid])
                poly_out.append(np.full(n, poly_id))
                epoca_out.append(np.full(n, epoca))
                frac_out.append(np.ones(n))

        if touches.any():
            inter_area = shapely.area(shapely.intersection(poly, cells[touches]))
            frac_t = inter_area / cell_area_deg2
            keep = frac_t > min_frac
            if keep.any():
                pix_ids = grid.lookup(ii[touches][keep], jj[touches][keep])
                valid = pix_ids != -1
                n = int(valid.sum())
                if n:
                    ids_out.append(pix_ids[valid])
                    poly_out.append(np.full(n, poly_id))
                    epoca_out.append(np.full(n, epoca))
                    frac_out.append(frac_t[keep][valid])

    if not ids_out:
        return pd.DataFrame({
            "poly_id": pd.Series(dtype=np.int64), "ID": pd.Series(dtype=np.int64),
            "epoca": pd.Series(dtype=np.int64), "frac": pd.Series(dtype=np.float64),
        })

    return pd.DataFrame({
        "poly_id": np.concatenate(poly_out),
        "ID": np.concatenate(ids_out),
        "epoca": np.concatenate(epoca_out),
        "frac": np.concatenate(frac_out),
    })


def pixel_fractions(
    gdf: gpd.GeoDataFrame, grid: ZoneGrid, *, min_frac: float = 1e-4, dissolve: bool = False,
) -> pd.DataFrame:
    """Agregado por (ID, epoca) de `pixel_poly_fractions` -- columnas `ID, epoca, frac, frac_raw,
    n_pol`. `frac` es `frac_raw` recortada a 1 (los solapes entre polígonos de la misma época
    pueden hacer que la suma agregada supere 1); `frac_raw` queda para poder reportar cuánta masa
    se recortó.

    `dissolve=True` une los polígonos de cada época ANTES del cruce (`gdf.dissolve(by="epoca")`)
    en vez de sumarlos después -- geométricamente correcto para los solapes, pero pierde la
    identidad de polígono individual (`n_pol` queda en NaN) y es más caro. Pensado para activar
    solo si el diagnóstico de `frac_raw` muestra más de ~1% de masa recortada por solapes."""
    if dissolve:
        dissolved = gdf.dissolve(by="epoca", as_index=False)
        dissolved = dissolved.set_index("epoca", drop=False)
        long = pixel_poly_fractions(dissolved, grid, min_frac=min_frac)
        out = long.rename(columns={"frac": "frac_raw"}).drop(columns="poly_id")
        out["n_pol"] = np.nan
        out["frac"] = out["frac_raw"].clip(upper=1.0)
        return out[["ID", "epoca", "frac", "frac_raw", "n_pol"]]

    long = pixel_poly_fractions(gdf, grid, min_frac=min_frac)
    if long.empty:
        return pd.DataFrame({
            "ID": pd.Series(dtype=np.int64), "epoca": pd.Series(dtype=np.int64),
            "frac": pd.Series(dtype=np.float64), "frac_raw": pd.Series(dtype=np.float64),
            "n_pol": pd.Series(dtype=np.int64),
        })
    agg = long.groupby(["ID", "epoca"], as_index=False).agg(
        frac_raw=("frac", "sum"), n_pol=("poly_id", "nunique"),
    )
    agg["frac"] = agg["frac_raw"].clip(upper=1.0)
    return agg[["ID", "epoca", "frac", "frac_raw", "n_pol"]]


# ---------------------------------------------------------------------------
# Máscara de área relevada (y proxy del límite de la ecorregión -- ver
# docs/v2/paper_metodologia.md §5.8: Colección 13.0 no tiene polígonos al este del
# límite Chaco Seco / Chaco Húmedo dentro del bbox de chaco_santiago_frontier)
# ---------------------------------------------------------------------------

def surveyed_mask(gdf: gpd.GeoDataFrame, grid: ZoneGrid, *, cell_deg: float = 0.1, dilate: int = 1) -> np.ndarray:
    """Máscara booleana (n_i, n_j) sobre la grilla fina de `grid`: True donde el relevamiento
    manual de Colección 13.0 efectivamente llegó. Grilla gruesa de `cell_deg` (default 0,1°,
    ~11 km -- muy por encima del tamaño de cualquier polígono individual, así que ubicar cada
    polígono por su centroide en vez de por su forma completa no cambia el resultado), celda
    marcada si contiene el centroide de algún polígono de **cualquier** época 1976-2024 (definen
    la extensión del relevamiento, no del período de interés). Cierre morfológico + relleno de
    huecos + dilatación (`scipy.ndimage`) para no fragmentar en islas por el submuestreo de la
    digitalización manual, luego upsample a la grilla fina por repetición."""
    from scipy import ndimage

    lon_o, lat_s, lon_e, lat_n = pixel_bounds(
        np.array([grid.i0, grid.i0 + grid.n_i - 1]),
        np.array([grid.j0, grid.j0 + grid.n_j - 1]),
    )
    zone_minx, zone_maxx = float(lon_o[0]), float(lon_e[1])
    zone_maxy, zone_miny = float(lat_n[0]), float(lat_s[1])

    n_cols = int(np.ceil((zone_maxx - zone_minx) / cell_deg))
    n_rows = int(np.ceil((zone_maxy - zone_miny) / cell_deg))
    coarse = np.zeros((n_rows, n_cols), dtype=bool)

    centroids = gdf.geometry.centroid
    cx = np.floor((centroids.x.to_numpy() - zone_minx) / cell_deg).astype(np.int64)
    cy = np.floor((zone_maxy - centroids.y.to_numpy()) / cell_deg).astype(np.int64)
    valid = (cx >= 0) & (cx < n_cols) & (cy >= 0) & (cy < n_rows)
    coarse[cy[valid], cx[valid]] = True

    coarse = ndimage.binary_closing(coarse)
    coarse = ndimage.binary_fill_holes(coarse)
    if dilate:
        coarse = ndimage.binary_dilation(coarse, iterations=dilate)

    scale = int(round(cell_deg / PIXEL_DEG))
    fine = np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)
    return fine[: grid.n_i, : grid.n_j]
