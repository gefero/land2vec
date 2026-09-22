"""PAUSADO: el compuesto de mediana entre escenas de distintas órbitas sale con
un desalineamiento visible (rayado) en zonas grandes -- cada escena se
reproyecta a una grilla EPSG:4326 común (ver `target_transform`) pero el
resultado visual no convenció. Se optó por capturas manuales de Google Earth
Pro en su lugar (`scripts/make_zone_kml.py` + `scripts/import_manual_imagery.py`,
ver `viz/clusters/README.md`). Se deja el código como referencia por si vale
la pena retomarlo (con Earth Engine y sus algoritmos de mosaico en vez de esto
armado a mano, probablemente).

Genera imágenes satelitales de inicio/fin de período (2000 / 2022) por zona,
para superponer en el visor `viz/clusters/` como referencia visual del paisaje.

Compuesto de mediana anual libre de nubes por zona, vía el catálogo STAC
público de Microsoft Planetary Computer (sin cuenta ni token propios):

    - 2000: Landsat 5 TM (`landsat-c2-l2`), bandas red/green/blue + qa_pixel
      para la máscara de nubes/sombra. Reflectancia de superficie escalada
      (factor 0.0000275, offset -0.2) y estirada por percentiles (2, 98).
    - 2022 (o el año que se pida): Sentinel-2 L2A (`sentinel-2-l2a`), asset
      `visual` (RGB 8 bits ya compuesto) + `SCL` para la máscara de nubes.

Si la mediana del año calendario queda con cobertura pobre (zonas con mucha
nube, ej. selva misionera / Iberá), la ventana de fechas se ensancha
automáticamente en pasos de +/-2 meses hasta un máximo, y ese rango real
queda registrado en el manifiesto -- no se ensancha para todas las zonas por
igual, solo para las que lo necesitan.

Las zonas y sus bounds se leen de `viz/clusters/data/index.json` (el
manifiesto que ya genera `build_cluster_map.py`), así no se duplica la lista
de las 7 zonas OOD ni se recalculan bounds.

Salida (gitignoreada, igual que el resto de `viz/clusters/data/`):

    viz/clusters/data/imagery/{zona}_{anio}.png   RGBA, huecos sin datos transparentes
    viz/clusters/data/imagery/index.json          manifiesto propio: bounds,
                                                    rango de fechas real usado,
                                                    cobertura, fuente

Manifiesto separado del de `build_cluster_map.py` a propósito: si se vuelve a
correr ese script no pisa esto (y viceversa).

Uso:
    python scripts/fetch_zone_imagery.py                          # las 7 zonas, 2000 y 2022
    python scripts/fetch_zone_imagery.py --zone periurbano_cordoba --zone ibera
    python scripts/fetch_zone_imagery.py --year 2022 --max-side 1200
    python scripts/fetch_zone_imagery.py --workers 24 --min-coverage 0.9

Requiere `pystac-client`, `planetary-computer`, `rasterio` (ver
requirements.txt) además de Pillow y numpy, que ya son parte del entorno.
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.transform
import rasterio.warp
from PIL import Image
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_MANIFEST = ROOT / "viz" / "clusters" / "data" / "index.json"
OUT_DIR = ROOT / "viz" / "clusters" / "data" / "imagery"
LATLON_DIR = ROOT / "data"

# zonas sin clustering (training, no las 7 OOD de build_cluster_map.py) para
# las que igual queremos la imagen satelital de referencia -- bbox calculado
# de su lat_long_df_*.zip, no de index.json.
EXTRA_ZONE_LABELS = {
    "chaco_santiago_frontier": "Chaco-Santiago (frontera, train)",
}

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# timeouts explícitos: sin esto, una conexión colgada a Azure Blob deja
# esperando al hilo para siempre y el pipeline nunca termina.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_TIMEOUT=30,
    GDAL_HTTP_CONNECTTIMEOUT=15,
    GDAL_HTTP_MAX_RETRY=2,
    GDAL_HTTP_RETRY_DELAY=2,
)

YEARS = [2000, 2022]
MAX_SIDE_DEFAULT = 2000
CLOUD_COVER_MAX = 50
MIN_COVERAGE_DEFAULT = 0.85
WINDOW_STEPS_MONTHS = [2, 4, 6, 9, 12]   # ensanches sucesivos a +/- N meses del año pedido
WORKERS_DEFAULT = 16
MAX_SCENES = 30   # tope duro por composite (tiempo/memoria acotados); prioriza las menos nubosas

# Landsat Collection 2 L2 qa_pixel: bit 0 = sin datos (fill, borde de escena),
# 1 = dilated cloud, 3 = cloud, 4 = shadow. Sin bit 0, el negro de relleno del
# borde de la escena entra al compuesto como si fuera reflectancia real.
LANDSAT_CLOUD_BITS = (0, 1, 3, 4)
LANDSAT_SCALE, LANDSAT_OFFSET = 0.0000275, -0.2
# Sentinel-2 SCL: 0 sin datos (borde de escena/tile), 1 saturado/defectuoso,
# 3 sombra, 8/9 nube media/alta, 10 cirrus delgado. Mismo motivo que arriba.
SENTINEL_CLOUD_CLASSES = {0, 1, 3, 8, 9, 10}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", action="append", help="repetible; default: todas las de index.json")
    p.add_argument("--year", type=int, action="append", help="repetible; default: 2000 y 2022")
    p.add_argument("--max-side", type=int, default=MAX_SIDE_DEFAULT)
    p.add_argument("--cloud-cover", type=int, default=CLOUD_COVER_MAX)
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE_DEFAULT,
                    help="fracción de píxeles con >=1 observación válida antes de ensanchar la ventana")
    p.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def bbox_from_latlon_zip(zone_id):
    """Bounds de una zona sin clustering, calculados de su lat_long_df_*.zip
    (mismos archivos que usa build_cluster_map.py para las zonas OOD, pero acá
    leídos directo porque la zona no pasa por ese script)."""
    import csv
    import io
    import zipfile

    path = LATLON_DIR / f"lat_long_df_{zone_id}.zip"
    if not path.exists():
        return None
    lat_min = lat_max = lon_min = lon_max = None
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
            lat, lon = float(row["latitude"]), float(row["longitude"])
            lat_min = lat if lat_min is None else min(lat_min, lat)
            lat_max = lat if lat_max is None else max(lat_max, lat)
            lon_min = lon if lon_min is None else min(lon_min, lon)
            lon_max = lon if lon_max is None else max(lon_max, lon)
    return [[lat_min, lon_min], [lat_max, lon_max]]


def load_zones(only=None):
    manifest = json.loads(CLUSTER_MANIFEST.read_text())
    zones = manifest["zones"]
    if only:
        wanted = set(only)
        zones = [z for z in zones if z["id"] in wanted]
        missing = wanted - {z["id"] for z in zones}
        for zid in sorted(missing):
            bounds = bbox_from_latlon_zip(zid)
            if bounds is None:
                raise SystemExit(
                    f"zona '{zid}' no está en {CLUSTER_MANIFEST.name} ni tiene "
                    f"data/lat_long_df_{zid}.zip para calcularle el bbox"
                )
            zones.append({"id": zid, "label": EXTRA_ZONE_LABELS.get(zid, zid), "bounds": bounds})
    return zones


def bbox_of(bounds):
    (south, west), (north, east) = bounds
    return [west, south, east, north]


def output_shape(bbox, max_side):
    west, south, east, north = bbox
    dx, dy = east - west, north - south
    if dx >= dy:
        w = max_side
        h = max(1, round(max_side * dy / dx))
    else:
        h = max_side
        w = max(1, round(max_side * dx / dy))
    return h, w


def date_window(year, pad_months):
    if pad_months == 0:
        return f"{year}-01-01/{year}-12-31"
    start_month = 1 - pad_months
    end_month = 12 + pad_months
    start_year = year + (start_month - 1) // 12
    start_month = (start_month - 1) % 12 + 1
    end_year = year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    return f"{start_year:04d}-{start_month:02d}-01/{end_year:04d}-{end_month:02d}-28"


def search_items(catalog, collection, bbox, datetime_range, cloud_cover, platform=None):
    query = {"eo:cloud_cover": {"lt": cloud_cover}}
    if platform:
        query["platform"] = {"in": [platform]}
    search = catalog.search(collections=[collection], bbox=bbox, datetime=datetime_range, query=query)
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items[:MAX_SCENES]


def with_retries(fn, *args, attempts=3, delay=3):
    # GDAL_HTTP_MAX_RETRY solo cubre errores de red a nivel curl; un tile
    # truncado (TIFFFillTile/TIFFReadEncodedTile) llega como excepción de
    # rasterio después de una respuesta HTTP "exitosa" pero incompleta, y ese
    # no lo reintenta GDAL solo -- pasa unas pocas veces por corrida en zonas
    # grandes con muchas lecturas concurrentes.
    for attempt in range(attempts):
        try:
            return fn(*args)
        except rasterio.errors.RasterioIOError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def target_transform(bbox, shape):
    """Transform EPSG:4326 fijo para (H, W) sobre bbox -- MISMO para todas las
    escenas de un composite, sin importar su CRS/órbita nativa. Antes cada
    escena se leía con una ventana en SU PROPIO CRS nativo (más rápido, un
    solo read por banda), pero eso alinea las escenas solo aproximadamente:
    a resolución alta, y en zonas que cruzan más de una pasada satelital, el
    desalineamiento entre tomas se veía como rayado vertical en la mediana."""
    west, south, east, north = bbox
    h, w = shape
    return rasterio.transform.from_bounds(west, south, east, north, w, h)


def read_band(href, bbox, shape, resampling):
    def _read():
        dst = np.full(shape, np.nan, dtype="float32")
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(href) as src:
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1), destination=dst,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=target_transform(bbox, shape), dst_crs="EPSG:4326",
                    resampling=resampling, src_nodata=None, dst_nodata=np.nan,
                )
        return dst
    return with_retries(_read)


def read_visual(href, bbox, shape):
    def _read():
        dst = np.full((3, *shape), np.nan, dtype="float32")
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(href) as src:
                rasterio.warp.reproject(
                    source=rasterio.band(src, (1, 2, 3)), destination=dst,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=target_transform(bbox, shape), dst_crs="EPSG:4326",
                    resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan,
                )
        return dst
    return with_retries(_read)


def submit_landsat(executor, item, bbox, shape):
    return {b: executor.submit(read_band, item.assets[b].href, bbox, shape,
                                Resampling.nearest if b == "qa_pixel" else Resampling.average)
            for b in ("red", "green", "blue", "qa_pixel")}


def assemble_landsat(futs, shape):
    qa = futs["qa_pixel"].result().astype("uint16")
    cloud = np.zeros(shape, dtype=bool)
    for bit in LANDSAT_CLOUD_BITS:
        cloud |= (qa & (1 << bit)) > 0
    rgb = np.stack([futs[b].result().astype("float32") for b in ("red", "green", "blue")])
    rgb = rgb * LANDSAT_SCALE + LANDSAT_OFFSET
    return np.where(cloud, np.nan, np.clip(rgb, 0, 1))  # (3, H, W) reflectancia 0-1, NaN = nube/sombra


def submit_sentinel(executor, item, bbox, shape):
    return {
        "rgb": executor.submit(read_visual, item.assets["visual"].href, bbox, shape),
        "scl": executor.submit(read_band, item.assets["SCL"].href, bbox, shape, Resampling.nearest),
    }


def assemble_sentinel(futs, shape):
    rgb = futs["rgb"].result().astype("float32")
    cloud = np.isin(futs["scl"].result().astype("uint8"), list(SENTINEL_CLOUD_CLASSES))
    return np.where(cloud, np.nan, rgb)  # (3, H, W) 0-255, NaN = nube


def stretch_to_uint8(band, lo_pct=2, hi_pct=98):
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((band - lo) / (hi - lo), 0, 1) * 255
    return out


def build_composite(zone_id, year, bbox, shape, cloud_cover, min_coverage, workers):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace, timeout=30)
    is_landsat = year < 2013  # Landsat 5 dejó de operar en 2012; Sentinel-2 arranca en 2015
    collection = "landsat-c2-l2" if is_landsat else "sentinel-2-l2a"
    platform = "landsat-5" if is_landsat else None

    submit = submit_landsat if is_landsat else submit_sentinel
    assemble = assemble_landsat if is_landsat else assemble_sentinel

    used_pad = 0
    items, stack = [], None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pad in [0] + WINDOW_STEPS_MONTHS:
            dt = date_window(year, pad)
            items = search_items(catalog, collection, bbox, dt, cloud_cover, platform)
            used_pad = pad
            if not items:
                continue
            futs = [submit(executor, it, bbox, shape) for it in items]
            scenes = [assemble(f, shape) for f in futs]
            stack = np.stack(scenes, axis=0)  # (N, 3, H, W)
            coverage = float(np.mean(np.any(~np.isnan(stack[:, 0]), axis=0)))
            print(f"  {zone_id} {year}: ventana ±{pad}m, {len(items)} escenas, cobertura {coverage:.1%}")
            if coverage >= min_coverage:
                break

    if stack is None:
        raise RuntimeError(f"sin escenas para {zone_id} {year} ni ensanchando la ventana")

    # píxeles sin ninguna observación válida (bbox rectangular con esquinas
    # fuera de toda escena, típico en zonas grandes/irregulares) -> NaN de
    # sobra, se resuelven en alpha=0 (transparente) más abajo.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        composite = np.nanmedian(stack, axis=0)  # (3, H, W)
    alpha = np.any(~np.isnan(stack[:, 0]), axis=0).astype("uint8") * 255

    if is_landsat:
        rgb_u8 = np.stack([stretch_to_uint8(composite[c]) for c in range(3)]).astype("uint8")
    else:
        rgb_u8 = np.nan_to_num(composite, nan=0).astype("uint8")

    rgba = np.concatenate([rgb_u8, alpha[None]], axis=0)  # (4, H, W)
    dt_used = date_window(year, used_pad)
    return rgba, {
        "source": "landsat-5" if is_landsat else "sentinel-2-l2a",
        "n_scenes": len(items),
        "date_range": dt_used,
        "coverage": round(float(np.mean(alpha > 0)), 4),
    }


def save_png(rgba, path):
    arr = np.moveaxis(rgba, 0, -1)  # (H, W, 4)
    Image.fromarray(arr, mode="RGBA").save(path)


def main():
    args = parse_args()
    zones = load_zones(args.zone)
    years = args.year or YEARS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for zone in zones:
        bbox = bbox_of(zone["bounds"])
        shape = output_shape(bbox, args.max_side)
        zone_entry = manifest.setdefault(zone["id"], {"bounds": zone["bounds"], "label": zone["label"]})
        for year in years:
            t0 = time.time()
            print(f"{zone['id']} / {year} -- shape {shape}")
            try:
                rgba, meta = build_composite(
                    zone["id"], year, bbox, shape, args.cloud_cover, args.min_coverage, args.workers,
                )
            except Exception as e:
                # una escena corrupta/con timeout no debe tirar abajo el resto
                # de la corrida -- se loguea y se sigue con la próxima zona/año
                # (rerun puntual después con --zone X --year Y).
                print(f"  !! {zone['id']} {year} falló: {e!r}")
                continue
            fname = f"{zone['id']}_{year}.png"
            save_png(rgba, args.out_dir / fname)
            meta["file"] = f"imagery/{fname}"
            zone_entry[str(year)] = meta
            print(f"  -> {fname} ({time.time() - t0:.0f}s, {meta['n_scenes']} escenas, "
                  f"cobertura {meta['coverage']:.1%}, ventana {meta['date_range']})")
            # se escribe después de CADA año, no al final de la zona: si algo
            # falla más adelante, lo ya calculado no se pierde.
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nmanifiesto: {manifest_path}")


if __name__ == "__main__":
    main()
