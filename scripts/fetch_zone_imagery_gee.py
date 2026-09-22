"""Genera imágenes satelitales de inicio/fin de período (2000 / 2022) por zona,
para superponer en el visor `viz/clusters/` -- vía Google Earth Engine, no la
API pública de Google Earth (esa no existe: ver `viz/clusters/README.md`,
sección "Imagen satelital"). Earth Engine da el mismo Landsat/Sentinel que
`scripts/fetch_zone_imagery.py` (pausado, ver su docstring) pero resuelve el
mosaico y la reproyección puertas adentro -- no arma la mediana a mano leyendo
cada escena en su propia grilla nativa, que era la causa del desalineamiento
que se vio con ese script.

Setup, una sola vez (gratis para uso no comercial/investigación):

    pip install earthengine-api requests
    python -c "import ee; ee.Authenticate()"   # abre el navegador, guarda el token

Earth Engine pide un proyecto de Google Cloud vinculado -- se crea al mismo
tiempo que te das de alta en https://code.earthengine.google.com/register si
todavía no tenés uno. El token queda en ~/.config/earthengine/credentials;
corridas siguientes no vuelven a pedir el navegador.

No se corrió contra Earth Engine real en esta sesión (el entorno de desarrollo
no tiene forma de completar el login interactivo) -- probado solo sintaxis y
lógica de las funciones que no llaman a la API. Correlo vos y avisame
cualquier error para ajustar el pedido exacto a la API.

Fuentes, igual que el intento anterior:
    - 2000: Landsat 5 TM (`LANDSAT/LT05/C02/T1_L2`), SR_B3/B2/B1 = R/G/B,
      máscara de nubes/sombra con QA_PIXEL.
    - 2022 (o el año que se pida): Sentinel-2 SR armonizado
      (`COPERNICUS/S2_SR_HARMONIZED`), B4/B3/B2 = R/G/B, máscara con SCL.
Mediana sobre una ventana de +/- `--pad-months` alrededor del año pedido (por
defecto 3) -- Earth Engine no necesita el ensanche escalonado del script
anterior porque no paga un costo extra por escena de más, así que arranca
directo con una ventana generosa.

Salida: mismo formato que `fetch_zone_imagery.py` / `import_manual_imagery.py`
(`viz/clusters/data/imagery/{zona}_{año}.png` + `imagery/index.json`), así
`index.html` no distingue de dónde salió cada imagen.

Sin --zone/--year corre las 15 zonas (7 de evaluación + 8 de entrenamiento,
`land2vec.zones.ZONES_BY_GROUP`) × (2000, 2022) -- 30 combinaciones.
Si {zona}_{año}.png ya existe (y su entrada en el manifiesto), la salta; para
forzar que la rehaga, `--force`. Así se puede cortar la corrida a mitad de
camino y retomarla después sin perder lo ya bajado.

Uso:
    python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP
    python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP --zone ibera --year 2022
    python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP --zone delta_parana --force
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import ee
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from land2vec.zones import ZONE_LABELS, ZONES_BY_GROUP  # noqa: E402

CLUSTER_MANIFEST = ROOT / "viz" / "clusters" / "data" / "index.json"
OUT_DIR = ROOT / "viz" / "clusters" / "data" / "imagery"
LATLON_DIR = ROOT / "data"

# zonas de entrenamiento -- si build_cluster_map.py todavía no las procesó (no
# hay clusters_train_pooled*.zip), no están en index.json y hay que calcularles
# el bbox a mano con bbox_from_latlon_zip().
EXTRA_ZONE_LABELS = {z: ZONE_LABELS[z] for z in ZONES_BY_GROUP["train"]}

YEARS = [2000, 2022]
MAX_SIDE_DEFAULT = 2000
CLOUD_COVER_MAX = 50
PAD_MONTHS_DEFAULT = 5
MAX_SCENES = 160   # presupuesto total por composite (ver cap_per_tile para Sentinel-2)
FALLBACK_MAX_SCENES = 30   # el respaldo de Landsat 8/9 solo tapa un hueco puntual, no compite en calidad

# rangos de visualización fijos (no percentiles por zona, para no depender de
# una llamada extra a reduceRegion que no pude probar contra la API real).
# Landsat C2 L2 ya viene escalado a reflectancia (ver LANDSAT_SCALE/OFFSET);
# Sentinel-2 SR queda en DN crudo (0-10000), el rango de abajo es el que usan
# los ejemplos oficiales de Earth Engine para compuestos true-color.
LANDSAT_VIS = {"min": 0.0, "max": 0.3}
SENTINEL_VIS = {"min": 0.0, "max": 3000.0}
LANDSAT_SCALE, LANDSAT_OFFSET = 0.0000275, -0.2

# QA_PIXEL (Landsat C2 L2): 0 sin datos, 1 dilated cloud, 3 cloud, 4 shadow.
LANDSAT_CLOUD_BITS = (0, 1, 3, 4)
# SCL (Sentinel-2): 0 sin datos, 1 saturado/defectuoso, 3 sombra, 8/9 nube,
# 10 cirrus delgado. Mismos códigos que fetch_zone_imagery.py.
SENTINEL_CLOUD_CLASSES = [0, 1, 3, 8, 9, 10]


def bbox_from_latlon_zip(zone_id):
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
    else:
        # sin --zone: "todas" son las zonas del manifiesto (eval + train, una
        # vez que build_cluster_map.py procesó los clusters_train_pooled*.zip)
        # más las de EXTRA_ZONE_LABELS que todavía no estén ahí -- p. ej. antes
        # de correr assign_train_clusters.py, ninguna zona de train figura en
        # index.json y esta rama es la única fuente de sus bounds.
        missing = set(EXTRA_ZONE_LABELS) - {z["id"] for z in zones}
    for zid in sorted(missing):
        bounds = bbox_from_latlon_zip(zid)
        if bounds is None:
            if only:
                raise SystemExit(
                    f"zona '{zid}' no está en {CLUSTER_MANIFEST.name} ni tiene "
                    f"data/lat_long_df_{zid}.zip para calcularle el bbox"
                )
            continue  # "todas": si falta el zip de una extra, se salta sin frenar todo
        zones.append({"id": zid, "label": EXTRA_ZONE_LABELS.get(zid, zid), "bounds": bounds})
    return zones


def bbox_of(bounds):
    (south, west), (north, east) = bounds
    return [west, south, east, north]


def output_dims(bbox, max_side):
    west, south, east, north = bbox
    dx, dy = east - west, north - south
    if dx >= dy:
        return max_side, max(1, round(max_side * dy / dx))  # (w, h)
    return max(1, round(max_side * dx / dy)), max_side


def mask_landsat(img):
    qa = img.select("QA_PIXEL")
    bits = 0
    for b in LANDSAT_CLOUD_BITS:
        bits |= 1 << b
    mask = qa.bitwiseAnd(bits).eq(0)
    scaled = img.select(["SR_B3", "SR_B2", "SR_B1"], ["R", "G", "B"]).multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET)
    return scaled.updateMask(mask).copyProperties(img, ["system:time_start"])


def mask_landsat89(img):
    # igual que mask_landsat, pero Landsat 8/9 (OLI) numera las bandas RGB
    # como SR_B4/B3/B2 en vez de SR_B3/B2/B1 (TM, Landsat 5) -- mismo QA_PIXEL
    # y mismo factor de escala en Collection 2 L2.
    qa = img.select("QA_PIXEL")
    bits = 0
    for b in LANDSAT_CLOUD_BITS:
        bits |= 1 << b
    mask = qa.bitwiseAnd(bits).eq(0)
    scaled = img.select(["SR_B4", "SR_B3", "SR_B2"], ["R", "G", "B"]).multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET)
    return scaled.updateMask(mask).copyProperties(img, ["system:time_start"])


def mask_sentinel(img):
    scl = img.select("SCL")
    bad = ee.List(SENTINEL_CLOUD_CLASSES)
    mask = scl.remap(bad, ee.List.repeat(0, bad.size()), 1)
    return img.select(["B4", "B3", "B2"], ["R", "G", "B"]).updateMask(mask).copyProperties(img, ["system:time_start"])


def date_window(year, pad_months):
    start = ee.Date.fromYMD(year, 1, 1).advance(-pad_months, "month")
    end = ee.Date.fromYMD(year, 12, 31).advance(pad_months, "month")
    return start, end


def cap_per_tile(coll, tile_prop, cloud_prop, total_budget, min_per_tile=8):
    """Reparte `total_budget` escenas ENTRE LOS TILES que hagan falta, no un
    tope fijo por tile -- agrupa por `tile_prop`, calcula cuántos tiles
    distintos hay y le da a cada uno `total_budget / n_tiles` escenas,
    ordenadas por nubosidad. Necesario para Sentinel-2: una zona grande
    necesita varios tiles MGRS (~100x100 km) para cubrirla entera, y un tope
    GLOBAL ordenado por nubosidad de toda la colección le puede dar todo el
    cupo a 1-2 tiles con poca nube y dejar los demás enteros afuera -- eso se
    veía como agujeros negros rectangulares en el compuesto final, no ruido.
    Repartir por tile mantiene el mismo total (ya validado que no tira "User
    memory limit exceeded") pero distribuido.

    `min_per_tile` es solo el umbral bajo el cual avisar que el reparto puede
    dejar agujeros genuinos (no hay presupuesto para darle a cada tile ni esa
    mínima cantidad) -- nunca se usa como techo: con pocos tiles y presupuesto
    de sobra, cada uno se queda con `total_budget // n_tiles`, no con el
    mínimo. Un bug previo hacía justo eso (clampeaba al mínimo aun sobrando
    presupuesto) y dejaba agujeros por nubosidad en zonas de pocos tiles
    -- ver `puna_salta_catamarca`/`yungas` en viz/clusters/README.md."""
    tiles = coll.aggregate_array(tile_prop).distinct()
    n_tiles = max(1, tiles.size().getInfo())
    per_tile_cap = max(1, total_budget // n_tiles)
    if per_tile_cap < min_per_tile:
        print(f"    aviso: {n_tiles} tiles para {total_budget} escenas de presupuesto da solo "
              f"{per_tile_cap}/tile (mínimo recomendado {min_per_tile}) -- posibles agujeros por nubosidad")
    # cuántas escenas hay disponibles por tile ANTES del tope -- si algún
    # tile da 0, ese agujero es falta de dato real (nada por debajo del
    # --cloud-cover pedido en toda la ventana), no algo que un presupuesto
    # más alto vaya a arreglar.
    tile_names = tiles.getInfo()
    counts = [coll.filter(ee.Filter.eq(tile_prop, t)).size().getInfo() for t in tile_names]
    detail = ", ".join(f"{t}:{c}" for t, c in zip(tile_names, counts))
    print(f"    {n_tiles} tiles, {per_tile_cap} escenas/tile (~{n_tiles * per_tile_cap} total) -- {detail}")

    def images_for_tile(tile):
        return coll.filter(ee.Filter.eq(tile_prop, tile)).sort(cloud_prop).limit(per_tile_cap).toList(per_tile_cap)

    per_tile_lists = tiles.map(images_for_tile)
    return ee.ImageCollection(ee.List(per_tile_lists).flatten())


def add_landsat_tile_id(img):
    # Sentinel-2 ya trae MGRS_TILE como property nativa; Landsat no tiene un
    # id de tile único, hay que armarlo de WRS_PATH + WRS_ROW.
    tile_id = ee.Number(img.get("WRS_PATH")).multiply(1000).add(ee.Number(img.get("WRS_ROW")))
    return img.set("TILE_ID", tile_id)


def visualized_composite(coll, tile_prop, cloud_prop, mask_fn, vis, budget=MAX_SCENES):
    """Tope por tile + máscara + mediana + visualize -- el cuerpo común a
    cualquier composite (Landsat 5, Landsat 8/9 o Sentinel-2). Devuelve
    (imagen visualizada 0-255, n_scenes usadas)."""
    coll = cap_per_tile(coll, tile_prop, cloud_prop, budget)
    n_scenes = coll.size().getInfo()
    masked = coll.map(mask_fn)
    composite = masked.median().visualize(bands=["R", "G", "B"], min=vis["min"], max=vis["max"])
    return composite, n_scenes


def build_composite(region, year, pad_months, cloud_cover, landsat_fallback=False):
    is_landsat = year < 2013  # Landsat 5 dejó de operar en 2012
    start, end = date_window(year, pad_months)
    if is_landsat:
        cloud_prop, tile_prop = "CLOUD_COVER", "TILE_ID"
        coll = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt(cloud_prop, cloud_cover))
            .map(add_landsat_tile_id)
        )
        vis = LANDSAT_VIS
        source = "landsat-5-gee"
    else:
        cloud_prop, tile_prop = "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt(cloud_prop, cloud_cover))
        )
        vis = SENTINEL_VIS
        source = "sentinel-2-l2a-gee"
    n_scenes = coll.size().getInfo()
    if n_scenes == 0:
        raise RuntimeError(f"sin escenas en la ventana {start.format().getInfo()}/{end.format().getInfo()}")
    # reparto por tile (path/row de Landsat, MGRS de Sentinel) en vez de un
    # tope global ordenado por nubosidad -- una zona ancha puede cruzar más
    # de un tile/path-row (le pasó a chaco_santiago_frontier en Landsat pese
    # a su swath de 185 km), y un tope global le da todo el cupo a los tiles
    # con menos nube y deja los demás enteros afuera -- eso se veía como
    # agujeros negros rectangulares en el compuesto, no ruido.
    composite, n_scenes = visualized_composite(coll, tile_prop, cloud_prop, mask_landsat if is_landsat else mask_sentinel, vis)
    meta = {
        "source": source,
        "n_scenes": n_scenes,
        "date_range": f"{start.format('YYYY-MM-dd').getInfo()}/{end.format('YYYY-MM-dd').getInfo()}",
    }

    if landsat_fallback and not is_landsat:
        # Sentinel-2 puede quedar con una costura sin dato justo en el límite
        # entre dos husos UTM (su grilla de tiles MGRS es por huso; le pasó a
        # patagonia_estepa, chaco_santiago_frontier y pampa_nucleo, las 3
        # zonas que cruzan un límite). Landsat 8/9 usa una grilla totalmente
        # distinta (path/row, sin ese corte), así que se arma un segundo
        # compuesto solo para tapar esa costura -- se ve más grueso (30 m)
        # nada más que en el pedacito donde Sentinel-2 no tenía nada.
        fallback_coll = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover))
            .map(add_landsat_tile_id)
        )
        if fallback_coll.size().getInfo() > 0:
            fallback_vis, fallback_n = visualized_composite(
                fallback_coll, "TILE_ID", "CLOUD_COVER", mask_landsat89, LANDSAT_VIS,
                budget=FALLBACK_MAX_SCENES,
            )
            composite = ee.ImageCollection([fallback_vis, composite]).mosaic()
            meta["landsat_fallback_scenes"] = fallback_n

    return composite, meta


def export_png(image, region, dims, path):
    url = image.getThumbURL({
        "region": region, "dimensions": f"{dims[0]}x{dims[1]}", "format": "png",
    })
    r = requests.get(url, timeout=420)
    if not r.ok:
        # el cuerpo trae el motivo real (p. ej. límite de tamaño excedido);
        # sin esto solo se ve "400 Bad Request", que no dice nada.
        raise RuntimeError(f"{r.status_code} en getPixels: {r.text[:500]}")
    path.write_bytes(r.content)


def already_done(zone_entry, year, out_dir):
    entry = zone_entry.get(str(year))
    return bool(entry) and (out_dir / f"{entry['file'].split('/')[-1]}").exists()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="proyecto de Google Cloud vinculado a Earth Engine")
    p.add_argument("--zone", action="append", help="repetible; default: todas las de index.json")
    p.add_argument("--year", type=int, action="append", help="repetible; default: 2000 y 2022")
    p.add_argument("--max-side", type=int, default=MAX_SIDE_DEFAULT)
    p.add_argument("--cloud-cover", type=int, default=CLOUD_COVER_MAX)
    p.add_argument("--pad-months", type=int, default=PAD_MONTHS_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--force", action="store_true", help="reprocesa aunque ya exista {zona}_{año}.png")
    p.add_argument("--landsat-fallback", action="store_true",
                    help="tapa con Landsat 8/9 las costuras sin dato de Sentinel-2 en límites de huso UTM")
    return p.parse_args()


def main():
    args = parse_args()
    ee.Initialize(project=args.project)

    zones = load_zones(args.zone)
    years = args.year or YEARS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for zone in zones:
        bbox = bbox_of(zone["bounds"])
        region = ee.Geometry.Rectangle(bbox)
        dims = output_dims(bbox, args.max_side)
        zone_entry = manifest.setdefault(zone["id"], {"bounds": zone["bounds"], "label": zone["label"]})
        for year in years:
            if not args.force and already_done(zone_entry, year, args.out_dir):
                print(f"{zone['id']} / {year} -- ya está ({zone_entry[str(year)]['file']}), salteo")
                continue
            print(f"{zone['id']} / {year} -- dims {dims}")
            try:
                composite, meta = build_composite(
                    region, year, args.pad_months, args.cloud_cover, args.landsat_fallback,
                )
                fname = f"{zone['id']}_{year}.png"
                export_png(composite, region, dims, args.out_dir / fname)
            except Exception as e:
                print(f"  !! {zone['id']} {year} falló: {e!r}")
                continue
            meta["file"] = f"imagery/{fname}"
            zone_entry[str(year)] = meta
            print(f"  -> {fname} ({meta['n_scenes']} escenas, ventana {meta['date_range']})")
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nmanifiesto: {manifest_path}")


if __name__ == "__main__":
    main()
