"""Acomoda capturas manuales de Google Earth Pro (ver
viz/clusters/README.md, sección "Imagen satelital -- a mano con Google Earth
Pro") en la carpeta y el manifiesto que espera el visor -- mismo formato que
genera scripts/fetch_zone_imagery.py, así index.html no necesita saber de
dónde salió cada imagen.

Convención de nombres esperada en la carpeta de entrada (heredada del vuelo a
cada rectángulo de viz/clusters/zonas_imagenes.kml -- ver
scripts/make_zone_kml.py):

    {zona}_2000.png   (o .jpg)   -- captura en la fecha histórica más cercana a 2000
    {zona}_2022.png   (o .jpg)   -- ídem para 2022

`{zona}` es el id tal cual aparece en viz/clusters/data/index.json (puna_noa,
patagonia_estepa, periurbano_cordoba, ibera, delta_parana, pampa_nucleo,
misiones_selva) o chaco_santiago_frontier (la zona de training, sin
clustering -- ver scripts/fetch_zone_imagery.py::EXTRA_ZONE_LABELS).

Uso:
    python scripts/import_manual_imagery.py ~/Desktop/capturas_earth/
    python scripts/import_manual_imagery.py ~/Desktop/capturas_earth/ --date puna_noa_2000=1999-08-15

`--date {archivo sin extensión}=YYYY-MM-DD` es opcional, solo para que el
tooltip del visor muestre la fecha real de la imagen (Earth Pro no siempre
tiene una escena exactamente en el año pedido) -- sin esto, el manifiesto
igual queda funcional, solo sin esa fecha.

No corrige perspectiva ni recorta: asume que en Earth Pro volaste al
rectángulo del KML con inclinación 0 (top-down) antes de exportar, así el
encuadre coincide razonablemente con el bbox de la zona. Si el encuadre no da
justo, se nota como un corrimiento del overlay respecto de los puntos de
cluster -- ajustable a ojo desde Earth Pro repitiendo la captura, no hay
homografía ni calibración de esquinas en esta primera versión.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_MANIFEST = ROOT / "viz" / "clusters" / "data" / "index.json"
OUT_DIR = ROOT / "viz" / "clusters" / "data" / "imagery"
LATLON_DIR = ROOT / "data"

EXTRA_ZONE_LABELS = {
    "chaco_santiago_frontier": "Chaco-Santiago (frontera, train)",
}

FNAME_RE = re.compile(r"^(?P<zone>.+)_(?P<year>2000|2022)\.(png|jpg|jpeg)$", re.IGNORECASE)


def bbox_from_latlon_zip(zone_id):
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


def zone_bounds_and_label(zone_id, cluster_zones):
    if zone_id in cluster_zones:
        return cluster_zones[zone_id]["bounds"], cluster_zones[zone_id]["label"]
    bounds = bbox_from_latlon_zip(zone_id)
    if bounds is None:
        return None, None
    return bounds, EXTRA_ZONE_LABELS.get(zone_id, zone_id)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_dir", type=Path, help="carpeta con los {zona}_{2000|2022}.png exportados")
    p.add_argument("--date", action="append", default=[],
                    help="'{zona}_{año}=YYYY-MM-DD', repetible; fecha real de esa captura")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    dates = dict(d.split("=", 1) for d in args.date)

    cluster_manifest = json.loads(CLUSTER_MANIFEST.read_text())
    cluster_zones = {z["id"]: z for z in cluster_manifest["zones"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    found = list(args.in_dir.iterdir())
    matched = 0
    for path in sorted(found):
        m = FNAME_RE.match(path.name)
        if not m:
            print(f"  (salteado, no matchea la convención de nombres) {path.name}")
            continue
        zone_id, year = m.group("zone"), m.group("year")
        bounds, label = zone_bounds_and_label(zone_id, cluster_zones)
        if bounds is None:
            print(f"  !! zona desconocida '{zone_id}' (ni en index.json ni data/lat_long_df_{zone_id}.zip) -- {path.name}")
            continue

        fname = f"{zone_id}_{year}.png"
        img = Image.open(path).convert("RGB")
        img.save(args.out_dir / fname)

        zone_entry = manifest.setdefault(zone_id, {"bounds": bounds, "label": label})
        zone_entry[year] = {
            "source": "google_earth_pro",
            "date": dates.get(f"{zone_id}_{year}", None),
            "file": f"imagery/{fname}",
        }
        matched += 1
        print(f"  {path.name} -> {fname}"
              + (f" ({dates[f'{zone_id}_{year}']})" if f"{zone_id}_{year}" in dates else ""))

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n{matched} imagen(es) importada(s). manifiesto: {manifest_path}")


if __name__ == "__main__":
    main()
