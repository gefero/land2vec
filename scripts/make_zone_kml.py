"""Genera un KML con el rectángulo de cada zona (7 de evaluación OOD +
`chaco_santiago_frontier`, la base de entrenamiento) para volar al mismo
encuadre en Google Earth Pro al capturar las imágenes de inicio/fin de
período a mano (ver viz/clusters/README.md, sección "Imagen satelital -- a
mano con Google Earth Pro").

Los bounds de las 7 OOD salen de viz/clusters/data/index.json (los mismos que
usa el visor); el de chaco_santiago_frontier se calcula de su
data/lat_long_df_chaco_santiago_frontier.zip, igual que en
scripts/fetch_zone_imagery.py::bbox_from_latlon_zip.

Uso:
    python scripts/make_zone_kml.py
    -> viz/clusters/zonas_imagenes.kml
"""

import csv
import io
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_MANIFEST = ROOT / "viz" / "clusters" / "data" / "index.json"
LATLON_DIR = ROOT / "data"
OUT_PATH = ROOT / "viz" / "clusters" / "zonas_imagenes.kml"

# zonas sin clustering (training) para las que igual queremos encuadre --
# mismo criterio y misma etiqueta que scripts/fetch_zone_imagery.py.
EXTRA_ZONE_LABELS = {
    "chaco_santiago_frontier": "Chaco-Santiago (frontera, train)",
}


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


def zone_placemark(zone_id, label, bounds):
    (south, west), (north, east) = bounds
    coords = f"{west},{south},0 {east},{south},0 {east},{north},0 {west},{north},0 {west},{south},0"
    lon_c, lat_c = (west + east) / 2, (south + north) / 2
    name = escape(f"{label} ({zone_id})")
    return f"""    <Placemark>
      <name>{name}</name>
      <styleUrl>#zoneStyle</styleUrl>
      <LookAt>
        <longitude>{lon_c}</longitude>
        <latitude>{lat_c}</latitude>
        <tilt>0</tilt>
        <heading>0</heading>
        <range>{max(north - south, east - west) * 111000 * 1.3:.0f}</range>
      </LookAt>
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs>
          <LinearRing><coordinates>{coords}</coordinates></LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""


def main():
    manifest = json.loads(CLUSTER_MANIFEST.read_text())
    zones = [(z["id"], z["label"], z["bounds"]) for z in manifest["zones"]]
    for zone_id in EXTRA_ZONE_LABELS:
        bounds = bbox_from_latlon_zip(zone_id)
        if bounds is None:
            print(f"aviso: no encontré data/lat_long_df_{zone_id}.zip, la salteo")
            continue
        zones.append((zone_id, EXTRA_ZONE_LABELS[zone_id], bounds))

    placemarks = "\n".join(zone_placemark(*z) for z in zones)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>land2vec -- zonas para imagen satelital</name>
    <Style id="zoneStyle">
      <LineStyle><color>ff00ffff</color><width>3</width></LineStyle>
      <PolyStyle><fill>0</fill></PolyStyle>
    </Style>
{placemarks}
  </Document>
</kml>
"""
    OUT_PATH.write_text(kml)
    print(f"{OUT_PATH} ({len(zones)} zonas)")


if __name__ == "__main__":
    main()
