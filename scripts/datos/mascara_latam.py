"""Máscara de América Latina y el Caribe para el censo de trayectorias (autoencoder_v3).

Fuente: Natural Earth admin-0 *map units* 1:50m, versión fija (v5.1.2). Se usan las map units y no los países
porque los países ponen la Guayana Francesa, Guadalupe y Martinica dentro de Francia. Se toman las unidades con
REGION_WB == "Latin America & Caribbean" (51, incluidas las Malvinas).

La máscara se dilata --buffer-px píxeles de 300 m, como hace censo_mundial.py con los continentes (sin dilatar se
pierde la tierra costera que el polígono generaliza), y después se le resta el resto de las unidades SIN dilatar,
para que la dilatación no se meta en Estados Unidos ni en ningún otro vecino. Por eso el censo se corre con
--buffer-px 0:

    python scripts/datos/mascara_latam.py
    python scripts/datos/censo_mundial.py --completo --workers 6 --mascara data/geo/latam.geojson --buffer-px 0 \\
        --out-dir data/autoencoder_v3/latam --nombre universo_latam
"""
import argparse
import urllib.request
import warnings
from pathlib import Path
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402

import geopandas as gpd  # noqa: E402

VERSION = "v5.1.2"
URL = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{VERSION}/geojson/ne_50m_admin_0_map_units.geojson"
REGION = "Latin America & Caribbean"
DEG = 1 / 360   # tamaño del píxel ESA CCI en grados


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffer-px", type=int, default=16, help="dilatación hacia el mar, en píxeles de 300 m")
    ap.add_argument("--out", type=Path, default=P.GEO / "latam.geojson")
    a = ap.parse_args()

    fuente = P.GEO / f"ne_50m_admin_0_map_units_{VERSION}.geojson"
    if not fuente.exists():
        print("descargando", URL)
        urllib.request.urlretrieve(URL, fuente)
    g = gpd.read_file(fuente)
    al = g.REGION_WB == REGION
    print(f"{al.sum()} unidades de {REGION}:", ", ".join(sorted(g.NAME[al])))

    with warnings.catch_warnings():   # operar en grados es intencional (el píxel mide 1/360°)
        warnings.simplefilter("ignore", UserWarning)
        dentro = g[al].geometry.union_all().buffer(a.buffer_px * DEG)
        fuera = g[~al].geometry.union_all()
        m = dentro.difference(fuera)
    out = gpd.GeoDataFrame({"nombre": ["America Latina y el Caribe"], "fuente": [f"Natural Earth {VERSION} map units"],
                            "buffer_px": [a.buffer_px]}, geometry=[m], crs=g.crs)
    out.to_file(a.out, driver="GeoJSON")
    print(f"-> {P.rel(a.out)}")


if __name__ == "__main__":
    main()
