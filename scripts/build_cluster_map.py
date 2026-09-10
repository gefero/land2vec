"""Arma el payload del mapa interactivo de clusterizaciones (viz/clusters/).

Es el **productor**: junta las etiquetas de cluster de la matriz 3 granularidades
x 2 familias (`scripts/tune_clustering.py --select`, ver
`docs/v2_autoencoder_training.md` §7.2) con las coordenadas por parcela y escribe
JSONs compactos que `viz/clusters/index.html` consume tal cual (sin recalcular).

Dos "sets" por corrida, ver docs §7.2 (Nota metodológica):

- `dynamic`            -- las ~107k secuencias con transición sobre las que se
                         ajustó el clustering ("in-sample").
- `pooled_subsampled`  -- ~126k parcelas (incluye constantes submuestreadas al
                         15%), asignadas por centroide más cercano con un corte
                         "sin tipificar" (cluster = -1) ("aplicado").

Solo usa la librería estándar (csv/zipfile/json): las entradas ya son CSV dentro
de zips, no hace falta pandas/numpy. Corre con cualquier `python3`.

Uso:
    python scripts/build_cluster_map.py                 # las 6 corridas x 2 sets
    python scripts/build_cluster_map.py --only _medium  # una granularidad/familia
    python scripts/build_cluster_map.py --precision 4   # menos decimales, menos peso

Prerrequisitos (los deja `tune_clustering.py --select`):
    data/clusters_{dynamic,pooled_subsampled}{,_medium,_coarse}{,_parametric}.zip
    data/lat_long_df_{zona}.zip           (una por zona OOD)

Opcional, para etiquetas legibles en la leyenda (lo deja
`scripts/describe_clusters.py`, gitignoreado):
    viz/typology/typology_browser.json

Escribe (gitignoreado -- contiene coordenadas por parcela, ver
viz/clusters/README.md):
    viz/clusters/data/index.json           manifiesto (corridas, zonas, paleta)
    viz/clusters/data/{set}{suffix}.json   puntos por zona y por cluster
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "viz" / "clusters" / "data"
TYPOLOGY_JSON = ROOT / "viz" / "typology" / "typology_browser.json"

# Las 7 zonas de evaluación out-of-domain (land2vec.cluster.ZONES). El clustering
# se ajusta sobre las 7 juntas; acá solo las separamos para el mapa.
ZONES = [
    "puna_noa",
    "patagonia_estepa",
    "periurbano_cordoba",
    "ibera",
    "delta_parana",
    "pampa_nucleo",
    "misiones_selva",
]

ZONE_LABELS = {
    "puna_noa": "Puna (Jujuy)",
    "patagonia_estepa": "Estepa patagónica",
    "periurbano_cordoba": "Periurbano (Córdoba)",
    "ibera": "Iberá",
    "delta_parana": "Delta del Paraná",
    "pampa_nucleo": "Pampa núcleo",
    "misiones_selva": "Selva misionera",
}

# suffix -> nombre visible (mismo criterio que scripts/tune_clustering.py y
# viz/typology). "" = granularidad fina / familia HDBSCAN.
SUFFIXES = {
    "": "fina / HDBSCAN",
    "_parametric": "fina / paramétrico",
    "_medium": "media / HDBSCAN",
    "_medium_parametric": "media / paramétrico",
    "_coarse": "gruesa / HDBSCAN",
    "_coarse_parametric": "gruesa / paramétrico",
}

SETS = ["dynamic", "pooled_subsampled"]
SET_LABELS = {
    "dynamic": "dinámico (ajuste · in-sample)",
    "pooled_subsampled": "pool submuestreado (aplicado)",
}

# Paleta cualitativa (Tableau-20 + Set3 recortada) cicleada por id de cluster.
# El -1 ("sin tipificar") se pinta aparte, gris claro, en el visor.
PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#a6cee3", "#fb9a99", "#fdbf6f", "#cab2d6", "#b15928",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
    "#ffd92f", "#e5c494", "#b3b3b3", "#8dd3c7", "#bebada",
    "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
]


def read_zip_csv(path: Path) -> csv.DictReader:
    """Abre el único miembro de un .zip como csv.DictReader (todo en memoria:
    los archivos son chicos, < 3 MB descomprimidos)."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8")
    return csv.DictReader(io.StringIO(raw))


def load_zone_coords(zone: str, data_dir: Path) -> dict[str, tuple[float, float]]:
    """ID (str) -> (lat, lon) para una zona. El ID es el índice posicional por
    zona (land2vec.cluster.load_zone_coords)."""
    path = data_dir / f"lat_long_df_{zone}.zip"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}")
    out: dict[str, tuple[float, float]] = {}
    for row in read_zip_csv(path):
        out[row["ID"]] = (float(row["latitude"]), float(row["longitude"]))
    return out


def load_typology_labels() -> dict[str, dict[int, dict]]:
    """suffix -> {cluster_id -> {etiqueta, glosa, forma, anio_cambio}}.
    Vacío si no está viz/typology/typology_browser.json."""
    if not TYPOLOGY_JSON.exists():
        return {}
    payload = json.loads(TYPOLOGY_JSON.read_text())
    by_suffix: dict[str, dict[int, dict]] = {}
    for run in payload.get("runs", []):
        suffix = run.get("suffix", "")
        by_suffix[suffix] = {
            int(c["cluster"]): {
                "etiqueta": c.get("etiqueta"),
                "glosa": c.get("glosa"),
                "forma": c.get("forma"),
                "anio_cambio": c.get("anio_cambio"),
            }
            for c in run.get("clusters", [])
        }
    return by_suffix


def build_set(
    set_name: str,
    suffix: str,
    coords_cache: dict[str, dict[str, tuple[float, float]]],
    data_dir: Path,
    precision: int,
) -> dict | None:
    path = data_dir / f"clusters_{set_name}{suffix}.zip"
    if not path.exists():
        print(f"  {path.name}: no existe, se omite")
        return None

    zones: dict[str, dict] = {}
    clusters_present: set[int] = set()
    n_points = 0
    n_missing = 0

    for row in read_zip_csv(path):
        zone = row["zone"]
        cid = int(row["cluster"])
        if zone not in coords_cache:
            coords_cache[zone] = load_zone_coords(zone, data_dir)
        coord = coords_cache[zone].get(row["ID"])
        if coord is None:
            n_missing += 1
            continue
        lat, lon = round(coord[0], precision), round(coord[1], precision)

        z = zones.setdefault(zone, {"points": {}, "min_lat": lat, "max_lat": lat,
                                    "min_lon": lon, "max_lon": lon})
        z["points"].setdefault(str(cid), []).extend((lat, lon))
        z["min_lat"] = min(z["min_lat"], lat)
        z["max_lat"] = max(z["max_lat"], lat)
        z["min_lon"] = min(z["min_lon"], lon)
        z["max_lon"] = max(z["max_lon"], lon)
        clusters_present.add(cid)
        n_points += 1

    for zone, z in zones.items():
        z["bounds"] = [[z.pop("min_lat"), z.pop("min_lon")],
                       [z.pop("max_lat"), z.pop("max_lon")]]
        # cuentas por cluster, para la leyenda ordenada por tamaño
        z["counts"] = {cid: len(pts) // 2 for cid, pts in z["points"].items()}

    if n_missing:
        print(f"  aviso: {n_missing:,} filas sin coordenada (ID no encontrado)")

    return {
        "set": set_name,
        "suffix": suffix,
        "name": f"{SUFFIXES[suffix]} · {SET_LABELS[set_name]}",
        "n_points": n_points,
        "clusters_present": sorted(clusters_present),
        "zones": zones,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--precision", type=int, default=5,
                        help="decimales de lat/lon (5 ~1 m, 4 ~11 m; menos = archivo más liviano)")
    parser.add_argument("--only", choices=list(SUFFIXES), default=None,
                        help="procesar solo esta granularidad/familia")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffixes = [args.only] if args.only else list(SUFFIXES)
    typ_labels = load_typology_labels()
    if not typ_labels:
        print(f"nota: {TYPOLOGY_JSON.relative_to(ROOT)} no está; leyenda sin etiquetas automáticas")

    coords_cache: dict[str, dict[str, tuple[float, float]]] = {}
    runs: list[dict] = []
    zone_geo: dict[str, dict] = {}

    for suffix in suffixes:
        print(f"[{SUFFIXES[suffix]}]")
        run_entry: dict = {
            "suffix": suffix,
            "name": SUFFIXES[suffix],
            "labels": {str(k): v for k, v in typ_labels.get(suffix, {}).items()},
            "sets": {},
        }
        for set_name in SETS:
            built = build_set(set_name, suffix, coords_cache, args.data_dir, args.precision)
            if built is None:
                continue
            out_path = args.out_dir / f"{set_name}{suffix}.json"
            out_path.write_text(json.dumps(built, separators=(",", ":")))
            size_mb = out_path.stat().st_size / 1e6
            print(f"  {out_path.name}: {built['n_points']:,} puntos, "
                  f"{len(built['clusters_present'])} clusters, {size_mb:.1f} MB")
            run_entry["sets"][set_name] = {
                "file": out_path.name,
                "n_points": built["n_points"],
                "n_clusters": len(built["clusters_present"]),
                "clusters_present": built["clusters_present"],
            }
            for zone, z in built["zones"].items():
                zone_geo.setdefault(zone, z["bounds"])
        if run_entry["sets"]:
            runs.append(run_entry)

    if not runs:
        sys.exit("no se generó ninguna corrida -- ¿faltan los data/clusters_*.zip?")

    index = {
        "generated_from": "scripts/build_cluster_map.py",
        "zones": [
            {
                "id": z,
                "label": ZONE_LABELS.get(z, z),
                "bounds": zone_geo.get(z),
            }
            for z in ZONES if z in zone_geo
        ],
        "set_labels": SET_LABELS,
        "palette": PALETTE,
        "runs": runs,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"\nmanifiesto: {(args.out_dir / 'index.json').relative_to(ROOT)} "
          f"({len(runs)} corridas, {len(index['zones'])} zonas)")
    print(f"levantá el visor con:  python -m http.server -d {args.out_dir.parent.relative_to(ROOT)} 8001")


if __name__ == "__main__":
    main()
