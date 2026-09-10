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
    python scripts/build_cluster_map.py --no-constants  # sin el raster de fondo

Prerrequisitos (los deja `tune_clustering.py --select`):
    data/clusters_{dynamic,pooled_subsampled}{,_medium,_coarse}{,_parametric}.zip
    data/lat_long_df_{zona}.zip           (una por zona OOD)
    data/id_seqs_text_2000_2022_{zona}.zip  (para el raster de fondo de constantes)

Opcional, para etiquetas legibles en la leyenda (lo deja
`scripts/describe_clusters.py`, gitignoreado):
    viz/typology/typology_browser.json

Escribe (gitignoreado -- contiene coordenadas por parcela, ver
viz/clusters/README.md):
    viz/clusters/data/index.json           manifiesto (corridas, zonas, paletas)
    viz/clusters/data/{set}{suffix}.json   puntos (lat,lon,seqIdx) por zona y
                                           cluster + `seqs` (trayectorias crudas
                                           deduplicadas, para el popup por píxel)
    viz/clusters/data/constants_{zona}.png  raster de fondo: píxeles cuya
                                            trayectoria 2000-2022 no cambia,
                                            coloreados por su único estado
"""

import argparse
import csv
import io
import json
import math
import struct
import sys
import zlib
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
# El -1 ("sin tipificar") se pinta aparte, gris claro, en el visor. Es el modo de
# color "cluster"; el modo por defecto del visor es "proceso" (ver abajo).
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

# Copia de land2vec.typology.STATE_COLORS (paleta única de estados de uso del
# suelo). Se emite en index.json para un futuro modo "estado final" del visor y
# para la capa de trayectorias constantes (pendiente 3).
STATE_COLORS = {
    "[UNK]": "#d9d9d9", "A": "#e6c229", "F": "#1b7837", "G": "#a6d96a",
    "Wt": "#2b8cbe", "U": "#d7301f", "Sh": "#b8860b", "Sp": "#dfc27d",
    "B": "#8c6d31", "Wa": "#08519c", "Nd": "#bdbdbd",
}

# --------------------------------------------------------------------------- #
#  Color por proceso (modo por defecto del visor)
# --------------------------------------------------------------------------- #
# Cada cluster se agrupa en un "proceso" conceptual y se pinta con el hue de ese
# proceso (anclado a la semántica: rojo = pérdida de bosque ... verde =
# regeneración ... azul = agua), variando luminosidad y croma según la antigüedad
# del cambio (cambio viejo -> más oscuro y saturado, reciente -> claro y pálido).
#
# El color se genera en OKLCh (espacio perceptualmente ~uniforme) para que la
# rampa temporal de cada proceso tenga pasos de ΔE parejos -- se valida con
# scripts/check_cluster_palette.py contra la métrica de
# https://color-analyzer.streamlit.app/ (objetivo: cerca de viridis, lejos de Jet).
#
# La clasificación (classify_process) es determinista y se deriva de la secuencia
# modal inicio»fin del cluster (typology_browser.json). Las glosas agronómicas de
# cada par (inicio, fin) viven en src/land2vec/typology.py (GLOSSES).

PROCESSES = {
    "deforestacion": {
        "label": "Deforestación — pérdida de bosque",
        "gloss": "la trayectoria arranca en bosque (F→agricultura/pastizal); el impacto que la define es el bosque perdido, no en qué termina",
        "h": 29},
    "degradacion_forestal": {
        "label": "Degradación forestal (F→arbustal/esparso)",
        "gloss": "el bosque no desaparece del todo pero se abre: pasa a arbustal, cobertura esparsa o suelo desnudo",
        "h": 52},
    "expansion_agricola": {
        "label": "Expansión agrícola (sobre pastizal/estepa)",
        "gloss": "termina en agricultura y NO venía de bosque (pastizal/arbustal/esparso/suelo→A): avance de la frontera agrícola sin pérdida forestal",
        "h": 74},
    "perdida_vegetacion": {
        "label": "Pérdida de vegetación / aridización",
        "gloss": "pastizal o arbustal que pasa a suelo desnudo o cobertura esparsa (desertificación, sobrepastoreo)",
        "h": 106},
    "revegetacion": {
        "label": "Revegetación de suelo árido",
        "gloss": "suelo desnudo o esparso que gana cobertura (pasa a esparso, pastizal o arbustal)",
        "h": 130},
    "regeneracion_bosque": {
        "label": "Regeneración de bosque (→F)",
        "gloss": "cualquier cobertura no forestal que termina en bosque: rebrote, forestación, avance del monte",
        "h": 152},
    "dinamica_hidrica": {
        "label": "Dinámica de agua / humedal",
        "gloss": "agua o humedal en el inicio o el fin: anegamiento, desecación, avance/retroceso de cuerpos de agua",
        "h": 240},
    "urbanizacion": {
        "label": "Urbanización (→U)",
        "gloss": "cualquier cobertura que termina en suelo urbano",
        "h": 330},
    "oscilante": {
        "label": "Oscilante / múltiple",
        "gloss": "la trayectoria vuelve a su estado inicial o pasa por varios estados sin una dirección clara",
        "h": 300, "c_scale": 0.5},
    "otro": {
        "label": "Otro",
        "gloss": "transiciones que no caen en ninguna de las categorías anteriores",
        "h": 0, "c_scale": 0.0},
}

_RAMP_L = (0.80, 0.50)   # OKLab L: cambio reciente -> cambio viejo
_RAMP_C = (0.06, 0.14)   # croma OKLCh: reciente -> viejo


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _oklab_to_srgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    "OKLab -> sRGB (0..1), recortando al gamut. https://bottosson.github.io/posts/oklab/"
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b2 = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    return tuple(min(1.0, max(0.0, _linear_to_srgb(x))) for x in (r, g, b2))


def _srgb_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    "sRGB (0..1) -> OKLab."
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def _oklch_to_hex(L: float, C: float, h_deg: float) -> str:
    a = C * math.cos(math.radians(h_deg))
    b = C * math.sin(math.radians(h_deg))
    r, g, bb = _oklab_to_srgb(L, a, b)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(bb * 255))


def process_color(key: str, t_age: float | None) -> str:
    """Color de un cluster del proceso `key`. `t_age` in [0, 1]: 0 = cambio más
    reciente (claro, poco saturado), 1 = más viejo (oscuro, saturado). `None`
    (sin año de cambio) cae en el medio de la rampa."""
    t = 0.5 if t_age is None else max(0.0, min(1.0, t_age))
    p = PROCESSES[key]
    L = _RAMP_L[0] + (_RAMP_L[1] - _RAMP_L[0]) * t
    C = (_RAMP_C[0] + (_RAMP_C[1] - _RAMP_C[0]) * t) * p.get("c_scale", 1.0)
    return _oklch_to_hex(L, C, p["h"])


_DEFOR_FIN = {"A", "G"}
_DEGRAD_FIN = {"Sh", "Sp", "B"}


def classify_process(inicio: str, fin: str, forma: str | None) -> str:
    """Proceso conceptual de un cluster a partir de su secuencia modal colapsada
    (inicio»fin) y su forma. Reglas ordenadas, primera que matchea. Determinista."""
    if forma in ("oscilante", "múltiple") or inicio == fin:
        return "oscilante"
    if fin == "U":
        return "urbanizacion"
    if fin == "F" and inicio != "F":
        return "regeneracion_bosque"
    if inicio == "F" and fin in _DEFOR_FIN:
        return "deforestacion"
    if inicio == "F" and fin in _DEGRAD_FIN:
        return "degradacion_forestal"
    if {inicio, fin} & {"Wt", "Wa"}:
        return "dinamica_hidrica"
    if fin == "A":
        return "expansion_agricola"
    if inicio in ("B", "Sp") and fin in ("Sp", "G", "Sh"):
        return "revegetacion"
    if fin in ("B", "Sp"):
        return "perdida_vegetacion"
    return "otro"


def _dss(tokens: list[str]) -> list[str]:
    "Distinct Successive States: colapsa repeticiones consecutivas (land2vec.typology.dss)."
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
#  Paleta de estados constantes (fondo de trayectorias que nunca cambiaron)
# --------------------------------------------------------------------------- #
# Integrada al sistema de color de los procesos: mismo espacio OKLCh, y el hue de
# cada estado se comparte con el proceso semánticamente equivalente (bosque
# constante = verde pálido, mismo hue que "regeneración de bosque"; agua = azul,
# mismo hue que "dinámica de agua"). Muy claro y poco saturado para que el fondo
# retroceda y los puntos de cluster resalten.

VOCAB = ["[UNK]", "A", "F", "G", "Wt", "U", "Sh", "Sp", "B", "Wa", "Nd"]  # land2vec.tokenizer

STATE_LABELS = {
    "[UNK]": "sin clasificar", "A": "agricultura", "F": "forestal", "G": "pastizal",
    "Wt": "humedal", "U": "urbano", "Sh": "arbustal", "Sp": "vegetación esparsa",
    "B": "suelo desnudo", "Wa": "agua", "Nd": "sin dato",
}

# (L, C, hue OKLCh) por estado. Todos claros (fondo que retrocede), pero con
# suficiente separación de hue/L/C entre los beige áridos (A/Sp/B/Sh, dominantes
# en Puna/Patagonia) para que no se fundan. F/Wa/Wt/U/A/G comparten el hue de su
# proceso análogo (mismo sistema de color). Wa y U -- raros e importantes de
# ubicar -- salen más oscuros y saturados.
_CONST_OKLCH = {
    "F":  (0.83, 0.070, 152),   # forestal -- verde
    "G":  (0.87, 0.065, 128),   # pastizal -- verde-amarillo
    "A":  (0.87, 0.078,  80),   # agricultura -- oro
    "Sh": (0.86, 0.055, 110),   # arbustal -- oliva
    "Sp": (0.88, 0.055,  62),   # vegetación esparsa -- arena
    "B":  (0.92, 0.038,  30),   # suelo desnudo -- el más pálido, rosado
    "Wt": (0.86, 0.072, 210),   # humedal -- celeste
    "Wa": (0.77, 0.088, 245),   # agua -- azul, netamente más oscuro que el resto del fondo
    "U":  (0.82, 0.090, 330),   # urbano -- magenta
}


def _const_color(token: str) -> str:
    if token == "[UNK]":
        return _oklch_to_hex(0.90, 0.0, 0.0)
    if token == "Nd":
        return _oklch_to_hex(0.80, 0.0, 0.0)
    return _oklch_to_hex(*_CONST_OKLCH[token])


CONST_COLORS = {t: _const_color(t) for t in VOCAB}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# paleta indexada del PNG: idx 0 = vacío/transparente, idx 1..11 = VOCAB
CONST_PALETTE = [(0, 0, 0)] + [_hex_to_rgb(CONST_COLORS[t]) for t in VOCAB]
CONST_IDX = {t: i + 1 for i, t in enumerate(VOCAB)}


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _png_indexed(w: int, h: int, pixels: bytes, palette: list[tuple[int, int, int]],
                 transparent_idx: int = 0) -> bytes:
    """PNG de color indexado (color type 3, 8 bpp). `pixels` = w*h bytes de índices
    de paleta, fila por fila desde arriba. `transparent_idx` sale con alpha 0."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0)
    plte = b"".join(struct.pack(">BBB", *c) for c in palette)
    trns = bytes(0 if i == transparent_idx else 255 for i in range(len(palette)))
    raw = bytearray()
    for y in range(h):
        raw.append(0)                       # filtro 0 (None) por scanline
        raw += pixels[y * w:(y + 1) * w]
    idat = zlib.compress(bytes(raw), 9)
    return (sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"PLTE", plte)
            + _png_chunk(b"tRNS", trns) + _png_chunk(b"IDAT", idat)
            + _png_chunk(b"IEND", b""))


def build_constant_raster(zone: str, data_dir: Path) -> dict | None:
    """Raster PNG (grilla ESA CCI 300 m reconstruida) de los píxeles cuya
    trayectoria 2000-2022 no cambia -- coloreados por su único estado. Devuelve
    None si faltan las entradas."""
    seqs_path = data_dir / f"id_seqs_text_2000_2022_{zone}.zip"
    coords_path = data_dir / f"lat_long_df_{zone}.zip"
    if not seqs_path.exists() or not coords_path.exists():
        print(f"  {zone}: falta id_seqs/lat_long, se omite el fondo")
        return None

    constant: dict[str, str] = {}
    for row in read_zip_csv(seqs_path):
        toks = row["seqs"].split("-")
        if toks and all(t == toks[0] for t in toks):
            constant[row["ID"]] = toks[0]

    coords = [(r["ID"], round(float(r["latitude"]), 6), round(float(r["longitude"]), 6))
              for r in read_zip_csv(coords_path)]
    lons = sorted({lon for _, _, lon in coords})
    lats = sorted({lat for _, lat, _ in coords})
    W, H = len(lons), len(lats)
    lon_ix = {v: i for i, v in enumerate(lons)}
    lat_ix = {v: i for i, v in enumerate(lats)}

    buf = bytearray(W * H)
    placed = 0
    for cid, lat, lon in coords:
        tok = constant.get(cid)
        if tok is None:
            continue
        col, rowi = lon_ix.get(lon), lat_ix.get(lat)
        if col is None or rowi is None:
            continue
        buf[(H - 1 - rowi) * W + col] = CONST_IDX[tok]   # norte arriba
        placed += 1

    png = _png_indexed(W, H, bytes(buf), CONST_PALETTE)
    dlon = (lons[-1] - lons[0]) / (W - 1) if W > 1 else 0.0028
    dlat = (lats[-1] - lats[0]) / (H - 1) if H > 1 else 0.0028
    bounds = [[lats[0] - dlat / 2, lons[0] - dlon / 2],
              [lats[-1] + dlat / 2, lons[-1] + dlon / 2]]
    by_state: dict[str, int] = {}
    for tok in constant.values():
        by_state[tok] = by_state.get(tok, 0) + 1

    return {
        "png": png,
        "bounds": bounds,
        "by_state": by_state,
        "n_constant": len(constant),
        "placed": placed,
        "grid": [W, H],
        "rect": W * H == len(coords),
    }


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


def load_zone_seqs(zone: str, data_dir: Path) -> dict[str, str]:
    """ID (str) -> secuencia de tokens cruda 'F-F-...-A' para una zona
    (data/id_seqs_text_2000_2022_<zone>.zip). {} si falta el archivo."""
    path = data_dir / f"id_seqs_text_2000_2022_{zone}.zip"
    if not path.exists():
        return {}
    return {row["ID"]: row["seqs"] for row in read_zip_csv(path)}


def load_typology_labels() -> dict[str, dict[int, dict]]:
    """suffix -> {cluster_id -> {etiqueta, glosa, forma, anio_cambio, inicio, fin,
    proceso, color}}. Vacío si no está viz/typology/typology_browser.json.

    `inicio`/`fin` salen de colapsar la secuencia modal; `proceso` de
    classify_process(); `color` de process_color(proceso, antigüedad del cambio),
    con la antigüedad normalizada al rango de anio_cambio de esa corrida
    (cambio viejo -> color más oscuro)."""
    if not TYPOLOGY_JSON.exists():
        return {}
    payload = json.loads(TYPOLOGY_JSON.read_text())
    by_suffix: dict[str, dict[int, dict]] = {}
    for run in payload.get("runs", []):
        suffix = run.get("suffix", "")
        clusters = run.get("clusters", [])
        years = [c["anio_cambio"] for c in clusters
                 if isinstance(c.get("anio_cambio"), (int, float))]
        ymax = max(years) if years else 0
        span = (ymax - min(years)) if years else 0
        run_labels: dict[int, dict] = {}
        for c in clusters:
            d = _dss((c.get("modal_seq") or "").split("-"))
            forma = c.get("forma")
            if len(d) <= 1 or not d[0]:
                inicio = fin = ""
                proceso = "otro"
            else:
                inicio, fin = d[0], d[-1]
                proceso = classify_process(inicio, fin, forma)
            anio = c.get("anio_cambio")
            t_age = ((ymax - anio) / span) if (span and isinstance(anio, (int, float))) else None
            run_labels[int(c["cluster"])] = {
                "etiqueta": c.get("etiqueta"),
                "glosa": c.get("glosa"),
                "forma": forma,
                "anio_cambio": anio,
                "inicio": inicio,
                "fin": fin,
                "proceso": proceso,
                "color": process_color(proceso, t_age),
            }
        by_suffix[suffix] = run_labels
    return by_suffix


def build_set(
    set_name: str,
    suffix: str,
    coords_cache: dict[str, dict[str, tuple[float, float]]],
    seqs_cache: dict[str, dict[str, str]],
    data_dir: Path,
    precision: int,
) -> dict | None:
    path = data_dir / f"clusters_{set_name}{suffix}.zip"
    if not path.exists():
        print(f"  {path.name}: no existe, se omite")
        return None

    zones: dict[str, dict] = {}
    clusters_present: set[int] = set()
    seq_index: dict[str, int] = {}   # secuencia cruda -> índice en built["seqs"]
    seqs: list[str] = []
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

        if zone not in seqs_cache:
            seqs_cache[zone] = load_zone_seqs(zone, data_dir)
        seq = seqs_cache[zone].get(row["ID"], "")
        si = seq_index.get(seq)
        if si is None:
            si = seq_index[seq] = len(seqs)
            seqs.append(seq)

        z = zones.setdefault(zone, {"points": {}, "min_lat": lat, "max_lat": lat,
                                    "min_lon": lon, "max_lon": lon})
        z["points"].setdefault(str(cid), []).extend((lat, lon, si))
        z["min_lat"] = min(z["min_lat"], lat)
        z["max_lat"] = max(z["max_lat"], lat)
        z["min_lon"] = min(z["min_lon"], lon)
        z["max_lon"] = max(z["max_lon"], lon)
        clusters_present.add(cid)
        n_points += 1

    for zone, z in zones.items():
        z["bounds"] = [[z.pop("min_lat"), z.pop("min_lon")],
                       [z.pop("max_lat"), z.pop("max_lon")]]
        # cuentas por cluster (tripletes lat,lon,seqIdx), para la leyenda por tamaño
        z["counts"] = {cid: len(pts) // 3 for cid, pts in z["points"].items()}

    if n_missing:
        print(f"  aviso: {n_missing:,} filas sin coordenada (ID no encontrado)")

    return {
        "set": set_name,
        "suffix": suffix,
        "name": f"{SUFFIXES[suffix]} · {SET_LABELS[set_name]}",
        "n_points": n_points,
        "clusters_present": sorted(clusters_present),
        "seqs": seqs,
        "zones": zones,
    }


def build_all_constants(data_dir: Path, out_dir: Path) -> tuple[dict, dict]:
    "Genera los constants_<zona>.png y devuelve (constants_meta, conteos globales)."
    print("[fondo de trayectorias constantes]")
    constants: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for zone in ZONES:
        cr = build_constant_raster(zone, data_dir)
        if cr is None:
            continue
        (out_dir / f"constants_{zone}.png").write_bytes(cr["png"])
        constants[zone] = {
            "file": f"constants_{zone}.png",
            "bounds": cr["bounds"],
            "by_state": cr["by_state"],
        }
        for tok, n in cr["by_state"].items():
            counts[tok] = counts.get(tok, 0) + n
        warn = "" if cr["rect"] else "  ¡grilla no rectangular!"
        print(f"  constants_{zone}.png: {cr['grid'][0]}×{cr['grid'][1]}, "
              f"{cr['n_constant']:,} constantes, {len(cr['png']) / 1024:.0f} KB{warn}")
    return constants, counts


def _rebuild_constants_only(data_dir: Path, out_dir: Path) -> None:
    "Regenera solo el raster de fondo y actualiza esas claves en index.json."
    index_path = out_dir / "index.json"
    if not index_path.exists():
        sys.exit(f"falta {index_path} -- corré primero sin --only-constants")
    index = json.loads(index_path.read_text())
    constants, counts = build_all_constants(data_dir, out_dir)
    index["constant_colors"] = CONST_COLORS
    index["state_labels"] = STATE_LABELS
    index["constants"] = constants
    index["constant_state_counts"] = counts
    index_path.write_text(json.dumps(index, separators=(",", ":")))
    print(f"\nactualizado {index_path.relative_to(ROOT)} (solo fondo)")


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
    parser.add_argument("--no-constants", action="store_true",
                        help="no generar el raster de fondo de trayectorias constantes")
    parser.add_argument("--only-constants", action="store_true",
                        help="regenerar solo el raster de fondo (deja los JSON de puntos)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_constants:
        return _rebuild_constants_only(args.data_dir, args.out_dir)

    suffixes = [args.only] if args.only else list(SUFFIXES)
    typ_labels = load_typology_labels()
    if not typ_labels:
        print(f"nota: {TYPOLOGY_JSON.relative_to(ROOT)} no está; leyenda sin etiquetas automáticas")

    coords_cache: dict[str, dict[str, tuple[float, float]]] = {}
    seqs_cache: dict[str, dict[str, str]] = {}
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
            built = build_set(set_name, suffix, coords_cache, seqs_cache,
                              args.data_dir, args.precision)
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

    constants, const_counts = (
        ({}, {}) if args.no_constants else build_all_constants(args.data_dir, args.out_dir)
    )

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
        "processes": {
            k: {"label": v["label"], "gloss": v["gloss"], "color": process_color(k, 0.5)}
            for k, v in PROCESSES.items()
        },
        "state_colors": STATE_COLORS,
        "constant_colors": CONST_COLORS,
        "state_labels": STATE_LABELS,
        "constants": constants,
        "constant_state_counts": const_counts,
        "runs": runs,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"\nmanifiesto: {(args.out_dir / 'index.json').relative_to(ROOT)} "
          f"({len(runs)} corridas, {len(index['zones'])} zonas)")
    print(f"levantá el visor con:  python -m http.server -d {args.out_dir.parent.relative_to(ROOT)} 8001")


if __name__ == "__main__":
    main()
