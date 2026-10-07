"""Rutas del repo en un solo lugar.

Layout de `data/`:
    data/landcover_timeseries_2000-2022.nc   crudo (ESA CCI, LFS)
    data/geo/                                crudo (provincias, Sudamérica, polígonos de desmonte)
    data/zonas/                              id_seqs_text_2000_2022_<zona>.zip, lat_long_df_<zona>.zip
    data/zonas/full/                         idem sin submuestrear constantes (no versionado)
    data/desmonte/                           desmonte_px_<zona>.zip, desmonte_px_epoca_<zona>.zip, desmonte_poly_px_<zona>.zip
    data/v1/                                 seqs_short.csv, test_sample_*.zip (modelo autorregresivo original)
    data/v2/                                 embeddings_<zona>.zip, clusters_<set><sufijo>.zip
    data/autoencoder_v3/                     productos del ejercicio v3; landcover_timeseries_1992-2022_rebuild.nc es la serie de v3 (la de 2000-2022 quedó en el tag v3-2000-2022)
    data/ESA_data/raw_unzipped/              mapas anuales crudos del CDS (no versionado)

Las funciones de archivo aceptan `data`, la raíz de datos (default `DATA`), para que
los `--data-dir` de los scripts sigan funcionando con otra raíz.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
NC_FILE = DATA / "landcover_timeseries_2000-2022.nc"                               # v1/v2: original (LFS)
V3_YEARS = (1992, 2022)                                                            # v3 rehecho: período de la serie (v1/v2 usan 2000-2022)
NC_V3 = DATA / "autoencoder_v3" / f"landcover_timeseries_{V3_YEARS[0]}-{V3_YEARS[1]}_rebuild.nc"   # v3: reconstruido desde los crudos (versionado por LFS)
GEO = DATA / "geo"
DESMONTE_RAR = GEO / "data_validacion_chaco_Coleccion_13.0.rar"
ESA_RAW = DATA / "ESA_data" / "raw_unzipped"             # mapas anuales globales del CDS (no versionado)
ANUAL = DATA / "autoencoder_v3" / "anual"                # lccs_<año>.nc agrupados y recortados (no versionado)

MODELS = ROOT / "models"
MODELS_V1 = MODELS / "v1"
MODELS_V2 = MODELS / "v2"
MODELS_V3 = MODELS / "autoencoder_v3"
MODEL_V1 = MODELS_V1 / "full_model"
MODEL_V2 = MODELS_V2 / "autoencoder_v2"
CLUSTER_V2 = MODELS_V2 / "cluster_v2"

VIZ = ROOT / "viz"
IMGS = ROOT / "imgs"
DOCS = ROOT / "docs"


def rel(path: Path | str) -> str:
    "Ruta relativa al repo para mensajes; absoluta si cae fuera (p. ej. un --out-dir en /tmp)."
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --- zonas (insumo compartido por todas las versiones) ---
def zonas_dir(data: Path = DATA, full: bool = False) -> Path:
    d = Path(data) / "zonas"
    return d / "full" if full else d


def seqs_file(zone: str, data: Path = DATA, full: bool = False) -> Path:
    return zonas_dir(data, full) / f"id_seqs_text_2000_2022_{zone}.zip"


def latlon_file(zone: str, data: Path = DATA, full: bool = False) -> Path:
    return zonas_dir(data, full) / f"lat_long_df_{zone}.zip"


# --- referencia de desmonte ---
def desmonte_dir(data: Path = DATA) -> Path:
    return Path(data) / "desmonte"


def desmonte_px_file(zone: str, data: Path = DATA) -> Path:
    return desmonte_dir(data) / f"desmonte_px_{zone}.zip"


def desmonte_epoca_file(zone: str, data: Path = DATA) -> Path:
    return desmonte_dir(data) / f"desmonte_px_epoca_{zone}.zip"


def desmonte_poly_file(zone: str, data: Path = DATA) -> Path:
    return desmonte_dir(data) / f"desmonte_poly_px_{zone}.zip"


# --- productos de v2 (autoencoder_v2 + cluster_v2) ---
def v2_dir(data: Path = DATA) -> Path:
    return Path(data) / "v2"


def embeddings_file(zone: str, data: Path = DATA) -> Path:
    return v2_dir(data) / f"embeddings_{zone}.zip"


def clusters_file(set_name: str, suffix: str = "", data: Path = DATA, tag: str = "") -> Path:
    "data/v2/clusters_<set><sufijo><tag>.zip (p. ej. clusters_train_pooled_coarse_full.zip)."
    return v2_dir(data) / f"clusters_{set_name}{suffix}{tag}.zip"
