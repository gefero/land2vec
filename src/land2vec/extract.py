from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Mapeo de códigos numéricos del raster lccs_class a los tokens del vocabulario.
# Debe mantenerse en sync con land2vec.tokenizer.Tokenizer.VOCAB (salvo "[UNK]").
LCCS_CODE_TO_TOKEN: dict[int, str] = {
    0: "Nd",
    1: "A",
    2: "F",
    3: "G",
    4: "Wt",
    5: "U",
    6: "Sh",
    7: "Sp",
    8: "B",
    9: "Wa",
}


def load_landcover_dataset(path: Path | str = Path("data") / "landcover_timeseries_2000-2022.nc") -> xr.Dataset:
    "Abre el netCDF de land cover (ESA CCI, 300m, 2000-2022)."
    return xr.open_dataset(path)


def crop_bbox(ds: xr.Dataset, minx: float, miny: float, maxx: float, maxy: float) -> xr.Dataset:
    "Recorta el dataset a un bounding box en lon/lat."
    return ds.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))


def build_lat_long_df(ds: xr.Dataset) -> pd.DataFrame:
    "Arma un DataFrame ID/latitude/longitude, uno por píxel de la grilla de ds."
    lon_grid, lat_grid = np.meshgrid(ds["lon"].values, ds["lat"].values)
    return pd.DataFrame({
        "ID": np.arange(lat_grid.size),
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel(),
    })


def build_sequence_df(
    ds: xr.Dataset,
    code_mapping: dict[int, str] = LCCS_CODE_TO_TOKEN,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    "Arma un DataFrame ID/seqs con la secuencia anual de estados (2000-2022) por píxel de ds."
    years = ds["time"].dt.year.values
    n_pixels = ds.sizes["lat"] * ds.sizes["lon"]
    # (n_pixels, n_years), mismo orden de píxel (lon variando más rápido) que build_lat_long_df
    lccs = ds["lccs_class"].values.reshape(len(years), n_pixels).T

    max_code = max(code_mapping)
    lookup = np.full(max_code + 1, "Nd", dtype=object)
    for code, token in code_mapping.items():
        lookup[code] = token

    seqs: list[str] = []
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        # nan_to_num cubre píxeles sin dato (p. ej. bordes de costa) -> quedan como "Nd"
        codes = np.nan_to_num(lccs[start:end], nan=0).astype(int)
        tokens = lookup[np.clip(codes, 0, max_code)]
        seqs.extend("-".join(row) for row in tokens)

    return pd.DataFrame({"ID": np.arange(n_pixels), "seqs": seqs})


def extract_zone(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    code_mapping: dict[int, str] = LCCS_CODE_TO_TOKEN,
    chunk_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    "Recorta ds al bbox (minx, miny, maxx, maxy) y devuelve (lat_long_df, seqs_df)."
    minx, miny, maxx, maxy = bbox
    ds_cropped = crop_bbox(ds, minx, miny, maxx, maxy)
    lat_long_df = build_lat_long_df(ds_cropped)
    seqs_df = build_sequence_df(ds_cropped, code_mapping=code_mapping, chunk_size=chunk_size)
    return lat_long_df, seqs_df


def _constant_mask(seqs_df: pd.DataFrame, seq_col: str) -> pd.Series:
    return seqs_df[seq_col].str.split("-").apply(lambda states: len(set(states)) == 1)


def drop_constant_sequences(seqs_df: pd.DataFrame, seq_col: str = "seqs") -> pd.DataFrame:
    "Descarta píxeles cuya secuencia es constante en todo el período (p. ej. agua permanente)."
    return seqs_df[~_constant_mask(seqs_df, seq_col)]


def subsample_constant_sequences(
    seqs_df: pd.DataFrame,
    max_fraction: float = 0.15,
    seq_col: str = "seqs",
    seed: int = 42,
) -> pd.DataFrame:
    """Como drop_constant_sequences, pero submuestrea en vez de descartar del todo:
    una trayectoria estable (bosque o agua que no cambia en 23 años) es un tipo de
    dinámica legítimo y debe seguir presente en el dataset, solo que sin dominarlo
    -- útil para entrenar el autoencoder, donde la inmensa mayoría de los píxeles
    de cualquier zona es constante (ver notebooks/eval_ood_zones.ipynb) y sin
    balancear el modelo aprende poco más que reconstruir "23 años de lo mismo".

    Devuelve como mucho max_fraction del dataset resultante como secuencias
    constantes; el resto (secuencias con al menos una transición) queda intacto.
    """
    is_constant = _constant_mask(seqs_df, seq_col)
    varying_df, constant_df = seqs_df[~is_constant], seqs_df[is_constant]

    n_varying = len(varying_df)
    if n_varying == 0 or max_fraction >= 1:
        return seqs_df  # nada para balancear, o no se pidió balancear

    max_constant = int(max_fraction * n_varying / (1 - max_fraction))
    if len(constant_df) > max_constant:
        constant_df = constant_df.sample(n=max_constant, random_state=seed)

    return pd.concat([varying_df, constant_df]).sort_index()


def save_zone_csvs(
    lat_long_df: pd.DataFrame,
    seqs_df: pd.DataFrame,
    output_dir: Path,
    zone_name: str,
) -> tuple[Path, Path]:
    "Guarda (lat_long_df, seqs_df) con el mismo naming/formato que los archivos existentes en data/."
    seqs_path = output_dir / f"id_seqs_text_2000_2022_{zone_name}.zip"
    lat_long_path = output_dir / f"lat_long_df_{zone_name}.zip"
    seqs_df.to_csv(seqs_path, index=False, compression="zip")
    lat_long_df.to_csv(lat_long_path, index=False, compression="zip")
    return seqs_path, lat_long_path
