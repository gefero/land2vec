"""Valida las 6 clusterizaciones de land2vec v2 contra los polígonos de desmonte
de Colección 13.0 (monitoreodesmonte.com.ar) ya cruzados con la grilla por
`scripts/build_desmonte_labels.py`. Ver `docs/paper_metodologia.md` §5.8.

Fase 3 del plan (etiqueta de referencia por píxel) y Fase 4 (métricas) en un
solo script: la etiqueta no se persiste aparte porque depende de `tau` (tres
valores de sensibilidad) y del `suffix` (a través de `cluster == -1`), así que
recalcularla por corrida es más simple que versionarla.

**Unidad de reporte: por zona, sin poolear, por default.** `chaco_santiago_frontier`
es el resultado "Chaco Seco"; `yungas` y `periurbano_cordoba` van marcadas
`fuera_ecorregion=True` -- nunca se combinan numéricamente con Chaco, solo sirven
de contraste de generalización del encoder/clustering.

Prerrequisitos:
    data/clusters_train_pooled{suffix}_full.zip   ver scripts/assign_train_clusters.py
                                                    --group all --max-constant-fraction 1.0 --out-tag _full
    data/desmonte_px_<zona>.zip                    ver scripts/build_desmonte_labels.py
    data/desmonte_poly_px_<zona>.zip               ídem
    data/lat_long_df_<zona>.zip, data/id_seqs_text_2000_2022_<zona>.zip
    viz/typology/typology_browser.json             lo deja scripts/describe_clusters.py

Escribe:
    models/cluster_v2/desmonte_eval{suffix}.csv   una fila por (zona, corrida/baseline)
    models/cluster_v2/desmonte_eval_summary.json  selección externa de granularidad + metadatos

Uso:
    python scripts/eval_desmonte.py --zone periurbano_cordoba --only _medium
    python scripts/eval_desmonte.py                     # las 6 corridas x 3 zonas
    python scripts/eval_desmonte.py --n-boot 999         # bootstrap completo (lento)
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_cluster_map as BCM  # noqa: E402 -- classify_process, load_typology_labels (solo lectura, sin efectos de import)
from land2vec import geo as G  # noqa: E402

DATA_DIR = ROOT / "data"
CLUSTER_DIR = ROOT / "models" / "cluster_v2"

SUFFIXES = ["", "_parametric", "_medium", "_medium_parametric", "_coarse", "_coarse_parametric"]
ZONES = ["chaco_santiago_frontier", "yungas", "periurbano_cordoba"]
FUERA_ECORREGION = {"yungas", "periurbano_cordoba"}  # ver Contexto del plan -- nunca se poolean con Chaco

TAU = 0.5
YEAR0 = 2000  # toks[0] de id_seqs_text_2000_2022_* es el año 2000
DEFOR_PROCESOS = {"deforestacion", "degradacion_forestal"}


def _fmt_elapsed(seconds: float) -> str:
    "1234.5 -> '20m 34.5s' (o '45.2s' si no llega al minuto)."
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s" if m else f"{s:.1f}s"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_clusters(suffix: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"clusters_train_pooled{suffix}_full.zip"
    if not path.exists():
        sys.exit(
            f"falta {path.relative_to(ROOT)} -- corré antes:\n"
            "  python scripts/assign_train_clusters.py --group all "
            "--max-constant-fraction 1.0 --out-tag _full"
        )
    return pd.read_csv(path)


def load_desmonte_px(zone: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"desmonte_px_{zone}.zip"
    if not path.exists():
        sys.exit(f"falta {path.relative_to(ROOT)} -- corré antes:\n  python scripts/build_desmonte_labels.py --zone {zone}")
    return pd.read_csv(path)


def load_desmonte_poly_px(zone: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"desmonte_poly_px_{zone}.zip"
    if not path.exists():
        sys.exit(f"falta {path.relative_to(ROOT)} -- corré antes:\n  python scripts/build_desmonte_labels.py --zone {zone}")
    return pd.read_csv(path)


def load_coords(zone: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"lat_long_df_{zone}.zip")


def load_seqs(zone: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"id_seqs_text_2000_2022_{zone}.zip")


def load_cluster_info() -> dict[str, dict[int, dict]]:
    "suffix -> {cluster_id -> {..., anio_cambio, proceso}} -- reutiliza build_cluster_map.load_typology_labels()."
    info = BCM.load_typology_labels()
    if not info:
        sys.exit(
            "falta viz/typology/typology_browser.json -- corré antes:\n"
            "  python scripts/describe_clusters.py"
        )
    return info


# ---------------------------------------------------------------------------
# Fase 3 -- etiqueta de referencia por píxel
# ---------------------------------------------------------------------------

def label_reference(px: pd.DataFrame, tau: float = TAU) -> pd.Series:
    """positivo / negativo_limpio / control_pre / control_post / excluido, según
    frac_previo/frac_ventana/frac_post/relevado -- ver docs/paper_metodologia.md §5.8, tabla de
    Fase 3. Los positivos/controles no dependen de `relevado` porque un píxel tocado por un
    polígono cae por construcción dentro del área relevada (la máscara es la envolvente de los
    propios polígonos)."""
    fp, fv, fpost = px["frac_previo"], px["frac_ventana"], px["frac_post"]
    rel = px["relevado"].astype(bool)

    ref = pd.Series("excluido", index=px.index, dtype=object)
    negativo_limpio = (fp == 0) & (fv == 0) & (fpost == 0) & rel
    control_pre = (fp >= tau) & (fv == 0)
    control_post = (fpost >= tau) & (fp == 0) & (fv == 0)
    positivo = (fv >= tau) & (fp < 0.1)

    ref[negativo_limpio] = "negativo_limpio"
    ref[control_pre] = "control_pre"
    ref[control_post] = "control_post"
    ref[positivo] = "positivo"  # va al final: si por algún borde numérico coincide con un control, gana positivo
    return ref


def reference_year(poly_px: pd.DataFrame, min_frac_poly: float = 0.25) -> pd.Series:
    """Año de referencia por píxel: mínimo `anio` (FECHA_DESM) entre 2001-2022 de los polígonos
    que cubren >= `min_frac_poly` del píxel. Solo tiene valor para píxeles con al menos un
    polígono de ventana así de dominante -- típicamente coincide con los `positivo`."""
    sub = poly_px[(poly_px["anio"] >= 2001) & (poly_px["anio"] <= 2022) & (poly_px["frac"] >= min_frac_poly)]
    if sub.empty:
        return pd.Series(dtype=np.float64)
    return sub.groupby("ID")["anio"].min()


# ---------------------------------------------------------------------------
# Líneas de base sobre la secuencia cruda (R0, R1, â_i)
# ---------------------------------------------------------------------------

def _r0_year(toks: list[str]) -> float:
    "Existe t con x_t=F y t'>t con x_t' en {A,G}: año = primer A/G tras la ÚLTIMA F de la secuencia."
    f_pos = [i for i, t in enumerate(toks) if t == "F"]
    if not f_pos:
        return float("nan")
    last_f = f_pos[-1]
    after = [i for i in range(last_f + 1, len(toks)) if toks[i] in ("A", "G")]
    return float(YEAR0 + after[0]) if after else float("nan")


def _r1_year(toks: list[str]) -> float:
    """Como R0, exigiendo >=3 años de F consecutivos antes del cambio y >=3 años de A/G
    consecutivos después, sin retorno (la última F de la secuencia es única por construcción,
    así que "sin retorno" queda garantizado por definición)."""
    f_pos = [i for i, t in enumerate(toks) if t == "F"]
    if not f_pos:
        return float("nan")
    last_f = f_pos[-1]
    if last_f < 2 or toks[last_f - 2:last_f + 1] != ["F", "F", "F"]:
        return float("nan")
    after = toks[last_f + 1:]
    if len(after) < 3 or not all(t in ("A", "G") for t in after[:3]):
        return float("nan")
    return float(YEAR0 + last_f + 1)


def _own_transition_year(toks: list[str]) -> float:
    "â_i: primer año en que el píxel abandona F sin retornar, exigiendo >=2 años de permanencia."
    f_pos = [i for i, t in enumerate(toks) if t == "F"]
    if not f_pos:
        return float("nan")
    last_f = f_pos[-1]
    if len(toks) - (last_f + 1) < 2:
        return float("nan")
    return float(YEAR0 + last_f + 1)


def baseline_table(seqs_df: pd.DataFrame) -> pd.DataFrame:
    "ID, r0, r1, r0_year, r1_year, own_year -- sobre id_seqs_text_2000_2022_<zona>.zip."
    rows = []
    for id_, seq in zip(seqs_df["ID"], seqs_df["seqs"]):
        toks = seq.split("-")
        r0y, r1y, ownr = _r0_year(toks), _r1_year(toks), _own_transition_year(toks)
        rows.append((id_, not np.isnan(r0y), not np.isnan(r1y), r0y, r1y, ownr))
    return pd.DataFrame(rows, columns=["ID", "r0", "r1", "r0_year", "r1_year", "own_year"])


def compute_weights(seqs_df: pd.DataFrame, grid: G.ZoneGrid) -> pd.DataFrame:
    """Ponderación por inverso de la probabilidad de muestreo (Fase 1 del plan de validación
    externa): las zonas cuyo `lat_long_df`/`id_seqs_text` ya se guardaron submuestreados en la
    extracción (p. ej. `yungas`: solo quedaron las secuencias con transición + una submuestra al
    15% de las constantes, ver `land2vec.extract.subsample_constant_sequences`) subrepresentan
    sistemáticamente los píxeles constantes -- justo los que dominan la clase `negativo_limpio`.
    Las dinámicas nunca se descartan (`w=1`); cada constante existente representa
    `n_constantes_totales / n_constantes_guardadas` constantes reales de la grilla completa.

    Para una zona con grilla completa (`grid.n_i * grid.n_j == len(seqs_df)`, sin submuestreo,
    como `chaco_santiago_frontier` o `periurbano_cordoba`) esto da `w=1` para todas las filas --
    la fórmula es general, no hace falta un caso especial por zona."""
    from land2vec.cluster import dynamic_mask

    dinamica = dynamic_mask(seqs_df["seqs"])
    n_total_grilla = grid.n_i * grid.n_j
    n_dinamicas = int(dinamica.sum())
    n_existentes = len(seqs_df)
    n_constantes_guardadas = n_existentes - n_dinamicas
    n_constantes_totales = n_total_grilla - n_dinamicas

    w_constante = (n_constantes_totales / n_constantes_guardadas) if n_constantes_guardadas > 0 else 1.0
    weight = np.where(dinamica, 1.0, w_constante)
    return pd.DataFrame({"ID": seqs_df["ID"].to_numpy(), "weight": weight})


# ---------------------------------------------------------------------------
# Mesa maestra
# ---------------------------------------------------------------------------

def spatial_blocks(lat: np.ndarray, lon: np.ndarray, block_deg: float = 0.05) -> np.ndarray:
    "ID de bloque espacial (bloques de block_deg grados) -- para OOF y bootstrap."
    bi = np.floor(lat / block_deg).astype(np.int64)
    bj = np.floor(lon / block_deg).astype(np.int64)
    return bi * 1_000_000 + bj


def build_base(
    px: pd.DataFrame, coords_df: pd.DataFrame, baselines: pd.DataFrame, weights: pd.DataFrame,
    tau: float = TAU,
) -> pd.DataFrame:
    """Todo lo que NO depende de una corrida de clustering en particular: referencia, coordenadas,
    bloque espacial, baselines R0/R1/â_i, `anio_ref` y el `weight` de muestreo (`compute_weights`,
    =1 salvo en zonas submuestreadas como `yungas`). Se arma una sola vez por zona; cada corrida
    solo le agrega `cluster`/`proceso`/`anio_cambio` (ver `attach_cluster`)."""
    df = px.copy()
    df["referencia"] = label_reference(df, tau=tau).to_numpy()
    df = df.merge(coords_df[["ID", "latitude", "longitude"]], on="ID", how="left")
    df["block"] = spatial_blocks(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    df = df.merge(baselines, on="ID", how="left")
    df = df.merge(weights, on="ID", how="left")
    df["weight"] = df["weight"].fillna(1.0)
    return df


def attach_cluster(base: pd.DataFrame, zone: str, clusters_df: pd.DataFrame, cluster_info: dict[int, dict]) -> pd.DataFrame:
    "Agrega cluster/proceso/anio_cambio de una corrida sobre la mesa base (ver build_base)."
    cz = clusters_df.loc[clusters_df["zone"] == zone, ["ID", "cluster"]]
    df = base.merge(cz, on="ID", how="inner")
    if len(df) != len(base):
        print(f"    [aviso] {zone}: {len(base) - len(df):,} píxeles de desmonte_px sin cluster asignado "
              f"(¿corriste assign_train_clusters.py --group all --max-constant-fraction 1.0?)")

    info_df = pd.DataFrame.from_dict(cluster_info, orient="index")
    info_df.index.name = "cluster"
    df = df.merge(info_df[["proceso", "anio_cambio"]], left_on="cluster", right_index=True, how="left")
    df["proceso"] = df["proceso"].fillna("sin_info")  # cluster=-1 (sin tipificar) no está en la tipología
    return df


# ---------------------------------------------------------------------------
# Métrica 1 -- ganancia acumulada OOF
# ---------------------------------------------------------------------------

def gain_curve_oof(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Curva de ganancia acumulada + Gain@10%, con `p_c` estimado out-of-fold por bloque espacial
    (nunca en el mismo fold que evalúa). Todo ponderado por `weight` (`compute_weights` -- =1
    salvo en zonas submuestreadas): `p_c` es una media ponderada, y tanto el eje de área como el
    de positivos capturados acumulan `weight`, no conteo de filas -- si no, una zona submuestreada
    como `yungas` subestimaría la prevalencia real (sus filas `negativo_limpio` sobrevivientes son
    solo una fracción de las que realmente hay). Devuelve (curva con columnas
    area_cum/pos_cum/p_oof/y/weight, gain_at_10, tabla lift/p_c por cluster -- esta última
    estimada sobre TODO el conjunto restringido, no OOF, es la lectura sustantiva)."""
    cols_vacias = ["area_cum", "pos_cum", "p_oof", "y", "weight"]
    sub = df[df["referencia"].isin(["positivo", "negativo_limpio"])].copy()
    if sub.empty or sub["referencia"].nunique() < 2:
        return pd.DataFrame(columns=cols_vacias), float("nan"), pd.DataFrame()

    sub["y"] = (sub["referencia"] == "positivo").astype(int)
    sub["wy"] = sub["y"] * sub["weight"]
    blocks = sub["block"].unique()
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(blocks)
    fold_of_block = {b: i % n_folds for i, b in enumerate(shuffled)}
    sub["fold"] = sub["block"].map(fold_of_block)

    p_global = float(sub["wy"].sum() / sub["weight"].sum())
    p_oof = np.full(len(sub), p_global)
    for k in range(n_folds):
        train_mask = (sub["fold"] != k).to_numpy()
        eval_mask = ~train_mask
        if not eval_mask.any():
            continue
        g = sub.loc[train_mask].groupby("cluster").agg(wsum=("weight", "sum"), wysum=("wy", "sum"))
        p_c = g["wysum"] / g["wsum"]
        mapped = sub.loc[eval_mask, "cluster"].map(p_c)
        p_oof[eval_mask] = mapped.fillna(p_global).to_numpy()
    sub["p_oof"] = p_oof

    ordered = sub.sort_values("p_oof", ascending=False).reset_index(drop=True)
    cum_w = ordered["weight"].cumsum()
    total_w = float(cum_w.iloc[-1]) if len(cum_w) else 0.0
    ordered["area_cum"] = cum_w / total_w if total_w else np.nan
    cum_wy = ordered["wy"].cumsum()
    total_wy = max(float(cum_wy.iloc[-1]), 1e-9) if len(cum_wy) else 1e-9
    ordered["pos_cum"] = cum_wy / total_wy
    gain_at_10 = float(np.interp(0.10, ordered["area_cum"].to_numpy(), ordered["pos_cum"].to_numpy()))

    g_all = sub.groupby("cluster").agg(n=("wy", "size"), wsum=("weight", "sum"), wysum=("wy", "sum"))
    g_all["p_c"] = g_all["wysum"] / g_all["wsum"]
    g_all["lift"] = g_all["p_c"] / p_global if p_global else np.nan
    lift = g_all.reset_index()[["cluster", "n", "p_c", "lift"]]

    return ordered[cols_vacias], gain_at_10, lift


def auc_pr_from_scores(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    "AUC-PR ponderada (trapecios) a partir de scores continuos -- reutiliza p_oof/weight de gain_curve_oof."
    if weight is None:
        weight = np.ones_like(y, dtype=np.float64)
    order = np.argsort(-score)
    y_sorted = y[order]
    w_sorted = weight[order]
    tp = np.cumsum(y_sorted * w_sorted)
    fp = np.cumsum((1 - y_sorted) * w_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(tp[-1]), 1e-12)
    # inserta el punto (recall=0, precision=precision[0]) para la integral
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    # integral por trapecios a mano: np.trapz se removió en NumPy >=2.4 (renombrado a
    # np.trapezoid en 2.0); esto no depende de qué nombre exista en la versión instalada.
    return float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2.0))


# ---------------------------------------------------------------------------
# Métrica 2 -- MCC de la regla semántica
# ---------------------------------------------------------------------------

def weighted_prevalence(df: pd.DataFrame) -> float:
    "Fracción ponderada de positivo/negativo_limpio que es positivo -- usar en vez de .mean() sin peso."
    sub = df[df["referencia"].isin(["positivo", "negativo_limpio"])]
    if sub.empty:
        return float("nan")
    y = (sub["referencia"] == "positivo").to_numpy().astype(np.float64)
    w = sub["weight"].to_numpy()
    return float((y * w).sum() / w.sum())


def _mcc(vp: float, vn: float, fp: float, fn: float) -> float:
    num = vp * vn - fp * fn
    den = np.sqrt(float(vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    return float(num / den) if den else float("nan")


def semantic_confusion(df: pd.DataFrame, exclude_untyped: bool) -> dict:
    """MCC/precisión/exhaustividad de ŷ=proceso∈{deforestacion,degradacion_forestal} vs.
    y=referencia=='positivo'. `vp/vn/fp/fn` son sumas de `weight`, no conteos de filas (=conteos
    cuando `weight=1`, como en zonas sin submuestreo) -- ver `compute_weights`. `n` sigue siendo
    el conteo real de filas, solo para diagnóstico."""
    sub = df[df["referencia"].isin(["positivo", "negativo_limpio"])]
    if exclude_untyped:
        sub = sub[sub["cluster"] != -1]
    if sub.empty:
        return {"mcc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "vp": 0.0, "vn": 0.0, "fp": 0.0, "fn": 0.0, "n": 0}
    y = (sub["referencia"] == "positivo").to_numpy()
    yhat = sub["proceso"].isin(DEFOR_PROCESOS).to_numpy()
    w = sub["weight"].to_numpy()
    vp, vn = float(w[y & yhat].sum()), float(w[(~y) & (~yhat)].sum())
    fp, fn = float(w[(~y) & yhat].sum()), float(w[y & (~yhat)].sum())
    precision = vp / (vp + fp) if (vp + fp) else float("nan")
    recall = vp / (vp + fn) if (vp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) and precision == precision else float("nan")
    return {"mcc": _mcc(vp, vn, fp, fn), "precision": precision, "recall": recall, "f1": f1,
            "vp": vp, "vn": vn, "fp": fp, "fn": fn, "n": int(len(sub))}


def baseline_confusion(df: pd.DataFrame, rule: str) -> dict:
    "Igual que semantic_confusion pero con ŷ = baseline r0/r1 en vez del proceso del cluster."
    sub = df[df["referencia"].isin(["positivo", "negativo_limpio"])]
    if sub.empty:
        return {"mcc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "vp": 0.0, "vn": 0.0, "fp": 0.0, "fn": 0.0, "n": 0}
    y = (sub["referencia"] == "positivo").to_numpy()
    yhat = sub[rule].fillna(False).to_numpy().astype(bool)
    w = sub["weight"].to_numpy()
    vp, vn = float(w[y & yhat].sum()), float(w[(~y) & (~yhat)].sum())
    fp, fn = float(w[(~y) & yhat].sum()), float(w[y & (~yhat)].sum())
    precision = vp / (vp + fp) if (vp + fp) else float("nan")
    recall = vp / (vp + fn) if (vp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) and precision == precision else float("nan")
    return {"mcc": _mcc(vp, vn, fp, fn), "precision": precision, "recall": recall, "f1": f1,
            "vp": vp, "vn": vn, "fp": fp, "fn": fn, "n": int(len(sub))}


# ---------------------------------------------------------------------------
# Métrica 3 -- detección por polígono
# ---------------------------------------------------------------------------

def polygon_detection(poly_px: pd.DataFrame, master: pd.DataFrame, threshold: float = 0.5) -> dict:
    """`peso` = frac (del píxel) * pixel_area_ha(i) / SUPERF_ha (del polígono). Un polígono está
    "detectado" si la suma ponderada de `peso` sobre sus píxeles con proceso en
    {deforestacion, degradacion_forestal} es >= `threshold`. No usa el umbral tau de Fase 3: pesa
    TODA la superficie, incluidos los píxeles que a nivel-píxel quedaron `excluido`.

    No usa `weight` (`compute_weights`): acá el peso es área física real del polígono, no una
    corrección de representatividad muestral -- un píxel existe o no existe en el dataset (si no
    existe, ya no aporta `peso` porque no tiene fila en `poly_px`), no hay "cuántos píxeles como
    este hay en la grilla completa" que corregir."""
    win = poly_px[(poly_px["anio"] >= 2001) & (poly_px["anio"] <= 2022)].copy()
    if win.empty:
        return {"tasa_deteccion": float("nan"), "n_poligonos": 0, "por_decil": pd.DataFrame()}

    lat = master.set_index("ID")["latitude"]
    # área del píxel por fila de latitud -- se recalcula el índice i desde lat/lon del propio ID
    ids_win = win["ID"].to_numpy()
    lat_w = lat.reindex(ids_win).to_numpy()
    lon_w = master.set_index("ID")["longitude"].reindex(ids_win).to_numpy()
    i_w, _ = G.grid_index(lat_w, lon_w)
    area_ha = G.pixel_area_ha(i_w)
    win["peso"] = win["frac"].to_numpy() * area_ha / win["SUPERF_ha"].to_numpy()

    proc = master.set_index("ID")["proceso"]
    win["proceso"] = win["ID"].map(proc)
    win["matchea"] = win["proceso"].isin(DEFOR_PROCESOS)

    win["peso_matcheado"] = win["peso"].where(win["matchea"], 0.0)
    por_poligono = win.groupby("poly_id").agg(
        peso_total=("peso", "sum"),
        peso_matcheado=("peso_matcheado", "sum"),
        superf_ha=("SUPERF_ha", "first"),
    )
    por_poligono["tasa"] = por_poligono["peso_matcheado"] / por_poligono["peso_total"].clip(lower=1e-9)
    por_poligono["detectado"] = por_poligono["tasa"] >= threshold

    tasa_global = float(por_poligono["detectado"].mean())

    cortes = [0, 8.56, 25.7, 50, 100, np.inf]
    etiquetas = ["<1px (<8.56ha)", "1-3px", "3-6px (25.7-50ha)", "50-100ha", ">100ha"]
    por_poligono["decil"] = pd.cut(por_poligono["superf_ha"], bins=cortes, labels=etiquetas, right=False)
    por_decil = por_poligono.groupby("decil", observed=True).agg(
        n=("detectado", "size"), tasa_deteccion=("detectado", "mean"),
    ).reset_index()

    return {"tasa_deteccion": tasa_global, "n_poligonos": int(len(por_poligono)), "por_decil": por_decil}


# ---------------------------------------------------------------------------
# Casi-principal -- error temporal
# ---------------------------------------------------------------------------

def temporal_error(master: pd.DataFrame) -> dict:
    """Error cluster (anio_cambio - anio_ref) y error píxel (own_year - anio_ref) sobre filas
    positivo. No usa `weight`: `referencia == "positivo"` implica una transición real, y
    `compute_weights` nunca submuestrea trayectorias dinámicas (`w=1` siempre en este subconjunto,
    aunque la zona tenga huecos como `yungas`)."""
    sub = master[master["referencia"] == "positivo"].copy()
    sub = sub.dropna(subset=["anio_ref"])
    if sub.empty:
        return {"n": 0, "mediana_err_cluster": float("nan"), "sesgo_cluster": float("nan"),
                "mediana_err_pixel": float("nan"), "sesgo_pixel": float("nan"),
                "pct_2anios_cluster": float("nan"), "pct_2anios_pixel": float("nan")}

    err_c = sub["anio_cambio"] - sub["anio_ref"]
    err_p = sub["own_year"] - sub["anio_ref"]
    err_c_valid = err_c.dropna()
    err_p_valid = err_p.dropna()

    def _stats(err: pd.Series) -> tuple[float, float, float]:
        if err.empty:
            return float("nan"), float("nan"), float("nan")
        return float(err.abs().median()), float(err.median()), float((err.abs() <= 2).mean())

    med_c, sesgo_c, pct2_c = _stats(err_c_valid)
    med_p, sesgo_p, pct2_p = _stats(err_p_valid)
    return {
        "n": int(len(sub)), "n_con_anio_cambio": int(err_c_valid.shape[0]), "n_con_own_year": int(err_p_valid.shape[0]),
        "mediana_err_cluster": med_c, "sesgo_cluster": sesgo_c, "pct_2anios_cluster": pct2_c,
        "mediana_err_pixel": med_p, "sesgo_pixel": sesgo_p, "pct_2anios_pixel": pct2_p,
    }


# ---------------------------------------------------------------------------
# U(D|C) -- coeficiente de incertidumbre dirigido, ajustado por permutación de bloques
# ---------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def uncertainty_coefficient(df: pd.DataFrame, seed: int = 42) -> dict:
    """U(D|C) = [H(D) - H(D|C)] / H(D), ponderado por `weight` (cada grupo pesa por su fracción
    de `weight` total, no de conteo de filas -- si no, una zona submuestreada subestimaría cuánto
    "explican" los clusters dominados por trayectorias constantes), ajustado restando la media de
    49 permutaciones i.i.d. de la etiqueta de cluster -- línea de base estándar de una prueba de
    permutación para una medida de asociación (cuánta reducción de entropía da una partición al
    azar del mismo tamaño). No hace falta que la permutación sea por bloque acá, a diferencia del
    bootstrap de incertidumbre: la pregunta es "cuánto explica una partición cualquiera", no
    "cuánta autocorrelación espacial hay" (eso ya lo cubre `run_permutation_baseline`)."""
    sub = df[df["referencia"].isin(["positivo", "negativo_limpio"])].copy()
    if sub.empty:
        return {"u": float("nan"), "u_ajustado": float("nan")}
    y = (sub["referencia"] == "positivo").to_numpy().astype(np.float64)
    w = sub["weight"].to_numpy()
    p_overall = float((y * w).sum() / w.sum())
    h_d = _binary_entropy(p_overall)
    if h_d == 0:
        return {"u": 0.0, "u_ajustado": 0.0}

    def _h_d_given_c(labels: np.ndarray) -> float:
        tmp = pd.DataFrame({"wy": y * w, "w": w, "c": labels})
        g = tmp.groupby("c").agg(wsum=("w", "sum"), wysum=("wy", "sum"))
        g["p"] = g["wysum"] / g["wsum"]
        total_w = float(g["wsum"].sum())
        return float(((g["wsum"] / total_w) * g["p"].apply(_binary_entropy)).sum())

    u_obs = (h_d - _h_d_given_c(sub["cluster"].to_numpy())) / h_d

    rng = np.random.RandomState(seed)
    cluster_vals = sub["cluster"].to_numpy()
    perm_us = [
        (h_d - _h_d_given_c(rng.permutation(cluster_vals))) / h_d
        for _ in range(49)
    ]
    u_adj = float(u_obs - np.mean(perm_us))
    return {"u": float(u_obs), "u_ajustado": u_adj}


# ---------------------------------------------------------------------------
# Bootstrap por bloques espaciales
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    df: pd.DataFrame, point_fn, n_boot: int = 200, seed: int = 42,
) -> tuple[float, float, float, int]:
    """Bootstrap por bloques espaciales no solapados: remuestrea bloques con reposición hasta
    igualar el número de bloques observado, recomputa `point_fn(muestra)` en cada réplica.
    Devuelve (estimador puntual sobre los datos originales, IC 2.5%, IC 97.5%, n_bloques)."""
    point = point_fn(df)
    blocks = df["block"].unique()
    n_blocks = len(blocks)
    if n_blocks < 2 or not np.isfinite(point):
        return point, float("nan"), float("nan"), n_blocks

    rng = np.random.RandomState(seed)
    by_block = {b: idx.to_numpy() for b, idx in df.groupby("block").groups.items()}
    estimates = []
    for _ in range(n_boot):
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        idx = np.concatenate([by_block[b] for b in sampled_blocks])
        rep = df.loc[idx]
        val = point_fn(rep)
        if np.isfinite(val):
            estimates.append(val)
    if not estimates:
        return point, float("nan"), float("nan"), n_blocks
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return point, float(lo), float(hi), n_blocks


def _mcc_point_fn(exclude_untyped: bool):
    def fn(sample: pd.DataFrame) -> float:
        return semantic_confusion(sample, exclude_untyped)["mcc"]
    return fn


# ---------------------------------------------------------------------------
# Selección externa de granularidad
# ---------------------------------------------------------------------------

_CORRIDA_NIVELES = {"fina", "parametric", "medium", "medium_parametric", "coarse", "coarse_parametric"}


def select_best_suffix(rows: list[dict]) -> str | None:
    """Criterio a priori: mayor MCC (lectura -1=negativo), desempate por mayor U(D|C) ajustado.
    No hace falta OOF acá -- a diferencia de `p_c` en la ganancia acumulada, `proceso` sale de una
    regla determinista sobre la secuencia modal del cluster (`classify_process`), nunca vio el
    shapefile de desmonte, así que no hay circularidad que corregir con cross-fitting."""
    candidatos = [r for r in rows if r.get("nivel") in _CORRIDA_NIVELES
                  and np.isfinite(r.get("mcc_neg1_neg", float("nan")))]
    if not candidatos:
        return None
    candidatos.sort(key=lambda r: (r["mcc_neg1_neg"], r.get("u_ajustado", float("-inf"))), reverse=True)
    return candidatos[0]["suffix"]


# ---------------------------------------------------------------------------
# Corrida por (zona, suffix)
# ---------------------------------------------------------------------------

def run_one(zone: str, suffix: str, master: pd.DataFrame, poly_px: pd.DataFrame, n_boot: int) -> dict:
    label = suffix.strip("_") or "fina"
    print(f"  [{zone} / {label}] n={len(master):,}")

    cov_w = master.groupby("referencia")["weight"].sum()
    cov = (cov_w / cov_w.sum()).to_dict()
    prevalencia = weighted_prevalence(master)

    ordered, gain10, lift = gain_curve_oof(master)
    auc_pr = (
        auc_pr_from_scores(ordered["y"].to_numpy(), ordered["p_oof"].to_numpy(), ordered["weight"].to_numpy())
        if not ordered.empty else float("nan")
    )
    # el punto de block_bootstrap_ci recalcula el mismo MCC sobre los datos sin remuestrear (barato,
    # confusion matrix sobre ~n filas); se ignora acá y se usa semantic_confusion() para tener
    # también precisión/exhaustividad/f1 en un solo paso.
    _, mcc_neg_lo, mcc_neg_hi, n_bloques = block_bootstrap_ci(master, _mcc_point_fn(False), n_boot=n_boot)
    mcc_neg = semantic_confusion(master, exclude_untyped=False)
    mcc_tip = semantic_confusion(master, exclude_untyped=True)
    det = polygon_detection(poly_px, master)
    temp = temporal_error(master)
    unc = uncertainty_coefficient(master)

    row = {
        "zone": zone, "suffix": suffix, "nivel": label,
        "fuera_ecorregion": zone in FUERA_ECORREGION,
        "n": int(len(master)), "n_bloques": n_bloques,
        "cobertura_positivo": cov.get("positivo", 0.0),
        "cobertura_negativo_limpio": cov.get("negativo_limpio", 0.0),
        "cobertura_control_pre": cov.get("control_pre", 0.0),
        "cobertura_control_post": cov.get("control_post", 0.0),
        "cobertura_excluido": cov.get("excluido", 0.0),
        "prevalencia": prevalencia,
        "gain_at_10": gain10, "auc_pr_oof": auc_pr,
        "u_dc": unc["u"], "u_ajustado": unc["u_ajustado"],
        "mcc_neg1_neg": mcc_neg["mcc"], "mcc_neg1_neg_ci_lo": mcc_neg_lo, "mcc_neg1_neg_ci_hi": mcc_neg_hi,
        "precision_neg1_neg": mcc_neg["precision"], "recall_neg1_neg": mcc_neg["recall"], "f1_neg1_neg": mcc_neg["f1"],
        "mcc_tipificados": mcc_tip["mcc"], "precision_tipificados": mcc_tip["precision"],
        "recall_tipificados": mcc_tip["recall"], "f1_tipificados": mcc_tip["f1"],
        "deteccion_poligonos": det["tasa_deteccion"], "n_poligonos_ventana": det["n_poligonos"],
        "mediana_err_anio_cluster": temp["mediana_err_cluster"], "sesgo_anio_cluster": temp["sesgo_cluster"],
        "pct_2anios_cluster": temp["pct_2anios_cluster"],
        "mediana_err_anio_pixel": temp["mediana_err_pixel"], "sesgo_anio_pixel": temp["sesgo_pixel"],
        "pct_2anios_pixel": temp["pct_2anios_pixel"],
    }
    return row, lift, det["por_decil"]


def run_baseline(zone: str, base: pd.DataFrame, rule: str, n_boot: int) -> dict:
    """R0/R1: mismas métricas de detección/temporal que `run_one`, pero con ŷ = baseline sobre la
    secuencia cruda en vez del proceso de un cluster -- no dependen de ninguna corrida en
    particular, por eso toman `base` (`build_base`, sin cluster) y no `master`."""
    conf = baseline_confusion(base, rule)

    def point_fn(sample: pd.DataFrame) -> float:
        return baseline_confusion(sample, rule)["mcc"]

    _, lo, hi, n_bloques = block_bootstrap_ci(base, point_fn, n_boot=n_boot)

    sub = base[base["referencia"] == "positivo"].dropna(subset=["anio_ref"])
    year_col = "r0_year" if rule == "r0" else "r1_year"
    err = (sub[year_col] - sub["anio_ref"]).dropna()
    med = float(err.abs().median()) if not err.empty else float("nan")
    sesgo = float(err.median()) if not err.empty else float("nan")

    return {
        "zone": zone, "suffix": None, "nivel": rule, "fuera_ecorregion": zone in FUERA_ECORREGION,
        "n": int(len(base)), "n_bloques": n_bloques,
        "prevalencia": weighted_prevalence(base),
        "mcc_neg1_neg": conf["mcc"], "mcc_neg1_neg_ci_lo": lo, "mcc_neg1_neg_ci_hi": hi,
        "precision_neg1_neg": conf["precision"], "recall_neg1_neg": conf["recall"], "f1_neg1_neg": conf["f1"],
        "mediana_err_anio_cluster": med, "sesgo_anio_cluster": sesgo,
    }


def run_permutation_baseline(zone: str, suffix: str, master: pd.DataFrame, seed: int = 42) -> dict:
    """Línea de base de permutación espacial: reasigna el vector de `cluster` entero al azar entre
    píxeles (rompe la coherencia espacial preservando el tamaño de cada cluster) y recalcula el
    MCC semántico -- la diferencia contra el MCC real de esa corrida es la Δ que reporta la Tabla 1
    (percentil 95 de repetir esto, no solo una réplica -- acá una sola por simplicidad, ver
    limitaciones). Una rotación toroidal exacta requeriría reconstruir la grilla 2D completa."""
    rng = np.random.RandomState(seed)
    perm = master.copy()
    perm["cluster"] = rng.permutation(perm["cluster"].to_numpy())
    proc_by_cluster = master.drop_duplicates("cluster").set_index("cluster")["proceso"]
    perm["proceso"] = perm["cluster"].map(proc_by_cluster).fillna("sin_info")
    conf = semantic_confusion(perm, exclude_untyped=False)
    return {
        "zone": zone, "suffix": suffix, "nivel": "permutacion", "fuera_ecorregion": zone in FUERA_ECORREGION,
        "n": int(len(perm)), "mcc_neg1_neg": conf["mcc"], "precision_neg1_neg": conf["precision"],
        "recall_neg1_neg": conf["recall"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=CLUSTER_DIR)
    parser.add_argument("--zone", action="append", default=None, help="repetible; default: las 3 zonas")
    parser.add_argument("--only", choices=SUFFIXES, default=None, help="procesar solo esta corrida")
    parser.add_argument("--n-boot", type=int, default=200,
                         help="réplicas de bootstrap por bloques (el plan recomienda 999 para la corrida final)")
    parser.add_argument("--tau", type=float, default=TAU)
    args = parser.parse_args()

    t_inicio = time.perf_counter()
    zones = args.zone or ZONES
    suffixes = [args.only] if args.only else SUFFIXES

    cluster_info = load_cluster_info()

    all_rows: list[dict] = []
    all_lift: list[pd.DataFrame] = []
    all_decil: list[pd.DataFrame] = []

    for zone in zones:
        t_zona = time.perf_counter()
        print(f"\n=== {zone} {'(fuera de ecorregión)' if zone in FUERA_ECORREGION else '(Chaco Seco)'} ===")
        px = load_desmonte_px(zone, args.data_dir)
        poly_px = load_desmonte_poly_px(zone, args.data_dir)
        coords_df = load_coords(zone, args.data_dir)
        seqs_df = load_seqs(zone, args.data_dir)
        grid = G.ZoneGrid.from_coords(coords_df, zone)
        baselines = baseline_table(seqs_df)
        baselines = baselines.merge(reference_year(poly_px).rename("anio_ref"), left_on="ID", right_index=True, how="left")
        weights_df = compute_weights(seqs_df, grid)
        if not grid.complete:
            w = weights_df["weight"]
            print(f"  ponderación por submuestreo: w_constante={w[w > 1].iloc[0] if (w > 1).any() else 1.0:.2f} "
                  f"({int((w > 1).sum()):,} filas constantes reponderadas)")

        # todo lo que no depende de una corrida en particular (referencia, bloque, R0/R1/â_i,
        # anio_ref, weight) se arma una sola vez por zona -- ver build_base().
        base = build_base(px, coords_df, baselines, weights_df, tau=args.tau)

        # R0/R1 son independientes del clustering: se corren una sola vez por zona, no por suffix.
        all_rows.append(run_baseline(zone, base, "r0", n_boot=args.n_boot))
        all_rows.append(run_baseline(zone, base, "r1", n_boot=args.n_boot))

        for suffix in suffixes:
            t_corrida = time.perf_counter()
            clusters_df = load_clusters(suffix, args.data_dir)
            info = cluster_info.get(suffix, {})
            if not info:
                print(f"  [{suffix or 'fina'}] sin entrada en typology_browser.json, se omite")
                continue
            master = attach_cluster(base, zone, clusters_df, info)
            row, lift, decil = run_one(zone, suffix, master, poly_px, n_boot=args.n_boot)
            all_rows.append(row)
            lift["zone"], lift["suffix"] = zone, suffix
            all_lift.append(lift)
            decil["zone"], decil["suffix"] = zone, suffix
            all_decil.append(decil)
            all_rows.append(run_permutation_baseline(zone, suffix, master))
            print(f"    tiempo de la corrida: {_fmt_elapsed(time.perf_counter() - t_corrida)}")

        print(f"  tiempo total de {zone}: {_fmt_elapsed(time.perf_counter() - t_zona)}")

    if not all_rows:
        sys.exit("no se corrió ninguna combinación zona/corrida")

    out_df = pd.DataFrame(all_rows)
    args.cluster_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        sub = out_df[(out_df["suffix"] == suffix) | (out_df["suffix"].isna())]
        out_path = args.cluster_dir / f"desmonte_eval{suffix}.csv"
        sub.to_csv(out_path, index=False)
        print(f"\nguardado: {out_path.relative_to(ROOT)} ({len(sub)} filas)")

    if all_lift:
        pd.concat(all_lift, ignore_index=True).to_csv(args.cluster_dir / "desmonte_lift_por_cluster.csv", index=False)
    if all_decil:
        pd.concat(all_decil, ignore_index=True).to_csv(args.cluster_dir / "desmonte_deteccion_por_decil.csv", index=False)

    tiempo_total_seg = time.perf_counter() - t_inicio
    summary = {
        "zonas": zones, "suffixes": suffixes, "n_boot": args.n_boot, "tau": args.tau,
        "tiempo_total_seg": round(tiempo_total_seg, 1), "por_zona": {},
    }
    for zone in zones:
        rows_zone = [r for r in all_rows if r["zone"] == zone]
        best = select_best_suffix(rows_zone)
        summary["por_zona"][zone] = {
            "fuera_ecorregion": zone in FUERA_ECORREGION,
            "mejor_suffix_mcc_oof": best,
        }
    summary_path = args.cluster_dir / "desmonte_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"guardado: {summary_path.relative_to(ROOT)}")
    print(f"\ntiempo de corrida total = {_fmt_elapsed(tiempo_total_seg)}")


if __name__ == "__main__":
    main()
