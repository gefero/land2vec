"""Tipología de trayectorias sobre los embeddings z de land2vec v2.

Carga los embeddings ya extraídos (`data/embeddings_<zona>.zip`, ver
`scripts/extract_embeddings.py`), ofrece un despacho común a cuatro familias de
clustering (KMeans, GaussianMixture, HDBSCAN, aglomerativo/jerárquico) sobre
distintos preprocesados de `z`, y calcula el conjunto de métricas con el que
`scripts/tune_clustering.py` elige una configuración -- ver
`docs/v2_autoencoder_training.md` sección 7.2 para el criterio completo.

Todas las métricas de un `ClusterRunResult` se calculan en el mismo espacio en
que se ajustó el clustering (`space`), salvo `prototype_fidelity`, que siempre
decodifica centroides en el **z crudo** (`raw_centers`): el decoder del
autoencoder solo vio esa escala en entrenamiento, así que un centroide en
espacio estandarizado o L2-normalizado no es un input válido para `decode()`.

Nota de tractabilidad: `stability_ari` reajusta el clustering sobre pares de
submuestras al 80% (bootstrap), pero cada submuestra se recorta a lo sumo a
`boot_cap` filas -- sin ese techo, reajustar KMeans/HDBSCAN/jerárquico decenas
de veces por corrida del barrido, sobre las 107k secuencias con transición, no
termina en tiempo razonable en CPU. `boot_cap=20000` es el default.

Nota sobre el jerárquico: a diferencia de las otras tres familias, no se ajusta
directo sobre las 107k filas -- ni el Ward canónico de scipy (O(n^2) en
distancias) ni una versión restringida a un grafo de conectividad k-NN
(probada y descartada: esta base tiene tantas trayectorias idénticas que el
grafo queda fragmentado en cientos de componentes, y reconectarlos resultó
igual de costoso) terminan en tiempo razonable acá. `run_hierarchical_config`
en cambio ajusta sobre una submuestra estratificada por zona y extiende las
etiquetas al resto por centroide más cercano -- la mitigación estándar para
Ward a esta escala, no un atajo menos riguroso que las otras familias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from land2vec.extract import constant_mask
from land2vec.tokenizer import Tokenizer

ZONES = [
    "puna_noa",
    "patagonia_estepa",
    "periurbano_cordoba",
    "ibera",
    "delta_parana",
    "pampa_nucleo",
    "misiones_selva",
]

VALID_LABELS = [i for i in Tokenizer.VOCAB.values() if i != Tokenizer.VOCAB["[UNK]"]]
LABEL_NAMES = [Tokenizer.REVERSE_VOCAB[i] for i in VALID_LABELS]


# ---------------------------------------------------------------------------
# Carga y armado del pool
# ---------------------------------------------------------------------------


def _stratified_indices(zone: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    "Índices de una submuestra de tamaño ~n estratificada por zona (asignación proporcional, redondeada por zona)."
    frac = n / len(zone)
    chunks = []
    for z_name in np.unique(zone):
        zone_idx = np.flatnonzero(zone == z_name)
        take = max(1, round(len(zone_idx) * frac))
        take = min(take, len(zone_idx))
        chunks.append(rng.choice(zone_idx, size=take, replace=False))
    return np.concatenate(chunks)


@dataclass
class Pool:
    "Filas alineadas por posición: mismo orden en ids/seqs/z/zone/lat/lon."

    ids: np.ndarray
    seqs: pd.Series
    z: np.ndarray
    zone: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask: np.ndarray) -> "Pool":
        return Pool(
            ids=self.ids[mask],
            seqs=self.seqs[mask].reset_index(drop=True),
            z=self.z[mask],
            zone=self.zone[mask],
            lat=self.lat[mask],
            lon=self.lon[mask],
        )

    def sample(self, n: int, seed: int = 42, stratify_by_zone: bool = True) -> "Pool":
        "Submuestra sin reemplazo, opcionalmente estratificada por zona."
        rng = np.random.default_rng(seed)
        if n >= len(self):
            return self
        if not stratify_by_zone:
            idx = rng.choice(len(self), size=n, replace=False)
            return self.subset(np.isin(np.arange(len(self)), idx))
        idx = _stratified_indices(self.zone, n, rng)
        mask = np.zeros(len(self), dtype=bool)
        mask[idx] = True
        return self.subset(mask)


def load_zone_seqs(zone: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"id_seqs_text_2000_2022_{zone}.zip")


def load_zone_coords(zone: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"lat_long_df_{zone}.zip", usecols=["ID", "latitude", "longitude"])


def load_zone_embeddings(zone: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"embeddings_{zone}.zip")


def load_pool(zones: list[str], data_dir: Path) -> Pool:
    """Concatena secuencias + embeddings z + coordenadas de varias zonas.

    Valida, por zona, que las tres fuentes tengan el mismo largo y el mismo
    `ID` en el mismo orden -- `scripts/extract_embeddings.py` preserva el
    orden posicional del CSV de secuencias al escribir los embeddings, pero
    nada impide que una fuente se regenere de forma independiente y quede
    desalineada; sin este chequeo eso pasaría silencioso.
    """
    seqs_parts, zone_parts, z_parts, lat_parts, lon_parts, id_parts = [], [], [], [], [], []
    for zone in zones:
        seqs_df = load_zone_seqs(zone, data_dir)
        emb_df = load_zone_embeddings(zone, data_dir)
        coords_df = load_zone_coords(zone, data_dir)
        if len(seqs_df) != len(emb_df) or not (seqs_df["ID"].values == emb_df["ID"].values).all():
            raise ValueError(f"{zone}: secuencias y embeddings no están alineados por ID")
        if len(seqs_df) != len(coords_df) or not (seqs_df["ID"].values == coords_df["ID"].values).all():
            raise ValueError(f"{zone}: secuencias y coordenadas no están alineados por ID")

        z_cols = [c for c in emb_df.columns if c.startswith("z")]
        id_parts.append(seqs_df["ID"].values)
        seqs_parts.append(seqs_df["seqs"])
        z_parts.append(emb_df[z_cols].values.astype(np.float64))
        zone_parts.append(np.full(len(seqs_df), zone))
        lat_parts.append(coords_df["latitude"].values)
        lon_parts.append(coords_df["longitude"].values)

    return Pool(
        ids=np.concatenate(id_parts),
        seqs=pd.concat(seqs_parts, ignore_index=True),
        z=np.concatenate(z_parts),
        zone=np.concatenate(zone_parts),
        lat=np.concatenate(lat_parts),
        lon=np.concatenate(lon_parts),
    )


def load_pool_subsampled(zones: list[str], data_dir: Path, max_fraction: float = 0.15, seed: int = 42) -> Pool:
    """Como `load_pool`, pero recorta las secuencias constantes de cada zona a lo
    sumo a `max_fraction` del resultado (`land2vec.extract.subsample_constant_sequences`)
    -- para que el mapa de clusters cubra el pool completo (hoy 96,8% del territorio
    queda sin colorear si solo se etiquetan las secuencias con transición) sin que
    la inmensa mayoría de píxeles "sin cambio" ahogue el resto."""
    from land2vec.extract import subsample_constant_sequences

    seqs_parts, zone_parts, z_parts, lat_parts, lon_parts, id_parts = [], [], [], [], [], []
    for zone in zones:
        seqs_df = load_zone_seqs(zone, data_dir)
        emb_df = load_zone_embeddings(zone, data_dir)
        coords_df = load_zone_coords(zone, data_dir)
        if len(seqs_df) != len(emb_df) or not (seqs_df["ID"].values == emb_df["ID"].values).all():
            raise ValueError(f"{zone}: secuencias y embeddings no están alineados por ID")
        if len(seqs_df) != len(coords_df) or not (seqs_df["ID"].values == coords_df["ID"].values).all():
            raise ValueError(f"{zone}: secuencias y coordenadas no están alineados por ID")

        sub_df = subsample_constant_sequences(seqs_df, max_fraction=max_fraction, seed=seed)
        keep = seqs_df["ID"].isin(sub_df["ID"]).values
        z_cols = [c for c in emb_df.columns if c.startswith("z")]

        id_parts.append(seqs_df["ID"].values[keep])
        seqs_parts.append(seqs_df["seqs"][keep])
        z_parts.append(emb_df[z_cols].values.astype(np.float64)[keep])
        zone_parts.append(np.full(int(keep.sum()), zone))
        lat_parts.append(coords_df["latitude"].values[keep])
        lon_parts.append(coords_df["longitude"].values[keep])

    return Pool(
        ids=np.concatenate(id_parts),
        seqs=pd.concat(seqs_parts, ignore_index=True),
        z=np.concatenate(z_parts),
        zone=np.concatenate(zone_parts),
        lat=np.concatenate(lat_parts),
        lon=np.concatenate(lon_parts),
    )


def dynamic_mask(seqs: pd.Series) -> np.ndarray:
    "True para las secuencias con al menos una transición en los 23 años."
    return ~constant_mask(pd.DataFrame({"seqs": seqs.reset_index(drop=True)}), "seqs").values


# ---------------------------------------------------------------------------
# Preprocesado de z
# ---------------------------------------------------------------------------

Space = Literal["raw", "standard", "l2"]


@dataclass
class SpaceTransform:
    space: Space
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def apply(self, z: np.ndarray) -> np.ndarray:
        if self.space == "raw":
            return z
        if self.space == "standard":
            return (z - self.mean) / self.scale
        if self.space == "l2":
            return normalize(z)
        raise ValueError(f"space desconocido: {self.space!r}")

    def to_jsonable(self) -> dict:
        return {
            "space": self.space,
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @staticmethod
    def from_jsonable(d: dict) -> "SpaceTransform":
        mean = None if d["mean"] is None else np.array(d["mean"])
        scale = None if d["scale"] is None else np.array(d["scale"])
        return SpaceTransform(d["space"], mean=mean, scale=scale)


def fit_space(z: np.ndarray, space: Space) -> tuple[np.ndarray, SpaceTransform]:
    if space == "raw":
        t = SpaceTransform("raw")
    elif space == "standard":
        scaler = StandardScaler().fit(z)
        t = SpaceTransform("standard", mean=scaler.mean_, scale=scaler.scale_)
    elif space == "l2":
        t = SpaceTransform("l2")
    else:
        raise ValueError(f"space desconocido: {space!r}")
    return t.apply(z), t


# ---------------------------------------------------------------------------
# Ajuste de clustering: despacho común a las 4 familias
# ---------------------------------------------------------------------------


@dataclass
class ClusterFit:
    labels: np.ndarray
    centers: np.ndarray | None  # en el mismo espacio en que se ajustó (z_fit)
    extra: dict = field(default_factory=dict)


def _labeled_centroids(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    "Centroide (media) de cada etiqueta >=0, en el espacio de z recibido, ordenados 0..k-1."
    uniq = sorted(int(l) for l in set(labels.tolist()) if l != -1)
    if not uniq:
        return np.zeros((0, z.shape[1]))
    return np.stack([z[labels == l].mean(axis=0) for l in uniq])


def fit_kmeans(z: np.ndarray, k: int, seed: int = 42) -> ClusterFit:
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
    return ClusterFit(model.labels_, model.cluster_centers_)


def fit_gmm(z: np.ndarray, k: int, covariance_type: str = "full", seed: int = 42) -> ClusterFit:
    "n_init=10 para paridad con KMeans (fit_kmeans) -- el default de sklearn es 1, mucho más propenso a un óptimo local pobre."
    model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=10, random_state=seed).fit(z)
    labels = model.predict(z)
    return ClusterFit(labels, model.means_, {"bic": float(model.bic(z)), "aic": float(model.aic(z))})


def fit_hdbscan(z: np.ndarray, min_cluster_size: int, min_samples: int | None = None) -> ClusterFit:
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(z)
    labels = model.labels_
    centers = _labeled_centroids(z, labels)
    return ClusterFit(labels, centers)


def hierarchical_linkage(z: np.ndarray, method: str) -> np.ndarray:
    """Matriz de linkage de scipy -- el paso caro (O(n^2) en distancias). Se calcula
    una sola vez por (submuestra, method) y sirve para cualquier `k` vía
    `hierarchical_from_linkage`, en vez de recalcularla en cada punto del barrido
    de `k` (lo que haría `--sweep hierarchical` ~14x más lento sin ganar nada,
    ya que el árbol completo no depende de `k`)."""
    return scipy_linkage(z, method=method)


def hierarchical_from_linkage(z: np.ndarray, Z: np.ndarray, k: int) -> ClusterFit:
    "Corta un linkage ya calculado (`hierarchical_linkage`) en `k` clusters."
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # a 0-index, como el resto
    centers = _labeled_centroids(z, labels)

    n = z.shape[0]
    heights = Z[:, 2]
    idx_for_k = n - k - 1  # fila de Z que produce exactamente k clusters
    if 0 <= idx_for_k < len(heights) - 1:
        # salto de altura entre el merge que arma k clusters y el siguiente
        # (el que los reduciría a k-1): un salto grande dice que k es un buen corte.
        merge_gap = float((heights[idx_for_k + 1] - heights[idx_for_k]) / (heights[idx_for_k] + 1e-12))
    else:
        merge_gap = float("nan")

    return ClusterFit(labels, centers, {"merge_gap": merge_gap})


def cophenetic_correlation(z: np.ndarray, Z: np.ndarray) -> float:
    "Correlación entre las distancias originales y las del dendrograma -- diagnóstico estándar para elegir linkage."
    coph_corr, _ = cophenet(Z, pdist(z))
    return float(coph_corr)


def fit_hierarchical_full(z: np.ndarray, k: int, method: str) -> ClusterFit:
    """Ward/average/complete canónico vía scipy, sin restricción de conectividad.

    Solo pensado para una submuestra chica (unos pocos miles de filas) -- la
    matriz condensada de distancias es O(n^2), inviable sobre las 107k
    secuencias con transición. `run_hierarchical_config` es el punto de entrada
    real: ajusta esto sobre una submuestra estratificada y extiende las
    etiquetas al resto del pool por centroide más cercano (ver su docstring
    para por qué -- una restricción de conectividad k-NN sobre el dataset
    completo se probó y resultó igual de inviable en este entorno).
    """
    Z = hierarchical_linkage(z, method)
    fit = hierarchical_from_linkage(z, Z, k)
    fit.extra["cophenetic_corr"] = cophenetic_correlation(z, Z)
    return fit


def dispatch_fit(algo: str, z: np.ndarray, params: dict, seed: int = 42) -> ClusterFit:
    "Despacho común a las familias que ajustan directo sobre el pool completo -- no cubre 'hierarchical', ver run_hierarchical_config."
    if algo == "kmeans":
        return fit_kmeans(z, k=params["k"], seed=seed)
    if algo == "gmm":
        return fit_gmm(z, k=params["k"], covariance_type=params.get("covariance_type", "full"), seed=seed)
    if algo == "hdbscan":
        return fit_hdbscan(z, min_cluster_size=params["min_cluster_size"], min_samples=params.get("min_samples"))
    raise ValueError(f"algo desconocido: {algo!r}")


def assign_by_centroid(z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    "Asigna cada fila de z (mismo espacio que `centers`) al centroide más cercano."
    nn = NearestNeighbors(n_neighbors=1).fit(centers)
    _, idx = nn.kneighbors(z)
    return idx.ravel()


def assign_pool(raw_z: np.ndarray, transform: SpaceTransform, centers: np.ndarray) -> np.ndarray:
    "Etiqueta embeddings nuevos (z crudo) con una config ya elegida: transforma y asigna."
    return assign_by_centroid(transform.apply(raw_z), centers)


# ---------------------------------------------------------------------------
# Métricas de selección
# ---------------------------------------------------------------------------


def silhouette_repeated(
    z: np.ndarray, labels: np.ndarray, n_repeats: int = 5, sample_size: int = 20000, seed: int = 42
) -> tuple[float, float]:
    mask = labels != -1
    z_eff, labels_eff = z[mask], labels[mask]
    if len(set(labels_eff.tolist())) < 2:
        return float("nan"), float("nan")
    scores = [
        silhouette_score(z_eff, labels_eff, sample_size=min(sample_size, len(z_eff)), random_state=seed + i)
        for i in range(n_repeats)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def safe_calinski_harabasz(z: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return float("nan")
    return float(calinski_harabasz_score(z[mask], labels[mask]))


def safe_davies_bouldin(z: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return float("nan")
    return float(davies_bouldin_score(z[mask], labels[mask]))


def stability_ari(
    z: np.ndarray,
    fit_labels_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    n_boot: int = 5,
    frac: float = 0.8,
    boot_cap: int = 20000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> float:
    """ARI entre dos reajustes independientes del mismo clustering sobre pares de
    submuestras al `frac` (recortadas a `boot_cap` filas, ver docstring del
    módulo), comparados en la intersección de índices. Defensa contra un `k`
    que solo se sostiene por el azar de la muestra.

    `fit_labels_fn` recibe `(z_sub, groups_sub)` -- `groups_sub` es `None` si no
    se pasa `groups` acá. Lo usa `run_hierarchical_config` para que su resample
    interno (submuestra + linkage + corte) estratifique por zona igual que el
    ajuste real (`Pool.sample(..., stratify_by_zone=True)`); las demás familias
    (kmeans/gmm/hdbscan) no lo necesitan y simplemente lo ignoran."""
    rng = np.random.default_rng(seed)
    n = len(z)
    size = min(int(frac * n), boot_cap)
    scores = []
    for _ in range(n_boot):
        idx_a = rng.choice(n, size=size, replace=False)
        idx_b = rng.choice(n, size=size, replace=False)
        common, pos_a, pos_b = np.intersect1d(idx_a, idx_b, return_indices=True)
        if len(common) < 50:
            continue
        groups_a = groups[idx_a] if groups is not None else None
        groups_b = groups[idx_b] if groups is not None else None
        labels_a = fit_labels_fn(z[idx_a], groups_a)[pos_a]
        labels_b = fit_labels_fn(z[idx_b], groups_b)[pos_b]
        scores.append(adjusted_rand_score(labels_a, labels_b))
    return float(np.mean(scores)) if scores else float("nan")


def prototype_fidelity(
    model: torch.nn.Module, seqs: pd.Series, labels: np.ndarray, raw_centers: np.ndarray, device: str = "cpu"
) -> float:
    """Macro F1 (restringido a las clases con soporte>0 en cada cluster) de
    comparar, posición a posición, la secuencia real de cada miembro contra la
    trayectoria prototípica de su cluster (el centroide en z crudo, decodificado
    con `model.decode()`). Ponderado por tamaño de cluster. Reusa el criterio de
    macro F1 restringido a soporte pendiente en docs/v2_autoencoder_training.md
    §8 (ahí para reconstrucción por zona, acá para clustering)."""
    encoded = np.stack([Tokenizer.encode(s) for s in seqs])  # (N, 23)
    uniq = sorted(int(l) for l in set(labels.tolist()) if l != -1)
    if not uniq or raw_centers.shape[0] != len(uniq):
        return float("nan")

    with torch.inference_mode():
        centers_t = torch.tensor(raw_centers, dtype=torch.float32, device=device)
        proto_tokens = model.decode(centers_t).argmax(-1).cpu().numpy()  # (k, 23)

    f1s, weights = [], []
    for row, cluster_id in enumerate(uniq):
        member_mask = labels == cluster_id
        y_true = encoded[member_mask].reshape(-1)
        y_pred = np.tile(proto_tokens[row], int(member_mask.sum()))
        present = sorted(set(y_true.tolist()) & set(VALID_LABELS))
        if not present:
            continue
        f1s.append(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))
        weights.append(int(member_mask.sum()))
    return float(np.average(f1s, weights=weights)) if f1s else float("nan")


def spatial_coherence(
    lat: np.ndarray, lon: np.ndarray, zone: np.ndarray, labels: np.ndarray, k_neighbors: int = 8, seed: int = 42
) -> float:
    """Fracción de los `k_neighbors` vecinos geográficos más próximos (dentro de
    la misma zona, excluyendo ruido de HDBSCAN) que comparten cluster, menos la
    misma fracción con etiquetas permutadas al azar (línea de base de "vecinos
    comparten cluster por puro tamaño relativo de cada cluster", no por
    contigüidad real). Los puntos con label==-1 (ruido) se excluyen antes de
    buscar vecinos: sin este filtro, dos vecinos geográficos ambos "sin
    asignar" contarían como si compartieran un cluster real, inflando la
    métrica en las configs con mucho ruido (HDBSCAN con `min_cluster_size`
    alto) por la sola contigüidad espacial de las zonas ambiguas sin
    tipificar, no por una tipología genuina."""
    rng = np.random.default_rng(seed)
    same_scores, baseline_scores = [], []
    for z_name in np.unique(zone):
        m = (zone == z_name) & (labels != -1)
        if m.sum() < k_neighbors + 1:
            continue
        coords = np.column_stack([lat[m], lon[m]])
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k_neighbors + 1)
        neighbor_idx = idx[:, 1:]  # excluye el punto mismo (columna 0)
        lab = labels[m]
        same_scores.append(float((lab[neighbor_idx] == lab[:, None]).mean()))
        perm = rng.permutation(lab)
        baseline_scores.append(float((perm[neighbor_idx] == perm[:, None]).mean()))
    if not same_scores:
        return float("nan")
    return float(np.mean(same_scores) - np.mean(baseline_scores))


def size_stats(labels: np.ndarray) -> dict:
    valid = labels[labels != -1]
    noise_frac = float((labels == -1).mean()) if len(labels) else float("nan")
    if len(valid) == 0:
        return {"largest_share": float("nan"), "size_entropy": float("nan"), "noise_frac": noise_frac, "k_effective": 0}
    counts = np.bincount(valid)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    return {
        "largest_share": float(p.max()),
        "size_entropy": entropy,
        "noise_frac": noise_frac,
        "k_effective": int(len(p)),
    }


# ---------------------------------------------------------------------------
# Orquestación de una corrida
# ---------------------------------------------------------------------------


@dataclass
class ClusterRunResult:
    run_id: str
    algo: str
    params: dict
    space: Space
    labels: np.ndarray
    centers: np.ndarray  # en espacio de ajuste (z_fit) -- para assign_by_centroid
    raw_centers: np.ndarray  # media de z crudo por cluster -- para decode()
    transform: SpaceTransform
    metrics: dict
    eligible: bool  # puede ser la config ganadora final (ver docs §7.2 / plan §3.1)


def run_config(
    pool: Pool,
    algo: Literal["kmeans", "gmm", "hdbscan"],
    params: dict,
    space: Space,
    model: torch.nn.Module,
    device: str = "cpu",
    seed: int = 42,
    n_boot: int = 5,
    boot_cap: int = 20000,
) -> ClusterRunResult:
    "Ajusta y evalúa kmeans/gmm/hdbscan, los tres directo sobre `pool` completo. Para 'hierarchical' ver run_hierarchical_config."
    z_fit, transform = fit_space(pool.z, space)
    fit = dispatch_fit(algo, z_fit, params, seed=seed)
    labels = fit.labels
    raw_centers = _labeled_centroids(pool.z, labels)

    def _fit_labels(z_sub: np.ndarray, _groups: np.ndarray | None = None) -> np.ndarray:
        return dispatch_fit(algo, z_sub, params, seed=seed).labels

    sil_mean, sil_std = silhouette_repeated(z_fit, labels)
    metrics = {
        "n_fit": len(pool),
        "silhouette_mean": sil_mean,
        "silhouette_std": sil_std,
        "calinski_harabasz": safe_calinski_harabasz(z_fit, labels),
        "davies_bouldin": safe_davies_bouldin(z_fit, labels),
        "stability_ari": stability_ari(z_fit, _fit_labels, n_boot=n_boot, boot_cap=boot_cap, seed=seed),
        "prototype_fidelity": prototype_fidelity(model, pool.seqs, labels, raw_centers, device=device),
        "spatial_coherence": spatial_coherence(pool.lat, pool.lon, pool.zone, labels),
        **size_stats(labels),
        **fit.extra,
    }

    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    run_id = f"{algo}__{param_str}__space={space}"
    return ClusterRunResult(run_id, algo, params, space, labels, fit.centers, raw_centers, transform, metrics, eligible=True)


def run_hierarchical_config(
    eval_pool: Pool,
    method: str,
    k: int,
    space: Space,
    fit_sample_size: int,
    model: torch.nn.Module,
    device: str = "cpu",
    seed: int = 42,
    n_boot: int = 5,
    boot_cap: int = 20000,
    linkage_cache: dict[str, np.ndarray] | None = None,
) -> ClusterRunResult:
    """Ward/average/complete jerárquico, evaluado en igualdad de condiciones con
    kmeans/gmm/hdbscan pese a no poder ajustarse sobre las 107k filas del pool
    completo: la matriz condensada de distancias de scipy es O(n^2) (inviable
    directo), y restringir la conectividad a un grafo k-NN -- la mitigación
    habitual para eso -- se probó en este módulo y resultó igual de costosa,
    porque las muchísimas trayectorias idénticas de esta base fragmentan ese
    grafo en cientos de componentes que `AgglomerativeClustering` no logra
    resolver en tiempo razonable incluso ya reconectados.

    La solución estándar en su lugar: ajustar sobre una submuestra estratificada
    por zona (`fit_sample_size`, tratable -- unos segundos incluso en 25k filas)
    y extender las etiquetas al resto del pool por centroide más cercano
    (`assign_by_centroid`), igual que ya hace falta para HDBSCAN/jerárquico
    cuando se etiqueta el pool con constantes submuestreadas en `--select`. Por
    eso, a diferencia de la primera versión de este módulo, esta config SÍ es
    elegible como ganadora -- ya no es solo diagnóstico.

    `linkage_cache`, si se pasa, evita recalcular `hierarchical_linkage` (el
    paso caro) para cada `k` de un mismo `method` dentro de un barrido.
    """
    fit_pool = eval_pool.sample(fit_sample_size, seed=seed, stratify_by_zone=True)
    z_fit_sub, transform = fit_space(fit_pool.z, space)

    if linkage_cache is not None and method in linkage_cache:
        Z = linkage_cache[method]
    else:
        Z = hierarchical_linkage(z_fit_sub, method)
        if linkage_cache is not None:
            linkage_cache[method] = Z

    sub_fit = hierarchical_from_linkage(z_fit_sub, Z, k)
    z_eval_fit = transform.apply(eval_pool.z)
    labels = assign_by_centroid(z_eval_fit, sub_fit.centers)
    raw_centers = _labeled_centroids(eval_pool.z, labels)

    def _fit_labels(z_sub: np.ndarray, zone_sub: np.ndarray | None) -> np.ndarray:
        """Reajusta el mismo procedimiento (submuestra + linkage + corte) sobre una
        submuestra bootstrap y extiende al resto de esa submuestra. Estratifica el
        resample interno por zona cuando se dispone de `zone_sub` -- espeja
        exactamente el ajuste real (`eval_pool.sample(..., stratify_by_zone=True)`
        arriba), en vez de un muestreo simple que no lo hacía."""
        rng = np.random.default_rng(seed)
        n_fit = min(fit_sample_size, len(z_sub))
        if zone_sub is not None:
            fit_idx = _stratified_indices(zone_sub, n_fit, rng)
        else:
            fit_idx = rng.choice(len(z_sub), size=n_fit, replace=False)
        Zb = hierarchical_linkage(z_sub[fit_idx], method)
        labels_fit = hierarchical_from_linkage(z_sub[fit_idx], Zb, k).labels
        centers_fit = _labeled_centroids(z_sub[fit_idx], labels_fit)
        return assign_by_centroid(z_sub, centers_fit)

    sil_mean, sil_std = silhouette_repeated(z_eval_fit, labels)
    metrics = {
        "n_fit": len(fit_pool),
        "silhouette_mean": sil_mean,
        "silhouette_std": sil_std,
        "calinski_harabasz": safe_calinski_harabasz(z_eval_fit, labels),
        "davies_bouldin": safe_davies_bouldin(z_eval_fit, labels),
        "stability_ari": stability_ari(
            z_eval_fit, _fit_labels, n_boot=n_boot, boot_cap=boot_cap, seed=seed, groups=eval_pool.zone
        ),
        "prototype_fidelity": prototype_fidelity(model, eval_pool.seqs, labels, raw_centers, device=device),
        "spatial_coherence": spatial_coherence(eval_pool.lat, eval_pool.lon, eval_pool.zone, labels),
        "cophenetic_corr": cophenetic_correlation(z_fit_sub, Z),
        **sub_fit.extra,  # merge_gap
        **size_stats(labels),
    }

    params = {"k": k, "method": method, "fit_sample_size": fit_sample_size}
    param_str = "_".join(f"{key}={val}" for key, val in sorted(params.items()))
    run_id = f"hierarchical__{param_str}__space={space}"
    return ClusterRunResult(
        run_id, "hierarchical", params, space, labels, sub_fit.centers, raw_centers, transform, metrics, eligible=True
    )


def to_jsonable(obj):
    "Convierte recursivamente arrays/escalares de numpy a tipos nativos de JSON."
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
