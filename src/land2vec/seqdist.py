"""Disimilitud entre trayectorias y calidad de partición en el espacio de las
secuencias -- el complemento "analítico" de `land2vec.typology` (que es
descriptivo).

Motivo para tenerlo pese al presupuesto O(n·T) de `typology`: **el pool dinámico
de 107.362 trayectorias tiene solo 1.128 secuencias distintas**. Colapsado a ese
conjunto con pesos, toda la maquinaria de disimilitud de TraMineR es trivial --
una matriz 1128x1128 (~10 MB) y unos segundos de cómputo -- y da lo que la
batería descriptiva no puede:

- una **vara de calidad de partición ajena al espacio `z`** en el que se formaron
  los clusters: pseudo-R² (discrepancia, Studer/Ritschard) y ASW en espacio de
  secuencias, con los que comparar las 6 corridas en igualdad de condiciones;
- **secuencias representativas** por densidad de vecindad (`seqrplot`): un
  puñado de trayectorias reales no redundantes que cubren cada cluster.

Tres métricas de disimilitud, todas sobre secuencias de largo fijo 23 (años
calendario alineados 2000-2022):

- `hamming`  -- `method="HAM"` de TraMineR: nº de posiciones que diferen (opcio-
  nalmente con costo de sustitución). Sin indels: respeta el tiempo calendario.
- `dhd`      -- Hamming dinámico: costo de sustitución dependiente de la posición
  (`c_t(i,j) = 2 - f_t(i) - f_t(j)`, con `f_t` la frecuencia transversal
  ponderada del estado en el año `t`). Simplificación por frecuencia del DHD de
  Lesnard (2010), que además usa tasas de transición locales.
- `om`       -- Optimal Matching (`method="OM"`): alineación global con indels y
  matriz de costos de sustitución. Por defecto, costos TRATE
  (`c(i,j) = 2 - p(i|j) - p(j|i)` de las tasas de transición globales) e
  `indel = max(c)/2`. Fiel a TraMineR; la objeción es que los indels
  reintroducen desalineación temporal, informativa acá.

Los pesos (`weights`) que reciben casi todas las funciones son los conteos de
cada secuencia distinta: hacen que el resultado sea idéntico al de operar sobre
las 107k filas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from land2vec.tokenizer import Tokenizer
from land2vec.typology import N_SYMBOLS, encode_matrix

VOCAB_NAMES = [Tokenizer.REVERSE_VOCAB[i] for i in range(N_SYMBOLS)]


# ---------------------------------------------------------------------------
# Conjunto de secuencias distintas ponderadas
# ---------------------------------------------------------------------------


def distinct(seqs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Serie de strings -> (uniq (U, 23) tokens, inverse (n,), counts (U,))."
    X = encode_matrix(seqs)
    uniq, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return uniq, inv.ravel(), cnt


def labels_to_distinct(inverse: np.ndarray, labels: np.ndarray, n_uniq: int) -> tuple[np.ndarray, int]:
    """Etiqueta de cluster por secuencia distinta. Las secuencias idénticas tienen
    `z` idéntico, así que caen todas en el mismo cluster -- salvo empates de
    frontera; se toma la moda y se devuelve cuántas secuencias distintas tenían
    más de una etiqueta no-ruido (para avisar si es un número no trivial).
    """
    out = np.full(n_uniq, -1, dtype=np.int64)
    n_split = 0
    for u in range(n_uniq):
        vals = labels[inverse == u]
        real = vals[vals != -1]
        if real.size == 0:
            out[u] = -1
            continue
        v, c = np.unique(real, return_counts=True)
        out[u] = int(v[c.argmax()])
        if v.size > 1:
            n_split += 1
    return out, n_split


# ---------------------------------------------------------------------------
# Matrices de costo
# ---------------------------------------------------------------------------


def constant_costs(val: float = 2.0, n: int = N_SYMBOLS) -> np.ndarray:
    C = np.full((n, n), val)
    np.fill_diagonal(C, 0.0)
    return C


def trate_costs(X: np.ndarray, weights: np.ndarray, n: int = N_SYMBOLS) -> np.ndarray:
    "Costos de sustitución TRATE: c(i,j) = 2 - p(i|j) - p(j|i), tasas de transición globales ponderadas."
    P = np.zeros((n, n))
    for t in range(X.shape[1] - 1):
        np.add.at(P, (X[:, t], X[:, t + 1]), weights.astype(float))
    row = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row, out=np.zeros_like(P), where=row > 0)
    C = 2.0 - P - P.T
    np.fill_diagonal(C, 0.0)
    return np.maximum(C, 0.0)


def position_freq_costs(X: np.ndarray, weights: np.ndarray, n: int = N_SYMBOLS) -> np.ndarray:
    "Costos por posición para DHD: (T, n, n) con c_t(i,j) = 2 - f_t(i) - f_t(j)."
    T = X.shape[1]
    W = weights.sum()
    C = np.zeros((T, n, n))
    for t in range(T):
        f = np.bincount(X[:, t], weights=weights.astype(float), minlength=n) / W
        c = 2.0 - f[:, None] - f[None, :]
        np.fill_diagonal(c, 0.0)
        C[t] = np.maximum(c, 0.0)
    return C


# ---------------------------------------------------------------------------
# Distancias (todas devuelven una matriz simétrica (U, U))
# ---------------------------------------------------------------------------


def hamming(X: np.ndarray, sub: np.ndarray | None = None) -> np.ndarray:
    "method='HAM': nº de posiciones distintas, o suma de costos de sustitución si se pasa `sub` (n, n)."
    U, T = X.shape
    D = np.zeros((U, U))
    for t in range(T):
        col = X[:, t]
        ne = col[:, None] != col[None, :]
        D += ne if sub is None else np.where(ne, sub[np.ix_(col, col)], 0.0)
    return D


def dhd(X: np.ndarray, cost_by_pos: np.ndarray) -> np.ndarray:
    "Hamming dinámico: suma sobre posiciones de c_t(a_t, b_t). `cost_by_pos` es (T, n, n)."
    U, T = X.shape
    D = np.zeros((U, U))
    for t in range(T):
        col = X[:, t]
        D += cost_by_pos[t][np.ix_(col, col)]
    return D


def om(X: np.ndarray, sub: np.ndarray, indel: float) -> np.ndarray:
    """method='OM': alineación global (Needleman-Wunsch) que minimiza costo, con
    `sub` (n, n) de sustitución e `indel` de inserción/borrado. Vectorizado sobre
    el eje de referencia -- O(U²·T²), unos segundos a U~1000.
    """
    U, T = X.shape
    base = np.arange(T + 1) * indel
    # SX[s, u, t] = sub[s, X[u, t]] -- costo de sustituir el estado `s` por el que
    # la secuencia `u` tiene en la posición `t`. Precalcularlo evita reindexar
    # `sub` U·T·T veces dentro del triple loop.
    SX = sub[:, X]  # (n_symbols, U, T)
    D = np.zeros((U, U))
    for i in range(U):
        a = X[i]
        prev = np.tile(base, (U, 1))  # (U, T+1) -- fila 0 de la DP
        for ii in range(1, T + 1):
            cur = np.empty((U, T + 1))
            cur[:, 0] = ii * indel
            row_sub = SX[int(a[ii - 1])]  # (U, T) -- costo de a[ii-1] contra cada posición de cada ref
            for j in range(1, T + 1):
                cur[:, j] = np.minimum(
                    np.minimum(prev[:, j - 1] + row_sub[:, j - 1], prev[:, j] + indel),
                    cur[:, j - 1] + indel,
                )
            prev = cur
        D[i] = prev[:, T]
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def build_distance(
    X: np.ndarray, weights: np.ndarray, method: str = "om", sub: str = "trate"
) -> tuple[np.ndarray, dict]:
    "Despacho: method in {ham, dhd, om}; sub in {trate, constant} (ignorado por dhd). Devuelve (D, info)."
    if method == "ham":
        C = trate_costs(X, weights) if sub == "trate" else constant_costs() if sub == "constant" else None
        return hamming(X, C), {"method": "ham", "sub": sub if C is not None else "none"}
    if method == "dhd":
        return dhd(X, position_freq_costs(X, weights)), {"method": "dhd", "sub": "position_freq"}
    if method == "om":
        C = trate_costs(X, weights) if sub == "trate" else constant_costs()
        indel = float(C.max() / 2.0)
        return om(X, C, indel), {"method": "om", "sub": sub, "indel": indel}
    raise ValueError(f"method desconocido: {method!r}")


# ---------------------------------------------------------------------------
# Calidad de partición
# ---------------------------------------------------------------------------


def _weighted_ss(D: np.ndarray, w: np.ndarray) -> float:
    "Suma de cuadrados de discrepancia ponderada: (1/2W) ΣΣ wᵢ wⱼ Dᵢⱼ (Batagelj / Studer)."
    W = w.sum()
    return float((w[:, None] * w[None, :] * D).sum() / (2.0 * W)) if W > 0 else float("nan")


def pseudo_r2(D: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    """Análisis de discrepancia (equivalente a `TraMineR::dissassoc`): qué fracción
    de la discrepancia total de las trayectorias explica la partición.

    `pseudo_r2` = 1 - SS_within / SS_total ; `pseudo_f` = MS_between / MS_within.
    Las secuencias con `labels == -1` (ruido) se excluyen; `coverage` es la
    fracción de peso que queda.
    """
    m = labels != -1
    if m.sum() < 2:
        return {"pseudo_r2": float("nan"), "pseudo_f": float("nan"), "coverage": float(m.mean())}
    Dm, lab, w = D[np.ix_(m, m)], labels[m], weights[m].astype(float)
    W = w.sum()
    ss_t = _weighted_ss(Dm, w)
    groups = np.unique(lab)
    ss_w = sum(_weighted_ss(Dm[np.ix_(lab == g, lab == g)], w[lab == g]) for g in groups)
    k = len(groups)
    r2 = 1.0 - ss_w / ss_t if ss_t > 0 else float("nan")
    f = ((ss_t - ss_w) / (k - 1)) / (ss_w / (W - k)) if k > 1 and W > k and ss_w > 0 else float("nan")
    return {"pseudo_r2": float(r2), "pseudo_f": float(f), "coverage": float(m.mean()), "k": int(k)}


def asw(D: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    """Average Silhouette Width en el espacio de secuencias, ponderada por conteo.

    Para cada secuencia distinta: `a` = distancia media (ponderada) a su propio
    cluster, `b` = mínima distancia media a otro cluster, `s = (b-a)/max(a,b)`.
    Devuelve la ASW global y el promedio por cluster.
    """
    m = labels != -1
    if m.sum() < 2:
        return {"asw": float("nan"), "by_cluster": {}}
    Dm, lab, w = D[np.ix_(m, m)], labels[m], weights[m].astype(float)
    groups = np.unique(lab)
    if len(groups) < 2:
        return {"asw": float("nan"), "by_cluster": {}}

    grp_mask = {g: (lab == g) for g in groups}
    grp_w = {g: w[grp_mask[g]].sum() for g in groups}
    # distancia media ponderada de cada punto a cada grupo
    mean_to = {g: (Dm[:, grp_mask[g]] * w[grp_mask[g]][None, :]).sum(axis=1) for g in groups}

    s = np.zeros(len(lab))
    for idx in range(len(lab)):
        gi = lab[idx]
        denom_a = grp_w[gi] - w[idx]
        a = mean_to[gi][idx] / denom_a if denom_a > 0 else 0.0
        b = min(mean_to[g][idx] / grp_w[g] for g in groups if g != gi)
        s[idx] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    by_cluster = {int(g): float((s[grp_mask[g]] * w[grp_mask[g]]).sum() / grp_w[g]) for g in groups}
    return {"asw": float((s * w).sum() / w.sum()), "by_cluster": by_cluster}


# ---------------------------------------------------------------------------
# Secuencias representativas (seqrplot, criterio de densidad de vecindad)
# ---------------------------------------------------------------------------


def representative_sequences(
    D: np.ndarray, weights: np.ndarray, radius_frac: float = 0.10, coverage: float = 0.25, max_reps: int = 8
) -> dict:
    """Conjunto representativo por densidad de vecindad (`TraMineR::seqrplot`,
    criterio `dist`): se elige de forma codiciosa la secuencia con más peso de
    vecinos dentro de un radio (`radius_frac` * distancia máxima), se marca su
    vecindario como cubierto, y se repite hasta cubrir `coverage` del peso del
    cluster o llegar a `max_reps`.

    Devuelve los índices representativos (posiciones dentro de `D`), la cobertura
    lograda, y la distancia media ponderada de cada secuencia a su representante
    más cercano (estadístico de calidad del conjunto).
    """
    U = len(weights)
    if U == 0:
        return {"reps": [], "coverage": 0.0, "mean_dist": float("nan")}
    if U == 1:
        return {"reps": [{"idx": 0, "covered_weight": 1.0}], "coverage": 1.0, "mean_dist": 0.0}

    W = weights.sum()
    dmax = float(D.max())
    radius = radius_frac * dmax if dmax > 0 else 0.0
    within = D <= radius  # (U, U)
    covered = np.zeros(U, dtype=bool)
    reps: list[dict] = []

    while len(reps) < max_reps and weights[covered].sum() < coverage * W:
        density = (within & ~covered[None, :]).astype(float) @ weights
        density[[r["idx"] for r in reps]] = -1.0
        i = int(density.argmax())
        if density[i] <= 0:
            break
        newly = within[i] & ~covered
        reps.append({"idx": i, "covered_weight": float(weights[newly].sum() / W)})
        covered |= newly

    if reps:
        nearest = D[:, [r["idx"] for r in reps]].min(axis=1)
        mean_dist = float((nearest * weights).sum() / W)
    else:
        mean_dist = float("nan")
    return {"reps": reps, "coverage": float(weights[covered].sum() / W), "mean_dist": mean_dist}


def seq_text(tokens: np.ndarray) -> str:
    return "-".join(VOCAB_NAMES[int(t)] for t in tokens)


# ---------------------------------------------------------------------------
# Smoke test -- `python scripts/check_typology.py`
# ---------------------------------------------------------------------------


def _smoke() -> None:
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(3000):
        sw = rng.integers(3, 20)
        seqs.append("-".join(["F"] * sw + ["A"] * (23 - sw)))
    for _ in range(1500):
        seqs.append("-".join(["G"] * 12 + ["Sh"] * 11))
    seqs = pd.Series(seqs)

    uniqX, inv, cnt = distinct(seqs)
    print(f"{len(seqs)} secuencias, {len(uniqX)} distintas")

    D, info = build_distance(uniqX, cnt, method="om", sub="trate")
    print("OM:", info, "rango D:", round(float(D.min()), 2), "-", round(float(D.max()), 2))
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)

    # partición perfecta (F->A vs G->Sh) debe dar pseudo_r2 alto
    labels_u = np.array([0 if "F" in seq_text(x) else 1 for x in uniqX])
    pr = pseudo_r2(D, labels_u, cnt)
    print("pseudo_r2 (partición real):", round(pr["pseudo_r2"], 3), "pseudo_f:", round(pr["pseudo_f"], 1))
    assert pr["pseudo_r2"] > 0.4, pr

    a = asw(D, labels_u, cnt)
    print("ASW:", round(a["asw"], 3), "por cluster:", {k: round(v, 2) for k, v in a["by_cluster"].items()})

    rep = representative_sequences(D[np.ix_(labels_u == 0, labels_u == 0)], cnt[labels_u == 0])
    print("representantes cluster 0:", len(rep["reps"]), "cobertura", round(rep["coverage"], 2))
    print("seqdist: smoke test OK")


if __name__ == "__main__":
    _smoke()
