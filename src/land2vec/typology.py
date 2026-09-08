"""Interpretación de las tipologías de trayectoria (clustering v2), al estilo del
análisis de secuencias de TraMineR.

`scripts/tune_clustering.py --select` deja seis particiones sobre las 107k
trayectorias con transición -- la matriz de 3 granularidades (fina / media /
gruesa) x 2 familias (HDBSCAN / no-HDBSCAN), ver `docs/v2_autoencoder_training.md`
§7.2. Cada cluster viene descrito con muy poco: su trayectoria prototípica
(centroide en `z` crudo, decodificado), su `prototype_fidelity`, y los vecinos
reales más cercanos al centroide. Eso alcanza para *validar* la partición pero no
para *leerla*.

Este módulo aporta el paso siguiente al clustering en análisis de secuencias:
describir cada grupo en el espacio de las secuencias mismas -- cuándo cambian, a
qué, con qué sincronía, cuán homogéneo es el grupo -- y ponerle una etiqueta
legible (`F»A · monotónica · ~2008 · deforestación para agricultura`).

Todas las primitivas operan sobre `X`, una matriz `(n, T)` de ids de token
(`land2vec.tokenizer.Tokenizer.encode`) con las secuencias de los miembros de un
cluster, y son O(n·T): no se construye ninguna matriz de disimilitud entre pares
(decisión de diseño -- ver `position_diversity`, que da compacidad en espacio de
secuencias sin salir de ese presupuesto). El productor es
`scripts/describe_clusters.py`; el notebook `notebooks/cluster_evaluation.ipynb`
§5 consume su salida.

Equivalencias con TraMineR (funciones de su User's Guide):

| acá                      | TraMineR   | qué es                                        |
|--------------------------|------------|----------------------------------------------|
| `state_distribution`     | `seqdplot` | cronograma: proporción de cada estado por año|
| `modal_sequence`         | `seqmsplot`| estado más común en cada posición            |
| `transversal_entropy`    | `seqHtplot`| entropía de la distribución transversal      |
| `mean_time_per_state`    | `seqmtplot`| años promedio en cada estado                 |
| `top_distinct_sequences` | `seqfplot` | secuencias distintas más frecuentes          |
| `transition_rates`       | `seqtrate` | P(estado en t+1 | estado en t)               |
| `sequence_indices`       | `seqici`/`seqient`/`seqtransn` | índices longitudinales      |

Omisión consciente: la **turbulencia** de Elzinga (`seqST`) -- requiere contar
subsecuencias distintas del DSS (una DP aparte) y aporta poco sobre el índice de
complejidad cuando todas las secuencias tienen el mismo largo (23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from land2vec.tokenizer import Tokenizer

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

T_DEFAULT = 23
N_SYMBOLS = len(Tokenizer.VOCAB)  # 11: [UNK] + 10 estados
YEAR_0 = 2000

# Paleta única de estados, indexada por id de token -- usada en todos los gráficos
# de este módulo y del notebook. [UNK]/relleno en gris.
STATE_COLORS = {
    "[UNK]": "#d9d9d9",
    "A": "#e6c229",   # agricultura -- dorado
    "F": "#1b7837",   # forestal -- verde oscuro
    "G": "#a6d96a",   # pastizal -- verde claro
    "Wt": "#2b8cbe",  # humedal -- celeste
    "U": "#d7301f",   # urbano -- rojo
    "Sh": "#b8860b",  # arbustal -- ocre
    "Sp": "#dfc27d",  # esparso -- arena
    "B": "#8c6d31",   # suelo desnudo -- marrón
    "Wa": "#08519c",  # agua -- azul
    "Nd": "#bdbdbd",  # sin dato -- gris
}
STATE_COLOR_LIST = [STATE_COLORS[Tokenizer.REVERSE_VOCAB[i]] for i in range(N_SYMBOLS)]
NOISE_COLOR = "#f0f0f0"

# Glosas agronómicas de los pares (estado inicial, estado final) más frecuentes en
# el área de estudio (Chaco / Santiago del Estero / frontera). Se usan para la
# `glosa` de `auto_label`; el fallback es genérico ("X → Y").
GLOSSES: dict[tuple[str, str], str] = {
    ("F", "A"): "deforestación para agricultura",
    ("F", "G"): "deforestación para pastura",
    ("F", "Sh"): "degradación forestal",
    ("F", "U"): "deforestación para urbanización",
    ("F", "Wt"): "anegamiento de bosque",
    ("Sh", "A"): "habilitación de arbustal para agricultura",
    ("Sh", "G"): "arbustal a pastizal",
    ("Sh", "F"): "recuperación forestal desde arbustal",
    ("Sh", "U"): "urbanización sobre arbustal",
    ("G", "A"): "intensificación: pastizal a agricultura",
    ("G", "F"): "avance del bosque sobre pastizal",
    ("G", "Sh"): "arbustización de pastizal",
    ("G", "U"): "urbanización sobre pastizal",
    ("A", "U"): "urbanización sobre área agrícola",
    ("A", "F"): "abandono agrícola / reforestación",
    ("A", "Sh"): "abandono agrícola con arbustización",
    ("A", "G"): "reversión a pastura",
    ("Wt", "G"): "desecación de humedal a pastizal",
    ("Wt", "A"): "avance agrícola sobre humedal",
    ("Wt", "F"): "humedal a bosque",
    ("Wt", "Sh"): "desecación de humedal a arbustal",
    ("G", "Wt"): "anegamiento de pastizal",
    ("A", "Wt"): "anegamiento de área agrícola",
    ("B", "Sp"): "revegetación de suelo desnudo",
    ("Sp", "B"): "pérdida de cobertura esparsa",
    ("B", "G"): "revegetación de suelo desnudo a pastizal",
    ("Sp", "Sh"): "densificación de cobertura esparsa",
    ("Sh", "Sp"): "ralentización de arbustal",
}


# ---------------------------------------------------------------------------
# Utilidades base
# ---------------------------------------------------------------------------


def encode_matrix(seqs) -> np.ndarray:
    "Serie/iterable de strings 'F-Sh-F-...' -> matriz (n, T) de ids de token."
    return np.stack([np.asarray(Tokenizer.encode(s), dtype=np.int64) for s in seqs])


def _names(ids) -> list[str]:
    return [Tokenizer.REVERSE_VOCAB[int(i)] for i in ids]


def _shannon(p: np.ndarray, axis: int = -1) -> np.ndarray:
    "Entropía de Shannon (nats) de una distribución sobre `axis`, tolerando ceros."
    p = np.where(p > 0, p, 1.0)  # 0·log0 = 0
    return -(p * np.log(p)).sum(axis=axis)


def cluster_alphabet_size(X: np.ndarray) -> int:
    "Cantidad de estados distintos que aparecen en todo el cluster (para normalizar entropías)."
    return int(len(np.unique(X)))


# ---------------------------------------------------------------------------
# Bloque descriptivo (batería estilo TraMineR)
# ---------------------------------------------------------------------------


def state_distribution(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    "Cronograma (`seqdplot`): matriz (T, n_symbols) con la proporción de cada estado por posición."
    T = X.shape[1]
    counts = np.stack([np.bincount(X[:, t], minlength=n_symbols) for t in range(T)]).astype(np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def modal_sequence(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> tuple[np.ndarray, np.ndarray]:
    "Secuencia modal (`seqmsplot`): (tokens (T,), share (T,)) -- estado más común por posición y su frecuencia relativa."
    dist = state_distribution(X, n_symbols)
    return dist.argmax(axis=1), dist.max(axis=1)


def transversal_entropy(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    """Entropía transversal (`seqHtplot`): (T,) con la entropía de Shannon de la
    distribución de estados en cada posición, normalizada a [0, 1] por `log(A)`
    con `A` = cantidad de estados distintos en todo el cluster. 0 = todos los
    miembros comparten estado en ese año; cerca de 1 = máxima dispersión. El pico
    marca el año de la transición menos sincronizada del cluster.
    """
    A = cluster_alphabet_size(X)
    dist = state_distribution(X, n_symbols)
    h = _shannon(dist, axis=1)
    return h / np.log(A) if A > 1 else np.zeros(X.shape[1])


def position_diversity(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    """Diversidad de Gini `1 - Σpᵢ²` por posición: (T,). Es la probabilidad de que
    dos miembros del cluster tomados al azar difieran en ese año.

    Su suma sobre las T posiciones **es** la distancia de Hamming media entre
    pares dentro del cluster (aproximación con reemplazo; el sesgo de muestra
    finita es `n/(n-1)`, despreciable a estos `n`). Da compacidad en el espacio
    de secuencias sin construir ninguna matriz de disimilitud entre pares.
    """
    dist = state_distribution(X, n_symbols)
    return 1.0 - (dist ** 2).sum(axis=1)


def mean_time_per_state(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    "Tiempo medio por estado (`seqmtplot`): (n_symbols,) con los años promedio que un miembro pasa en cada estado."
    return state_distribution(X, n_symbols).sum(axis=0)


@dataclass
class DistinctSequence:
    tokens: tuple[int, ...]
    text: str
    count: int
    coverage: float  # fracción del cluster


def top_distinct_sequences(X: np.ndarray, n: int = 5) -> list[DistinctSequence]:
    "Secuencias distintas más frecuentes del cluster (`seqfplot`)."
    uniq, counts = np.unique(X, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:n]
    total = X.shape[0]
    return [
        DistinctSequence(
            tokens=tuple(int(t) for t in uniq[i]),
            text="-".join(_names(uniq[i])),
            count=int(counts[i]),
            coverage=float(counts[i] / total),
        )
        for i in order
    ]


def top_sequences_to_coverage(
    X: np.ndarray, target: float = 0.8, cap: int = 40
) -> tuple[list[DistinctSequence], float]:
    """Las secuencias distintas más frecuentes que, acumuladas, cubren al menos
    `target` de los miembros del cluster (o hasta `cap` secuencias). Devuelve la
    lista y la cobertura acumulada realmente alcanzada.
    """
    uniq, counts = np.unique(X, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = X.shape[0]
    out: list[DistinctSequence] = []
    cum = 0.0
    for i in order:
        cov = float(counts[i] / total)
        out.append(DistinctSequence(tuple(int(t) for t in uniq[i]), "-".join(_names(uniq[i])), int(counts[i]), cov))
        cum += cov
        if cum >= target or len(out) >= cap:
            break
    return out, cum


def transition_rates(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    """Matriz de tasas de transición (`seqtrate`): (n_symbols, n_symbols) con
    `P(estado j en t+1 | estado i en t)`, agregada sobre todas las posiciones.
    Las filas de estados nunca observados quedan en cero.
    """
    M = np.zeros((n_symbols, n_symbols), dtype=np.float64)
    for t in range(X.shape[1] - 1):
        np.add.at(M, (X[:, t], X[:, t + 1]), 1.0)
    row = M.sum(axis=1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def sequence_indices(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> pd.DataFrame:
    """Índices longitudinales, uno por secuencia (n filas):

    - `n_transiciones`  -- cambios de estado (`seqtransn`)
    - `n_estados`       -- estados distintos visitados
    - `entropia_long`   -- entropía de los tiempos-en-estado, normalizada por `log(A)`
      del cluster (`seqient`)
    - `indice_complejidad` -- `sqrt( (n_transiciones/(T-1)) · entropia_long )` (`seqici`,
      Gabadinho et al. 2010)
    - `spell_medio`, `spell_mas_largo` -- duración de los tramos de estado constante
    """
    n, T = X.shape
    A = cluster_alphabet_size(X)
    changes = X[:, 1:] != X[:, :-1]
    n_trans = changes.sum(axis=1)

    per_state = np.stack([(X == s).sum(axis=1) for s in range(n_symbols)], axis=1).astype(np.float64)
    n_states = (per_state > 0).sum(axis=1)
    p = per_state / T
    h = _shannon(p, axis=1)
    h_norm = h / np.log(A) if A > 1 else np.zeros(n)

    complexity = np.sqrt((n_trans / (T - 1)) * h_norm)
    spell_mean = T / (n_trans + 1)
    spell_max = np.array([_longest_run(row) for row in X])

    return pd.DataFrame(
        {
            "n_transiciones": n_trans.astype(int),
            "n_estados": n_states.astype(int),
            "entropia_long": h_norm,
            "indice_complejidad": complexity,
            "spell_medio": spell_mean,
            "spell_mas_largo": spell_max.astype(int),
        }
    )


def _longest_run(row: np.ndarray) -> int:
    best = run = 1
    for i in range(1, len(row)):
        run = run + 1 if row[i] == row[i - 1] else 1
        best = max(best, run)
    return best


def dss(tokens) -> tuple[int, ...]:
    "Distinct Successive States: colapsa repeticiones consecutivas. F-F-F-A-A -> (F, A)."
    out: list[int] = []
    for t in tokens:
        t = int(t)
        if not out or out[-1] != t:
            out.append(t)
    return tuple(out)


# ---------------------------------------------------------------------------
# Etiquetado automático
# ---------------------------------------------------------------------------


@dataclass
class ClusterLabel:
    dss_modal: tuple[int, ...]
    dss_text: str          # "F»A", "A»Sh»A", "F"
    forma: str             # estable / monotónica / oscilante / múltiple
    anio_cambio: int | None
    inicio: str
    fin: str
    glosa: str
    etiqueta: str          # string completo ensamblado

    def to_dict(self) -> dict:
        return {
            "dss_text": self.dss_text,
            "forma": self.forma,
            "anio_cambio": self.anio_cambio,
            "inicio": self.inicio,
            "fin": self.fin,
            "glosa": self.glosa,
            "etiqueta": self.etiqueta,
        }


def _forma(d: tuple[int, ...]) -> str:
    if len(d) == 1:
        return "estable"
    if len(d) == 2:
        return "monotónica"
    if len(set(d)) < len(d):
        return "oscilante"
    return "múltiple"


def _transition_year(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> int | None:
    """Año del cambio más marcado del cronograma: la posición t (1..T-1) que
    maximiza la distancia de variación total entre la distribución en t-1 y en t.
    `None` si el cluster es prácticamente estático (todos los saltos < 0.05).
    """
    dist = state_distribution(X, n_symbols)
    tv = 0.5 * np.abs(np.diff(dist, axis=0)).sum(axis=1)  # (T-1,)
    t = int(tv.argmax())
    if tv[t] < 0.05:
        return None
    return YEAR_0 + t + 1


def auto_label(X: np.ndarray, n_symbols: int = N_SYMBOLS) -> ClusterLabel:
    """Etiqueta legible para un cluster, derivada de su secuencia modal y su
    cronograma. Determinista, sin parámetros de ajuste.

    Ejemplos: `F»A · monotónica · ~2008 · deforestación para agricultura`,
    `A»Sh»A · oscilante · ~2011 · abandono agrícola con arbustización`,
    `Wt · estable · humedal sin cambio neto`.
    """
    modal, _ = modal_sequence(X, n_symbols)
    d = dss(modal)
    forma = _forma(d)
    inicio, fin = Tokenizer.REVERSE_VOCAB[d[0]], Tokenizer.REVERSE_VOCAB[d[-1]]
    dss_text = "»".join(_names(d))

    if forma == "estable":
        glosa = f"{inicio} sin cambio neto"
        etiqueta = f"{dss_text} · estable · {glosa}"
        return ClusterLabel(d, dss_text, forma, None, inicio, fin, glosa, etiqueta)

    anio = _transition_year(X, n_symbols)
    glosa = GLOSSES.get((inicio, fin), f"{inicio} → {fin}")
    if inicio == fin:  # oscilación que vuelve al origen
        mid = Tokenizer.REVERSE_VOCAB[d[1]] if len(d) > 1 else fin
        glosa = GLOSSES.get((inicio, mid), f"oscilación {inicio}↔{mid}")
    anio_txt = f" · ~{anio}" if anio is not None else ""
    etiqueta = f"{dss_text} · {forma}{anio_txt} · {glosa}"
    return ClusterLabel(d, dss_text, forma, anio, inicio, fin, glosa, etiqueta)


# ---------------------------------------------------------------------------
# Firma por cluster (una fila del CSV de tipología)
# ---------------------------------------------------------------------------


def prototype_agreement(modal_tokens: np.ndarray, proto_tokens: np.ndarray) -> float:
    "Fracción de posiciones donde la secuencia modal (libre de modelo) coincide con el prototipo decodificado."
    return float((np.asarray(modal_tokens) == np.asarray(proto_tokens)).mean())


def cluster_signature(
    X: np.ndarray,
    zones: np.ndarray | None = None,
    proto_tokens: np.ndarray | None = None,
    n_symbols: int = N_SYMBOLS,
) -> dict:
    """Resumen escalar de un cluster -- una fila para `typology_{suffix}.csv`.

    `X`      : (n, T) tokens de los miembros
    `zones`  : (n,) nombre de zona por miembro (opcional; agrega composición por zona)
    `proto_tokens` : (T,) prototipo decodificado del cluster (opcional; agrega la
                     tasa de coincidencia modal-vs-prototipo, un chequeo del decoder)
    """
    n, T = X.shape
    modal, modal_share = modal_sequence(X, n_symbols)
    idx = sequence_indices(X, n_symbols)
    tops = top_distinct_sequences(X, n=5)
    lab = auto_label(X, n_symbols)
    div = position_diversity(X, n_symbols)

    row = {
        "n": n,
        "modal_seq": "-".join(_names(modal)),
        "modal_share_medio": float(modal_share.mean()),
        "cobertura_top1": tops[0].coverage,
        "cobertura_top5": float(sum(t.coverage for t in tops)),
        "n_secuencias_distintas": int(np.unique(X, axis=0).shape[0]),
        "n_transiciones_medio": float(idx["n_transiciones"].mean()),
        "indice_complejidad_medio": float(idx["indice_complejidad"].mean()),
        "entropia_long_media": float(idx["entropia_long"].mean()),
        "hamming_media_intra": float(div.sum()),
        "entropia_transv_pico": float(transversal_entropy(X, n_symbols).max()),
        **lab.to_dict(),
    }
    if proto_tokens is not None:
        row["proto_seq"] = "-".join(_names(proto_tokens))
        row["modal_vs_proto"] = prototype_agreement(modal, proto_tokens)
    if zones is not None:
        vc = pd.Series(zones).value_counts(normalize=True)
        row["zona_dominante"] = str(vc.index[0])
        row["zona_dominante_share"] = float(vc.iloc[0])
        row["n_zonas"] = int((vc > 0).sum())
    return row


# ---------------------------------------------------------------------------
# Comparación entre corridas (solo por etiquetas -- las 6 comparten filas y orden)
# ---------------------------------------------------------------------------


def _pairwise_agreement(a: np.ndarray, b: np.ndarray, exclude_noise: bool) -> tuple[float, float, float]:
    "(ARI, NMI, cobertura) entre dos particiones alineadas por posición."
    if exclude_noise:
        m = (a != -1) & (b != -1)
        a, b, cov = a[m], b[m], float(m.mean())
    else:
        cov = 1.0
    if len(a) < 2 or len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return float("nan"), float("nan"), cov
    return adjusted_rand_score(a, b), normalized_mutual_info_score(a, b), cov


def crossrun_agreement(labels_by_run: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Matrices de acuerdo entre las particiones de las corridas.

    Devuelve cuatro DataFrames R×R (`ari`, `nmi`, `cobertura`, `ari_con_ruido`):

    - `ari` / `nmi` / `cobertura`: medida **primaria** -- se calcula sobre las
      filas que **ninguna** de las dos corridas dejó como ruido (`-1`), y
      `cobertura` es qué fracción del total es esa intersección. Con fracciones de
      ruido que van de 0% a 24% entre corridas, esa cobertura no es un detalle:
      hay que leer el ARI junto a ella.
    - `ari_con_ruido`: medida **secundaria** -- trata `-1` como una etiqueta más.

    (Mismo tipo de doble lectura del `-1` que documenta §7.2 para los archivos de
    etiquetas.)
    """
    runs = list(labels_by_run)
    R = len(runs)
    ari = np.eye(R)
    nmi = np.eye(R)
    cov = np.ones((R, R))
    ari_n = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            a, b = labels_by_run[runs[i]], labels_by_run[runs[j]]
            r_ari, r_nmi, r_cov = _pairwise_agreement(a, b, exclude_noise=True)
            ari[i, j] = ari[j, i] = r_ari
            nmi[i, j] = nmi[j, i] = r_nmi
            cov[i, j] = cov[j, i] = r_cov
            ari_n[i, j] = ari_n[j, i] = _pairwise_agreement(a, b, exclude_noise=False)[0]
    mk = lambda M: pd.DataFrame(M, index=runs, columns=runs)
    return {"ari": mk(ari), "nmi": mk(nmi), "cobertura": mk(cov), "ari_con_ruido": mk(ari_n)}


def nesting_table(fine: np.ndarray, coarse: np.ndarray) -> pd.DataFrame:
    """Para cada cluster de la partición gruesa: su tamaño, cuántos clusters finos
    lo componen, y su **pureza** (fracción de sus miembros en el cluster fino
    modal). Pureza alta y pocos finos por grueso = la tipología gruesa anida a la
    fina; pureza baja = las dos particiones cortan distinto.

    El ruido (`-1`) de cualquiera de las dos se excluye antes de tabular.
    """
    m = (fine != -1) & (coarse != -1)
    f, c = fine[m], coarse[m]
    rows = []
    for cid in sorted(set(c.tolist())):
        sub = f[c == cid]
        vc = pd.Series(sub).value_counts()
        rows.append(
            {
                "cluster_grueso": cid,
                "n": int(sub.size),
                "n_finos": int(vc.size),
                "pureza": float(vc.iloc[0] / sub.size),
                "fino_modal": int(vc.index[0]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gráficos (matplotlib -- import perezoso; el módulo no lo necesita para el resto)
# ---------------------------------------------------------------------------

YEARS = np.arange(YEAR_0, YEAR_0 + T_DEFAULT)


def plot_chronogram(ax: "plt.Axes", dist: np.ndarray, years: np.ndarray | None = None, legend: bool = False) -> None:
    "Cronograma apilado (`seqdplot`): `dist` es (T, N_SYMBOLS) de `state_distribution`."
    years = YEARS[: dist.shape[0]] if years is None else years
    bottom = np.zeros(dist.shape[0])
    for s in range(dist.shape[1]):
        h = dist[:, s]
        if h.sum() == 0:
            continue
        ax.bar(years, h, bottom=bottom, width=1.0, color=STATE_COLOR_LIST[s],
               label=Tokenizer.REVERSE_VOCAB[s], linewidth=0)
        bottom += h
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)


def plot_token_strip(ax: "plt.Axes", tokens: np.ndarray, years: np.ndarray | None = None, ylabel: str = "") -> None:
    "Una secuencia (tokens (T,)) como tira de celdas de color -- para la modal y el prototipo decodificado."
    years = YEARS[: len(tokens)] if years is None else years
    for i, t in enumerate(tokens):
        ax.add_patch(plt_rect((years[i] - 0.5, 0), 1, 1, STATE_COLOR_LIST[int(t)]))
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", fontsize=8)


def plt_rect(xy, w, h, color):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, w, h, facecolor=color, edgecolor="white", linewidth=0.3)


def plot_cluster_panel(X: np.ndarray, proto_tokens: np.ndarray | None, title: str) -> "plt.Figure":
    """Panel de un cluster: cronograma + entropía transversal + modal vs. prototipo
    decodificado + tiempo medio por estado. Devuelve la figura (el caller la guarda o muestra).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 0.5, 0.5, 1.4], width_ratios=[3, 1], hspace=0.5, wspace=0.25)

    ax_chr = fig.add_subplot(gs[0, 0])
    plot_chronogram(ax_chr, state_distribution(X), legend=True)
    ax_chr.set_title(title, fontsize=10, loc="left")
    ax_chr.set_ylabel("proporción")

    ax_ent = fig.add_subplot(gs[0, 1])
    ax_ent.plot(transversal_entropy(X), YEARS, color="#333", lw=1.2)
    ax_ent.set_xlim(0, 1)
    ax_ent.set_ylim(YEARS[-1], YEARS[0])
    ax_ent.set_title("entropía transv.", fontsize=8)
    ax_ent.tick_params(labelsize=7)

    modal, _ = modal_sequence(X)
    ax_m = fig.add_subplot(gs[1, 0])
    plot_token_strip(ax_m, modal, ylabel="modal")
    ax_m.set_xticks([])
    if proto_tokens is not None:
        ax_p = fig.add_subplot(gs[2, 0])
        plot_token_strip(ax_p, proto_tokens, ylabel="prototipo")
        ax_p.set_xticks([])

    ax_mt = fig.add_subplot(gs[3, 0])
    mt = mean_time_per_state(X)
    present = np.flatnonzero(mt > 0.05)
    ax_mt.bar(range(len(present)), mt[present], color=[STATE_COLOR_LIST[s] for s in present])
    ax_mt.set_xticks(range(len(present)))
    ax_mt.set_xticklabels([Tokenizer.REVERSE_VOCAB[s] for s in present], fontsize=8)
    ax_mt.set_ylabel("años medios", fontsize=8)

    ax_txt = fig.add_subplot(gs[3, 1])
    ax_txt.axis("off")
    tops = top_distinct_sequences(X, n=3)
    lab = auto_label(X)
    txt = f"{lab.etiqueta}\n\ntop secuencias distintas:\n" + "\n".join(
        f"  {t.coverage:.0%}  {t.text}" for t in tops
    )
    ax_txt.text(0, 1, txt, fontsize=6.5, va="top", family="monospace")
    return fig


def plot_run_atlas(
    chronos: np.ndarray, sizes: np.ndarray, labels: list[str], run_name: str, ncols: int = 8
) -> "plt.Figure":
    """Atlas de una corrida: un cronograma compacto por cluster, ordenados por
    tamaño descendente. `chronos` es (k, T, N_SYMBOLS). Legible como grilla a
    k<=40 y como atlas de texturas a k~120.
    """
    import matplotlib.pyplot as plt

    k = chronos.shape[0]
    order = np.argsort(sizes)[::-1]
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 1.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for pos, cid in enumerate(order):
        ax = axes[pos // ncols][pos % ncols]
        ax.axis("on")
        plot_chronogram(ax, chronos[cid])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{cid} · n={int(sizes[cid]):,}\n{labels[cid]}", fontsize=5.5, loc="left")
    fig.suptitle(f"Atlas de cronogramas — {run_name} (k={k})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_alluvial(ax: "plt.Axes", left: np.ndarray, right: np.ndarray, left_name: str, right_name: str,
                  min_flow: int = 30) -> None:
    """Diagrama aluvial entre dos particiones alineadas por posición (Sankey a mano).
    El ruido (`-1`) de cualquiera de las dos se descarta. Los flujos con menos de
    `min_flow` miembros no se dibujan.
    """
    m = (left != -1) & (right != -1)
    l, r = left[m], right[m]
    lu = sorted(set(l.tolist()))
    ru = sorted(set(r.tolist()))

    def _layout(vals, uniq):
        "Posición vertical de cada cluster, apilado por tamaño (centro de su banda)."
        counts = np.array([(vals == c).sum() for c in uniq], dtype=float)
        order = np.argsort(-counts)
        y, cum = {}, 0.0
        total = counts.sum()
        for oi in order:
            y[uniq[oi]] = cum + counts[oi] / 2 / total
            cum += counts[oi] / total
        return y

    ly, ry = _layout(l, lu), _layout(r, ru)
    for lc in lu:
        for rc in ru:
            flow = int(((l == lc) & (r == rc)).sum())
            if flow < min_flow:
                continue
            ax.plot([0, 1], [ly[lc], ry[rc]], lw=max(0.3, flow / m.sum() * 60),
                    color=STATE_COLOR_LIST[lc % N_SYMBOLS], alpha=0.35, solid_capstyle="round")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_name, right_name], fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"{left_name} → {right_name}", fontsize=9)


def plot_crossrun(ax: "plt.Axes", matrix: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    "Heatmap R×R de acuerdo entre corridas (ARI/NMI)."
    import matplotlib.pyplot as plt

    M = matrix.values
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(matrix.index, fontsize=7)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.6 else "black")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)


# ---------------------------------------------------------------------------
# Smoke test -- `python scripts/check_typology.py`
# ---------------------------------------------------------------------------


def _smoke() -> None:
    rng = np.random.default_rng(0)
    A, F, Sh = Tokenizer.VOCAB["A"], Tokenizer.VOCAB["F"], Tokenizer.VOCAB["Sh"]
    # cluster sintético: F -> A alrededor del año 8, con ruido de bordes
    n = 4000
    switch = 8 + rng.integers(-2, 3, size=n)
    X = np.full((n, T_DEFAULT), F, dtype=np.int64)
    for i in range(n):
        X[i, switch[i]:] = A
    X[rng.integers(0, n, 200), rng.integers(0, T_DEFAULT, 200)] = Sh  # 200 celdas espurias

    dist = state_distribution(X)
    assert np.allclose(dist.sum(axis=1), 1.0), "las filas del cronograma deben sumar 1"

    # identidad: suma de la diversidad de Gini por posición == Hamming media entre pares
    sub = X[:400]
    ham = np.mean([(sub[i] != sub[j]).sum() for i in range(len(sub)) for j in range(i + 1, len(sub))])
    approx = position_diversity(sub).sum()
    assert abs(ham - approx) < 0.05, f"identidad de Hamming rota: {ham:.4f} vs {approx:.4f}"

    lab = auto_label(X)
    assert lab.forma == "monotónica" and lab.dss_text == "F»A", lab
    assert lab.glosa == GLOSSES[("F", "A")], lab
    print("auto_label:", lab.etiqueta)

    sig = cluster_signature(X, zones=np.array(["puna_noa"] * n), proto_tokens=modal_sequence(X)[0])
    print("firma:", {k: sig[k] for k in ("n", "modal_seq", "cobertura_top1", "n_transiciones_medio",
                                         "hamming_media_intra", "modal_vs_proto")})

    tr = transition_rates(X)
    assert np.allclose(tr.sum(axis=1)[[F, A]], 1.0), "las filas observadas de seqtrate deben sumar 1"
    print("typology: smoke test OK")


if __name__ == "__main__":
    _smoke()
