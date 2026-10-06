"""P2: otros métodos de clustering sobre los mismos espacios (plan autoencoder_v3 §4.2).

Compara, con el mismo pipeline que `p2_tipologias.py` (calidad en el espacio de secuencias, estabilidad con
submuestras del 80 % de los píxeles, qué agrupa, control de costura), cinco familias sobre las 4.545
trayectorias dinámicas:

    kmedoides   el de p2_tipologias.py (línea de base; se vuelve a correr acá con el protocolo de estabilidad de este script)
    avg         jerárquico aglomerativo, enlace promedio (UPGMA ponderado por superficie) sobre la distancia del espacio; todos los espacios
    ward        jerárquico de Ward ponderado por superficie sobre las coordenadas (onehot, pca8, ae_d*); no aplica a OM
    kmeans      k-means ponderado por superficie (sklearn, n_init=10) sobre las coordenadas
    gmm         mezcla gaussiana (covarianza completa) ajustada a una muestra de píxeles proporcional a la superficie (sklearn no pondera); pca8 y ae_d*
    hdbscan     HDBSCAN sobre la matriz de distancias, **sin ponderar** (sklearn no admite pesos); k no se fija: para cada k objetivo
                se elige la configuración (min_cluster_size, min_samples, eom/leaf) cuyo n.º de clusters sea el más cercano (desempate: menos ruido). Los tipos sin cluster quedan en -1 (ruido)

Peso = n_px; universos completo y sin_costura. La estabilidad se calcula sobre las trayectorias que quedan
en cada submuestra (adelgazamiento binomial del 80 % de n_px), ARI ponderado contra el ajuste completo, con el ruido
como clase.

Escribe en data/autoencoder_v3/p2/: p2_labels<tag>.csv.gz, p2_metricas<tag>.csv, p2_acuerdo<tag>.csv (tag por defecto `_metodos`).

Uso: python scripts/clustering/p2_metodos.py --om <OM .npy> [--reps 20] [--methods avg ward ...] [--spaces ...]
"""
import argparse
import importlib.util
import itertools
import time
import warnings
from pathlib import Path
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec import seqdist  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

_spec = importlib.util.spec_from_file_location("p2t", ROOT / "scripts" / "clustering" / "p2_tipologias.py")
P2T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2T)

KS, OUT = P2T.KS, P2T.OUT
SPACES = ["om", "onehot", "pca8", "ae_d3", "ae_d8", "ae_d16"]
COORD_SPACES = [s for s in SPACES if s != "om"]
METHODS_SPACES = {"kmedoides": SPACES, "avg": SPACES, "ward": COORD_SPACES, "kmeans": COORD_SPACES,
                  "gmm": ["pca8", "ae_d3", "ae_d8", "ae_d16"], "hdbscan": SPACES}
MCS_GRID = [2, 3, 5, 10, 20, 40]
MS_GRID = [1, 2, 5]


# ---------------------------------------------------------------------------
# Jerárquico ponderado (cadena de vecinos más cercanos + Lance-Williams con masas)
# ---------------------------------------------------------------------------
def hier_tree(D: np.ndarray, mass: np.ndarray, link: str) -> np.ndarray:
    """Dendrograma aglomerativo con masas. `link`: 'average' (D = distancias) o 'ward' (D = distancias euclídeas
    al cuadrado; la altura devuelta es el aumento de la suma de cuadrados). Devuelve merges (n-1, 3): (a, b, altura), a se funde en b. Ambos criterios son reducibles,
    así que la cadena de vecinos da el árbol exacto y ordenar por altura es válido."""
    n = len(mass)
    D = D.astype(np.float64).copy()
    m = mass.astype(np.float64).copy()
    if link == "ward":  # se trabaja con el costo de fusión Δ_ij = m_i m_j / (m_i + m_j) · ||x_i − x_j||², que es lo que actualiza Lance-Williams
        D *= m[:, None] * m[None, :] / (m[:, None] + m[None, :])
    np.fill_diagonal(D, np.inf)
    active = np.ones(n, bool)
    merges, chain = [], []
    while len(merges) < n - 1:
        if not chain:
            chain = [int(np.flatnonzero(active)[0])]
        while True:
            a = chain[-1]
            row = D[a]
            j = int(row.argmin())
            if len(chain) > 1 and row[chain[-2]] <= row[j]:
                j = chain[-2]
            if len(chain) > 1 and j == chain[-2]:
                break
            chain.append(j)
        a, b = chain.pop(), chain.pop()
        d = D[a, b]
        h = d
        ma, mb, mk = m[a], m[b], m
        if link == "ward":
            new = ((ma + mk) * D[a] + (mb + mk) * D[b] - mk * d) / (ma + mb + mk)
        else:
            new = (ma * D[a] + mb * D[b]) / (ma + mb)
        new[~active] = np.inf
        new[a] = new[b] = np.inf
        D[b, :], D[:, b] = new, new
        D[a, :], D[:, a] = np.inf, np.inf
        m[b] = ma + mb
        active[a] = False
        merges.append((a, b, h))
    return np.array(merges)


def cut_tree(merges: np.ndarray, n: int, k: int) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in merges[np.argsort(merges[:, 2], kind="stable")][: n - k]:
        parent[find(int(a))] = find(int(b))
    return pd.factorize(np.array([find(i) for i in range(n)]))[0]


# ---------------------------------------------------------------------------
# Ajuste de un método sobre los tipos con peso > 0; devuelve {k: etiquetas sobre `idx`}
# ---------------------------------------------------------------------------
def fit_method(method, D, coords, w, idx, ks, rng, state):
    "D: distancias del espacio (n, n); coords: (n, d) o None; w: pesos (n,); idx: tipos del ajuste. Etiquetas: -1 = ruido."
    wi = w[idx]
    if method == "kmedoides":
        return {k: P2T.kmedoids(D[np.ix_(idx, idx)], wi, k, rng, n_init=5)[1] for k in ks}
    if method in ("avg", "ward"):
        Dm = D[np.ix_(idx, idx)]
        tree = hier_tree(Dm ** 2 if method == "ward" else Dm, wi, "ward" if method == "ward" else "average")
        return {k: cut_tree(tree, len(idx), k) for k in ks}
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return {k: KMeans(k, n_init=10, random_state=int(rng.integers(1 << 31))).fit(coords[idx], sample_weight=wi).labels_ for k in ks}
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        samp = rng.choice(len(idx), 20000, p=wi / wi.sum())
        out = {}
        for k in ks:
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, max_iter=100, n_init=1,
                                random_state=int(rng.integers(1 << 31))).fit(coords[idx][samp])
            out[k] = g.predict(coords[idx])
        return out
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        Dm = D[np.ix_(idx, idx)].astype(np.float64)
        state.setdefault("hdb", {})  # configuración por k: se busca una vez con el ajuste completo y se reusa
        out = {}

        def run(cfg):
            return HDBSCAN(min_cluster_size=cfg[0], min_samples=cfg[1], cluster_selection_method=cfg[2], metric="precomputed").fit(Dm).labels_
        for k in ks:
            if k not in state["hdb"]:
                cand = []
                for cfg in itertools.product(MCS_GRID, MS_GRID, ("eom", "leaf")):
                    lab = run(cfg)
                    cand.append((abs(len(set(lab) - {-1}) - k), (lab == -1).mean(), cfg, lab))
                best = min(cand, key=lambda c: (c[0], c[1]))
                state["hdb"][k] = best[2]
                out[k] = best[3]
            else:
                out[k] = run(state["hdb"][k])
        return out
    raise ValueError(method)


def medoids(D, w, lab):
    "traj (posición en el array) del medoide de cada etiqueta >= 0, ponderado por w; devuelve array n con el medoide de la etiqueta de cada fila (-1 si ruido)."
    out = np.full(len(lab), -1)
    for g in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero((lab == g) & (w > 0))
        if len(idx):
            out[lab == g] = idx[(D[np.ix_(idx, idx)] * w[idx][None, :]).sum(1).argmin()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--om", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--gmm-reps", type=int, default=10, help="submuestras para la estabilidad del GMM (es el más lento)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS_SPACES))
    ap.add_argument("--spaces", nargs="+", default=SPACES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_metodos", help="sufijo de los CSV de salida")
    ap.add_argument("--ks", nargs="+", type=int, default=KS, help="n.º de clusters (para el barrido de k)")
    ap.add_argument("--solo-completo", action="store_true", help="sólo el universo completo (sin la variante sin costura)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore")

    u, ud, X, Xd, dyn = P2T.dynamic_universe()
    w_fit = P2T.p1c.train_weights(u)
    Dom = np.load(args.om).astype(np.float64)[np.ix_(dyn, dyn)]
    wpx = ud.n_px.values.astype(float)
    universos = {"completo": np.ones(len(ud), bool), "sin_costura": ~ud.costura.values}
    if args.solo_completo:
        universos.pop("sin_costura")

    metr, labels_out, acuerdo, labs = [], [], [], {}
    for sp in args.spaces:
        coords = None if sp == "om" else P2T.space_coords(sp, X, w_fit, dyn)
        D = Dom if sp == "om" else cdist(coords, coords)
        state = {}
        for method in args.methods:
            if sp not in METHODS_SPACES[method]:
                continue
            t0 = time.time()
            for uni, mu in universos.items():
                idx = np.flatnonzero(mu)
                fits = fit_method(method, D, coords, wpx, idx, args.ks, rng, state)
                for k, lab_i in fits.items():
                    lab = np.full(len(ud), -2)
                    lab[idx] = lab_i
                    labs[(method, sp, uni, k)] = lab
                    mk = mu
                    nreal = len(set(lab_i) - {-1})
                    base = {"metodo": method, "espacio": sp, "universo": uni, "peso": "px", "k": k}
                    if method == "hdbscan":
                        cfg = state["hdb"][k]
                        for kk, v in (("mcs", cfg[0]), ("min_samples", cfg[1]), ("leaf", float(cfg[2] == "leaf"))):
                            metr.append(base | {"metrica": kk, "valor": v})
                    metr.append(base | {"metrica": "k_real", "valor": nreal})
                    metr.append(base | {"metrica": "ruido_share_px", "valor": float(wpx[mk][lab[mk] == -1].sum() / wpx[mk].sum())})
                    r2 = seqdist.pseudo_r2(Dom[np.ix_(mk, mk)], lab[mk], wpx[mk])
                    a = seqdist.asw(Dom[np.ix_(mk, mk)], lab[mk], wpx[mk], correct=True)
                    for kk, v in (("pseudo_r2", r2["pseudo_r2"]), ("pseudo_f", r2["pseudo_f"]), ("asw_corr", a["asw"])):
                        metr.append(base | {"metrica": kk, "valor": v})
                    for kk, v in P2T.describe(lab[mk], ud[mk].reset_index(drop=True), wpx[mk], Dom).items():
                        metr.append(base | {"metrica": kk, "valor": v})
                    sizes = np.bincount(lab[mk][lab[mk] >= 0], weights=wpx[mk][lab[mk] >= 0], minlength=max(nreal, 1)) / wpx[mk].sum()
                    metr.append(base | {"metrica": "cluster_mayor_share", "valor": float(sizes.max())})
                    med = medoids(D, np.where(mk, wpx, 0.0), lab)
                    labels_out.append(pd.DataFrame({"traj_id": ud.traj_id, "metodo": method, "espacio": sp, "universo": uni, "peso": "px",
                                                    "k": k, "etiqueta": lab, "medoide": np.where(med >= 0, ud.traj_id.values[np.maximum(med, 0)], -1)}))
            # estabilidad: submuestras del 80 % de los píxeles, ARI sobre las trayectorias presentes en la submuestra
            reps = args.gmm_reps if method == "gmm" else args.reps
            aris = {k: [] for k in args.ks}
            for _ in range(reps):
                w2 = rng.binomial(wpx.astype(np.int64), 0.8).astype(float)
                idx = np.flatnonzero(w2 > 0)
                fits = fit_method(method, D, coords, w2, idx, args.ks, rng, state)
                for k, lab_i in fits.items():
                    aris[k].append(P2T.ari_w(labs[(method, sp, "completo", k)][idx], lab_i, wpx[idx]))
            for k in args.ks:
                base = {"metodo": method, "espacio": sp, "universo": "completo", "peso": "px", "k": k}
                metr.append(base | {"metrica": "estabilidad_ari_80", "valor": float(np.mean(aris[k]))})
                metr.append(base | {"metrica": "estabilidad_ari_80_sd", "valor": float(np.std(aris[k]))})
            print(f"{sp}/{method}: {time.time() - t0:.0f}s", flush=True)

    for k in args.ks:
        for sp in args.spaces:
            ms = [m for m in args.methods if (m, sp, "completo", k) in labs]
            for a_, b_ in itertools.combinations(ms, 2):
                acuerdo.append({"tipo": "entre_metodos", "espacio": sp, "a": a_, "b": b_, "k": k,
                                "ari": P2T.ari_w(labs[(a_, sp, "completo", k)], labs[(b_, sp, "completo", k)], wpx)})
    pd.DataFrame(metr).to_csv(OUT / f"p2_metricas{args.tag}.csv", index=False)
    pd.DataFrame(acuerdo).to_csv(OUT / f"p2_acuerdo{args.tag}.csv", index=False)
    pd.concat(labels_out).to_csv(OUT / f"p2_labels{args.tag}.csv.gz", index=False)
    print("->", P.rel(OUT))


if __name__ == "__main__":
    main()
