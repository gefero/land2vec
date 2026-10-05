"""P2 (plan autoencoder_v3 §4.2): tipologías de las dinámicas de cobertura en tres espacios.

Se tipifican las trayectorias dinámicas del universo de Argentina (4.545 tipos; las 9 constantes
quedan como clases estables aparte) con el mismo algoritmo en todos los espacios: k-medoides
ponderado (k-medoides++ y varios reinicios) sobre la matriz de distancias de cada espacio:

    om        OM con costos TRATE (la distancia clásica del análisis de secuencias)
    onehot    euclídea sobre el one-hot completo
    pca8      euclídea sobre las 8 primeras componentes del one-hot
    ae_d3/8/16  euclídea sobre z del autoencoder de P1 (semilla 0; las semillas 1 y 2 miden la estabilidad entre réplicas)

Granularidad: k ∈ {6, 12, 24}. Peso = n_px (superficie); la variante `tipo` pesa 1 por trayectoria.
Los resultados se reportan sobre el universo completo y sin las trayectorias de la costura 2015/16 (plan §7).

Salidas (data/autoencoder_v3/p2/):
    p2_metricas.csv   calidad (pseudo-R², ASW corregida), estabilidad (submuestras binomiales del 80 % de los
                      píxeles), qué agrupa cada tipología (NMI con la secuencia de estados y con el año del
                      primer cambio) y control de costura
    p2_acuerdo.csv    ARI ponderado entre espacios, entre semillas de z y entre universos
    p2_labels.csv     etiqueta de cada trayectoria dinámica por espacio, k y universo

Uso, desde la raíz del repo:
    python scripts/clustering/p2_tipologias.py --om <distancia OM .npy> [--reps 20]
"""
import argparse
import itertools
import time
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402
from land2vec import seqdist  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("p1c", Path(__file__).resolve().parents[1] / "modelo" / "p1_compresion.py")
p1c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1c)

KS = [6, 12, 24]
OUT = P.DATA / "autoencoder_v3" / "p2"


# ---------------------------------------------------------------------------
# k-medoides ponderado
# ---------------------------------------------------------------------------
def kmedoids(D: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10, max_iter: int = 50):
    "Devuelve (medoides, etiquetas, costo). D (n, n), w (n,) >= 0; las filas con w == 0 no influyen."
    n = len(w)
    best = (None, None, np.inf)
    for _ in range(n_init):
        # k-medoides++ ponderado
        med = [rng.choice(n, p=w / w.sum())]
        d2 = D[med[0]].astype(float) ** 2
        for _ in range(k - 1):
            pr = w * d2
            med.append(rng.choice(n, p=pr / pr.sum()) if pr.sum() > 0 else rng.integers(n))
            d2 = np.minimum(d2, D[med[-1]].astype(float) ** 2)
        med = np.array(med)
        for _ in range(max_iter):
            lab = D[:, med].argmin(1)
            new = med.copy()
            for g in range(k):
                idx = np.flatnonzero((lab == g) & (w > 0))
                if len(idx):
                    new[g] = idx[(D[np.ix_(idx, idx)] * w[idx][None, :]).sum(1).argmin()]
            if (new == med).all():
                break
            med = new
        lab = D[:, med].argmin(1)
        cost = float((w * D[np.arange(n), med[lab]]).sum())
        if cost < best[2]:
            best = (med, lab, cost)
    return best


# ---------------------------------------------------------------------------
# Comparación de particiones (ponderada)
# ---------------------------------------------------------------------------
def contingency(a, b, w):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia, ib), w)
    return C


def ari_w(a, b, w) -> float:
    C = contingency(a, b, w)
    c2 = lambda x: x * (x - 1) / 2.0
    sij, sa, sb, n = c2(C).sum(), c2(C.sum(1)).sum(), c2(C.sum(0)).sum(), c2(C.sum())
    exp = sa * sb / n
    den = 0.5 * (sa + sb) - exp
    return float((sij - exp) / den) if den > 0 else float("nan")


def nmi_w(a, b, w) -> float:
    C = contingency(a, b, w)
    p = C / C.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / np.sqrt(ha * hb)) if ha > 0 and hb > 0 else float("nan")


# ---------------------------------------------------------------------------
# Datos y espacios
# ---------------------------------------------------------------------------
def dynamic_universe():
    u, X = p1c.load_universe()
    dyn = ~u.constante.values
    ud, Xd = u[dyn].reset_index(drop=True), X[dyn]
    # descriptores de "qué agrupa": secuencia de estados y año del primer cambio
    states, first = [], []
    for row in Xd:
        st, ch = p1c.runs_of(row)
        states.append("-".join(Tokenizer.REVERSE_VOCAB[t] for t in st))
        first.append(2000 + int(ch[0]))
    ud["estados"], ud["anio_primer_cambio"] = states, first
    return u, ud, X, Xd, dyn


def space_coords(name: str, X: np.ndarray, w_fit: np.ndarray, dyn: np.ndarray, seed: int = 0) -> np.ndarray:
    if name == "onehot":
        return p1c.onehot(X)[dyn]
    if name == "pca8":
        m = p1c.pca_fit(X, w_fit, 8)
        return p1c.pca_embed_recon(m, 8)[0][dyn]
    if name.startswith("ae_d"):
        d = int(name[4:].split("_s")[0])
        s = int(name.split("_s")[1]) if "_s" in name else 0
        return np.load(p1c.OUT_MODELS / f"ae_d{d}_s{s}.npz")["z"][dyn]
    raise ValueError(name)


def describe(labels, ud, w, Dom):
    "Qué agrupa la tipología: NMI con la secuencia de estados y con el año del primer cambio; control de costura."
    out = {"nmi_estados": nmi_w(labels, ud.estados.values, w),
           "nmi_anio_primer_cambio": nmi_w(labels, ud.anio_primer_cambio.values, w)}
    sc = ud.costura.values
    share = np.array([(w[(labels == g) & sc].sum() / w[labels == g].sum()) for g in np.unique(labels)])
    solo = ud.solo_costura.values
    share_solo = np.array([(w[(labels == g) & solo].sum() / w[labels == g].sum()) for g in np.unique(labels)])
    out["costura_max_share_cluster"] = float(share.max())
    out["solo_costura_max_share_cluster"] = float(share_solo.max())
    # ¿algún cluster está definido por la costura? (>= 80 % de su superficie son trayectorias de costura)
    out["clusters_definidos_por_costura"] = int((share >= 0.8).sum())
    return out


def stability(D_full, w_px, k, reps, rng, n_init=5):
    "Submuestras reales del 80 % de los píxeles (adelgazamiento binomial); ARI ponderado contra el ajuste completo."
    med0, lab0, _ = kmedoids(D_full, w_px, k, rng, n_init=n_init)
    aris = []
    for _ in range(reps):
        w2 = rng.binomial(w_px.astype(np.int64), 0.8).astype(float)
        med, _, _ = kmedoids(D_full, w2, k, rng, n_init=n_init)
        lab = D_full[:, med].argmin(1)  # todos los tipos, por el medoide más cercano
        aris.append(ari_w(lab0, lab, w_px))
    return float(np.mean(aris)), float(np.std(aris))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--om", type=Path, required=True, help="matriz OM (4.554 x 4.554) de P1; ver p1_compresion.py om")
    ap.add_argument("--reps", type=int, default=20, help="submuestras del 80 %% para la estabilidad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spaces", nargs="+", default=["om", "onehot", "pca8", "ae_d3", "ae_d8", "ae_d16"])
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    u, ud, X, Xd, dyn = dynamic_universe()
    w_fit = p1c.train_weights(u)
    Dom_all = np.load(args.om).astype(np.float64)
    Dom = Dom_all[np.ix_(dyn, dyn)]  # distancia OM entre dinámicas: la vara común de calidad
    print(f"dinámicas: {len(ud):,} tipos, {ud.n_px.sum():,} px; con costura: {ud.costura.sum()} tipos", flush=True)

    universos = {"completo": np.ones(len(ud), bool), "sin_costura": ~ud.costura.values}
    pesos = {"px": ud.n_px.values.astype(float), "tipo": np.ones(len(ud))}

    metr, labels_out, acuerdo = [], [], []
    labs: dict[tuple, np.ndarray] = {}
    for sp in args.spaces:
        t0 = time.time()
        D = Dom if sp == "om" else cdist(c := space_coords(sp, X, w_fit, dyn), c).astype(np.float32)
        for uni, mu in universos.items():
            for pw, wfull in pesos.items():
                w = np.where(mu, wfull, 0.0)
                for k in KS:
                    med, lab, cost = kmedoids(D, w, k, rng)
                    labs[(sp, uni, pw, k)] = lab
                    mk = mu & (w > 0)
                    base = {"espacio": sp, "universo": uni, "peso": pw, "k": k}
                    r2 = seqdist.pseudo_r2(Dom[np.ix_(mk, mk)], lab[mk], w[mk])
                    a = seqdist.asw(Dom[np.ix_(mk, mk)], lab[mk], w[mk], correct=True)
                    for kk, v in (("pseudo_r2", r2["pseudo_r2"]), ("pseudo_f", r2["pseudo_f"]), ("asw_corr", a["asw"])):
                        metr.append(base | {"metrica": kk, "valor": v})
                    for kk, v in describe(lab[mk], ud[mk].reset_index(drop=True), w[mk], Dom).items():
                        metr.append(base | {"metrica": kk, "valor": v})
                    sizes = np.bincount(lab[mk], weights=w[mk], minlength=k) / w[mk].sum()
                    metr.append(base | {"metrica": "cluster_mayor_share", "valor": float(sizes.max())})
                    if uni == "completo" and pw == "px":
                        labels_out.append(pd.DataFrame({"traj_id": ud.traj_id, "espacio": sp, "k": k, "etiqueta": lab}))
        # estabilidad con submuestras reales del 80 % (universo completo, peso px)
        for k in KS:
            m, s = stability(D, pesos["px"], k, args.reps, rng)
            metr.append({"espacio": sp, "universo": "completo", "peso": "px", "k": k, "metrica": "estabilidad_ari_80", "valor": m})
            metr.append({"espacio": sp, "universo": "completo", "peso": "px", "k": k, "metrica": "estabilidad_ari_80_sd", "valor": s})
        print(f"{sp}: {time.time() - t0:.0f}s", flush=True)

    wpx = pesos["px"]
    for k in KS:
        for a_, b_ in itertools.combinations(args.spaces, 2):
            acuerdo.append({"tipo": "entre_espacios", "a": a_, "b": b_, "k": k, "ari": ari_w(labs[(a_, "completo", "px", k)], labs[(b_, "completo", "px", k)], wpx)})
        for sp in args.spaces:
            acuerdo.append({"tipo": "completo_vs_sin_costura", "a": sp, "b": sp, "k": k,
                            "ari": ari_w(labs[(sp, "completo", "px", k)][universos["sin_costura"]], labs[(sp, "sin_costura", "px", k)][universos["sin_costura"]], wpx[universos["sin_costura"]])})
            acuerdo.append({"tipo": "px_vs_tipo", "a": sp, "b": sp, "k": k, "ari": ari_w(labs[(sp, "completo", "px", k)], labs[(sp, "completo", "tipo", k)], wpx)})
    # entre semillas de z (réplicas del autoencoder): ruido de la tipología por el embedding
    for sp in [s for s in args.spaces if s.startswith("ae_d")]:
        D_s = {s: (lambda c: cdist(c, c).astype(np.float32))(space_coords(f"{sp}_s{s}", X, w_fit, dyn)) for s in (0, 1, 2)}
        for k in KS:
            L = {s: kmedoids(D_s[s], wpx, k, rng)[1] for s in D_s}
            for s1, s2 in itertools.combinations(L, 2):
                acuerdo.append({"tipo": "entre_semillas", "a": f"{sp}_s{s1}", "b": f"{sp}_s{s2}", "k": k, "ari": ari_w(L[s1], L[s2], wpx)})

    pd.DataFrame(metr).to_csv(OUT / "p2_metricas.csv", index=False)
    pd.DataFrame(acuerdo).to_csv(OUT / "p2_acuerdo.csv", index=False)
    pd.concat(labels_out).to_csv(OUT / "p2_labels.csv", index=False)
    print("->", P.rel(OUT))


if __name__ == "__main__":
    main()
