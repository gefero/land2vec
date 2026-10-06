"""Interpretación de las tipologías de P2 (plan autoencoder_v3 §4.2, "qué agrupa cada una").

Para cada (espacio, k) pedido (universo completo, peso por superficie) describe cada cluster dinámico:
superficie y n.º de tipos, etiqueta automática (`typology.auto_label`: secuencia modal, forma, año de
cambio, glosa), secuencia de estados dominante y qué fracción de la superficie del cluster la comparte,
año del primer cambio (mediana y rango intercuartil por superficie), % de la superficie que es costura
2015/16, las 3 trayectorias de mayor superficie y el medoide. Además cruza dos espacios (qué cluster del
segundo espacio absorbe más superficie de cada cluster del primero).

Escribe data/autoencoder_v3/p2/p2_clusters_<espacio>_k<k>.csv y p2_cruce_<a>_<b>_k<k>.csv.

Uso: python scripts/clustering/p2_interpretacion.py [--spaces om ae_d8] [--k 12]
"""
import argparse
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402
from land2vec import typology as T  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = P.DATA / "autoencoder_v3" / "p2"


def wquantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, q)])


def describe(lab: pd.DataFrame, ud: pd.DataFrame, Xd: np.ndarray) -> pd.DataFrame:
    rows = []
    wtot = ud.n_px.sum()
    for g in sorted(lab.etiqueta.unique()):
        m = (lab.etiqueta == g).to_numpy()
        w, sub = ud.n_px.values[m].astype(float), ud[m]
        reps = np.repeat(np.flatnonzero(m), np.maximum(1, np.round(w / w.sum() * 20000)).astype(int))
        al = T.auto_label(Xd[reps])
        dss_share = sub.groupby("dss").n_px.sum().sort_values(ascending=False)
        top = sub.sort_values("n_px", ascending=False).head(3)
        med = ud.set_index("traj_id").loc[lab.medoide.values[m][0]]
        rows.append({
            "cluster": g, "n_tipos": int(m.sum()), "n_px": int(w.sum()), "share_px": w.sum() / wtot,
            "etiqueta": al.etiqueta, "dss_dominante": dss_share.index[0], "dss_dominante_share": dss_share.iloc[0] / w.sum(),
            "n_dss": int(len(dss_share)),
            "anio_1er_cambio_mediana": wquantile(sub.anio_primer_cambio.values, w, 0.5),
            "anio_1er_cambio_q25": wquantile(sub.anio_primer_cambio.values, w, 0.25),
            "anio_1er_cambio_q75": wquantile(sub.anio_primer_cambio.values, w, 0.75),
            "n_cambios_medio": float((sub.n_cambios.values * w).sum() / w.sum()),
            "share_costura": float(w[sub.costura.values].sum() / w.sum()),
            "top3": " | ".join(f"{r.seqs} ({r.n_px / w.sum():.0%})" for r in top.itertuples()),
            "medoide": med.seqs,
        })
    return pd.DataFrame(rows).sort_values("n_px", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spaces", nargs="+", default=["om", "ae_d8"])
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    uni = pd.read_csv(P.DATA / "autoencoder_v3" / "universo_argentina.csv")
    ud = uni[~uni.constante].reset_index(drop=True)
    Xd = np.array([Tokenizer.encode(s) for s in ud.seqs])
    ud["dss"] = ["»".join(Tokenizer.REVERSE_VOCAB[t] for t in T.dss(r)) for r in Xd]
    ud["anio_primer_cambio"] = [2000 + int(np.flatnonzero(r[1:] != r[:-1])[0]) + 1 for r in Xd]
    labels = pd.read_csv(OUT / "p2_labels.csv")
    labels = labels[(labels.universo == "completo") & (labels.peso == "px") & (labels.k == args.k)]
    L = {sp: labels[labels.espacio == sp].set_index("traj_id").loc[ud.traj_id].reset_index() for sp in args.spaces}

    for sp, lab in L.items():
        tab = describe(lab, ud, Xd)
        tab.to_csv(OUT / f"p2_clusters_{sp}_k{args.k}.csv", index=False)
        print(f"\n== {sp} k={args.k} ==")
        print(tab[["cluster", "n_tipos", "share_px", "etiqueta", "dss_dominante_share", "anio_1er_cambio_mediana", "share_costura"]].round(3).to_string(index=False))
    for a, b in zip(args.spaces, args.spaces[1:]):
        C = pd.crosstab(L[a].etiqueta, L[b].etiqueta, values=ud.n_px, aggfunc="sum").fillna(0)
        rows = []
        for g in C.index:
            r = C.loc[g]
            rows.append({f"cluster_{a}": g, "share_px": r.sum() / C.values.sum(), f"mejor_{b}": r.idxmax(),
                         "fraccion_en_mejor": r.max() / r.sum(),
                         f"segundo_{b}": r.drop(r.idxmax()).idxmax(), "fraccion_en_segundo": r.drop(r.idxmax()).max() / r.sum()})
        pd.DataFrame(rows).to_csv(OUT / f"p2_cruce_{a}_{b}_k{args.k}.csv", index=False)
        print(f"\ncruce {a} -> {b}:\n", pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
