"""Tablas markdown de la comparación de métodos de clustering de P2 (filas = método, columnas = espacio).

Lee p2_metricas_metodos.csv, p2_metricas_hdbscan.csv, p2_desmonte_metodos.csv y p2_desmonte_hdbscan.csv
(universo completo, peso por superficie, k pedido; el control con desmonte es de Chaco) e imprime una tabla
por métrica, más los promedios por espacio y por k. Se pega en docs/autoencoder_v3/p2_resultados.md.

Uso: python scripts/clustering/p2_tablas_metodos.py [--k 12 | --barrido]
"""
import argparse
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = P.DATA / "autoencoder_v3" / "p2"
SP = ["om", "onehot", "pca8", "ae_d3", "ae_d8", "ae_d16"]
ME = ["kmedoides", "kmeans", "ward", "avg", "gmm", "hdbscan"]
VALIDOS = ["kmedoides", "kmeans", "ward"]  # los que dan una partición comparable (ver docs)
HEAD = ["OM", "one-hot", "PCA-8", "AE d=3", "AE d=8", "AE d=16"]


def load():
    m = pd.concat([pd.read_csv(D / "p2_metricas_metodos.csv"), pd.read_csv(D / "p2_metricas_hdbscan.csv")])
    t = pd.concat([pd.read_csv(D / "p2_desmonte_metodos.csv"), pd.read_csv(D / "p2_desmonte_hdbscan.csv")])
    t = t[(t.zona == "chaco_santiago_frontier") & (t.universo == "completo") & (t.peso == "px")]
    return m[m.universo == "completo"], t


def piv(m, t, met, k):
    if met in ("u_ratio", "mcc_cv_ratio_techo", "mcc_sem_menos_r0p"):
        x = t[t.k == k].pivot_table(index="metodo", columns="espacio", values=met)
    else:
        x = m[(m.metrica == met) & (m.k == k)].pivot_table(index="metodo", columns="espacio", values="valor")
    return x.reindex(index=ME, columns=SP)


def md(p, dec=2, best=None):
    "best: 'max' resalta el mayor de cada columna entre los métodos VALIDOS + avg + gmm (no hdbscan)."
    fmt = lambda v: "—" if pd.isna(v) else f"{v:.{dec}f}".replace(".", ",")
    ref = p.drop(index="hdbscan", errors="ignore")
    rows = ["| método | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(SP)]
    for me in ME:
        cells = []
        for sp in SP:
            s = fmt(p.loc[me, sp])
            if best == "max" and me != "hdbscan" and not pd.isna(p.loc[me, sp]) and p.loc[me, sp] == ref[sp].max():
                s = f"**{s}**"
            cells.append(s)
        rows.append(f"| {me} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


METRICAS = [("pseudo_r2", 2, "max"), ("asw_corr", 2, "max"), ("estabilidad_ari_80", 2, "max"), ("nmi_estados", 2, None),
            ("nmi_anio_primer_cambio", 2, None), ("cluster_mayor_share", 2, None), ("ruido_share_px", 2, None),
            ("k_real", 0, None), ("u_ratio", 3, "max"), ("mcc_cv_ratio_techo", 3, "max"), ("mcc_sem_menos_r0p", 3, "max")]


def barrido():
    "Curvas contra k (kmedoides, kmeans y ward, promedio de métodos y espacios) y reglas de elección de k por método y espacio."
    m = pd.read_csv(D / "p2_metricas_barrido.csv")
    t = pd.read_csv(D / "p2_desmonte_barrido.csv")
    t = t[t.peso == "px"]
    ks = np.array(sorted(m.k.unique()))
    cur = lambda met: m[m.metrica == met].groupby("k").valor.mean().reindex(ks)
    tcur = lambda met: t.groupby("k")[met].mean().reindex(ks)
    tab = pd.DataFrame({"pseudo-R²": cur("pseudo_r2"), "pseudo-F (miles)": cur("pseudo_f") / 1000, "ASW": cur("asw_corr"),
                        "estabilidad": cur("estabilidad_ari_80"), "NMI estados": cur("nmi_estados"), "NMI año": cur("nmi_anio_primer_cambio"),
                        "mayor cluster": cur("cluster_mayor_share"), "U ratio": tcur("u_ratio"), "MCC/techo": tcur("mcc_cv_ratio_techo")})
    print("#### curvas contra k\n\n" + tab.round(3).to_markdown())

    def knee(y, logx):  # codo: mayor distancia a la recta entre los extremos (curva creciente y cóncava)
        x = np.log(ks) if logx else ks.astype(float)
        xn, yn = (x - x[0]) / (x[-1] - x[0]), (y.values - y.min()) / (y.max() - y.min())
        return int(ks[np.argmax(yn - xn)])
    print("\n#### codo (k en escala log / lineal)\n")
    for name, y in (("pseudo-R²", cur("pseudo_r2")), ("U ratio", tcur("u_ratio")), ("MCC/techo", tcur("mcc_cv_ratio_techo"))):
        print(f"- {name}: {knee(y, True)} / {knee(y, False)}")
    rows = []
    for (me, sp), g in m.groupby(["metodo", "espacio"]):
        a = g[g.metrica == "asw_corr"].set_index("k").valor
        f = g[g.metrica == "pseudo_f"].set_index("k").valor
        tt = t[(t.metodo == me) & (t.espacio == sp)].set_index("k")
        ok = tt[(tt.mcc_cv_ratio_techo >= 0.98) & (tt.u_ratio >= 0.95)].index.min()
        rows.append({"metodo": me, "espacio": sp, "k_ASW_max(<=40)": int(a[a.index <= 40].idxmax()), "k_pseudoF_max": int(f.idxmax()), "k_regla_externa": ok})
    r = pd.DataFrame(rows)
    for col in ("k_ASW_max(<=40)", "k_pseudoF_max", "k_regla_externa"):
        print(f"\n#### {col}\n\n" + r.pivot(index="espacio", columns="metodo", values=col).reindex(SP[:]).dropna(how="all").to_markdown())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--barrido", action="store_true", help="en vez de las tablas de un k, las curvas contra k del barrido (p2_*_barrido.csv)")
    a = ap.parse_args()
    if a.barrido:
        return barrido()
    k = a.k
    m, t = load()
    for met, dec, best in METRICAS:
        print(f"\n#### {met}\n\n" + md(piv(m, t, met, k), dec, best))
    print("\n#### promedio por espacio (kmedoides, kmeans y ward)\n")
    rows = {}
    for met, _, _ in METRICAS[:5] + METRICAS[8:10]:
        rows[met] = piv(m, t, met, k).loc[VALIDOS].mean()
    print(pd.DataFrame(rows).T.round(3).to_markdown())
    print("\n#### promedio por método según k\n")
    for met in ("pseudo_r2", "estabilidad_ari_80", "cluster_mayor_share", "u_ratio", "mcc_cv_ratio_techo"):
        print(met)
        print(pd.DataFrame({kk: piv(m, t, met, kk).mean(axis=1) for kk in (6, 12, 24)}).round(3).T.to_markdown())


if __name__ == "__main__":
    main()
