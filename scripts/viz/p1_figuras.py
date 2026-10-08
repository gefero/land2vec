"""Figuras del informe de P1 (docs/autoencoder_v3/p1_resultados.md), a partir de data/autoencoder_v3/p1_metricas.csv
y universo_argentina.csv.

    python scripts/viz/p1_figuras.py            # -> docs/autoencoder_v3/figuras/p1/*.png

Colores: los 4 primeros espacios de la paleta categórica de referencia (azul, naranja, aqua, amarillo), en orden fijo
y validados con validate_palette.js; los tres marcadores distintos (círculo, cuadrado, triángulo, rombo) y las
etiquetas directas cumplen la regla de relevo para aqua y amarillo, que quedan por debajo de 3:1 sobre el fondo.
"""
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "autoencoder_v3" / "figuras" / "p1"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
METODOS = [("ae", "Autoencoder", "#2a78d6", "o"), ("lin", "AE lineal", "#eb6834", "s"),
           ("pca", "PCA", "#1baf7a", "^"), ("mca", "MCA", "#eda100", "D")]
D_ELEGIDA = 4

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "regular", "axes.titlelocation": "left",
                     "legend.frameon": False, "font.family": "DejaVu Sans"})


def load():
    m = pd.read_csv(P.DATA / "autoencoder_v3" / "p1_metricas.csv")
    u = pd.read_csv(P.DATA / "autoencoder_v3" / "universo_argentina.csv")
    return m, u


def serie(m, sub, peso, met):
    t = m[(m.subset == sub) & (m.peso == peso) & (m.metrica == met)]
    g = t.groupby(["metodo", "d"]).valor.agg(["mean", "min", "max"]).reset_index()
    return {k: g[g.metodo == k].set_index("d") for k, *_ in METODOS}


def lineas(ax, s, ds, banda=("ae",)):
    for k, nom, col, mk in METODOS:
        t = s[k].loc[ds]
        ax.plot(t.index, t["mean"], color=col, marker=mk, ms=6, lw=2, label=nom, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        if k in banda:
            ax.fill_between(t.index, t["min"], t["max"], color=col, alpha=0.18, lw=0, zorder=2)
    ax.set_xticks(ds)
    ax.set_xlabel("dimensión del embedding d")
    ax.axvline(D_ELEGIDA, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)


def fig_universo(u):
    d = u[~u.constante].sort_values("n_px", ascending=False)
    cs = d.n_px.cumsum() / d.n_px.sum()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(np.arange(1, len(cs) + 1), cs, color="#2a78d6", lw=2)
    for q in (0.5, 0.9, 0.99):
        n = int(np.searchsorted(cs.values, q) + 1)
        ax.plot([n], [q], "o", color="#2a78d6", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{n:,} trayectorias cubren el {int(q * 100)} %".replace(",", "."), xy=(n, q), xytext=(n * 1.5, q - 0.07),
                    fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("trayectorias dinámicas, ordenadas de más a menos superficie (escala log)")
    ax.set_ylabel("fracción de la superficie dinámica")
    ax.set_title("El universo es concentrado: pocas trayectorias cubren casi toda la superficie dinámica")
    fig.tight_layout()
    fig.savefig(OUT / "fig0_universo.png", dpi=160)
    plt.close(fig)


def fig_exacta(m, peso, nombre, titulo, leyenda=(1.0, 0.42), loc="center right"):
    s = serie(m, "dinamicas", peso, "exacta")
    ds = sorted(m.d.unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_ylim(-0.02, 1.04)
    lineas(ax, s, ds)
    ax.axhline(0.99, color=INK2, lw=1, ls=":", zorder=1)
    ax.annotate("umbral 0,99", xy=(ds[-1], 0.99), xytext=(ds[-1], 0.955), fontsize=8.5, color=INK2, ha="right")
    ax.annotate(f"d = {D_ELEGIDA}", xy=(D_ELEGIDA, 0.62), xytext=(D_ELEGIDA + 0.4, 0.62), fontsize=9, color=INK2)
    ax.set_ylabel("fracción de trayectorias reconstruidas exactas")
    ax.set_title(titulo)
    ax.legend(loc=loc, bbox_to_anchor=leyenda)
    fig.tight_layout()
    fig.savefig(OUT / nombre, dpi=160)
    plt.close(fig)


def fig_fidelidad(m):
    ds = sorted(m.d.unique())
    paneles = [("n_cambios_ok", "número de cambios correcto", False), ("estados_ok", "secuencia de estados correcta", False),
               ("err_anio_cambio", "error del año del cambio (años)", True)]
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, (met, tit, sl) in zip(axs, paneles):
        s = serie(m, "dinamicas", "tipo", met)
        lineas(ax, s, ds)
        ax.set_title(tit)
        if sl:
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_ylim(0, 12)
            ax.set_ylabel("años de error (escala symlog)")
        else:
            ax.set_ylim(-0.02, 1.03)
            ax.set_ylabel("fracción de tipos")
        ax.set_xticks([1, 4, 10, 16, 22, 31])
    axs[0].legend(loc="lower right")
    fig.suptitle("Qué se pierde al comprimir (píxeles dinámicos, cada tipo pesa 1)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_fidelidad_por_aspecto.png", dpi=160)
    plt.close(fig)


def fig_semillas(m):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (k, ds, tit) in zip(axs, [("ae", [1, 4, 7, 10], "Autoencoder: tres semillas por d"),
                                      ("lin", sorted(m.d.unique()), "AE lineal: tres semillas por d")]):
        _, nom, col, mk = [x for x in METODOS if x[0] == k][0]
        t = m[(m.subset == "dinamicas") & (m.peso == "tipo") & (m.metrica == "exacta") & (m.metodo == k) & (m.d.isin(ds))]
        for sem, off in zip((0, 1, 2), (-0.35, 0, 0.35)):
            q = t[t.semilla == sem]
            ax.plot(q.d + off, q.valor, ls="none", marker=mk, ms=7, color=col, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=None)
        mu = t.groupby("d").valor.mean()
        ax.plot(mu.index, mu.values, color=col, lw=1.2, alpha=0.6, zorder=1)
        ax.set_xticks(ds)
        ax.set_xlabel("dimensión del embedding d")
        ax.set_ylabel("fracción de tipos reconstruidos exactos")
        ax.set_title(tit)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ruido_semillas.png", dpi=160)
    plt.close(fig)


def fig_estructura(m):
    ds = sorted(m.d.unique())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (met, tit) in zip(axs, [("spearman_om", "Correlación de Spearman con la distancia OM"),
                                    ("knn10_om", "Vecinos en común con OM (10 vecinos más cercanos)")]):
        s = serie(m, "todas", "tipo", met)
        lineas(ax, s, ds)
        ax.set_ylim(0, 1)
        ax.set_ylabel("valor (cada tipo pesa 1)")
        ax.set_title(tit)
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_estructura_om.png", dpi=160)
    plt.close(fig)


def fig_recall(m):
    clases = ["A", "F", "G", "Wt", "U", "Sh", "Sp", "B", "Wa"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, d in zip(axs, (1, 4)):
        t = m[(m.metrica.str.startswith("recall_")) & (m.subset == "dinamicas") & (m.peso == "tipo") & (m.d == d)]
        t = t.assign(clase=t.metrica.str[7:]).groupby(["clase", "metodo"]).valor.mean().unstack()
        for i, c in enumerate(clases):
            xs = [t.loc[c, k] for k, *_ in METODOS]
            ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=2, zorder=1)
            for (k, nom, col, mk), x in zip(METODOS, xs):
                ax.plot(x, i, marker=mk, ms=7, color=col, ls="none", markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.set_yticks(range(len(clases)))
        ax.set_yticklabels(clases)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.05)
        ax.set_xlabel("recall de la clase (fracción de años-píxel bien reconstruidos)")
        ax.set_title(f"d = {d}")
        ax.grid(axis="y", visible=False)
    axs[0].set_ylabel("clase de cobertura")
    handles = [plt.Line2D([], [], color=col, marker=mk, ls="none", ms=7, label=nom) for k, nom, col, mk in METODOS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Recall por clase en trayectorias dinámicas (cada tipo pesa 1)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig6_recall_por_clase.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, u = load()
    fig_universo(u)
    fig_exacta(m, "px", "fig1_exacta_por_superficie.png",
               "Reconstrucción exacta, ponderada por superficie (píxeles dinámicos)")
    fig_exacta(m, "tipo", "fig2_exacta_por_tipo.png",
               "Reconstrucción exacta, cada tipo pesa 1 (trayectorias dinámicas)", leyenda=(0.19, 0.93), loc="upper left")
    fig_fidelidad(m)
    fig_semillas(m)
    fig_estructura(m)
    fig_recall(m)
    print("figuras ->", OUT)


if __name__ == "__main__":
    main()
