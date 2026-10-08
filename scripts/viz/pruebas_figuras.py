"""Figuras de las pruebas AE contra OM (docs/autoencoder_v3/pruebas_ae_vs_om.md §5), a partir de data/autoencoder_v3/pruebas/*.csv.

    python scripts/viz/pruebas_figuras.py      # -> docs/autoencoder_v3/figuras/pruebas/*.png

Paleta: los cinco primeros espacios de la paleta categórica de referencia (validada); marcadores distintos por método.
OM y el one-hot, que no tienen d, van como líneas horizontales de referencia.
"""
from pathlib import Path
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))
from land2vec import paths as P  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

IN = P.DATA / "autoencoder_v3" / "pruebas"
OUT = _ROOT / "docs" / "autoencoder_v3" / "figuras" / "pruebas"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ESP = [("AE", "#2a78d6", "o"), ("AE lineal", "#eb6834", "s"), ("PCA", "#1baf7a", "^"), ("MCA", "#eda100", "D"),
       ("OM-MDS", "#e87ba4", "P")]
REF = [("OM", INK, (0, (5, 3))), ("One-hot", INK2, ":")]
DS = [1, 4, 7, 31]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "regular", "axes.titlelocation": "left", "legend.frameon": False})


def lineas(ax, df, col, ref_vals, ylim=None):
    for nom, c, mk in ESP:
        g = df[df.espacio == nom].groupby("d")[col].agg(["mean", "min", "max"]).reindex(DS)
        ax.plot(DS, g["mean"], color=c, marker=mk, ms=6, lw=2, label=nom, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        if nom in ("AE", "AE lineal"):
            ax.fill_between(DS, g["min"], g["max"], color=c, alpha=0.18, lw=0, zorder=2)
    for nom, c, ls in REF:
        if nom in ref_vals:
            ax.axhline(ref_vals[nom], color=c, lw=1.2, ls=ls, label=nom + " (referencia)", zorder=1)
    ax.set_xticks(DS)
    ax.set_xlabel("dimensión del embedding d")
    if ylim:
        ax.set_ylim(*ylim)


def fig_desfase():
    t = pd.read_csv(IN / "t1_desfase.csv")
    t = t[(t.n_cambios == 1) & (t.negativos == "mismo_anio")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, k in zip(axs, (5, 10)):
        s = t[t.k == k]
        ref = {nm: s[s.espacio == nm].auc_tipo.mean() for nm, *_ in REF}
        lineas(ax, s, "auc_tipo", ref, (0.3, 1.02))
        ax.set_title(f"desfase de {k} años")
        ax.set_ylabel("AUC (probabilidad de acertar el mismo proceso)")
    axs[1].legend(loc="lower left", ncol=2, fontsize=9)
    fig.suptitle("Mismo proceso desfasado contra otro proceso en el mismo año (trayectorias de un cambio, cada tipo pesa 1)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_desfase_auc.png", dpi=160)
    plt.close(fig)


def fig_tolerancia():
    t = pd.read_csv(IN / "t1_desfase.csv")
    t = t[(t.n_cambios == 1) & (t.negativos == "general")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, d in zip(axs, (4, 31)):
        for nom, c, mk in ESP:
            g = t[(t.espacio == nom) & (t.d == d)].groupby("k").dist_rel_mediana.mean()
            ax.plot(g.index, g.values, color=c, marker=mk, ms=6, lw=2, label=nom, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        for nom, c, ls in REF:
            g = t[t.espacio == nom].groupby("k").dist_rel_mediana.mean()
            ax.plot(g.index, g.values, color=c, lw=1.4, ls=ls, label=nom + " (referencia)", zorder=2)
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xlabel("desfase entre las dos trayectorias (años)")
        ax.set_title(f"d = {d}")
        ax.set_ylabel("distancia mediana / distancia entre pares al azar")
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Tolerancia al desfase: cuánto se separan dos trayectorias del mismo proceso a medida que crece el desfase",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_desfase_tolerancia.png", dpi=160)
    plt.close(fig)


def fig_semillas():
    t = pd.read_csv(IN / "t2_semillas.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (nom, c, mk) in zip(axs, ESP[:2]):
        s = t[t.espacio == nom]
        for tipo, off, fill, lab in (("entre_semillas", -0.4, c, "entre semillas"), ("con_OM", 0.4, SURFACE, "con OM")):
            for d in DS:
                v = s[(s.d == d) & (s.tipo == tipo)].knn10.values
                x = d + off
                ax.plot([x] * len(v), v, ls="none", marker=mk, ms=7, mfc=fill, mec=c, mew=1.6, label=lab if d == DS[0] else None, zorder=3)
        ax.set_xticks(DS)
        ax.set_xlabel("dimensión del embedding d")
        ax.set_title(nom)
        ax.set_ylabel("vecinos en común (de 10)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("¿Los vecinos coinciden más entre semillas (macizo) o con OM (hueco)?", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_semillas.png", dpi=160)
    plt.close(fig)


def fig_espacial():
    t = pd.read_csv(IN / "t5_espacial.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, col, tit in zip(axs, ("percentil_medio", "frac_bajo_p10"),
                            ("Percentil medio del par adyacente (menor = más coherente)", "Fracción de pares adyacentes bajo el percentil 10 (mayor = más coherente)")):
        ref = {nm: t[t.espacio == nm][col].mean() for nm, *_ in REF}
        lineas(ax, t, col, ref)
        ax.set_title(tit, fontsize=10)
    axs[0].set_ylabel("sólo píxeles adyacentes con trayectorias distintas")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig5_espacial.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_desfase()
    fig_tolerancia()
    fig_semillas()
    fig_espacial()
    print("figuras ->", OUT)


if __name__ == "__main__":
    main()
