"""Figuras del informe de la Pregunta 1 (docs/autoencoder_v3/p1_resultados.md), a partir de data/autoencoder_v3/pregunta1/*.csv.

    python scripts/viz/pregunta1_figuras.py      # -> docs/autoencoder_v3/figuras/p1/*.png

Paleta: espacios 1-4 de la paleta categórica de referencia (validada con validate_palette.js), en orden fijo por método,
con marcadores distintos (relevo para aqua y amarillo, por debajo de 3:1 de contraste).
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

IN = P.DATA / "autoencoder_v3" / "pregunta1"
OUT = _ROOT / "docs" / "autoencoder_v3" / "figuras" / "p1"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MET = [("AE", "#2a78d6", "o"), ("AE lineal", "#eb6834", "s"), ("PCA", "#1baf7a", "^"), ("MCA", "#eda100", "D")]
D_ELEGIDA = 4
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "regular", "axes.titlelocation": "left", "legend.frameon": False})
PANELES = [("Argentina (ajuste)", "-", "Argentina: trayectorias de ajuste (referencia)"),
           ("mundo no visto", "visto", "Mundo no visto: proceso visto en Argentina"),
           ("mundo no visto", "nuevo", "Mundo no visto: proceso nuevo")]


def curva(ax, df, x, nom, col, mk, banda=True):
    g = df.groupby(x).valor.agg(["mean", "min", "max"]).sort_index()
    ax.plot(g.index, g["mean"], color=col, marker=mk, ms=5.5, lw=2, label=nom, markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
    if banda and (g["max"] > g["min"]).any():
        ax.fill_between(g.index, g["min"], g["max"], color=col, alpha=0.18, lw=0, zorder=2)


def eje_d(ax, ds, fontsize=8):
    "d en posiciones equiespaciadas (la grilla 1, 2, 3, 4, 7, ..., 31 no es regular), con todas las etiquetas."
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([str(d) for d in ds], fontsize=fontsize)
    ax.minorticks_off()
    if D_ELEGIDA in ds:
        ax.axvline(ds.index(D_ELEGIDA), color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)


def con_x(df, ds):
    return df.assign(x=df.d.map({d: i for i, d in enumerate(ds)}))


def leyenda_abajo(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def fig_reconstruccion(r, metrica, peso, nombre, titulo, ylab):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (conj, a, tit) in zip(axs, PANELES):
        s = r[(r.conjunto == conj) & (r.A == a if a != "-" else r.A == "-") & (r.B.isin(["todos", "-"])) & (r.peso == peso) & (r.metrica == metrica)]
        ds = sorted(s.d.unique())
        s = con_x(s, ds)
        for nom, col, mk in MET:
            curva(ax, s[s.metodo == nom], "x", nom, col, mk)
        eje_d(ax, ds)
        ax.set_xlabel("dimensión del embedding d")
        ax.set_title(tit, fontsize=10)
        ax.set_ylim(-0.02, 1.03)
    axs[0].set_ylabel(ylab)
    leyenda_abajo(fig, axs[0], 4)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / nombre, dpi=160)
    plt.close(fig)


def fig_gradiente(r, d=D_ELEGIDA):
    orden = ["1", "2", "3", "4-5", "6+"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey="row")
    for i, (metrica, ylab) in enumerate((("acc_anio", "exactitud por año"), ("estados_ok", "secuencia de estados correcta"))):
        for j, a in enumerate(("visto", "nuevo")):
            ax = axs[i, j]
            s = r[(r.conjunto == "mundo no visto") & (r.A == a) & (r.B.isin(orden)) & (r.peso == "tipo") & (r.metrica == metrica) & (r.d == d)]
            s = s.assign(x=s.B.map({b: n for n, b in enumerate(orden)}))
            for nom, col, mk in MET:
                curva(ax, s[s.metodo == nom], "x", nom, col, mk)
            ax.set_xticks(range(len(orden)))
            ax.set_xticklabels(orden)
            ax.set_ylim(-0.02, 1.03)
            if i == 0:
                ax.set_title(f"Proceso {a} en Argentina", fontsize=10)
            if i == 1:
                ax.set_xlabel("años distintos respecto de la trayectoria argentina más parecida (h)")
            if j == 0:
                ax.set_ylabel(ylab)
    leyenda_abajo(fig, axs[1, 0], 4)
    fig.suptitle(f"Cómo cae la reconstrucción con la novedad de la trayectoria (d = {d}, cada tipo pesa 1)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig5_gradiente_novedad.png", dpi=160)
    plt.close(fig)


def fig_gradiente_todas_d(r, metrica, nombre, titulo):
    "Mapa de calor d × h para cada método y estrato A: todos los d en una figura."
    orden = ["1", "2", "3", "4-5", "6+"]
    s = r[(r.conjunto == "mundo no visto") & (r.B.isin(orden)) & (r.peso == "tipo") & (r.metrica == metrica)]
    ds = sorted(s.d.unique())
    fig, axs = plt.subplots(2, 4, figsize=(15, 9), sharex=True, sharey=True)
    for i, a in enumerate(("visto", "nuevo")):
        for j, (nom, _, _) in enumerate(MET):
            ax = axs[i, j]
            q = s[(s.A == a) & (s.metodo == nom)].groupby(["d", "B"]).valor.mean().unstack("B").reindex(index=ds, columns=orden)
            im = ax.imshow(q.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for (y, x), v in np.ndenumerate(q.values):
                ax.text(x, y, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7,
                        color=SURFACE if v > 0.55 else INK)
            ax.grid(False)
            ax.set_xticks(range(len(orden)))
            ax.set_xticklabels(orden)
            ax.set_yticks(range(len(ds)))
            ax.set_yticklabels(ds)
            ax.set_title(f"{nom}, proceso {a}", fontsize=10)
            if i == 1:
                ax.set_xlabel("h (años distintos)")
            if j == 0:
                ax.set_ylabel("d")
    fig.colorbar(im, ax=axs, shrink=0.6, label=titulo.split(" (")[0].lower())
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11.5)
    fig.savefig(OUT / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()   # las figuras del informe anterior se reemplazan
    r = pd.read_csv(IN / "p11_reconstruccion.csv")
    fig_reconstruccion(r, "acc_anio", "tipo", "fig1_exactitud_por_anio.png",
                       "Exactitud por año (cuándo), cada tipo pesa 1", "fracción de años correctos")
    fig_reconstruccion(r, "estados_ok", "tipo", "fig2_secuencia_de_estados.png",
                       "Secuencia de estados correcta (qué), cada tipo pesa 1", "fracción de trayectorias")
    fig_reconstruccion(r, "exacta", "tipo", "fig3_exacta_por_tipo.png",
                       "Reconstrucción exacta, cada tipo pesa 1", "fracción de trayectorias reconstruidas exactas")
    fig_reconstruccion(r, "exacta", "superficie", "fig4_exacta_por_superficie.png",
                       "Reconstrucción exacta, ponderada por superficie", "fracción de la superficie reconstruida exacta")
    fig_reconstruccion(r, "acc_anio", "superficie", "fig1b_exactitud_por_anio_superficie.png",
                       "Exactitud por año (cuándo), ponderada por superficie", "fracción de años correctos")
    fig_reconstruccion(r, "estados_ok", "superficie", "fig2b_secuencia_de_estados_superficie.png",
                       "Secuencia de estados correcta (qué), ponderada por superficie", "fracción de la superficie")
    fig_gradiente(r)
    fig_gradiente_todas_d(r, "estados_ok", "fig5b_gradiente_secuencia_todas_d.png",
                          "Secuencia de estados correcta por d y h (mundo no visto; cada tipo pesa 1; media de semillas)")
    fig_gradiente_todas_d(r, "acc_anio", "fig5c_gradiente_exactitud_todas_d.png",
                          "Exactitud por año por d y h (mundo no visto; cada tipo pesa 1; media de semillas)")
    fig_gradiente_todas_d(r, "exacta", "fig5d_gradiente_exacta_todas_d.png",
                          "Reconstrucción exacta por d y h (mundo no visto; cada tipo pesa 1; media de semillas)")
    print("figuras ->", OUT)


if __name__ == "__main__":
    main()
