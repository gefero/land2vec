"""Figuras del catálogo de procesos de la Pregunta 2 (docs/autoencoder_v3/p2_catalogo.md), a partir de
data/autoencoder_v3/pregunta2/catalogo_*.csv.

    python scripts/viz/pregunta2_figuras.py      # -> docs/autoencoder_v3/figuras/p2/*.png

Paleta: espacios 1-2 de la paleta categórica de referencia (la misma de las figuras de la Pregunta 1), con marcadores
distintos. Los años de costura y de cambio de sensor van sombreados en gris.
"""
from pathlib import Path
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec.procesos import COSTURAS, NOMBRE, PROCESOS, SENSOR  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

IN = P.DATA / "autoencoder_v3" / "pregunta2"
OUT = _ROOT / "docs" / "autoencoder_v3" / "figuras" / "p2"
SURFACE, INK, INK2, GRID, SOMBRA = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#ecebe6"
CONJ = [("Argentina", "#2a78d6", "o"), ("mundo", "#eb6834", "s")]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5,
                     "axes.titlelocation": "left", "legend.frameon": False})


def fig_anios(a):
    "Distribución del año de los eventos de cada proceso (fracción de la superficie del proceso), Argentina y mundo."
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, p in zip(axs.ravel(), PROCESOS):
        for y in COSTURAS + SENSOR:
            ax.axvspan(y - 0.5, y + 0.5, color=SOMBRA, lw=0, zorder=0)
        for conj, col, mk in CONJ:
            s = a[(a.proceso == p) & (a.conjunto == conj)].set_index("anio").superficie
            s = (s / s.sum()).reindex(range(1993, 2023), fill_value=0)
            ax.plot(s.index, s.values, color=col, marker=mk, ms=4.5, lw=2, label=conj,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
        ax.set_title(NOMBRE[p])
        ax.set_xticks([1995, 2000, 2005, 2010, 2016, 2022])
    for ax in axs[:, 0]:
        ax.set_ylabel("fracción de la superficie del proceso")
    for ax in axs[1]:
        ax.set_xlabel("año del evento")
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color=SOMBRA))
    l.append("años de costura (1995, 2016) y de cambio de sensor (1999, 2000)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Año de los eventos de cada proceso (superficie de cada año sobre el total del proceso)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig1_anio_de_los_eventos.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_anios(pd.read_csv(IN / "catalogo_anios.csv"))
    print("figuras ->", OUT)


if __name__ == "__main__":
    main()
