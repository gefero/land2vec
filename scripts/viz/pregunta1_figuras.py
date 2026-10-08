"""Figuras del informe de la Pregunta 1 (docs/autoencoder_v3/p1_resultados.md), a partir de data/autoencoder_v3/pregunta1/*.csv.

    python scripts/viz/pregunta1_figuras.py      # -> docs/autoencoder_v3/figuras/p1/*.png

Paleta: espacios 1-5 de la paleta categórica de referencia (validada con validate_palette.js), en orden fijo por método,
con marcadores distintos (relevo para aqua, amarillo y magenta, por debajo de 3:1 de contraste). Las referencias
(one-hot con Hamming, partición al azar, líneas de base) van en gris, con línea discontinua o punteada.
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
SURFACE, INK, INK2, GRID, REFC = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8983"
MET = [("AE", "#2a78d6", "o"), ("AE lineal", "#eb6834", "s"), ("PCA", "#1baf7a", "^"), ("MCA", "#eda100", "D")]
ESP12 = MET + [("OM", "#e87ba4", "P")]
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


def leyenda_abajo(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def fig_reconstruccion(r, metrica, peso, nombre, titulo, ylab):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (conj, a, tit) in zip(axs, PANELES):
        s = r[(r.conjunto == conj) & (r.A == a if a != "-" else r.A == "-") & (r.B.isin(["todos", "-"])) & (r.peso == peso) & (r.metrica == metrica)]
        for nom, col, mk in MET:
            curva(ax, s[s.metodo == nom], "d", nom, col, mk)
        ax.axvline(D_ELEGIDA, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        ds = sorted(s.d.unique())
        ax.set_xticks(ds)
        ax.set_xticklabels([str(d) if d in (1, 2, 3, 4, 7, 10, 16, 31) else "" for d in ds], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("dimensión del embedding d (escala log)")
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


def fig_accesibilidad(acc, sonda="vecinos"):
    objs = [("estado_inicial", "estado inicial"), ("estado_final", "estado final"), ("n_cambios", "número de cambios"),
            ("proceso", "proceso"), ("anio_primer_cambio", "año del primer cambio (error, años)")]
    fig, axs = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (obj, tit) in zip(axs, objs):
        s = acc[(acc.objetivo == obj) & (acc.A == "visto") & (acc.B == "todos") & (acc.peso == "tipo")]
        for nom, col, mk in MET:
            curva(ax, s[(s.metodo == nom) & (s.sonda == sonda)], "d", nom, col, mk)
        b = s[s.metodo == "línea de base"].valor
        if len(b):
            ax.axhline(b.iloc[0], color=REFC, lw=1.3, ls=":", label="línea de base", zorder=1)
        ax.axvline(D_ELEGIDA, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        ds = sorted(s[s.metodo != "línea de base"].d.unique())
        ax.set_xticks(ds)
        ax.set_xticklabels([str(d) if d in (1, 2, 4, 7, 16, 31) else "" for d in ds], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("d (escala log)")
        ax.set_title(tit, fontsize=10)
        if obj != "anio_primer_cambio":
            ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("exactitud balanceada")
    leyenda_abajo(fig, axs[0], 5)
    fig.suptitle(f"Accesibilidad en z, sonda de {sonda} (mundo no visto, proceso visto; cada tipo pesa 1)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / f"fig6_accesibilidad_{sonda}.png", dpi=160)
    plt.close(fig)


def fig_tipologias(tp, algoritmo="k-medoides", d=D_ELEGIDA):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (col_m, tit) in zip(axs, (("exactitud_prototipo", "Exactitud por año del prototipo (cuándo)"),
                                      ("pureza_proceso", "Pureza de proceso (qué)"),
                                      ("dispersion_anio", "Dispersión del año del primer cambio (años)"))):
        s = tp[tp.algoritmo == algoritmo]
        for nom, col, mk in ESP12:
            q = s[(s.espacio == nom) & ((s.d == d) | (nom == "OM"))].rename(columns={col_m: "valor"})
            curva(ax, q, "k", nom, col, mk)
        oh = s[s.espacio == "One-hot (Hamming)"].rename(columns={col_m: "valor"})
        ax.plot(oh.k, oh.valor, color=INK2, lw=1.4, ls=(0, (5, 3)), label="One-hot, Hamming (referencia)", zorder=2)
        az = s.groupby("k")[f"azar_{col_m}"].mean()
        ax.plot(az.index, az.values, color=REFC, lw=1.4, ls=":", label="partición al azar (piso)", zorder=1)
        ax.set_xticks(sorted(s.k.unique()))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("número de grupos k")
        ax.set_title(tit, fontsize=10)
    leyenda_abajo(fig, axs[0], 7)
    fig.suptitle(f"Tipologías de las trayectorias dinámicas de Argentina ({algoritmo}; embeddings con d = {d}; cada tipo pesa 1)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    nombre = "fig7_tipologias_kmedoides.png" if algoritmo == "k-medoides" else "fig8_tipologias_jerarquico.png"
    fig.savefig(OUT / nombre, dpi=160)
    plt.close(fig)


def fig_estabilidad(sem, tp, d=D_ELEGIDA):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ax = axs[0]
    for nom, col, mk in MET[:2]:
        s = sem[(sem.espacio == nom) & (sem.d == d)].sort_values("k")
        ax.plot(s.k, s.ari_entre_semillas_media, color=col, marker=mk, ms=5.5, lw=2, label=nom, markeredgecolor=SURFACE, zorder=3)
        ax.fill_between(s.k, s.ari_min, s.ari_max, color=col, alpha=0.18, lw=0)
    ax.set_title("Entre las tres semillas del modelo", fontsize=10)
    ax.set_ylabel("índice de Rand ajustado")
    ax = axs[1]
    s = tp[(tp.algoritmo == "k-medoides")]
    for nom, col, mk in ESP12:
        q = s[(s.espacio == nom) & ((s.d == d) | (nom == "OM"))].rename(columns={"ari_entre_arranques": "valor"})
        curva(ax, q, "k", nom, col, mk)
    oh = s[s.espacio == "One-hot (Hamming)"]
    ax.plot(oh.k, oh.ari_entre_arranques, color=INK2, lw=1.4, ls=(0, (5, 3)), label="One-hot, Hamming (referencia)")
    ax.set_title("Entre los 10 arranques de k-medoides", fontsize=10)
    for ax in axs:
        ax.set_xticks(sorted(tp.k.unique()))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("número de grupos k")
        ax.set_ylim(0, 1.02)
    leyenda_abajo(fig, axs[1], 6)
    fig.suptitle(f"Estabilidad de las tipologías (k-medoides; embeddings con d = {d})", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig9_estabilidad.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()   # las figuras del informe anterior se reemplazan
    r = pd.read_csv(IN / "p11_reconstruccion.csv")
    acc = pd.read_csv(IN / "p11_accesibilidad.csv")
    tp = pd.read_csv(IN / "p12_tipologias.csv")
    sem = pd.read_csv(IN / "p12_semillas.csv")
    fig_reconstruccion(r, "acc_anio", "tipo", "fig1_exactitud_por_anio.png",
                       "Exactitud por año (cuándo), cada tipo pesa 1", "fracción de años correctos")
    fig_reconstruccion(r, "estados_ok", "tipo", "fig2_secuencia_de_estados.png",
                       "Secuencia de estados correcta (qué), cada tipo pesa 1", "fracción de trayectorias")
    fig_reconstruccion(r, "exacta", "tipo", "fig3_exacta_por_tipo.png",
                       "Reconstrucción exacta, cada tipo pesa 1", "fracción de trayectorias reconstruidas exactas")
    fig_reconstruccion(r, "exacta", "superficie", "fig4_exacta_por_superficie.png",
                       "Reconstrucción exacta, ponderada por superficie", "fracción de la superficie reconstruida exacta")
    fig_gradiente(r)
    fig_accesibilidad(acc, "vecinos")
    fig_accesibilidad(acc, "lineal")
    fig_tipologias(tp, "k-medoides")
    fig_tipologias(tp, "jerarquico completo")
    fig_estabilidad(sem, tp)
    print("figuras ->", OUT)


if __name__ == "__main__":
    main()
