"""Paper-quality English renders of the typology chronogram atlases.

Reads what `scripts/describe_clusters.py` already computed -- no model, no
torch, nothing recomputed:

    models/cluster_v2/typology{suffix}.csv
    models/cluster_v2/typology_chronograms.npz

and writes, per run, a PNG (300 dpi) + a vector PDF under `imgs/paper/`. Runs
above `--per-page` clusters (the two "fina" ones, k~118-120) are split into
several pages so panel titles stay legible; the rest (coarse, medium) fit on
one page each. Those same paginated runs also get a single-page "top-N"
summary atlas with only their most frequent clusters (`--top-n`).

Kept deliberately separate from `land2vec.typology` (which pulls in torch via
the tokenizer) so this can run in a bare matplotlib/pandas/numpy env.

Usage:
    python scripts/plot_paper_atlas.py                # las 6 corridas
    python scripts/plot_paper_atlas.py --only _coarse
    python scripts/plot_paper_atlas.py --per-page 36
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from land2vec.paths import rel  # noqa: E402
DATA_DIR = ROOT / "models" / "cluster_v2"
OUT_DIR = ROOT / "imgs" / "paper"

# same matrix as scripts/describe_clusters.py::LEVEL_SPECS
LEVEL_SPECS = [
    ("coarse / HDBSCAN", "_coarse"),
    ("coarse / non-HDBSCAN", "_coarse_parametric"),
    ("medium / HDBSCAN", "_medium"),
    ("medium / non-HDBSCAN", "_medium_parametric"),
    ("fine / HDBSCAN", ""),
    ("fine / non-HDBSCAN", "_parametric"),
]
NPZ_KEY = {"": "fina", "_parametric": "_parametric", "_medium": "_medium",
           "_medium_parametric": "_medium_parametric", "_coarse": "_coarse",
           "_coarse_parametric": "_coarse_parametric"}

# id -> (code, English name, color) -- same ids/colors as land2vec.typology.STATE_COLORS
STATES = [
    (0, "[UNK]", "unknown", "#d9d9d9"),
    (1, "A", "Agriculture", "#e6c229"),
    (2, "F", "Forest", "#1b7837"),
    (3, "G", "Grassland", "#a6d96a"),
    (4, "Wt", "Wetland", "#2b8cbe"),
    (5, "U", "Urban", "#d7301f"),
    (6, "Sh", "Shrubland", "#b8860b"),
    (7, "Sp", "Sparse cover", "#dfc27d"),
    (8, "B", "Bare soil", "#8c6d31"),
    (9, "Wa", "Water", "#08519c"),
    (10, "Nd", "No data", "#bdbdbd"),
]
COLOR_LIST = [s[3] for s in STATES]
YEAR_0 = 2000

FORMA_EN = {
    "estable": "stable",
    "monotónica": "monotonic",
    "oscilante": "oscillating",
    "múltiple": "multi-state",
}

GLOSA_EN = {
    "abandono agrícola / reforestación": "agricultural abandonment / reforestation",
    "anegamiento de bosque": "forest flooding",
    "anegamiento de área agrícola": "cropland flooding",
    "anegamiento de pastizal": "grassland flooding",
    "arbustal a pastizal": "shrubland to grassland",
    "arbustización de pastizal": "grassland shrubification",
    "avance agrícola sobre humedal": "agricultural encroachment on wetland",
    "avance del bosque sobre pastizal": "forest encroachment on grassland",
    "deforestación para agricultura": "deforestation for agriculture",
    "deforestación para pastura": "deforestation for pasture",
    "deforestación para urbanización": "deforestation for urbanization",
    "degradación forestal": "forest degradation",
    "densificación de cobertura esparsa": "sparse-cover densification",
    "desecación de humedal a arbustal": "wetland desiccation to shrubland",
    "desecación de humedal a pastizal": "wetland desiccation to grassland",
    "habilitación de arbustal para agricultura": "shrubland clearing for agriculture",
    "humedal a bosque": "wetland to forest",
    "intensificación: pastizal a agricultura": "intensification: grassland to agriculture",
    "pérdida de cobertura esparsa": "loss of sparse cover",
    "ralentización de arbustal": "shrubland thinning",
    "recuperación forestal desde arbustal": "forest recovery from shrubland",
    "revegetación de suelo desnudo": "revegetation of bare soil",
    "revegetación de suelo desnudo a pastizal": "revegetation of bare soil to grassland",
    "reversión a pastura": "reversion to pasture",
    "urbanización sobre arbustal": "urbanization over shrubland",
    "urbanización sobre área agrícola": "urbanization over cropland",
    "urbanización sobre pastizal": "urbanization over grassland",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,   # embed as real text, not paths -- editable/selectable in the PDF
    "ps.fonttype": 42,
})


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace("-", "").lower()


def translate_glosa(g: str) -> str:
    if g.startswith("oscilación "):
        return "oscillation " + g[len("oscilación "):]
    return GLOSA_EN.get(g, g)


def panel_title(row: pd.Series, total_n: int, wrap: int) -> str:
    share = row["n"] / total_n
    header = f"#{int(row['cluster'])} · n={int(row['n']):,} ({share:.1%})"
    dss = str(row["dss_text"]).replace("»", " → ")
    forma_en = FORMA_EN.get(row["forma"], row["forma"])
    anio = row["anio_cambio"]
    meta = f"{dss} · {forma_en}" + (f" · ~{int(anio)}" if pd.notna(anio) else "")
    glosa_en = translate_glosa(str(row["glosa"]))
    glosa_lines = [] if glosa_en == dss else textwrap.wrap(glosa_en, width=wrap, break_long_words=False)
    return "\n".join([header, meta, *glosa_lines])


def plot_chronogram(ax: plt.Axes, dist: np.ndarray, show_xticks: bool) -> None:
    years = YEAR_0 + np.arange(dist.shape[0])
    bottom = np.zeros(dist.shape[0])
    for s in range(dist.shape[1]):
        h = dist[:, s]
        if h.sum() == 0:
            continue
        ax.bar(years, h, bottom=bottom, width=1.0, color=COLOR_LIST[s], linewidth=0)
        bottom += h
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    if show_xticks:
        ticks = [years[0], years[len(years) // 2], years[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=6)
        ax.tick_params(axis="x", length=2, pad=1)
    else:
        ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#888888")


def active_states(chronos: np.ndarray) -> list[int]:
    mass = chronos.sum(axis=(0, 1))
    return [i for i in range(len(STATES)) if mass[i] > 0]


def draw_legend(fig: plt.Figure, state_ids: list[int]) -> None:
    handles = [Patch(facecolor=STATES[i][3], edgecolor="#888888", linewidth=0.5,
                      label=f"{STATES[i][1]} — {STATES[i][2]}")
               for i in state_ids]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8,
               handlelength=1.2, handleheight=1.2, columnspacing=1.4)


def plot_page(df_page: pd.DataFrame, chronos_page: np.ndarray, total_n: int, ncols: int,
              suptitle: str, state_ids: list[int], wrap: int) -> plt.Figure:
    k = len(df_page)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.05),
                              squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.10, wspace=0.04, hspace=0.10)
    for ax in axes.flat:
        ax.axis("off")
    for pos in range(k):
        r, c = divmod(pos, ncols)
        ax = axes[r][c]
        ax.axis("on")
        row = df_page.iloc[pos]
        is_bottom = (r == nrows - 1) or (pos + ncols >= k)
        plot_chronogram(ax, chronos_page[pos], show_xticks=is_bottom)
        ax.set_title(panel_title(row, total_n, wrap), fontsize=6.4, loc="left", linespacing=1.35)
    fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.94))
    draw_legend(fig, state_ids)
    return fig


def save(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=300)
    fig.savefig(path_no_ext.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  {rel(path_no_ext)}.{{png,pdf}}")


def run(name: str, suffix: str, per_page: int, top_n: int, summary_only: bool = False) -> None:
    csv_path = DATA_DIR / f"typology{suffix}.csv"
    if not csv_path.exists():
        print(f"skip {name}: {csv_path.name} not found")
        return
    df = pd.read_csv(csv_path)
    z = np.load(DATA_DIR / "typology_chronograms.npz")
    chronos = z[NPZ_KEY[suffix]]
    assert chronos.shape[0] == len(df), f"{name}: {chronos.shape[0]} chronograms vs {len(df)} csv rows"

    order = df["n"].values.argsort()[::-1]
    df = df.iloc[order].reset_index(drop=True)
    chronos = chronos[order]
    total_n = int(df["n"].sum())
    state_ids = active_states(chronos)
    k = len(df)

    slug = _slug(name)
    print(f"\n{name} (k={k})")

    if k <= per_page:
        if not summary_only:
            ncols = 5 if k <= 20 else 6
            wrap = 34 if ncols == 5 else 30
            fig = plot_page(df, chronos, total_n, ncols, f"Land-use trajectory typology — {name} (k={k})",
                             state_ids, wrap)
            save(fig, OUT_DIR / f"typology_atlas_{slug}")
        return

    if not summary_only:
        ncols, wrap = 6, 30
        n_pages = int(np.ceil(k / per_page))
        for p in range(n_pages):
            lo, hi = p * per_page, min((p + 1) * per_page, k)
            title = f"Land-use trajectory typology — {name} (k={k}), page {p + 1}/{n_pages}"
            fig = plot_page(df.iloc[lo:hi], chronos[lo:hi], total_n, ncols, title, state_ids, wrap)
            save(fig, OUT_DIR / f"typology_atlas_{slug}_p{p + 1}of{n_pages}")

    top_n = min(top_n, k)
    df_top, chronos_top = df.iloc[:top_n], chronos[:top_n]
    share = df_top["n"].sum() / total_n
    ncols = 5 if top_n <= 20 else 6
    wrap = 34 if ncols == 5 else 30
    title = f"Land-use trajectory typology — {name}, top {top_n} of k={k} clusters ({share:.1%} of records)"
    fig = plot_page(df_top, chronos_top, total_n, ncols, title, state_ids, wrap)
    save(fig, OUT_DIR / f"typology_atlas_{slug}_top{top_n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="un solo sufijo, p. ej. _coarse")
    ap.add_argument("--per-page", type=int, default=36, help="umbral de paginación (clusters por página)")
    ap.add_argument("--top-n", type=int, default=24,
                    help="clusters más frecuentes a mostrar en el resumen de las corridas paginadas")
    ap.add_argument("--summary-only", action="store_true",
                    help="generar sólo el resumen top-N de las corridas paginadas, sin re-escribir los atlas completos")
    args = ap.parse_args()

    specs = [(n, s) for n, s in LEVEL_SPECS if args.only is None or s == args.only]
    if not specs:
        raise SystemExit(f"--only {args.only!r} no coincide con ningún sufijo de LEVEL_SPECS")

    for name, suffix in specs:
        run(name, suffix, args.per_page, args.top_n, args.summary_only)


if __name__ == "__main__":
    main()
