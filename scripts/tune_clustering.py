"""Barre y elige una configuración de clustering sobre los embeddings z de la v2
(land2vec, `models/autoencoder_v2`) -- ver `docs/v2_autoencoder_training.md`
sección 7.2 para el criterio completo, `select_winner()` más abajo para la
regla de decisión exacta, y `src/land2vec/cluster.py` para las métricas.

Corre sobre las secuencias con al menos una transición (3,2% del pool de las 7
zonas de evaluación, 107.362 filas) de las 4 familias de clustering (KMeans,
GaussianMixture, HDBSCAN, aglomerativo/jerárquico). Todo en CPU -- torch solo
hace falta para decodificar centroides (`prototype_fidelity`).

Uso:
    # barridos (acumulan filas en <out-dir>/summary.csv)
    python scripts/tune_clustering.py --sweep kmeans       --out-dir models/cluster_v2
    python scripts/tune_clustering.py --sweep gmm          --out-dir models/cluster_v2
    python scripts/tune_clustering.py --sweep hdbscan      --out-dir models/cluster_v2
    python scripts/tune_clustering.py --sweep hierarchical --out-dir models/cluster_v2

    # elige ganadoras (ver select_winner() y run_select() más abajo): matriz de
    # 3 niveles de granularidad (fina / media k<=40 / gruesa k<=20) x 2 familias
    # (HDBSCAN / no-HDBSCAN), y para cada una etiqueta las dinámicas + el pool con
    # constantes submuestreadas al 15% -> chosen{_medium,_coarse}{,_parametric}.json
    python scripts/tune_clustering.py --select --out-dir models/cluster_v2

Smoke test rápido antes de un barrido completo:
    python scripts/tune_clustering.py --sweep kmeans --k-values 2 4 8 \\
        --spaces standard --out-dir /tmp/cluster_smoke
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# land2vec vive en src/; corré este script directo desde el repo, sin `pip install`
# ni PYTHONPATH (misma convención que la celda bootstrap de los notebooks, ver
# commit f4f4700). Si además hiciste `pip install -e .`, esto es inocuo.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from land2vec import cluster as C  # noqa: E402
from land2vec.utils import load_config, load_model  # noqa: E402

DATA_DIR = ROOT / "data"
IMGS_DIR = ROOT / "imgs"

# La banda alta (>20) existe para poder comparar KMeans/GMM/jerárquico contra el
# k efectivo de HDBSCAN (que llega a ~120 con min_cluster_size chico) -- sin ella
# el ganador fino de HDBSCAN no tenía contrincante a su k. Ver docs §7.2.
DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                    25, 30, 40, 50, 65, 80, 100, 120]

# Umbral de estabilidad (ARI de bootstrap) para entrar en la selección final --
# ver criterio de decisión en docs/v2_autoencoder_training.md §7.2.
STABILITY_THRESHOLD = 0.75


def _parse_min_samples(values: list[str]) -> list[int | None]:
    return [None if v.lower() == "none" else int(v) for v in values]


def load_model_and_dynamic_pool(model_dir: Path, data_dir: Path, device: str) -> tuple[torch.nn.Module, C.Pool]:
    print("Cargando modelo y embeddings de las 7 zonas de evaluación...")
    config = dataclasses.replace(load_config(model_dir), device=device)
    model = load_model(config, model_dir)
    pool = C.load_pool(C.ZONES, data_dir)
    dyn = pool.subset(C.dynamic_mask(pool.seqs))
    print(f"  secuencias con transición: {len(dyn):,} / {len(pool):,} ({len(dyn) / len(pool) * 100:.1f}%)")
    return model, dyn


def append_summary(out_dir: Path, rows: list[dict]) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    new_df = pd.DataFrame(rows)
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        combined = pd.concat([old_df[~old_df["run_id"].isin(new_df["run_id"])], new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(summary_path, index=False)
    return combined


def result_row(result: C.ClusterRunResult) -> dict:
    return {
        "run_id": result.run_id,
        "algo": result.algo,
        "space": result.space,
        "params": json.dumps(result.params, sort_keys=True),
        "eligible": result.eligible,
        **result.metrics,
    }


def sweep_kmeans(pool, model, args) -> list[dict]:
    rows = []
    for space in args.spaces:
        for k in args.k_values:
            t0 = time.time()
            result = C.run_config(
                pool, "kmeans", {"k": k}, space, model,
                device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
            )
            rows.append(result_row(result))
            print(f"  kmeans k={k:>2d} space={space:<8s} "
                  f"silhouette={result.metrics['silhouette_mean']:.4f} "
                  f"stability_ari={result.metrics['stability_ari']:.4f} "
                  f"proto_fidelity={result.metrics['prototype_fidelity']:.4f} "
                  f"({time.time() - t0:.1f}s)")
    return rows


def sweep_gmm(pool, model, args) -> list[dict]:
    rows = []
    for space in args.gmm_space:
        for cov in args.covariance_types:
            for k in args.k_values:
                t0 = time.time()
                result = C.run_config(
                    pool, "gmm", {"k": k, "covariance_type": cov}, space, model,
                    device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
                )
                rows.append(result_row(result))
                print(f"  gmm k={k:>2d} cov={cov:<5s} space={space:<8s} "
                      f"silhouette={result.metrics['silhouette_mean']:.4f} bic={result.metrics.get('bic'):.0f} "
                      f"stability_ari={result.metrics['stability_ari']:.4f} ({time.time() - t0:.1f}s)")
    return rows


def sweep_hdbscan(pool, model, args) -> list[dict]:
    rows = []
    min_samples_values = _parse_min_samples(args.min_samples)
    for space in args.hdbscan_space:
        for mcs in args.min_cluster_sizes:
            for ms in min_samples_values:
                t0 = time.time()
                result = C.run_config(
                    pool, "hdbscan", {"min_cluster_size": mcs, "min_samples": ms}, space, model,
                    device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
                )
                rows.append(result_row(result))
                print(f"  hdbscan min_cluster_size={mcs:>5d} min_samples={str(ms):<5s} space={space:<8s} "
                      f"k_effective={result.metrics['k_effective']} noise_frac={result.metrics['noise_frac']:.3f} "
                      f"silhouette={result.metrics['silhouette_mean']:.4f} ({time.time() - t0:.1f}s)")
    return rows


def sweep_hierarchical(pool, model, args) -> list[dict]:
    """Ward/average/complete, ajustados sobre una submuestra estratificada
    (`args.hier_sample`) y evaluados sobre el pool completo de 107k filas por
    centroide más cercano -- ver el docstring de `land2vec.cluster.
    run_hierarchical_config` para por qué (Ward directo sobre el pool completo,
    con o sin restricción de conectividad k-NN, se probó infeasible en este
    entorno). El linkage se calcula una sola vez por (method, space) -- cacheado
    en un dict nuevo por cada `space`, ya que el árbol depende de en qué espacio
    se ajustó -- y se reusa para todo el barrido de `k`, ya que el árbol
    completo no depende de k."""
    rows = []
    dendrogram_payload = None
    best_coph = -1.0

    for space in args.hier_space:
        linkage_cache: dict[str, np.ndarray] = {}
        for method in args.hier_linkages:
            for k in args.k_values:
                t0 = time.time()
                result = C.run_hierarchical_config(
                    pool, method, k, space, args.hier_sample, model,
                    device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
                    linkage_cache=linkage_cache,
                )
                rows.append(result_row(result))
                print(f"  hierarchical method={method:<9s} k={k:>2d} space={space:<8s} "
                      f"silhouette={result.metrics['silhouette_mean']:.4f} "
                      f"cophenetic_corr={result.metrics['cophenetic_corr']:.4f} "
                      f"stability_ari={result.metrics['stability_ari']:.4f} "
                      f"proto_fidelity={result.metrics['prototype_fidelity']:.4f} ({time.time() - t0:.1f}s)")
                if result.metrics["cophenetic_corr"] > best_coph:
                    best_coph = result.metrics["cophenetic_corr"]
                    dendrogram_payload = (f"{method}/{space}", linkage_cache[method])

    if dendrogram_payload is not None and args.dendrogram_out is not None:
        plot_dendrogram(dendrogram_payload, args.dendrogram_out)

    return rows


def plot_dendrogram(payload: tuple[str, np.ndarray], out_path: Path) -> None:
    method, Z = payload
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=40, ax=ax, color_threshold=0)
    ax.set_title(f"Dendrograma (submuestra, linkage={method}, últimas 40 fusiones)")
    ax.set_xlabel("tamaño del cluster (o índice si es una hoja)")
    ax.set_ylabel("distancia de fusión")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  dendrograma guardado: {out_path}")


def plot_selection_curves(summary: pd.DataFrame, out_path: Path) -> None:
    """Curvas de las métricas principales vs. k_effective, un panel por métrica, una
    línea por (algo, space/variant). Usa `k_effective` (calculado por
    `land2vec.cluster.size_stats` para las 4 familias) en vez de `params["k"]` --
    HDBSCAN no tiene una clave "k" en sus params (solo min_cluster_size/min_samples),
    así que extraer "k" de params lo dejaba afuera del gráfico por completo pese a
    ser la familia ganadora (ver docs/v2_autoencoder_training.md §7.2)."""
    df = summary.copy()
    parsed_params = df["params"].apply(json.loads)
    df["k"] = df["k_effective"]
    df = df[df["k"].notna() & (df["k"] > 0)].copy()
    df["k"] = df["k"].astype(int)
    method_suffix = parsed_params.apply(lambda p: f"/{p['method']}" if "method" in p else "")
    df["series"] = df["algo"] + "/" + df["space"].fillna("") + method_suffix[df.index]

    metrics = ["silhouette_mean", "stability_ari", "prototype_fidelity", "spatial_coherence"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics):
        if metric not in df.columns:
            continue
        for series, sub in df.groupby("series"):
            sub = sub.sort_values("k")
            if sub[metric].notna().sum() == 0:
                continue
            ax.plot(sub["k"], sub[metric], marker="o", markersize=3, label=series, alpha=0.8)
        ax.set_xlabel("k efectivo")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("Barrido de clustering: métricas de selección vs. k efectivo")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curvas guardadas: {out_path}")


def select_winner(
    summary: pd.DataFrame, k_max: int | None = None, max_noise_frac: float | None = None
) -> tuple[pd.Series, bool]:
    """Aplica el criterio de docs/v2_autoencoder_training.md §7.2: entre las
    config elegibles con stability_ari >= STABILITY_THRESHOLD (con k_effective
    <= k_max si se pasa -- usado por run_select para los niveles medio/grueso --
    y noise_frac <= max_noise_frac si se pasa), la de
    mejor prototype_fidelity; desempate por silhouette_mean y, dentro del
    ruido, por menor k_effective. Si ninguna alcanza el umbral de estabilidad,
    cae a la de mayor stability_ari entre las candidatas (fallback=True, hay
    que revisarlo a mano).

    `max_noise_frac` existe porque `prototype_fidelity` solo se calcula sobre
    los miembros no-ruido de cada cluster (ver `land2vec.cluster.
    prototype_fidelity`) -- sin un tope, el criterio recompensa mecánicamente a
    HDBSCAN por descartar como ruido los puntos difíciles (hasta 45% de las
    filas en configs con `min_cluster_size` alto), no solo por tener una
    estructura de cluster genuinamente mejor. El tope filtra esas configs
    *antes* de rankear por fidelidad, para que la comparación entre familias
    sea más pareja.

    El desempate final usa `k_effective` (no `params["k"]`): HDBSCAN no tiene una
    clave "k" en sus params, así que desempatar por `params["k"]` la dejaba con
    NaN y, por cómo ordena pandas, siempre al final del desempate -- aunque en la
    práctica nunca se llegó a necesitar (HDBSCAN ganó por prototype_fidelity sin
    empates), ver docs/v2_autoencoder_training.md §7.2."""
    elig = summary[summary["eligible"].astype(bool)].dropna(subset=["stability_ari", "prototype_fidelity"]).copy()
    if k_max is not None:
        elig = elig[elig["k_effective"] <= k_max]
    if max_noise_frac is not None:
        elig = elig[elig["noise_frac"].fillna(0.0) <= max_noise_frac]
    if elig.empty:
        raise ValueError(
            f"summary.csv no tiene ninguna corrida elegible con métricas completas "
            f"(k_max={k_max}, max_noise_frac={max_noise_frac})"
        )
    elig["k"] = elig["k_effective"]

    stable = elig[elig["stability_ari"] >= STABILITY_THRESHOLD]
    fallback = stable.empty
    pool_for_pick = elig if fallback else stable
    if fallback:
        print(f"AVISO: ninguna config candidata (k_max={k_max}, max_noise_frac={max_noise_frac}) alcanza "
              f"stability_ari >= {STABILITY_THRESHOLD}; se elige por mayor stability_ari entre las candidatas. "
              f"Revisar a mano.")
        pool_for_pick = pool_for_pick.sort_values("stability_ari", ascending=False)
    else:
        pool_for_pick = pool_for_pick.sort_values(
            ["prototype_fidelity", "silhouette_mean", "k"], ascending=[False, False, True]
        )
    return pool_for_pick.iloc[0], fallback


def refit_and_save(
    label: str, winner: pd.Series, fallback: bool, model, dyn_pool: "C.Pool", pooled: "C.Pool", args, suffix: str
) -> None:
    """Reajusta una fila ganadora de summary.csv (con más bootstraps, para un
    número de estabilidad más confiable en la config final que en el barrido) y
    guarda chosen{suffix}.json + clusters_dynamic{suffix}.zip +
    clusters_pooled_subsampled{suffix}.zip. `suffix` combina nivel de granularidad
    ("" fina / "_medium" media / "_coarse" gruesa) y familia ("" HDBSCAN /
    "_parametric" no-HDBSCAN) -- ver run_select. `pooled` (el pool con constantes
    submuestreadas al 15%) se pasa ya cargado: es el mismo para las 6 celdas."""
    params = json.loads(winner["params"])
    print(f"\nGanadora ({label}): {winner['run_id']}")
    print(f"  algo={winner['algo']} space={winner['space']} params={params}")
    print(f"  silhouette={winner['silhouette_mean']:.4f} stability_ari={winner['stability_ari']:.4f} "
          f"prototype_fidelity={winner['prototype_fidelity']:.4f} spatial_coherence={winner['spatial_coherence']:.4f}")

    print("Reajustando (con más bootstraps para el número final de estabilidad)...")
    if winner["algo"] == "hierarchical":
        final = C.run_hierarchical_config(
            dyn_pool, params["method"], params["k"], winner["space"], params["fit_sample_size"], model,
            device=args.device, seed=args.seed, n_boot=args.select_n_boot, boot_cap=args.boot_cap,
        )
    else:
        final = C.run_config(
            dyn_pool, winner["algo"], params, winner["space"], model,
            device=args.device, seed=args.seed, n_boot=args.select_n_boot, boot_cap=args.boot_cap,
        )
    print(f"  stability_ari (n_boot={args.select_n_boot}): {final.metrics['stability_ari']:.4f}")

    # Tope "sin tipificar" para el mapa del pool: el pool se etiqueta por Voronoi
    # (assign_pool) incluso para HDBSCAN/jerárquico, que no tienen centroides ni
    # clase de ruido propia -- sin un tope de distancia, cada punto del pool queda
    # asignado a *algún* cluster y el mapa sobrestima cuánto territorio está
    # genuinamente tipificado frente a clusters_dynamic (que preserva el -1
    # honesto de HDBSCAN). El tope se calibra sobre la geometría real de esta
    # config: percentil `--untyped-dist-pct` de la distancia al centroide más
    # cercano entre los puntos del pool dinámico que la config NO dejó como ruido.
    z_fit_dyn = final.transform.apply(dyn_pool.z)
    _, dyn_dist = C.assign_by_centroid_with_dist(z_fit_dyn, final.centers)
    real = final.labels != -1
    untyped_thresh = float(np.percentile(dyn_dist[real], args.untyped_dist_pct))
    print(f"  tope 'sin tipificar' (P{args.untyped_dist_pct:g} de la dist. no-ruido): {untyped_thresh:.4f}")

    chosen = {
        "level": label,
        "run_id": final.run_id,
        "algo": final.algo,
        "params": final.params,
        "space": final.space,
        "transform": final.transform.to_jsonable(),
        "centers": C.to_jsonable(final.centers),
        "raw_centers": C.to_jsonable(final.raw_centers),
        "metrics": C.to_jsonable(final.metrics),
        "fallback": fallback,
        "stability_threshold": STABILITY_THRESHOLD,
        "untyped_dist_pct": args.untyped_dist_pct,
        "untyped_dist_threshold": untyped_thresh,
    }
    chosen_path = args.out_dir / f"chosen{suffix}.json"
    chosen_path.write_text(json.dumps(chosen, indent=2))
    print(f"Config elegida guardada: {chosen_path}")

    dyn_out = pd.DataFrame({"ID": dyn_pool.ids, "zone": dyn_pool.zone, "cluster": final.labels})
    dyn_out_path = args.data_dir / f"clusters_dynamic{suffix}.zip"
    dyn_out.to_csv(dyn_out_path, index=False, compression="zip")
    print(f"Etiquetas (dinámicas): {dyn_out_path} ({len(dyn_out):,} filas)")

    print("Etiquetando el pool con constantes submuestreadas al 15%...")
    pooled_idx, pooled_dist = C.assign_pool(
        pooled.z, final.transform, final.centers, return_dist=True
    )
    pooled_labels = np.where(pooled_dist > untyped_thresh, -1, pooled_idx)
    n_untyped = int((pooled_labels == -1).sum())
    print(f"  sin tipificar (dist > {untyped_thresh:.4f}): {n_untyped:,} / {len(pooled_labels):,} "
          f"({n_untyped / len(pooled_labels):.1%})")
    pooled_out = pd.DataFrame({"ID": pooled.ids, "zone": pooled.zone, "cluster": pooled_labels})
    pooled_out_path = args.data_dir / f"clusters_pooled_subsampled{suffix}.zip"
    pooled_out.to_csv(pooled_out_path, index=False, compression="zip")
    print(f"Etiquetas (pool submuestreado): {pooled_out_path} ({len(pooled_out):,} filas)")


def run_select(args) -> None:
    out_dir = args.out_dir
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} no existe -- corré al menos un --sweep antes de --select")
    summary = pd.read_csv(summary_path)

    # Matriz de 3 niveles de granularidad x 2 familias. Para cada celda se aplica
    # el mismo criterio de select_winner (ver §7.2) restringido a ese tope de k,
    # ese tope de ruido y esa familia. Sirve para comparar HDBSCAN (basado en
    # densidad, con clase de ruido) contra lo mejor que da el resto (KMeans/GMM/
    # jerárquico, que asignan todos los puntos) a granularidad pareja -- no solo
    # sobre la fila de summary.csv sino con el tratamiento completo (prototipos
    # decodificados, mapa del pool). Una celda se omite si su ganadora coincide
    # con una ya guardada (p. ej. si la HDBSCAN fina ya cae bajo el tope medio).
    level_specs = [
        ("fina",   "",        None,               args.max_noise_frac),
        ("media",  "_medium", args.medium_k_max,  args.medium_max_noise_frac),
        ("gruesa", "_coarse", args.coarse_k_max,  args.coarse_max_noise_frac),
    ]
    family_specs = [
        ("HDBSCAN",    "",            lambda df: df[df["algo"] == "hdbscan"]),
        ("no-HDBSCAN", "_parametric", lambda df: df[df["algo"] != "hdbscan"]),
    ]

    model, dyn_pool = load_model_and_dynamic_pool(args.model, args.data_dir, args.device)
    print("Cargando el pool con constantes submuestreadas al 15% (una vez para las 6 celdas)...")
    pooled = C.load_pool_subsampled(C.ZONES, args.data_dir, max_fraction=0.15, seed=args.seed)

    saved: dict[str, str] = {}  # run_id -> etiqueta con la que se guardó
    for lvl_name, lvl_suffix, k_max, max_noise in level_specs:
        for fam_name, fam_suffix, fam_filter in family_specs:
            k_lbl = "" if k_max is None else f" (k<={k_max})"
            label = f"{lvl_name}{k_lbl} / {fam_name}"
            suffix = f"{lvl_suffix}{fam_suffix}"
            try:
                winner, fallback = select_winner(fam_filter(summary), k_max=k_max, max_noise_frac=max_noise)
            except ValueError as exc:
                print(f"{label}: sin config candidata ({exc}); se omite.")
                continue
            if winner["run_id"] in saved:
                print(f"{label}: la ganadora ({winner['run_id']}) ya se guardó como '{saved[winner['run_id']]}'; se omite.")
                continue
            refit_and_save(label, winner, fallback, model, dyn_pool, pooled, args, suffix=suffix)
            saved[winner["run_id"]] = label

    plot_selection_curves(summary, IMGS_DIR / "v2_cluster_selection.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=["kmeans", "gmm", "hdbscan", "hierarchical"], default=None)
    parser.add_argument("--select", action="store_true")

    parser.add_argument("--model", type=Path, default=Path("models/autoencoder_v2"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("models/cluster_v2"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--spaces", choices=["raw", "standard", "l2"], nargs="+", default=["raw", "standard", "l2"],
                         help="usado por --sweep kmeans")
    parser.add_argument("--gmm-space", choices=["raw", "standard", "l2"], nargs="+", default=["raw", "standard", "l2"])
    parser.add_argument("--covariance-types", nargs="+", default=["full", "diag"])
    parser.add_argument("--hdbscan-space", choices=["raw", "standard", "l2"], nargs="+", default=["raw", "standard", "l2"])
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[250, 500, 1000, 2500])
    parser.add_argument("--min-samples", nargs="+", default=["none", "25"])
    parser.add_argument("--hier-space", choices=["raw", "standard", "l2"], nargs="+", default=["raw", "standard", "l2"])
    parser.add_argument("--hier-sample", type=int, default=5000, help="tamaño de la submuestra estratificada para ajustar ward/average/complete (se extiende al pool completo por centroide más cercano) -- 25000 midió memoria estable en una corrida aislada pero la acumuló sin liberarla a través de las ~6 refit de estabilidad por config del barrido completo, hasta hacer OOM-kill (ver commit); 5000 mantiene el RSS plano y cada config en ~27s")
    parser.add_argument("--hier-linkages", nargs="+", default=["ward", "average", "complete"])
    parser.add_argument("--dendrogram-out", type=Path, default=IMGS_DIR / "v2_cluster_dendrogram.png")
    parser.add_argument("--coarse-k-max", type=int, default=20,
                         help="--select elige tres niveles de tipología: fina (sin tope de k), media "
                              "(k_effective <= --medium-k-max) y gruesa (k_effective <= este valor). La gruesa "
                              "es la más apretada, pensada para un mapa/narrativa legible")
    parser.add_argument("--medium-k-max", type=int, default=40,
                         help="techo de k_effective del nivel intermedio (chosen_medium.json): entre la fina "
                              "y la gruesa. ~30-40 clusters suele ser el sweet spot donde prototype_fidelity "
                              "sube fuerte sin que k se vuelva ilegible. Un nivel se omite si su ganadora "
                              "coincide con la de otro ya guardado")
    parser.add_argument("--max-noise-frac", type=float, default=0.5,
                         help="--select descarta de la ganadora fina cualquier config con noise_frac por encima de "
                              "este valor, antes de rankear por prototype_fidelity -- sin este tope, HDBSCAN gana "
                              "en parte por poder descartar como ruido los puntos difíciles (prototype_fidelity solo "
                              "se calcula sobre miembros no-ruido), no solo por tener mejor estructura de cluster")
    parser.add_argument("--coarse-max-noise-frac", type=float, default=0.25,
                         help="igual que --max-noise-frac pero para la ganadora gruesa/interpretable (k<=coarse-k-max) "
                              "-- más estricto porque una tipología pensada para mapas/narrativa pierde sentido si "
                              "una fracción grande del territorio queda sin tipificar")
    parser.add_argument("--medium-max-noise-frac", type=float, default=0.25,
                         help="igual que --coarse-max-noise-frac pero para el nivel intermedio (por defecto el "
                              "mismo tope; subilo si querés dejar entrar configs de k medio un poco más ruidosas "
                              "pero de mayor fidelidad)")
    parser.add_argument("--untyped-dist-pct", type=float, default=95.0,
                         help="percentil de la distancia al centroide más cercano (medida sobre los puntos no-ruido "
                              "del pool dinámico) que define el tope 'sin tipificar' del mapa del pool: en "
                              "clusters_pooled_subsampled{,_coarse}.zip, los puntos más lejos que ese tope salen "
                              "como -1 en vez de asignados por Voronoi, para que ese archivo no sobrestime la "
                              "cobertura frente al -1 honesto de clusters_dynamic (ver docs §7.2, Nota metodológica)")

    parser.add_argument("--n-boot", type=int, default=3, help="bootstraps de stability_ari durante el barrido (menos que en --select por tiempo de cómputo)")
    parser.add_argument("--select-n-boot", type=int, default=10, help="bootstraps de stability_ari al re-ajustar la config ganadora en --select")
    parser.add_argument("--boot-cap", type=int, default=20000, help="tope de filas por submuestra de bootstrap")

    args = parser.parse_args()
    if not args.sweep and not args.select:
        parser.error("pasá --sweep <algo> y/o --select")

    if args.sweep:
        model, pool = load_model_and_dynamic_pool(args.model, args.data_dir, args.device)
        print(f"\n=== barrido: {args.sweep} ===")
        t0 = time.time()
        sweep_fn = {"kmeans": sweep_kmeans, "gmm": sweep_gmm, "hdbscan": sweep_hdbscan, "hierarchical": sweep_hierarchical}[args.sweep]
        rows = sweep_fn(pool, model, args)
        summary = append_summary(args.out_dir, rows)
        print(f"\nBarrido {args.sweep} terminado en {time.time() - t0:.1f}s -- {len(rows)} corridas nuevas, "
              f"{len(summary)} filas totales en {args.out_dir / 'summary.csv'}")

    if args.select:
        run_select(args)


if __name__ == "__main__":
    main()
