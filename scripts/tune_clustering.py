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

    # elige la config ganadora (ver select_winner() más abajo) y etiqueta
    # las dinámicas + el pool con constantes submuestreadas al 15%
    python scripts/tune_clustering.py --select --out-dir models/cluster_v2

Smoke test rápido antes de un barrido completo:
    python scripts/tune_clustering.py --sweep kmeans --k-values 2 4 8 \\
        --spaces standard --out-dir /tmp/cluster_smoke
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from land2vec import cluster as C
from land2vec.utils import load_config, load_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMGS_DIR = Path(__file__).resolve().parent.parent / "imgs"

DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

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
    for cov in args.covariance_types:
        for k in args.k_values:
            t0 = time.time()
            result = C.run_config(
                pool, "gmm", {"k": k, "covariance_type": cov}, args.gmm_space, model,
                device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
            )
            rows.append(result_row(result))
            print(f"  gmm k={k:>2d} cov={cov:<5s} "
                  f"silhouette={result.metrics['silhouette_mean']:.4f} bic={result.metrics.get('bic'):.0f} "
                  f"stability_ari={result.metrics['stability_ari']:.4f} ({time.time() - t0:.1f}s)")
    return rows


def sweep_hdbscan(pool, model, args) -> list[dict]:
    rows = []
    min_samples_values = _parse_min_samples(args.min_samples)
    for mcs in args.min_cluster_sizes:
        for ms in min_samples_values:
            t0 = time.time()
            result = C.run_config(
                pool, "hdbscan", {"min_cluster_size": mcs, "min_samples": ms}, args.hdbscan_space, model,
                device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
            )
            rows.append(result_row(result))
            print(f"  hdbscan min_cluster_size={mcs:>5d} min_samples={str(ms):<5s} "
                  f"k_effective={result.metrics['k_effective']} noise_frac={result.metrics['noise_frac']:.3f} "
                  f"silhouette={result.metrics['silhouette_mean']:.4f} ({time.time() - t0:.1f}s)")
    return rows


def sweep_hierarchical(pool, model, args) -> list[dict]:
    """Ward/average/complete, ajustados sobre una submuestra estratificada
    (`args.hier_sample`) y evaluados sobre el pool completo de 107k filas por
    centroide más cercano -- ver el docstring de `land2vec.cluster.
    run_hierarchical_config` para por qué (Ward directo sobre el pool completo,
    con o sin restricción de conectividad k-NN, se probó infeasible en este
    entorno). El linkage se calcula una sola vez por `method` (cacheado) y se
    reusa para todo el barrido de `k`, ya que el árbol completo no depende de k."""
    rows = []
    dendrogram_payload = None
    best_coph = -1.0
    linkage_cache: dict[str, np.ndarray] = {}

    for method in args.hier_linkages:
        linkage_cache.clear()
        for k in args.k_values:
            t0 = time.time()
            result = C.run_hierarchical_config(
                pool, method, k, args.hier_space, args.hier_sample, model,
                device=args.device, seed=args.seed, n_boot=args.n_boot, boot_cap=args.boot_cap,
                linkage_cache=linkage_cache,
            )
            rows.append(result_row(result))
            print(f"  hierarchical method={method:<9s} k={k:>2d} "
                  f"silhouette={result.metrics['silhouette_mean']:.4f} "
                  f"cophenetic_corr={result.metrics['cophenetic_corr']:.4f} "
                  f"stability_ari={result.metrics['stability_ari']:.4f} "
                  f"proto_fidelity={result.metrics['prototype_fidelity']:.4f} ({time.time() - t0:.1f}s)")
            if result.metrics["cophenetic_corr"] > best_coph:
                best_coph = result.metrics["cophenetic_corr"]
                dendrogram_payload = (method, linkage_cache[method])

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
    "Curvas de las métricas principales vs. k, un panel por métrica, una línea por (algo, space/variant)."
    df = summary.copy()
    parsed_params = df["params"].apply(json.loads)
    df["k"] = parsed_params.apply(lambda p: p.get("k"))
    df = df[df["k"].notna()].copy()
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
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("Barrido de clustering: métricas de selección vs. k")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curvas guardadas: {out_path}")


def select_winner(summary: pd.DataFrame, k_max: int | None = None) -> tuple[pd.Series, bool]:
    """Aplica el criterio de docs/v2_autoencoder_training.md §7.2: entre las
    config elegibles con stability_ari >= STABILITY_THRESHOLD (y, si `k_max` no
    es None, con k_effective <= k_max -- usado para la selección "gruesa"
    interpretable, ver select_coarse_winner), la de mejor prototype_fidelity;
    desempate por silhouette_mean y, dentro del ruido, por menor k. Si ninguna
    alcanza el umbral, cae a la de mayor stability_ari entre las candidatas
    (fallback=True, hay que revisarlo a mano)."""
    elig = summary[summary["eligible"].astype(bool)].dropna(subset=["stability_ari", "prototype_fidelity"]).copy()
    if k_max is not None:
        elig = elig[elig["k_effective"] <= k_max]
    if elig.empty:
        raise ValueError(f"summary.csv no tiene ninguna corrida elegible con métricas completas (k_max={k_max})")
    elig["k"] = elig["params"].apply(lambda p: json.loads(p).get("k", np.nan))

    stable = elig[elig["stability_ari"] >= STABILITY_THRESHOLD]
    fallback = stable.empty
    pool_for_pick = elig if fallback else stable
    if fallback:
        print(f"AVISO: ninguna config candidata (k_max={k_max}) alcanza stability_ari >= {STABILITY_THRESHOLD}; "
              f"se elige por mayor stability_ari entre las candidatas. Revisar a mano.")
        pool_for_pick = pool_for_pick.sort_values("stability_ari", ascending=False)
    else:
        pool_for_pick = pool_for_pick.sort_values(
            ["prototype_fidelity", "silhouette_mean", "k"], ascending=[False, False, True]
        )
    return pool_for_pick.iloc[0], fallback


def select_coarse_winner(summary: pd.DataFrame, k_max: int) -> tuple[pd.Series, bool]:
    """La ganadora fina (select_winner sin tope) puede tener un k grande y poco
    legible como tipología resumida (HDBSCAN con min_cluster_size chico, p. ej.,
    encuentra ~100+ clusters finos pero de alta fidelidad). Esta variante aplica
    el mismo criterio restringido a k_effective <= k_max, para tener también una
    versión "gruesa" pensada para mapas/narrativa en vez de solo precisión."""
    return select_winner(summary, k_max=k_max)


def refit_and_save(
    label: str, winner: pd.Series, fallback: bool, model, dyn_pool: "C.Pool", args, suffix: str
) -> None:
    """Reajusta una fila ganadora de summary.csv (con más bootstraps, para un
    número de estabilidad más confiable en la config final que en el barrido) y
    guarda chosen{suffix}.json + clusters_dynamic{suffix}.zip +
    clusters_pooled_subsampled{suffix}.zip. `suffix` distingue el nivel fino
    ("") del grueso interpretable ("_coarse") -- ver select_coarse_winner."""
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
    }
    chosen_path = args.out_dir / f"chosen{suffix}.json"
    chosen_path.write_text(json.dumps(chosen, indent=2))
    print(f"Config elegida guardada: {chosen_path}")

    dyn_out = pd.DataFrame({"ID": dyn_pool.ids, "zone": dyn_pool.zone, "cluster": final.labels})
    dyn_out_path = args.data_dir / f"clusters_dynamic{suffix}.zip"
    dyn_out.to_csv(dyn_out_path, index=False, compression="zip")
    print(f"Etiquetas (dinámicas): {dyn_out_path} ({len(dyn_out):,} filas)")

    print("Etiquetando el pool con constantes submuestreadas al 15%...")
    pooled = C.load_pool_subsampled(C.ZONES, args.data_dir, max_fraction=0.15, seed=args.seed)
    pooled_labels = C.assign_pool(pooled.z, final.transform, final.centers)
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

    fine_winner, fine_fallback = select_winner(summary)
    coarse_winner, coarse_fallback = select_coarse_winner(summary, k_max=args.coarse_k_max)
    same_winner = fine_winner["run_id"] == coarse_winner["run_id"]
    if same_winner:
        print(f"La ganadora fina ya cumple k_effective <= {args.coarse_k_max}: un solo nivel, no hace falta el grueso.")

    model, dyn_pool = load_model_and_dynamic_pool(args.model, args.data_dir, args.device)

    refit_and_save("fina", fine_winner, fine_fallback, model, dyn_pool, args, suffix="")
    if not same_winner:
        refit_and_save("gruesa (k<=%d)" % args.coarse_k_max, coarse_winner, coarse_fallback, model, dyn_pool, args, suffix="_coarse")

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
    parser.add_argument("--gmm-space", choices=["raw", "standard", "l2"], default="standard")
    parser.add_argument("--covariance-types", nargs="+", default=["full", "diag"])
    parser.add_argument("--hdbscan-space", choices=["raw", "standard", "l2"], default="standard")
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[250, 500, 1000, 2500])
    parser.add_argument("--min-samples", nargs="+", default=["none", "25"])
    parser.add_argument("--hier-space", choices=["raw", "standard", "l2"], default="standard")
    parser.add_argument("--hier-sample", type=int, default=5000, help="tamaño de la submuestra estratificada para ajustar ward/average/complete (se extiende al pool completo por centroide más cercano) -- 25000 midió memoria estable en una corrida aislada pero la acumuló sin liberarla a través de las ~6 refit de estabilidad por config del barrido completo, hasta hacer OOM-kill (ver commit); 5000 mantiene el RSS plano y cada config en ~27s")
    parser.add_argument("--hier-linkages", nargs="+", default=["ward", "average", "complete"])
    parser.add_argument("--dendrogram-out", type=Path, default=IMGS_DIR / "v2_cluster_dendrogram.png")
    parser.add_argument("--coarse-k-max", type=int, default=20,
                         help="--select también elige, aparte de la ganadora sin tope, la mejor config con "
                              "k_effective <= este valor -- una tipología gruesa/interpretable además de la fina")

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
