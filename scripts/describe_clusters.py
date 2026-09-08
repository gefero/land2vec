"""Describe e interpreta las seis tipologías de trayectoria de la v2 (la matriz de
3 granularidades x 2 familias que deja `scripts/tune_clustering.py --select`, ver
`docs/v2_autoencoder_training.md` §7.2 / §7.5).

Es el **productor**: `notebooks/cluster_evaluation.ipynb` §5 y el navegador
`docs/typology/index.html` consumen su salida y no recalculan nada. El trabajo
está en dos módulos:

- `land2vec.typology` -- batería descriptiva estilo TraMineR sobre las secuencias
  de cada cluster (cronograma, secuencia modal, tasas de transición, índices de
  complejidad) + etiquetado automático + comparación entre corridas;
- `land2vec.seqdist` -- disimilitud entre secuencias (Optimal Matching / DHD /
  Hamming) sobre las 1.128 secuencias distintas ponderadas del pool dinámico, y
  con ella: pseudo-R² de discrepancia, ASW en espacio de secuencias y secuencias
  representativas (`seqrplot`).

Uso:
    python scripts/describe_clusters.py                       # las 6 corridas
    python scripts/describe_clusters.py --only _medium        # una sola
    python scripts/describe_clusters.py --seqdist-method dhd   # (default: om)
    python scripts/describe_clusters.py --no-seqdist           # solo descriptivo

Prerrequisitos (los deja `tune_clustering.py --select`):
    models/cluster_v2/chosen{,_medium,_coarse}{,_parametric}.json
    data/clusters_dynamic{,_medium,_coarse}{,_parametric}.zip

Escribe:
    models/cluster_v2/typology{suffix}.csv        una fila por cluster (344 en total)
    models/cluster_v2/typology_chronograms.npz    tensores (k, 23, 11) por corrida
    models/cluster_v2/typology_crossrun.csv       ARI/NMI/cobertura entre corridas
    models/cluster_v2/typology_seqdist.csv        pseudo-R² / ASW por corrida (OM)
    docs/typology/typology_browser.json           payload del navegador (GitHub Pages)
    imgs/v2_typology_atlas_{slug}.png             un atlas de cronogramas por corrida
    imgs/v2_typology_crossrun.png                 heatmaps ARI/NMI
    imgs/v2_typology_alluvial_{a}__{b}.png        diagramas aluviales entre corridas
    imgs/v2_typology_seqdist.png                  pseudo-R² y ASW vs. k (las 6 corridas)
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# land2vec vive en src/; corré este script directo desde el repo, sin `pip install`
# ni PYTHONPATH (misma convención que la celda bootstrap de los notebooks, ver
# commit f4f4700). Si además hiciste `pip install -e .`, esto es inocuo.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from land2vec import cluster as C  # noqa: E402
from land2vec import seqdist as SD  # noqa: E402
from land2vec import typology as TY  # noqa: E402
from land2vec.utils import load_config, load_model  # noqa: E402

DATA_DIR = ROOT / "data"
IMGS_DIR = ROOT / "imgs"
PAGES_DIR = ROOT / "docs" / "typology"

# nombre legible -> sufijo de archivo (combina granularidad y familia), igual que
# LEVEL_SPECS en la celda 7 del notebook.
LEVEL_SPECS = [
    ("fina / HDBSCAN", ""),
    ("fina / no-HDBSCAN", "_parametric"),
    ("media / HDBSCAN", "_medium"),
    ("media / no-HDBSCAN", "_medium_parametric"),
    ("gruesa / HDBSCAN", "_coarse"),
    ("gruesa / no-HDBSCAN", "_coarse_parametric"),
]


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace("-", "").lower()


def load_model_and_dynamic_pool(model_dir: Path, data_dir: Path, device: str):
    "Mismo pool alineado que usa tune_clustering.py: 7 zonas -> subset con transición."
    print("Cargando modelo y embeddings de las 7 zonas...")
    config = dataclasses.replace(load_config(model_dir), device=device)
    model = load_model(config, model_dir)
    pool = C.load_pool(C.ZONES, data_dir)
    dyn = pool.subset(C.dynamic_mask(pool.seqs))
    print(f"  secuencias con transición: {len(dyn):,} / {len(pool):,}")
    return model, dyn


def decode_prototypes(model: torch.nn.Module, raw_centers: np.ndarray, device: str) -> np.ndarray:
    "Centroides en z crudo -> trayectoria prototípica por argmax del decoder. (k, 23)."
    with torch.inference_mode():
        logits = model.decode(torch.tensor(raw_centers, dtype=torch.float32, device=device))
    return logits.argmax(-1).cpu().numpy()


def nearest_member_seqs(z_members: np.ndarray, raw_center: np.ndarray, seqs_members: np.ndarray, n: int = 3):
    "Las `n` secuencias reales del cluster más cercanas al centroide (en z crudo)."
    d = np.linalg.norm(z_members - raw_center, axis=1)
    return [str(seqs_members[i]) for i in np.argsort(d)[:n]]


def read_labels(suffix: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"clusters_dynamic{suffix}.zip")


# ---------------------------------------------------------------------------
# Una corrida -- bloque descriptivo
# ---------------------------------------------------------------------------


def describe_run(
    name: str, suffix: str, X: np.ndarray, dyn: "C.Pool", model: torch.nn.Module, device: str, out_dir: Path
) -> dict:
    chosen = json.loads((out_dir / f"chosen{suffix}.json").read_text())
    labels_df = read_labels(suffix, DATA_DIR)
    if not (labels_df["ID"].values == dyn.ids).all() or not (labels_df["zone"].values == dyn.zone).all():
        raise ValueError(
            f"{name}: clusters_dynamic{suffix}.zip no está alineado con el pool dinámico actual "
            "-- ¿hace falta re-correr scripts/tune_clustering.py --select?"
        )
    labels = labels_df["cluster"].values.astype(int)
    raw_centers = np.asarray(chosen["raw_centers"], dtype=np.float64)
    k = raw_centers.shape[0]
    protos = decode_prototypes(model, raw_centers, device)
    noise_frac = float((labels == -1).mean())
    print(f"\n{name}  (algo={chosen['algo']} space={chosen['space']} k={k} ruido={noise_frac:.1%})")

    rows, chronos, browser_clusters = [], [], []
    for cid in range(k):
        mask = labels == cid
        Xc = X[mask]
        sig = TY.cluster_signature(Xc, zones=dyn.zone[mask], proto_tokens=protos[cid])
        vecinos = nearest_member_seqs(dyn.z[mask], raw_centers[cid], dyn.seqs[mask].values)
        top80, cov80 = TY.top_sequences_to_coverage(Xc, target=0.8)
        rows.append(
            {
                "cluster": cid,
                **sig,
                "n_seqs_hasta_80pct": len(top80),
                "vecinos": " | ".join(vecinos),
            }
        )

        dist = TY.state_distribution(Xc)
        chronos.append(dist)
        browser_clusters.append(
            {
                "cluster": cid,
                "n": sig["n"],
                "share": sig["n"] / len(labels),
                "etiqueta": sig["etiqueta"],
                "forma": sig["forma"],
                "anio_cambio": sig["anio_cambio"],
                "glosa": sig["glosa"],
                "modal_seq": sig["modal_seq"],
                "proto_seq": sig["proto_seq"],
                "modal_vs_proto": round(sig["modal_vs_proto"], 3),
                "cobertura_top1": round(sig["cobertura_top1"], 3),
                "cobertura_top5": round(sig["cobertura_top5"], 3),
                "n_transiciones_medio": round(sig["n_transiciones_medio"], 2),
                "indice_complejidad_medio": round(sig["indice_complejidad_medio"], 3),
                "hamming_media_intra": round(sig["hamming_media_intra"], 2),
                "entropia_transv_pico": round(sig["entropia_transv_pico"], 3),
                "zona_dominante": sig["zona_dominante"],
                "zona_dominante_share": round(sig["zona_dominante_share"], 3),
                "n_zonas": sig["n_zonas"],
                "chronogram": np.round(dist, 3).tolist(),
                # trayectorias distintas más frecuentes que acumulan >=80% del cluster
                "top_seqs_80": [
                    {"text": t.text, "count": t.count, "coverage": round(t.coverage, 4)} for t in top80
                ],
                "cobertura_acumulada_80": round(cov80, 4),
                "vecinos": vecinos,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"typology{suffix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  {csv_path.name}: {len(df)} clusters")

    chronos = np.stack(chronos)
    fig = TY.plot_run_atlas(chronos, df["n"].values, df["etiqueta"].tolist(), name)
    atlas_path = IMGS_DIR / f"v2_typology_atlas_{_slug(name)}.png"
    fig.savefig(atlas_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {atlas_path.name}")

    w = df["n"].values
    top1_w = float(np.average(df["cobertura_top1"].values, weights=w))
    print(f"  cobertura media (pond.) de la secuencia distinta más frecuente por cluster: {top1_w:.2f}")

    return {
        "name": name,
        "suffix": suffix,
        "k": k,
        "algo": chosen["algo"],
        "space": chosen["space"],
        "noise_frac": noise_frac,
        "metrics": chosen["metrics"],
        "top1_coverage_weighted": top1_w,
        "labels": labels,
        "chronos": chronos,
        "df": df,
        "browser_clusters": browser_clusters,
    }


# ---------------------------------------------------------------------------
# Disimilitud entre secuencias (land2vec.seqdist)
# ---------------------------------------------------------------------------


def seqdist_pass(run_infos: list[dict], dyn: "C.Pool", method: str, sub: str, out_dir: Path) -> None:
    """Sobre las secuencias distintas ponderadas del pool dinámico: calcula la
    matriz de disimilitud una vez y, por corrida, pseudo-R² de discrepancia + ASW
    en espacio de secuencias; por cluster, las secuencias representativas
    (`seqrplot`). Escribe `typology_seqdist.csv` y `imgs/v2_typology_seqdist.png`,
    y agrega `representantes` + `seqdist` a cada `browser_cluster` / `df`.
    """
    uniqX, inv, cnt = SD.distinct(dyn.seqs)
    print(f"\nDisimilitud entre secuencias: {len(uniqX)} secuencias distintas, method={method} sub={sub}")
    D, info = SD.build_distance(uniqX, cnt, method=method, sub=sub)
    print(f"  D lista ({info}); rango {D.min():.2f}–{D.max():.2f}")

    seqdist_rows = []
    for ri in run_infos:
        lab_u, n_split = SD.labels_to_distinct(inv, ri["labels"], len(uniqX))
        if n_split:
            print(f"  aviso ({ri['name']}): {n_split} secuencias distintas con >1 etiqueta no-ruido (se tomó la moda)")
        pr = SD.pseudo_r2(D, lab_u, cnt)
        aw = SD.asw(D, lab_u, cnt)
        seqdist_rows.append(
            {
                "corrida": ri["name"], "suffix": ri["suffix"], "k": ri["k"],
                "pseudo_r2": pr["pseudo_r2"], "pseudo_f": pr["pseudo_f"], "cobertura": pr["coverage"],
                "asw_secuencias": aw["asw"], "metodo": method,
            }
        )
        ri["seqdist"] = seqdist_rows[-1]

        # representantes por cluster
        reps_by_cluster = {}
        for cid in sorted(g for g in set(lab_u.tolist()) if g != -1):
            cm = lab_u == cid
            rep = SD.representative_sequences(D[np.ix_(cm, cm)], cnt[cm])
            idx_global = np.flatnonzero(cm)
            reps_by_cluster[cid] = {
                "coverage": round(rep["coverage"], 3),
                "mean_dist": round(rep["mean_dist"], 3),
                "asw": round(aw["by_cluster"].get(cid, float("nan")), 3),
                "seqs": [
                    {"text": SD.seq_text(uniqX[idx_global[r["idx"]]]),
                     "covered": round(r["covered_weight"], 3)}
                    for r in rep["reps"]
                ],
            }
        for bc in ri["browser_clusters"]:
            r = reps_by_cluster.get(bc["cluster"])
            if r:
                bc["representantes"] = r["seqs"]
                bc["rep_coverage"] = r["coverage"]
                bc["asw"] = r["asw"]
        rb = reps_by_cluster  # alias corto para los .map de abajo
        ri["df"]["asw_secuencias"] = ri["df"]["cluster"].map(lambda c: rb.get(c, {}).get("asw", float("nan")))
        ri["df"]["rep_coverage"] = ri["df"]["cluster"].map(lambda c: rb.get(c, {}).get("coverage", float("nan")))
        ri["df"]["representantes"] = ri["df"]["cluster"].map(
            lambda c: " | ".join(s["text"] for s in rb.get(c, {}).get("seqs", []))
        )
        ri["df"].to_csv(out_dir / f"typology{ri['suffix']}.csv", index=False, encoding="utf-8")

    sd = pd.DataFrame(seqdist_rows)
    sd.to_csv(out_dir / "typology_seqdist.csv", index=False, encoding="utf-8")
    print("\ntypology_seqdist.csv")
    print(sd.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(sd["k"], sd["pseudo_r2"], s=60, c=range(len(sd)), cmap="viridis")
    for _, r in sd.iterrows():
        axes[0].annotate(r["corrida"], (r["k"], r["pseudo_r2"]), fontsize=7, xytext=(4, 3),
                         textcoords="offset points")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("pseudo-R² (discrepancia, espacio de secuencias)")
    axes[0].set_title(f"pseudo-R² vs. k — disimilitud {method.upper()}"); axes[0].grid(alpha=.3)
    axes[1].bar(range(len(sd)), sd["asw_secuencias"], color="#40682f")
    axes[1].set_xticks(range(len(sd))); axes[1].set_xticklabels(sd["corrida"], rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("ASW (espacio de secuencias)"); axes[1].set_title("ASW por corrida"); axes[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMGS_DIR / "v2_typology_seqdist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("v2_typology_seqdist.png")


# ---------------------------------------------------------------------------
# Comparación entre corridas
# ---------------------------------------------------------------------------


def compare_runs(specs: list[tuple[str, str]], out_dir: Path) -> None:
    labels_by_run = {name: read_labels(suffix, DATA_DIR)["cluster"].values.astype(int) for name, suffix in specs}
    agree = TY.crossrun_agreement(labels_by_run)

    long = []
    for key, M in agree.items():
        m = M.reset_index().melt(id_vars="index", var_name="corrida_b", value_name=key).rename(
            columns={"index": "corrida_a"}
        )
        long.append(m.set_index(["corrida_a", "corrida_b"]))
    pd.concat(long, axis=1).reset_index().to_csv(out_dir / "typology_crossrun.csv", index=False, encoding="utf-8")
    print("\ntypology_crossrun.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    TY.plot_crossrun(axes[0], agree["ari"], "ARI entre corridas (filas no-ruido en ambas)")
    TY.plot_crossrun(axes[1], agree["nmi"], "NMI entre corridas (filas no-ruido en ambas)")
    fig.tight_layout()
    fig.savefig(IMGS_DIR / "v2_typology_crossrun.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("v2_typology_crossrun.png")

    by = dict(specs)
    pairs = [
        ("gruesa / HDBSCAN", "media / HDBSCAN"),
        ("media / HDBSCAN", "fina / HDBSCAN"),
        ("gruesa / no-HDBSCAN", "media / no-HDBSCAN"),
        ("media / no-HDBSCAN", "fina / no-HDBSCAN"),
        ("media / HDBSCAN", "media / no-HDBSCAN"),
    ]
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        fig, ax = plt.subplots(figsize=(6, 7))
        TY.plot_alluvial(ax, labels_by_run[a], labels_by_run[b], a, b)
        fig.tight_layout()
        fig.savefig(IMGS_DIR / f"v2_typology_alluvial_{_slug(a)}__{_slug(b)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("aluviales guardados")


# ---------------------------------------------------------------------------


def write_browser_json(run_infos: list[dict], seqdist_method: str | None) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    browser = {
        "generated_from": "scripts/describe_clusters.py",
        "state_colors": TY.STATE_COLORS,
        "year_0": TY.YEAR_0,
        "seqdist_method": seqdist_method,
        "runs": [
            {k: ri[k] for k in ("name", "suffix", "k", "algo", "space", "noise_frac", "metrics",
                                "top1_coverage_weighted")}
            | {"seqdist": ri.get("seqdist"), "clusters": ri["browser_clusters"]}
            for ri in run_infos
        ],
    }
    out = PAGES_DIR / "typology_browser.json"
    out.write_text(json.dumps(browser, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "autoencoder_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "cluster_v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None, help="procesar un solo sufijo, p. ej. _medium")
    parser.add_argument("--seqdist-method", choices=["om", "dhd", "ham"], default="om")
    parser.add_argument("--seqdist-sub", choices=["trate", "constant"], default="trate")
    parser.add_argument("--no-seqdist", action="store_true", help="saltar el bloque de disimilitud entre secuencias")
    args = parser.parse_args()

    IMGS_DIR.mkdir(exist_ok=True)
    specs = [(n, s) for n, s in LEVEL_SPECS if args.only is None or s == args.only]
    if not specs:
        raise SystemExit(f"--only {args.only!r} no coincide con ningún sufijo de LEVEL_SPECS")

    model, dyn = load_model_and_dynamic_pool(args.model, DATA_DIR, args.device)
    print("Codificando el pool dinámico a matriz de tokens...")
    X = TY.encode_matrix(dyn.seqs)

    run_infos = [describe_run(name, suffix, X, dyn, model, args.device, args.out_dir) for name, suffix in specs]

    np.savez_compressed(
        args.out_dir / "typology_chronograms.npz", **{ri["suffix"] or "fina": ri["chronos"] for ri in run_infos}
    )
    print(f"\ntypology_chronograms.npz: {len(run_infos)} corridas")

    method = None
    if not args.no_seqdist:
        seqdist_pass(run_infos, dyn, args.seqdist_method, args.seqdist_sub, args.out_dir)
        method = args.seqdist_method

    write_browser_json(run_infos, method)

    if len(specs) == len(LEVEL_SPECS):
        compare_runs(specs, args.out_dir)

    print("\nListo.")


if __name__ == "__main__":
    main()
