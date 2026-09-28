"""Atlas estático por zona: satelital 2000 | satelital 2022 | píxeles clusterizados.

Un PNG de 3 paneles por zona (de las 15 ya disponibles en
`viz/clusters/data/`), en el estilo de un atlas de trayectorias LUCC: dos
paneles con la imagen satelital de inicio y fin de período, y un tercero con
los píxeles clusterizados coloreados por proceso sobre el fondo de
trayectorias constantes -- exactamente lo que muestra el visor interactivo
(`viz/clusters/index.html`) con "color: proceso" + "fondo: trayectorias
constantes" activados y el ruido (cluster -1) oculto. Donde no hay
clasificación (ni constante ni cluster visible) se ve la imagen satelital
2022 de fondo, igual que en el visor (el raster va *sobre* los tiles del
mapa, no reemplaza los huecos por negro). No re-corre el modelo ni el
clustering: reusa tal cual los artefactos que ya arma
`scripts/build_cluster_map.py` y `scripts/fetch_zone_imagery_gee.py`.

Prerrequisitos (ya generados, ver viz/clusters/README.md):
    viz/clusters/data/index.json
    viz/clusters/data/imagery/index.json
    viz/clusters/data/constants_<zona>.png
    viz/clusters/data/{pooled_subsampled,train_pooled}<suffix>.json

Corre LOCAL (necesita numpy + matplotlib; no corre en el contenedor de
desarrollo, que no los tiene instalados).

Uso:
    python scripts/plot_zone_atlas.py                              # medium + medium_parametric, las 15 zonas
    python scripts/plot_zone_atlas.py --corrida medium
    python scripts/plot_zone_atlas.py --corrida fine coarse_parametric
    python scripts/plot_zone_atlas.py --zonas chaco_santiago_frontier yungas
    python scripts/plot_zone_atlas.py --dpi 200 --out-dir imgs/atlas_borrador
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
VIZ_DATA = ROOT / "viz" / "clusters" / "data"
IMAGERY_INDEX = VIZ_DATA / "imagery" / "index.json"
DEFAULT_OUT_DIR = ROOT / "imgs" / "zone_atlas"

# mismo criterio que scripts/build_cluster_map.py / scripts/tune_clustering.py
SUFFIX_BY_CORRIDA = {
    "fine": "",
    "fine_parametric": "_parametric",
    "medium": "_medium",
    "medium_parametric": "_medium_parametric",
    "coarse": "_coarse",
    "coarse_parametric": "_coarse_parametric",
}
CORRIDA_LABEL = {
    "fine": "Fina / HDBSCAN",
    "fine_parametric": "Fina / paramétrico",
    "medium": "Media / HDBSCAN",
    "medium_parametric": "Media / paramétrico",
    "coarse": "Gruesa / HDBSCAN",
    "coarse_parametric": "Gruesa / paramétrico",
}
DEFAULT_CORRIDAS = ["medium", "medium_parametric"]

NOISE_COLOR = "#9a9a9a"


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def to_rgba_u8(arr: np.ndarray) -> np.ndarray:
    """Normaliza a uint8 (H,W,4). plt.imread devuelve float32 en [0,1] u uint8
    según el PNG, y con o sin canal alfa. Preservo el alfa en vez de
    recortarlo: tanto el fondo de constantes (paleta + tRNS, índice 0 =
    "no es constante" con alfa 0) como el satelital Landsat (RGBA, huecos de
    nubes/sin dato con alfa 0) lo usan para marcar "sin dato" -- descartarlo
    los deja en RGB (0,0,0) opaco, negro."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr


def zone_info(manifest: dict, zone_id: str) -> dict:
    for z in manifest["zones"]:
        if z["id"] == zone_id:
            return z
    raise KeyError(f"zona desconocida en el manifiesto: {zone_id}")


def run_for_suffix(manifest: dict, suffix: str) -> dict:
    for r in manifest["runs"]:
        if r["suffix"] == suffix:
            return r
    raise KeyError(f"corrida desconocida (suffix={suffix!r}) en el manifiesto")


def rasterize_clusters(bg_rgba: np.ndarray, bounds, points_by_cluster: dict,
                        labels: dict, show_noise: bool):
    """Pinta los píxeles clusterizados sobre una copia del fondo de constantes.

    `points_by_cluster`: {"<cid>": [lat,lon,seqIdx, lat,lon,seqIdx, ...]}, tal
    cual lo consume viz/clusters/index.html. `labels`: {"<cid>": {..., color,
    proceso}} del run elegido (sin entrada para -1: no tiene tipología).
    Los píxeles no tocados (no constantes y no clusterizados/ocultos) quedan
    con el alfa del fondo (0 = transparente), para componer encima de una
    imagen satelital de contexto en vez de rellenarlos con negro.
    Devuelve (imagen (H,W,4) uint8, conteo de píxeles pintados por proceso).
    """
    h, w = bg_rgba.shape[:2]
    (south, west), (north, east) = bounds
    lat_span = north - south
    lon_span = east - west
    out = bg_rgba.copy()
    proc_counts: dict = {}

    for cid_str, flat in points_by_cluster.items():
        cid = int(cid_str)
        if cid == -1 and not show_noise:
            continue
        info = labels.get(str(cid))
        if info is not None:
            color = hex_to_rgb(info["color"])
            proc_key = info.get("proceso") or "otro"
        elif cid == -1:
            color = hex_to_rgb(NOISE_COLOR)
            proc_key = "__noise__"
        else:
            continue  # cluster sin etiqueta y no es ruido: no debería pasar

        pts = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            continue
        rows = np.round((north - pts[:, 0]) / lat_span * (h - 1)).astype(np.int64)
        cols = np.round((pts[:, 1] - west) / lon_span * (w - 1)).astype(np.int64)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        n = int(ok.sum())
        if n == 0:
            continue
        out[rows[ok], cols[ok]] = (*color, 255)
        proc_counts[proc_key] = proc_counts.get(proc_key, 0) + n

    return out, proc_counts


def build_legend_handles(manifest: dict, const_meta: dict, proc_counts: dict):
    processes = manifest["processes"]
    handles = []
    for key, _cnt in sorted(proc_counts.items(), key=lambda kv: -kv[1]):
        if key == "__noise__":
            handles.append(Patch(facecolor=NOISE_COLOR, edgecolor="none",
                                  label="sin tipificar (−1)"))
        else:
            info = processes.get(key, {"label": key, "color": "#999999"})
            handles.append(Patch(facecolor=info["color"], edgecolor="none",
                                  label=info["label"]))
    for token, _cnt in sorted(const_meta.get("by_state", {}).items(), key=lambda kv: -kv[1]):
        color = manifest["constant_colors"].get(token, "#cccccc")
        label = manifest["state_labels"].get(token, token) + " (estable)"
        handles.append(Patch(facecolor=color, edgecolor="#00000022", label=label))
    return handles


def render_zone(zone_id: str, corrida: str, manifest: dict, imagery_meta: dict,
                 out_dir: Path, dpi: int, show_noise: bool) -> Path | None:
    zinfo = zone_info(manifest, zone_id)
    group = zinfo["group"]
    set_name = "train_pooled" if group == "train" else "pooled_subsampled"
    suffix = SUFFIX_BY_CORRIDA[corrida]

    img_meta = imagery_meta.get(zone_id)
    if img_meta is None:
        print(f"  [aviso] {zone_id}: sin imagen satelital, salteo")
        return None
    const_meta = manifest["constants"].get(zone_id)
    if const_meta is None:
        print(f"  [aviso] {zone_id}: sin fondo de trayectorias constantes, salteo")
        return None

    points_path = VIZ_DATA / f"{set_name}{suffix}.json"
    if not points_path.exists():
        print(f"  [aviso] {zone_id}: falta {points_path.name}, salteo")
        return None
    pts_doc = load_json(points_path)
    zone_pts = pts_doc["zones"].get(zone_id)
    if zone_pts is None:
        print(f"  [aviso] {zone_id}: sin puntos en {points_path.name}, salteo")
        return None

    run = run_for_suffix(manifest, suffix)
    labels = run["labels"]

    sat2000 = to_rgba_u8(plt.imread(VIZ_DATA / img_meta["2000"]["file"]))
    sat2022 = to_rgba_u8(plt.imread(VIZ_DATA / img_meta["2022"]["file"]))
    (isouth, iwest), (inorth, ieast) = img_meta["bounds"]

    const_bg = to_rgba_u8(plt.imread(VIZ_DATA / const_meta["file"]))
    layer_rgba, proc_counts = rasterize_clusters(
        const_bg, const_meta["bounds"], zone_pts["points"], labels, show_noise)
    (csouth, cwest), (cnorth, ceast) = const_meta["bounds"]

    center_lat = (csouth + cnorth) / 2.0
    aspect = 1.0 / max(0.05, math.cos(math.radians(center_lat)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ax0, ax1, ax2 = axes

    ax0.imshow(sat2000, extent=(iwest, ieast, isouth, inorth))
    ax0.set_title("Imagen satelital · 2000", fontsize=10)

    ax1.imshow(sat2022, extent=(iwest, ieast, isouth, inorth))
    ax1.set_title("Imagen satelital · 2022", fontsize=10)

    # satelital 2022 de fondo (igual que el visor: el raster de trayectorias
    # va *sobre* los tiles del mapa) para que los píxeles sin dato -- no
    # constantes y sin cluster visible, alfa 0 en `layer_rgba` -- muestren el
    # paisaje real en vez de quedar en blanco
    ax2.imshow(sat2022, extent=(iwest, ieast, isouth, inorth))
    ax2.imshow(layer_rgba, extent=(cwest, ceast, csouth, cnorth))
    ax2.set_title(f"Píxeles clusterizados · {CORRIDA_LABEL[corrida]}", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect(aspect)
        ax.set_facecolor("white")  # huecos de nubes/sin dato (alfa 0) -> blanco, no negro

    handles = build_legend_handles(manifest, const_meta, proc_counts)
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.02),
               fontsize=7, frameon=False, title="Tipo de trayectoria LUCC",
               title_fontsize=8)

    suptitle = zinfo["label"]
    if group == "train":
        suptitle += " — in-sample (encoder)"
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.text(0.01, 0.01,
              f"{img_meta['2000']['source']} (2000) / {img_meta['2022']['source']} (2022) "
              "vía Google Earth Engine · ESA CCI Land Cover 300 m · land2vec v2",
              fontsize=7, color="#666666")

    fig.tight_layout(rect=(0, 0.02, 0.86, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{zone_id}.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrida", nargs="+", default=DEFAULT_CORRIDAS,
                     choices=list(SUFFIX_BY_CORRIDA), metavar="CORRIDA",
                     help=f"una o más de {list(SUFFIX_BY_CORRIDA)} (default: {DEFAULT_CORRIDAS})")
    ap.add_argument("--zonas", nargs="+", default=None,
                     help="ids de zona puntuales (default: las 15 del manifiesto)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                     help=f"raíz de salida, un subdirectorio por corrida (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show-noise", action="store_true",
                     help="incluir el cluster -1 (sin tipificar), oculto por defecto")
    args = ap.parse_args()

    manifest = load_json(VIZ_DATA / "index.json")
    imagery_meta = load_json(IMAGERY_INDEX)
    zone_ids = args.zonas or [z["id"] for z in manifest["zones"]]

    for corrida in args.corrida:
        out_dir = args.out_dir / corrida
        print(f"=== {CORRIDA_LABEL[corrida]} ({corrida}) -> {out_dir} ===")
        for zone_id in zone_ids:
            path = render_zone(zone_id, corrida, manifest, imagery_meta,
                                out_dir, args.dpi, args.show_noise)
            if path is not None:
                print(f"  {zone_id} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
