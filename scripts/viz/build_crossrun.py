"""Payload del visor de concordancia entre clusterizaciones (`viz/crossrun/`).

Las 6 corridas de `tune_clustering.py --select` (granularidad fina/media/gruesa ×
familia HDBSCAN/paramétrico) etiquetan **los mismos píxeles en el mismo orden**:
`data/v2/clusters_dynamic{suffix}.zip` son 6 archivos de 107.362 filas alineadas fila
a fila (las escribe una sola invocación de `--select` desde el mismo `dyn_pool` en
memoria). Eso hace que comparar dos particiones sea columna contra columna.

Este script materializa esa comparación, que hasta ahora solo existía como píxeles:
`land2vec.typology.plot_alluvial` calcula los flujos, los dibuja y los descarta, y
además saca el `-1` de los dos lados -- escondiendo entre el 10% y el 31% de la masa
según el par, sin decirlo en el gráfico. Acá el `-1` es una categoría más
("sin tipificar"), porque a dónde va el ruido de HDBSCAN bajo el paramétrico (que no
tiene ruido) es justamente la pregunta interesante.

Salida: un único `viz/crossrun/crossrun.json` (~98 KB) con, para cada uno de los 15
pares no ordenados, la tabla de contingencia a nivel **cluster** y a nivel **proceso**
más 8 medidas de acuerdo. El visor transpone en JS para el sentido inverso.

Sobre qué lleva el payload: nunca coordenadas por parcela (a diferencia de
`viz/clusters/data/`) y solo **una** secuencia verbatim por cluster -- la trayectoria
modal más su peso, para el tooltip --, es decir 344 sobre las 1.128 trayectorias
distintas del pool. `viz/typology/typology_browser.json`, en cambio, lleva 1.286.
Con `--no-modal-seq` no va ninguna y el payload queda estrictamente agregado
(contingencias + etiquetas + el `inicio»fin` ya colapsado).

Solo-stdlib: el ARI y el NMI se calculan a mano desde la tabla de contingencia (ver
`_ari_from_contingency`), no con sklearn. Coinciden con sklearn a ~1e-16; si está
`models/v2/cluster_v2/typology_crossrun.csv` se contrasta contra él y se avisa del delta.

Prerrequisitos:
    data/v2/clusters_dynamic{,_parametric,_medium,_medium_parametric,_coarse,_coarse_parametric}.zip
        (scripts/clustering/tune_clustering.py --select)
    viz/typology/typology_browser.json   (scripts/clustering/describe_clusters.py) -- de acá salen
        la etiqueta y el proceso conceptual de cada cluster

Uso:
    python scripts/viz/build_crossrun.py
    python scripts/viz/build_crossrun.py --out /tmp/crossrun.json
    python scripts/viz/build_crossrun.py --no-modal-seq      # payload estrictamente agregado
    python scripts/viz/build_crossrun.py --no-check-crossrun
"""

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
if str(ROOT / "scripts" / "viz") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "viz"))
from land2vec.paths import rel  # noqa: E402

# build_cluster_map no tiene efectos de import (todo detrás de main()), así que se
# puede usar como librería -- mismo patrón que plot_process_maps.py / eval_desmonte.py.
from build_cluster_map import (  # noqa: E402
    PROCESSES, SUFFIXES, load_typology_labels, process_color, read_zip_csv,
)

DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "viz" / "crossrun" / "crossrun.json"
CROSSRUN_CSV = P.CLUSTER_V2 / "typology_crossrun.csv"
TYPOLOGY_JSON = ROOT / "viz" / "typology" / "typology_browser.json"

# Los dos sets comparables: las mismas 6 corridas sobre las mismas 7 zonas de
# evaluación, alineados fila a fila dentro de cada set. Ojo: el -1 NO significa lo
# mismo en los dos -- en `dynamic` es ruido del ajuste de HDBSCAN (y por eso las
# corridas paramétricas no tienen ninguno), en `pooled_subsampled` es el corte por
# distancia al centroide, que se aplica a las seis. Ver docs §7.2.
SETS = {
    "dynamic": "dinámico · sobre el que se ajustó (in-sample)",
    "pooled_subsampled": "pool aplicado por centroide (incl. constantes al 15%)",
}

# id de proceso = índice en PROCESSES; el id siguiente es "sin tipificar" (-1),
# que no es un proceso sino la ausencia de tipología.
PROC_KEYS = list(PROCESSES)
PROC_ID = {k: i for i, k in enumerate(PROC_KEYS)}
NOISE_PID = len(PROC_KEYS)

# nombre visible -> (granularidad, familia), para los selectores del visor
_GRAN_FAM = {
    "": ("fina", "HDBSCAN"),
    "_parametric": ("fina", "paramétrico"),
    "_medium": ("media", "HDBSCAN"),
    "_medium_parametric": ("media", "paramétrico"),
    "_coarse": ("gruesa", "HDBSCAN"),
    "_coarse_parametric": ("gruesa", "paramétrico"),
}


# ---------------------------------------------------------------------------
# Acuerdo entre particiones, desde la tabla de contingencia (sin sklearn)
# ---------------------------------------------------------------------------

def _rel(p: Path) -> str:
    "Ruta relativa al repo si cae adentro; si no (p. ej. --out /tmp/…), absoluta."
    try:
        return str(rel(p))
    except ValueError:
        return str(p)


def _c2(k: int) -> int:
    "Combinaciones de a 2. Entero exacto: no pasa por float como sklearn."
    return k * (k - 1) // 2


def _marginals(cont: dict) -> tuple[dict, dict, int]:
    a_sum: dict = {}
    b_sum: dict = {}
    n = 0
    for (a, b), v in cont.items():
        a_sum[a] = a_sum.get(a, 0) + v
        b_sum[b] = b_sum.get(b, 0) + v
        n += v
    return a_sum, b_sum, n


def _ari_from_contingency(cont: dict) -> float:
    """Adjusted Rand Index desde {(a,b): n}. Idéntico a
    sklearn.metrics.adjusted_rand_score sobre las mismas etiquetas."""
    a_sum, b_sum, n = _marginals(cont)
    if n < 2:
        return float("nan")
    sij = sum(_c2(v) for v in cont.values())
    sa = sum(_c2(v) for v in a_sum.values())
    sb = sum(_c2(v) for v in b_sum.values())
    c2n = _c2(n)
    exp = sa * sb / c2n
    denom = (sa + sb) / 2 - exp
    if denom == 0:
        # ambas particiones triviales (una sola clase cada una): acuerdo perfecto
        return 1.0
    return (sij - exp) / denom


def _nmi_from_contingency(cont: dict) -> float:
    """NMI con average_method='arithmetic' (el default de sklearn, que es lo que
    hay en models/v2/cluster_v2/typology_crossrun.csv)."""
    a_sum, b_sum, n = _marginals(cont)
    if n == 0:
        return float("nan")
    mi = 0.0
    for (a, b), v in cont.items():
        if v == 0:
            continue
        mi += (v / n) * math.log((v * n) / (a_sum[a] * b_sum[b]))
    ha = -sum((v / n) * math.log(v / n) for v in a_sum.values() if v)
    hb = -sum((v / n) * math.log(v / n) for v in b_sum.values() if v)
    if ha == 0.0 and hb == 0.0:
        return 1.0
    denom = (ha + hb) / 2
    if denom == 0:
        return 0.0
    return mi / denom


def _contingency(la: list, lb: list) -> dict:
    "{(a, b): n} sobre dos listas de etiquetas alineadas por posición."
    cont: dict = {}
    for a, b in zip(la, lb):
        key = (a, b)
        cont[key] = cont.get(key, 0) + 1
    return cont


def _agreement(cont: dict) -> float:
    "Fracción de la masa en la diagonal (solo tiene sentido con nomenclatura común)."
    total = sum(cont.values())
    if total == 0:
        return float("nan")
    diag = sum(v for (a, b), v in cont.items() if a == b)
    return diag / total


def _nesting(cont: dict, side: int) -> tuple:
    """Anidamiento de una partición dentro de la otra, sobre la contingencia ya
    recortada a la intersección no-ruido.

    `side` 0 = A dentro de B (cada cluster de A, ¿cae entero en uno de B?), 1 = al revés.
    Devuelve (pureza ponderada, nº de clusters de origen, nº que se reparten entre 2+).
    Es la misma pureza que land2vec.typology.nesting_table, agregada y ponderada."""
    g: dict = {}
    for (a, b), v in cont.items():
        src, dst = (a, b) if side == 0 else (b, a)
        g.setdefault(src, {})
        g[src][dst] = g[src].get(dst, 0) + v
    tot = sum(sum(d.values()) for d in g.values())
    if not tot:
        return float("nan"), 0, 0
    pur = sum(max(d.values()) for d in g.values())
    multi = sum(1 for d in g.values() if len(d) > 1)
    return pur / tot, len(g), multi


def _flat(cont: dict) -> list:
    "{(a,b): n} -> [a, b, n, ...] ordenado por (a, b). Solo celdas > 0."
    out: list = []
    for (a, b) in sorted(cont):
        v = cont[(a, b)]
        if v:
            out.extend((a, b, v))
    return out


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_modal_seqs() -> tuple[dict, dict]:
    """suffix -> {cluster_id -> (modal_seq, cobertura_top1)}, y la paleta de estados.

    La trayectoria modal es la más frecuente del cluster (es exactamente
    `top_seqs_80[0]`) y `cobertura_top1` su peso dentro del cluster. Sale de
    `viz/typology/typology_browser.json`; `load_typology_labels` no la expone
    porque el visor de mapas no la usa.

    Ojo con esto: son 344 secuencias verbatim (una por cluster) sobre las 1.128
    trayectorias distintas del pool, así que el payload deja de ser estrictamente
    "solo agregado" -- ver --no-modal-seq y viz/crossrun/README.md."""
    if not TYPOLOGY_JSON.exists():
        return {}, {}
    doc = json.loads(TYPOLOGY_JSON.read_text(encoding="utf-8"))
    out: dict = {}
    for run in doc.get("runs", []):
        per: dict = {}
        for c in run.get("clusters", []):
            seq = c.get("modal_seq")
            if seq:
                per[int(c["cluster"])] = (seq, c.get("cobertura_top1"))
        out[run.get("suffix", "")] = per
    return out, doc.get("state_colors", {})


def load_run_labels(set_name: str, suffix: str, data_dir: Path) -> tuple[list, list, list]:
    "(ids, zones, clusters) de data/v2/clusters_{set}{suffix}.zip, en orden de archivo."
    path = P.clusters_file(set_name, suffix, data_dir)
    if not path.exists():
        sys.exit(f"falta {rel(path)} -- corré scripts/clustering/tune_clustering.py --select")
    ids, zones, clusters = [], [], []
    for row in read_zip_csv(path):
        ids.append(row["ID"])
        zones.append(row["zone"])
        clusters.append(int(row["cluster"]))
    return ids, zones, clusters


def check_alignment(keys_by_suffix: dict) -> None:
    """Las 6 corridas tienen que traer las mismas (ID, zone) en el mismo orden -- es
    lo que habilita comparar columna contra columna. Lo garantiza tune_clustering.py
    (un solo dyn_pool), pero si alguien regeneró una sola corrida deja de valer."""
    ref_suffix = ""
    ref = keys_by_suffix[ref_suffix]
    for suffix, keys in keys_by_suffix.items():
        if suffix == ref_suffix:
            continue
        if len(keys) != len(ref):
            sys.exit(f"desalineadas: '{SUFFIXES[suffix]}' tiene {len(keys):,} filas "
                     f"y '{SUFFIXES[ref_suffix]}' {len(ref):,}")
        for i, (k, r) in enumerate(zip(keys, ref)):
            if k != r:
                sys.exit(f"desalineadas: '{SUFFIXES[suffix]}' difiere de "
                         f"'{SUFFIXES[ref_suffix]}' en la fila {i}: {k} != {r}. "
                         f"Re-corré scripts/clustering/tune_clustering.py --select.")
    print(f"alineamiento: las 6 corridas comparten las mismas {len(ref):,} filas (ID, zone)")


def build_run_entry(suffix: str, clusters: list, labels: dict, modal: dict) -> dict:
    "Metadatos + nodos (un nodo por cluster presente, con su n, proceso, color y etiqueta)."
    counts: dict = {}
    for c in clusters:
        counts[c] = counts.get(c, 0) + 1
    lab = labels.get(suffix, {})

    nodes = []
    for cid in sorted(counts):
        node = {"id": cid, "n": counts[cid]}
        if cid == -1:
            node["p"] = NOISE_PID
        else:
            info = lab.get(cid)
            if info is None:
                # cluster sin fila en la tipología: cae en "otro", como load_typology_labels
                node["p"] = PROC_ID["otro"]
                node["c"] = process_color("otro", None)
                node["lab"] = f"#{cid}"
            else:
                key = info.get("proceso") or "otro"
                node["p"] = PROC_ID.get(key, PROC_ID["otro"])
                node["c"] = info.get("color") or process_color(key, None)
                node["lab"] = info.get("etiqueta") or f"#{cid}"
                if info.get("inicio"):
                    node["ini"] = info["inicio"]
                if info.get("fin"):
                    node["fin"] = info["fin"]
                if info.get("anio_cambio") is not None:
                    node["yr"] = info["anio_cambio"]
            ms = modal.get(suffix, {}).get(cid)
            if ms:
                node["seq"] = ms[0]                       # trayectoria modal, 23 tokens
                if ms[1] is not None:
                    node["cov"] = round(ms[1], 4)         # su peso dentro del cluster
        nodes.append(node)

    gran, fam = _GRAN_FAM[suffix]
    n_noise = counts.get(-1, 0)
    n = len(clusters)
    return {
        "suffix": suffix,
        "name": SUFFIXES[suffix],
        "gran": gran,
        "fam": fam,
        "k": sum(1 for c in counts if c != -1),
        "n_noise": n_noise,
        "noise": round(n_noise / n, 6) if n else 0.0,
        "nodes": nodes,
    }


def build_pair(ca: list, cb: list, pa: list, pb: list) -> dict:
    """Contingencias (cluster y proceso, sobre las N filas, con el -1 incluido) y las
    8 medidas de acuerdo."""
    cont_cluster = _contingency(ca, cb)
    cont_proc = _contingency(pa, pb)

    # intersección no-ruido: filas que NINGUNA de las dos corridas dejó en -1.
    # Es la lectura primaria (misma que land2vec.typology.crossrun_agreement).
    keep = [i for i, (a, b) in enumerate(zip(ca, cb)) if a != -1 and b != -1]
    n = len(ca)
    cob = len(keep) / n if n else 0.0
    ca_k = [ca[i] for i in keep]
    cb_k = [cb[i] for i in keep]
    pa_k = [pa[i] for i in keep]
    pb_k = [pb[i] for i in keep]

    cont_cluster_k = _contingency(ca_k, cb_k)
    cont_proc_k = _contingency(pa_k, pb_k)
    pur_ab, k_ab, spl_ab = _nesting(cont_cluster_k, 0)
    pur_ba, k_ba, spl_ba = _nesting(cont_cluster_k, 1)

    return {
        "cluster": _flat(cont_cluster),
        "proceso": _flat(cont_proc),
        "m": {
            # nivel cluster, intersección no-ruido (los números publicados)
            "ari": round(_ari_from_contingency(cont_cluster_k), 6),
            "nmi": round(_nmi_from_contingency(cont_cluster_k), 6),
            "cob": round(cob, 6),
            # nivel cluster, -1 como una etiqueta más
            "ari_ruido": round(_ari_from_contingency(cont_cluster), 6),
            # nivel proceso, intersección no-ruido
            "acu_proc": round(_agreement(cont_proc_k), 6),
            "ari_proc": round(_ari_from_contingency(cont_proc_k), 6),
            # nivel proceso, "sin tipificar" como 11ª categoría (denominador = N)
            "acu_proc11": round(_agreement(cont_proc), 6),
            "ari_proc11": round(_ari_from_contingency(cont_proc), 6),
            # anidamiento: ¿cada cluster de un lado cae entero en uno del otro?
            # Asimétrico a propósito: fino->grueso es alto si la escalera es
            # jerárquica, grueso->fino es bajo por construcción.
            "pur_ab": round(pur_ab, 6), "spl_ab": spl_ab, "k_ab": k_ab,
            "pur_ba": round(pur_ba, 6), "spl_ba": spl_ba, "k_ba": k_ba,
        },
    }


def check_against_sklearn(payload: dict, suffixes: list) -> None:
    """Contraste de regresión contra models/v2/cluster_v2/typology_crossrun.csv, que lo
    calculó con sklearn. Es opcional (el CSV está gitignoreado).

    Ojo con los nombres: describe_clusters.py llama "no-HDBSCAN" a lo que
    build_cluster_map.SUFFIXES llama "paramétrico" -- son la misma corrida."""
    if not CROSSRUN_CSV.exists():
        print(f"nota: {rel(CROSSRUN_CSV)} no está; salteo el contraste con sklearn")
        return
    by_pair: dict = {}
    with open(CROSSRUN_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_pair[(row["corrida_a"], row["corrida_b"])] = row

    def _aliases(name: str) -> list:
        alt = name.replace("paramétrico", "no-HDBSCAN")
        return [name] if alt == name else [name, alt]

    def _lookup(name_a: str, name_b: str):
        for a in _aliases(name_a):
            for b in _aliases(name_b):
                row = by_pair.get((a, b)) or by_pair.get((b, a))
                if row is not None:
                    return row
        return None

    worst = 0.0
    n_cmp = 0
    missing = 0
    for i, j in combinations(range(len(suffixes)), 2):
        name_a, name_b = SUFFIXES[suffixes[i]], SUFFIXES[suffixes[j]]
        row = _lookup(name_a, name_b)
        if row is None:
            missing += 1
            continue
        mine = payload["pairs"][f"{i}-{j}"]["m"]
        for key, col in (("ari", "ari"), ("nmi", "nmi"), ("cob", "cobertura"),
                         ("ari_ruido", "ari_con_ruido")):
            try:
                ref = float(row[col])
            except (KeyError, TypeError, ValueError):
                continue
            delta = abs(mine[key] - ref)
            worst = max(worst, delta)
            n_cmp += 1
    if n_cmp == 0:
        print("nota: el CSV no tiene ningún par comparable; salteo el contraste")
        return
    # el payload va redondeado a 6 decimales, así que 1e-6 es el piso del contraste
    status = "OK" if worst < 1e-6 else "REVISAR"
    faltan = f", {missing} pares sin fila en el CSV" if missing else ""
    print(f"contraste con sklearn ({n_cmp} valores{faltan}): "
          f"delta máximo {worst:.2e}  [{status}]")
    if worst >= 1e-6:
        print("  ¡el acuerdo calculado a mano no coincide con typology_crossrun.csv!")


def build_set_payload(set_name: str, suffixes: list, labels: dict, modal: dict,
                       data_dir: Path) -> dict:
    "n + runs + pairs de un set. Aborta si las 6 corridas no están alineadas."
    clusters_by_suffix: dict = {}
    keys_by_suffix: dict = {}
    for suffix in suffixes:
        ids, zones, clusters = load_run_labels(set_name, suffix, data_dir)
        clusters_by_suffix[suffix] = clusters
        keys_by_suffix[suffix] = list(zip(zones, ids))
    check_alignment(keys_by_suffix)

    runs = [build_run_entry(s, clusters_by_suffix[s], labels, modal) for s in suffixes]
    for r in runs:
        print(f"    {r['name']:22} k={r['k']:>3}  sin tipificar {100 * r['noise']:>5.2f}%")

    # proceso por fila (el -1 va a NOISE_PID)
    pid_of = [{node["id"]: node["p"] for node in r["nodes"]} for r in runs]
    procs_by_suffix = {
        s: [pid_of[i][c] for c in clusters_by_suffix[s]] for i, s in enumerate(suffixes)
    }

    pairs: dict = {}
    for i, j in combinations(range(len(suffixes)), 2):
        si, sj = suffixes[i], suffixes[j]
        pairs[f"{i}-{j}"] = build_pair(
            clusters_by_suffix[si], clusters_by_suffix[sj],
            procs_by_suffix[si], procs_by_suffix[sj])

    return {"n": len(clusters_by_suffix[""]), "runs": runs, "pairs": pairs}


def verify_set(set_name: str, block: dict) -> None:
    n, pairs, runs = block["n"], block["pairs"], block["runs"]
    for key, pair in pairs.items():
        for level in ("cluster", "proceso"):
            total = sum(pair[level][2::3])
            if total != n:
                sys.exit(f"[{set_name}] par {key}, nivel {level}: la contingencia suma "
                         f"{total:,}, no {n:,} -- se perdió masa")
    for i, r in enumerate(runs):
        # cualquier par que involucre a la corrida i sirve para reconstruir su marginal
        j, col = (i + 1, 0) if i + 1 < len(runs) else (0, 1)
        flat = pairs[f"{min(i, j)}-{max(i, j)}"]["cluster"]
        marg: dict = {}
        for t in range(0, len(flat), 3):
            cid = flat[t + col]
            marg[cid] = marg.get(cid, 0) + flat[t + 2]
        for node in r["nodes"]:
            if marg.get(node["id"], 0) != node["n"]:
                sys.exit(f"[{set_name}] {r['name']}: el marginal del cluster "
                         f"{node['id']} ({marg.get(node['id'], 0):,}) no coincide con "
                         f"su n ({node['n']:,})")
    print(f"    conservación de masa y marginales: OK ({n:,} px, {len(pairs)} pares)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS),
                    metavar="SET", help=f"sets a incluir (default: {list(SETS)})")
    ap.add_argument("--no-check-crossrun", action="store_true",
                    help="no contrastar contra models/v2/cluster_v2/typology_crossrun.csv")
    ap.add_argument("--no-modal-seq", action="store_true",
                    help="no incluir la trayectoria modal de cada cluster; el payload "
                         "queda estrictamente agregado (sin ninguna secuencia verbatim)")
    args = ap.parse_args()

    suffixes = list(SUFFIXES)

    labels = load_typology_labels()
    if not labels:
        sys.exit("falta viz/typology/typology_browser.json -- corré "
                 "scripts/clustering/describe_clusters.py (de ahí salen el proceso y la etiqueta "
                 "de cada cluster, que es lo que compara este visor)")

    modal, state_colors = ({}, {}) if args.no_modal_seq else load_modal_seqs()

    blocks: dict = {}
    for set_name in args.sets:
        print(f"[{set_name}]")
        blocks[set_name] = build_set_payload(set_name, suffixes, labels, modal, args.data_dir)
        verify_set(set_name, blocks[set_name])

    payload = {
        "generated_from": "scripts/viz/build_crossrun.py",
        "processes": [
            {"key": k, "label": PROCESSES[k]["label"], "gloss": PROCESSES[k]["gloss"],
             "color": process_color(k, None)}
            for k in PROC_KEYS
        ],
        "noise_pid": NOISE_PID,
        "noise_label": "sin tipificar",
        "state_colors": state_colors,   # token -> hex, para la tira de la trayectoria modal
        "year_0": 2000,
        "set_labels": {s: SETS[s] for s in args.sets},
        "sets": blocks,
    }

    # el CSV de referencia se calculó sobre `dynamic`; solo ahí tiene sentido contrastar
    if not args.no_check_crossrun and "dynamic" in blocks:
        check_against_sklearn(blocks["dynamic"], suffixes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                        encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    cells = sum(len(p[lv]) // 3 for b in blocks.values() for p in b["pairs"].values()
                for lv in ("cluster", "proceso"))
    print(f"\n{_rel(args.out)}: {kb:.1f} KB, {len(blocks)} set(s), "
          f"{cells:,} celdas no nulas")
    if kb > 400:
        print("  aviso: el payload creció mucho más de lo esperado -- ¿algo se duplicó?")


if __name__ == "__main__":
    main()
