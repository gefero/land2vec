"""Evaluación de la Pregunta 2: detección de procesos (protocolo: docs/autoencoder_v3/protocolo_evaluacion.md, Parte II).

Subcomandos (desde la raíz del repo; salidas en data/autoencoder_v3/pregunta2/):
    python scripts/validacion/evaluacion_pregunta2.py catalogo    # catálogo descriptivo de procesos (§7.6), Argentina y mundo
    python scripts/validacion/evaluacion_pregunta2.py tipologias  # particiones de cada espacio (§8): k-medoides x10 arranques y jerárquico completo
    python scripts/validacion/evaluacion_pregunta2.py nivel1      # Nivel 1 (§9): recuperación de los procesos -> nivel1.csv

`tipologias` requiere los códigos de la Pregunta 1 (`evaluacion_pregunta1.py codificar`) y la matriz OM (`p1_compresion.py om`).

Los eventos y procesos se definen en src/land2vec/procesos.py.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo", _ROOT / "scripts" / "datos", _ROOT / "scripts" / "clustering"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import p1_compresion as p1  # noqa: E402
from land2vec.procesos import PROCESOS, eventos_de, procesos_por_evento  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

OUT = P.DATA / "autoencoder_v3" / "pregunta2"
TIPOLOGIAS = OUT / "tipologias"
CODIGOS = P.DATA / "autoencoder_v3" / "pregunta1" / "codigos"
MUNDO = P.DATA / "autoencoder_v3" / "mundo" / "universo_mundo.csv.gz"
ORDEN = list(PROCESOS)


# ---------------------------------------------------------------------------
# Universos
# ---------------------------------------------------------------------------
def universo_argentina():
    "Trayectorias dinámicas de Argentina; superficie en píxeles (el censo de Argentina no guarda km²)."
    u, X = p1.load_universe()
    din = ~u.constante.values
    return X[din], u.n_px.values[din].astype(float), "px"


def universo_mundo():
    "Trayectorias dinámicas del censo mundial (incluye las de Argentina); superficie en km²."
    from censo_trayectorias import decode
    from land2vec.extract import LCCS_CODE_TO_TOKEN
    m = pd.read_csv(MUNDO)
    m = m[m.n_cambios > 0].reset_index(drop=True)
    tok = np.array([Tokenizer.VOCAB[LCCS_CODE_TO_TOKEN[c]] if c in LCCS_CODE_TO_TOKEN else -1 for c in range(16)])
    T = P.V3_YEARS[1] - P.V3_YEARS[0] + 1
    X = tok[np.array([decode(int(h), int(lo), T) for h, lo in zip(m.hi.astype("uint64"), m.lo.astype("uint64"))])]
    assert (X > 0).all(), "estados fuera del vocabulario en el mundo"
    return X, m.area_km2.values.astype(float), "km2"


# ---------------------------------------------------------------------------
# Catálogo (§7.6)
# ---------------------------------------------------------------------------
def mediana_ponderada(v, w):
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def transiciones_crudas(X):
    "Pares (origen, destino) de años consecutivos presentes en cada fila, sin persistencia."
    rev = Tokenizer.REVERSE_VOCAB
    out = []
    for i, row in enumerate(X):
        ch = np.flatnonzero(row[1:] != row[:-1])
        out.append({(rev[int(row[c])], rev[int(row[c + 1])]) for c in ch})
    return out


def catalogo(nombre, X, w, unidad):
    ev, osc = eventos_de(X)
    ev["w"] = w[ev.i.values]
    pe = procesos_por_evento(ev)
    total = w.sum()
    crudas = transiciones_crudas(X)
    filas, anios = [], []
    tray = {}
    for p in ORDEN:
        q = pe[pe.proceso == p]
        idx = np.unique(q.i.values)
        tray[p] = idx
        sup = w[idx].sum()
        pares = PROCESOS[p]
        cr = np.array([any((o, d) in pares or ("*", d) in pares and o != d for o, d in s) for s in crudas])
        wq = q.w.values
        marca = lambda col: float((wq * q[col].values).sum() / wq.sum()) if len(q) else np.nan  # noqa: E731
        sin_marca = q[~(q.costura | q.sensor | q.censurado)]
        filas.append({"conjunto": nombre, "proceso": p, "n_tipos": len(idx), "superficie": sup, "unidad": unidad,
                      "frac_superficie_dinamica": sup / total, "n_eventos": len(q),
                      "frac_eventos_costura": marca("costura"), "frac_eventos_sensor": marca("sensor"),
                      "frac_eventos_censura": marca("censurado"),
                      "superficie_sin_marcas": w[np.unique(sin_marca.i.values)].sum(),
                      "superficie_transicion_cruda": w[cr].sum(),
                      "anio_mediano": mediana_ponderada(q.anio.values, wq)})
        g = q.groupby("anio").w.sum()
        anios += [{"conjunto": nombre, "proceso": p, "anio": int(a), "superficie": float(v)} for a, v in g.items()]
    # superposición entre procesos (trayectorias con ambos)
    sup = []
    for a in ORDEN:
        for b in ORDEN:
            inter = np.intersect1d(tray[a], tray[b])
            sup.append({"conjunto": nombre, "proceso_a": a, "proceso_b": b, "superficie": w[inter].sum(),
                        "frac_de_a": w[inter].sum() / max(w[tray[a]].sum(), 1e-12)})
    # superficie sin ningún proceso y sus eventos más frecuentes
    con = np.unique(pe.i.values)
    sin = np.setdiff1d(np.arange(len(X)), con)
    sin_ev = np.setdiff1d(sin, ev.i.values)
    otros = ev[ev.i.isin(sin)].groupby(["origen", "destino"]).w.sum().sort_values(ascending=False)
    resto = [{"conjunto": nombre, "categoria": "sin proceso: total", "superficie": w[sin].sum(), "frac": w[sin].sum() / total},
             {"conjunto": nombre, "categoria": "sin proceso: sin ningún evento persistente", "superficie": w[sin_ev].sum(),
              "frac": w[sin_ev].sum() / total}]
    resto += [{"conjunto": nombre, "categoria": f"sin proceso: evento {o}→{d}", "superficie": v, "frac": v / total}
              for (o, d), v in otros.head(15).items()]
    # oscilaciones
    osc["w"] = w[osc.i.values]
    os_ = osc.groupby(["base", "transitorio"]).agg(superficie=("w", "sum"), n=("i", "size")).reset_index()
    os_["frac_superficie_dinamica"] = os_.superficie / total
    os_.insert(0, "conjunto", nombre)
    tot_osc = w[np.unique(osc.i.values)].sum()
    resto.append({"conjunto": nombre, "categoria": "trayectorias con alguna oscilación", "superficie": tot_osc, "frac": tot_osc / total})
    return (pd.DataFrame(filas), pd.DataFrame(anios), pd.DataFrame(sup), pd.DataFrame(resto),
            os_.sort_values("superficie", ascending=False), {"conjunto": nombre, "superficie_dinamica": total, "unidad": unidad,
                                                               "n_tipos_dinamicos": len(X)})


def cmd_catalogo(args):
    partes = [catalogo("Argentina", *universo_argentina()), catalogo("mundo", *universo_mundo())]
    OUT.mkdir(parents=True, exist_ok=True)
    nombres = ("catalogo_procesos", "catalogo_anios", "catalogo_superposicion", "catalogo_resto", "catalogo_oscilaciones")
    for j, nm in enumerate(nombres):
        pd.concat([p[j] for p in partes]).to_csv(OUT / f"{nm}.csv", index=False)
    pd.DataFrame([p[5] for p in partes]).to_csv(OUT / "catalogo_universos.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.concat([p[0] for p in partes])[["conjunto", "proceso", "n_tipos", "superficie", "frac_superficie_dinamica",
                                             "frac_eventos_costura", "frac_eventos_censura", "superficie_transicion_cruda"]])
    print("->", P.rel(OUT))


# ---------------------------------------------------------------------------
# Tipologías (§8)
# ---------------------------------------------------------------------------
D_ESPACIOS = (4, 16)
K = tuple(range(4, 49, 4))
SEEDS = (0, 1, 2)
NOMBRE_ESPACIO = {"ae": "AE", "lin": "AE lineal", "pca": "PCA", "mca": "MCA", "om": "OM", "onehot": "One-hot (Hamming)"}
UMBRAL_FRAC, UMBRAL_N = 0.01, 100   # §13: proceso evaluable


def espacios():
    "(espacio, d, semilla) de §8.1."
    for d in D_ESPACIOS:
        for met in ("ae", "lin", "pca", "mca"):
            for s in (SEEDS if met in ("ae", "lin") else (None,)):
                yield met, d, s
    yield "om", 0, None
    yield "onehot", 0, None


def tag(met, d, s):
    return met + (f"_d{d}" if d else "") + (f"_s{s}" if s is not None else "")


def distancias(met, d, s, din):
    "Matriz de distancias (float32) entre las trayectorias dinámicas de Argentina."
    if met == "om":
        D = np.load(p1.OUT_DATA / "om_trate.npy", mmap_mode="r")
        return np.ascontiguousarray(D[np.ix_(din, din)], dtype=np.float32)
    if met == "onehot":
        _, Xa = p1.load_universe()
        OH = p1.onehot(Xa[din]).astype(np.float32)
        return (Xa.shape[1] - OH @ OH.T).astype(np.float32)
    from scipy.spatial.distance import cdist
    z = np.load(CODIGOS / f"{tag(met, d, s)}.npz")["z_arg"][din]
    return cdist(z, z).astype(np.float32)


def _tipologia(job):
    met, d, s, arranques, rehacer = job
    f = TIPOLOGIAS / f"{tag(met, d, s)}.npz"
    if f.exists() and not rehacer:
        return f"{tag(met, d, s):12s} ya estaba"
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import squareform
    from p2_tipologias import kmedoids
    t0 = time.time()
    u, _ = p1.load_universe()
    din = np.flatnonzero(~u.constante.values)
    D = distancias(met, d, s, din)
    n = len(D)
    km = np.zeros((len(K), arranques, n), np.int16)
    costo = np.zeros((len(K), arranques))
    med = np.full((len(K), arranques, max(K)), -1, np.int32)
    w = np.ones(n)                                   # §8.3: el agrupamiento no se pondera
    for a, k in enumerate(K):
        rng = np.random.default_rng(k)
        for r in range(arranques):
            m, lab, c = kmedoids(D, w, k, rng, n_init=1)
            km[a, r], costo[a, r], med[a, r, :k] = lab, c, m
    # cut_tree corta en exactamente k grupos (fcluster "maxclust" da menos con alturas empatadas, p. ej. Hamming)
    jq = cut_tree(linkage(squareform(D, checks=False), "complete"), n_clusters=list(K)).T.astype(np.int16)
    TIPOLOGIAS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, k=np.array(K), km=km, costo=costo, medoides=med, jq=jq, din=din)
    return f"{tag(met, d, s):12s} {time.time() - t0:5.0f}s"


def cmd_tipologias(args):
    from multiprocessing import Pool
    jobs = [(m, d, s, args.arranques, args.rehacer) for m, d, s in espacios()]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(_tipologia, jobs):
            print(msg, flush=True)
    print("->", P.rel(TIPOLOGIAS))


def cargar_tipologia(met, d, s):
    "{'k-medoides': (len(K), n) mejor arranque, 'jerarquico completo': (len(K), n), 'arranques': (len(K), R, n), ...}"
    z = np.load(TIPOLOGIAS / f"{tag(met, d, s)}.npz")
    best = z["costo"].argmin(1)
    return {"k": z["k"], "k-medoides": z["km"][np.arange(len(best)), best].astype(int),
            "jerarquico completo": z["jq"].astype(int), "arranques": z["km"], "medoides": z["medoides"], "din": z["din"]}


# ---------------------------------------------------------------------------
# Nivel 1 (§9)
# ---------------------------------------------------------------------------
def etiquetas_procesos(X):
    """Para cada variante de eventos ('todos' | 'sin_marcas', §7.5) y proceso: positivos (bool) y año del primer
    evento del proceso (nan si no es positivo)."""
    ev, _ = eventos_de(X)
    pe = procesos_por_evento(ev)
    marcado = (pe.costura | pe.sensor | pe.censurado).values
    out = {}
    for var, q in (("todos", pe), ("sin_marcas", pe[~marcado])):
        for p in ORDEN:
            a = q[q.proceso == p].groupby("i").anio.min()
            anio = np.full(len(X), np.nan)
            anio[a.index.values] = a.values
            out[var, p] = (~np.isnan(anio), anio)
    return out


def recuperacion(lab, w, pos, anio, mitad):
    """Métricas del Nivel 1 con ajuste cruzado (§9.1): la asignación de grupos al proceso se estima con una mitad
    y se evalúa con la otra, en los dos sentidos; se devuelve el promedio."""
    G = lab.max() + 1
    res = []
    for m in (0, 1):
        fit, ev = mitad == m, mitad != m
        wg = np.bincount(lab[fit], w[fit], G)
        wp = np.bincount(lab[fit], w[fit] * pos[fit], G)
        asig = (wg > 0) & (wp >= 0.5 * wg)
        dentro = asig[lab[ev]]
        we, pe_ = w[ev], pos[ev]
        tp, npos, nasig = (we * pe_ * dentro).sum(), (we * pe_).sum(), (we * dentro).sum()
        rec = tp / npos if npos > 0 else np.nan
        prec = tp / nasig if nasig > 0 else np.nan
        f1 = 2 * prec * rec / (prec + rec) if nasig > 0 and tp > 0 else 0.0
        # fechado: año mediano (ponderado) del primer evento entre los positivos del grupo, estimado con la mitad de ajuste
        fe = np.nan
        if nasig > 0 and tp > 0:
            med = np.full(G, np.nan)
            for g in np.flatnonzero(asig):
                sel = fit & pos & (lab == g)
                med[g] = mediana_ponderada(anio[sel], w[sel])
            sel = ev & pos & asig[lab]
            fe = float((w[sel] * np.abs(anio[sel] - med[lab[sel]])).sum() / w[sel].sum())
        res.append((rec, prec, f1, fe, int(asig.sum())))
    r = np.array(res, float)
    with warnings.catch_warnings():   # precisión sin grupo asignado en las dos mitades -> nan
        warnings.simplefilter("ignore", RuntimeWarning)
        prom = np.nanmean(r, 0)
    return {"recall": prom[0], "precision": prom[1], "f1": prom[2], "fechado_mae": prom[3], "n_grupos_asignados": prom[4],
            "sin_grupo": int(np.isnan(r[:, 1]).sum())}


def _nivel1(job):
    met, d, s, X_, w_px, etiq, mitad, n_azar = job
    t = cargar_tipologia(met, d, s)
    rng = np.random.default_rng(1)
    filas = []
    for alg in ("k-medoides", "jerarquico completo"):
        for a, k in enumerate(t["k"]):
            lab = t[alg][a]
            perms = [rng.permutation(lab) for _ in range(n_azar)]
            for pond, w in (("superficie", w_px), ("tipo", np.ones_like(w_px))):
                for (var, p), (pos, anio) in etiq.items():
                    r = recuperacion(lab, w, pos, anio, mitad)
                    az = [recuperacion(q, w, pos, anio, mitad) for q in perms]
                    azar = {f"azar_{c}": float(np.nanmean([x[c] for x in az])) if not all(np.isnan(x[c]) for x in az) else np.nan
                            for c in ("recall", "precision", "f1")}
                    filas.append({"espacio": NOMBRE_ESPACIO[met], "metodo": met, "d": d or np.nan, "semilla": s,
                                  "algoritmo": alg, "k": int(k), "ponderacion": pond, "eventos": var, "proceso": p} | r | azar)
    return filas


def cmd_nivel1(args):
    from multiprocessing import Pool
    X, w, _ = universo_argentina()
    etiq = etiquetas_procesos(X)
    for p in ORDEN:   # §13: umbral de evaluabilidad (Argentina, todos los eventos)
        pos = etiq["todos", p][0]
        ok = w[pos].sum() / w.sum() >= UMBRAL_FRAC and pos.sum() >= UMBRAL_N
        print(f"{p:18s} {w[pos].sum() / w.sum():6.1%} {pos.sum():5d} trayectorias  {'evaluable' if ok else 'NO evaluable'}")
        if not ok:
            for var in ("todos", "sin_marcas"):
                etiq.pop((var, p))
    mitad = np.zeros(len(X), int)
    mitad[np.random.default_rng(0).permutation(len(X))[len(X) // 2:]] = 1
    jobs = [(m, d, s, X, w, etiq, mitad, args.azar) for m, d, s in espacios()]
    filas = []
    with Pool(args.workers) as pool:
        for f in pool.imap_unordered(_nivel1, jobs):
            filas += f
            print(f"{f[0]['espacio']:18s} d={f[0]['d']} s={f[0]['semilla']}", flush=True)
    df = pd.DataFrame(filas).sort_values(["eventos", "ponderacion", "proceso", "algoritmo", "metodo", "d", "semilla", "k"])
    df.to_csv(OUT / "nivel1.csv", index=False)
    print("->", P.rel(OUT / "nivel1.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("catalogo")
    sp.set_defaults(fn=cmd_catalogo)
    sp = sub.add_parser("tipologias")
    sp.add_argument("--arranques", type=int, default=10)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--rehacer", action="store_true")
    sp.set_defaults(fn=cmd_tipologias)
    sp = sub.add_parser("nivel1")
    sp.add_argument("--azar", type=int, default=20, help="permutaciones para el piso al azar (§9.3)")
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(fn=cmd_nivel1)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
