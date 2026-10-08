"""Evaluación de la Pregunta 1 (protocolo: docs/autoencoder_v3/protocolo_evaluacion.md, §2-§7 y §9).

Subcomandos, en orden (desde la raíz del repo; salidas en data/autoencoder_v3/pregunta1/):
    python scripts/validacion/evaluacion_pregunta1.py no_vistos       # trayectorias del mundo que no están en Argentina y sus estratos A y B (§3)
    python scripts/validacion/evaluacion_pregunta1.py codificar       # z y reconstrucción de cada modelo, para el mundo y para Argentina (caché)
    python scripts/validacion/evaluacion_pregunta1.py reconstruccion  # 1.1 reconstrucción (§4)  -> p11_reconstruccion.csv, p11_parametros.csv
    python scripts/validacion/evaluacion_pregunta1.py accesibilidad   # 1.1 accesibilidad (§5)   -> p11_accesibilidad.csv
    python scripts/validacion/evaluacion_pregunta1.py tipologias      # 1.2 tipologías (§9)      -> p12_tipologias.csv, p12_semillas.csv

Requiere los modelos de models/autoencoder_v3/p1/: autoencoders (`train`), autoencoders lineales con pesos (`linae`)
y el ajuste de PCA y MCA (`linear`), todos de scripts/modelo/p1_compresion.py.
"""
import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo", _ROOT / "scripts" / "datos", _ROOT / "scripts" / "clustering"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import p1_compresion as p1  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

OUT = P.DATA / "autoencoder_v3" / "pregunta1"
CACHE = OUT / "codigos"
MUNDO = P.DATA / "autoencoder_v3" / "mundo" / "universo_mundo.csv.gz"
METODOS = ("ae", "lin", "pca", "mca")
NOMBRE = {"ae": "AE", "lin": "AE lineal", "pca": "PCA", "mca": "MCA", "om": "OM", "onehot": "One-hot (Hamming)"}
SEEDS = (0, 1, 2)
B_CORTES = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 31, "6+")]
YEAR0 = 1992
CLASES = [Tokenizer.VOCAB[t] for t in ("A", "F", "G", "Wt", "U", "Sh", "Sp", "B", "Wa")]
K_TIPOLOGIA = list(range(4, 49, 4))
D_TIPOLOGIA = (4, 2, 7)


# ---------------------------------------------------------------------------
# Trayectorias
# ---------------------------------------------------------------------------
def procesos(X: np.ndarray):
    "Para cada trayectoria (fila de tokens): proceso (tupla de estados sucesivos) y año del primer cambio (0 si es constante)."
    proc, y1 = [], np.zeros(len(X), int)
    for i, row in enumerate(X):
        ch = np.flatnonzero(row[1:] != row[:-1]) + 1
        proc.append(tuple(row[np.r_[0, ch]].tolist()))
        y1[i] = YEAR0 + ch[0] if len(ch) else 0
    return proc, y1


def estrato_B(h):
    out = np.empty(len(h), object)
    for lo, hi, lab in B_CORTES:
        out[(h >= lo) & (h <= hi)] = lab
    return out


def cargar_no_vistos():
    z = np.load(OUT / "no_vistos.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def cmd_no_vistos(args):
    from censo_trayectorias import decode
    from land2vec.extract import LCCS_CODE_TO_TOKEN
    u, Xa = p1.load_universe()
    m = pd.read_csv(MUNDO)
    m = m[m.n_cambios > 0].reset_index(drop=True)
    tok = np.array([Tokenizer.VOCAB[LCCS_CODE_TO_TOKEN[c]] if c in LCCS_CODE_TO_TOKEN else -1 for c in range(16)])
    Xw = tok[np.array([decode(int(h), int(lo), 31) for h, lo in zip(m.hi.astype("uint64"), m.lo.astype("uint64"))])]
    assert (Xw > 0).all(), "estados fuera del vocabulario en el mundo"
    vistas = {r.tobytes() for r in Xa.astype(np.int64)}
    nuevo = np.array([r.tobytes() not in vistas for r in Xw.astype(np.int64)])
    Xw, m = Xw[nuevo], m[nuevo].reset_index(drop=True)
    proc_w, _ = procesos(Xw)
    proc_a = set(procesos(Xa)[0])
    visto = np.array([p in proc_a for p in proc_w])
    h = np.empty(len(Xw), np.int16)
    for c in range(0, len(Xw), 256):
        h[c:c + 256] = (Xw[c:c + 256, None, :] != Xa[None, :, :]).sum(2).min(1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "no_vistos.npz", X=Xw.astype(np.uint8), visto=visto, h=h,
                        area=m.area_km2.values, n_px=m.n_px.values)
    t = pd.crosstab(estrato_B(h), np.where(visto, "proceso visto", "proceso nuevo"), margins=True)
    print(f"{len(Xw):,} trayectorias dinámicas del mundo que no están en Argentina\n{t}")


# ---------------------------------------------------------------------------
# Codificación (caché)
# ---------------------------------------------------------------------------
def modelos(metodos=METODOS, dims=None):
    for d in dims or p1.DIMS:
        for met in metodos:
            for s in (SEEDS if met in ("ae", "lin") else (None,)):
                yield met, d, s


def tag(met, d, s):
    return f"{met}_d{d}" + (f"_s{s}" if s is not None else "")


def codificar(met, d, s, X):
    if met == "ae":
        return p1.ae_apply(tag(met, d, s), X)
    if met == "lin":
        mdl = np.load(p1.OUT_MODELS / f"{tag(met, d, s)}.npz")
        assert "We" in mdl.files, f"{tag(met, d, s)} no tiene pesos: correr `p1_compresion.py linae`"
        return p1.lin_apply(mdl, X)
    ajuste = dict(np.load(p1.OUT_MODELS / "lineales" / f"{met}.npz"))
    return (p1.pca_apply if met == "pca" else p1.mca_apply)(ajuste, X, d)


def cmd_codificar(args):
    import torch
    torch.set_num_threads(args.threads)
    _, Xa = p1.load_universe()
    Xw = cargar_no_vistos()["X"].astype(np.int64)
    CACHE.mkdir(parents=True, exist_ok=True)
    for met, d, s in modelos(args.metodos, args.dims):
        f = CACHE / f"{tag(met, d, s)}.npz"
        if f.exists() and not args.rehacer:
            continue
        t0 = time.time()
        zw, Rw = codificar(met, d, s, Xw)
        za, Ra = codificar(met, d, s, Xa)
        np.savez_compressed(f, z_mundo=zw.astype(np.float32), R_mundo=Rw.astype(np.uint8),
                            z_arg=za.astype(np.float32), R_arg=Ra.astype(np.uint8))
        print(f"{tag(met, d, s):12s} {time.time() - t0:5.1f}s  exactitud por año: mundo {(Rw == Xw).mean():.4f}, Argentina {(Ra == Xa).mean():.4f}", flush=True)


def cargar_codigos(met, d, s):
    return np.load(CACHE / f"{tag(met, d, s)}.npz")


# ---------------------------------------------------------------------------
# 1.1 Reconstrucción (§4)
# ---------------------------------------------------------------------------
def grupos_mundo(nv):
    "Conjuntos de evaluación: (A, B, máscara)."
    A = np.where(nv["visto"], "visto", "nuevo")
    B = estrato_B(nv["h"])
    out = [("todos", "todos", np.ones(len(A), bool))]
    for a in ("visto", "nuevo"):
        out.append((a, "todos", A == a))
        for _, _, b in B_CORTES:
            out.append((a, b, (A == a) & (B == b)))
    return out


def wmean(v, w):
    ok = ~np.isnan(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else np.nan


def metricas_reconstruccion(X, R, pesos):
    from sklearn.metrics import f1_score
    pt = p1.per_traj(X, R)
    out = []
    for wn, w in pesos.items():
        for col in ("acc_anio", "estados_ok", "exacta", "n_cambios_ok", "err_anio_cambio"):
            out.append((wn, col, wmean(pt[col].values.astype(float), w)))
        out.append((wn, "f1_macro", float(f1_score(X.ravel(), R.ravel(), labels=CLASES, average="macro",
                                                    sample_weight=np.repeat(w, X.shape[1]), zero_division=0))))
    return out


def cmd_reconstruccion(args):
    u, Xa = p1.load_universe()
    din = ~u.constante.values
    nv = cargar_no_vistos()
    Xw = nv["X"].astype(np.int64)
    gm = grupos_mundo(nv)
    rows, par = [], []
    for met, d, s in modelos(args.metodos, args.dims):
        c = cargar_codigos(met, d, s)
        base = {"metodo": NOMBRE[met], "d": d, "semilla": s}
        for a, b, mk in gm:
            for wn, mt, v in metricas_reconstruccion(Xw[mk], c["R_mundo"][mk].astype(np.int64),
                                                     {"tipo": np.ones(mk.sum()), "superficie": nv["area"][mk]}):
                rows.append(base | {"conjunto": "mundo no visto", "A": a, "B": b, "n_tipos": int(mk.sum()), "peso": wn, "metrica": mt, "valor": v})
        for wn, mt, v in metricas_reconstruccion(Xa[din], c["R_arg"][din].astype(np.int64),
                                                 {"tipo": np.ones(din.sum()), "superficie": u.n_px.values[din].astype(float)}):
            rows.append(base | {"conjunto": "Argentina (ajuste)", "A": "-", "B": "-", "n_tipos": int(din.sum()), "peso": wn, "metrica": mt, "valor": v})
        par.append(base | {"parametros": p1.n_parametros(met, d, tag(met, d, s))})
        print(tag(met, d, s), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "p11_reconstruccion.csv", index=False)
    pd.DataFrame(par).to_csv(OUT / "p11_parametros.csv", index=False)
    print("->", P.rel(OUT / "p11_reconstruccion.csv"))


# ---------------------------------------------------------------------------
# 1.1 Accesibilidad (§5)
# ---------------------------------------------------------------------------
def objetivos(X):
    proc, y1 = procesos(X)
    pid = np.array(["-".join(map(str, p)) for p in proc])
    return {"estado_inicial": X[:, 0], "estado_final": X[:, -1], "n_cambios": np.array([len(p) - 1 for p in proc]),
            "proceso": pid, "anio_primer_cambio": y1}


def sondear(job):
    "Ajusta las sondas con Argentina y evalúa en el mundo por estrato. job = (met, d, s, za, zw, ya, yw, gm, area)."
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    met, d, s, za, zw, ya, yw, gm, area = job
    sc = StandardScaler().fit(za)
    out = []
    for obj in ya:
        reg = obj == "anio_primer_cambio"
        sondas = {"vecinos": (KNeighborsRegressor(5) if reg else KNeighborsClassifier(5), za, zw),
                  "lineal": (Ridge(alpha=1.0) if reg else LogisticRegression(C=1.0, max_iter=1000), sc.transform(za), sc.transform(zw))}
        for nombre, (mdl, ta, tw) in sondas.items():
            pred = mdl.fit(ta, ya[obj]).predict(tw)
            for a, b, mk in gm:
                if obj == "proceso" and a != "visto":
                    continue
                for wn, w in (("tipo", np.ones(mk.sum())), ("superficie", area[mk])):
                    if reg:
                        v = float((np.abs(pred[mk] - yw[obj][mk]) * w).sum() / w.sum())
                        mt = "error_medio_anios"
                    else:
                        v = float(balanced_accuracy_score(yw[obj][mk], pred[mk], sample_weight=w))
                        mt = "exactitud_balanceada"
                    out.append({"metodo": NOMBRE[met], "d": d, "semilla": s, "objetivo": obj, "sonda": nombre, "A": a, "B": b,
                                "n_tipos": int(mk.sum()), "peso": wn, "metrica": mt, "valor": v})
    return out


def cmd_accesibilidad(args):
    import warnings
    from joblib import Parallel, delayed
    from sklearn.metrics import balanced_accuracy_score
    warnings.filterwarnings("ignore")   # balanced_accuracy avisa cuando el predictor usa clases ausentes del estrato
    u, Xa = p1.load_universe()
    din = ~u.constante.values
    nv = cargar_no_vistos()
    Xw = nv["X"].astype(np.int64)
    gm = grupos_mundo(nv)
    ya, yw = objetivos(Xa[din]), objetivos(Xw)
    # líneas de base: clase más frecuente / año promedio de Argentina
    rows = []
    for obj in ya:
        reg = obj == "anio_primer_cambio"
        if reg:
            pred_w = np.full(len(Xw), ya[obj].mean())
        else:
            vals, cnt = np.unique(ya[obj], return_counts=True)
            pred_w = np.full(len(Xw), vals[cnt.argmax()], dtype=ya[obj].dtype)
        for a, b, mk in gm:
            if obj == "proceso" and a != "visto":
                continue
            v = float(np.abs(pred_w[mk] - yw[obj][mk]).mean()) if reg else float(balanced_accuracy_score(yw[obj][mk], pred_w[mk]))
            rows.append({"metodo": "línea de base", "d": 0, "semilla": None, "objetivo": obj, "sonda": "-", "A": a, "B": b,
                         "n_tipos": int(mk.sum()), "peso": "tipo", "metrica": "error_medio_anios" if reg else "exactitud_balanceada", "valor": v})
    jobs = []
    for met, d, s in modelos(args.metodos, args.dims):
        c = cargar_codigos(met, d, s)
        jobs.append((met, d, s, c["z_arg"][din], c["z_mundo"], ya, yw, gm, nv["area"]))
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(sondear)(j) for j in jobs)
    rows += [r for rr in res for r in rr]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "p11_accesibilidad.csv", index=False)
    print("->", P.rel(OUT / "p11_accesibilidad.csv"))


# ---------------------------------------------------------------------------
# 1.2 Tipologías (§9)
# ---------------------------------------------------------------------------
def metricas_tipologia(lab, X, pid, y1):
    "Exactitud por año del prototipo, pureza de proceso y dispersión del año del primer cambio; todas por tipo."
    n, T = X.shape
    k = lab.max() + 1
    cnt = np.zeros((k, T, p1.V), np.int32)
    np.add.at(cnt, (np.repeat(lab, T), np.tile(np.arange(T), n), X.ravel()), 1)
    proto = cnt.argmax(2)
    acc = float((X == proto[lab]).mean())
    pure = 0
    disp = np.empty(n)
    for g in range(k):
        idx = np.flatnonzero(lab == g)
        if not len(idx):
            continue
        _, c = np.unique(pid[idx], return_counts=True)
        pure += c.max()
        disp[idx] = np.abs(y1[idx] - np.median(y1[idx]))
    return {"exactitud_prototipo": acc, "pureza_proceso": pure / n, "dispersion_anio": float(disp.mean())}


def tamanos(lab):
    c = np.bincount(lab)
    c = c[c > 0]
    return {"n_grupos": len(c), "grupos_menores_5": int((c < 5).sum()), "frac_grupo_mayor": float(c.max() / c.sum())}


def espacios_tipologia(u, din, args):
    from scipy.spatial.distance import cdist
    for d in D_TIPOLOGIA:
        for met in METODOS:
            for s in (SEEDS if met in ("ae", "lin") else (None,)):
                z = cargar_codigos(met, d, s)["z_arg"][din]
                yield met, d, s, cdist(z, z).astype(np.float32)
    D = np.load(p1.OUT_DATA / "om_trate.npy", mmap_mode="r")
    yield "om", 0, None, np.ascontiguousarray(D[np.ix_(din, din)], dtype=np.float32)
    _, Xa = p1.load_universe()
    OH = p1.onehot(Xa[din]).astype(np.float32)
    yield "onehot", 0, None, (Xa.shape[1] - OH @ OH.T).astype(np.float32)


def cmd_tipologias(args):
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score
    from p2_tipologias import kmedoids
    u, Xa = p1.load_universe()
    din = np.flatnonzero(~u.constante.values)
    X = Xa[din]
    proc, y1 = procesos(X)
    pid = np.array(["-".join(map(str, p)) for p in proc])
    rng_perm = np.random.default_rng(0)
    rows, mejores = [], {}
    for met, d, s, D in espacios_tipologia(u, din, args):
        t0 = time.time()
        base = {"espacio": NOMBRE[met], "d": d, "semilla": s}
        Zl = linkage(squareform(D, checks=False), "complete")
        for k in K_TIPOLOGIA:
            rng = np.random.default_rng(k)
            corridas = [kmedoids(D, np.ones(len(D)), k, rng, n_init=1) for _ in range(args.arranques)]
            costos = [c[2] for c in corridas]
            lab_km = corridas[int(np.argmin(costos))][1]
            mejores[(met, d, s, k)] = lab_km
            ari = float(np.mean([adjusted_rand_score(corridas[i][1], corridas[j][1]) for i, j in combinations(range(len(corridas)), 2)]))
            # cut_tree corta en exactamente k grupos siguiendo el orden de las uniones; fcluster("maxclust") devuelve
            # menos grupos cuando hay alturas empatadas (pasa con Hamming, que es entera: con k = 4 daba un solo grupo)
            lab_jq = cut_tree(Zl, n_clusters=k).ravel()
            for alg, lab, extra in (("k-medoides", lab_km, {"ari_entre_arranques": ari}), ("jerarquico completo", lab_jq, {})):
                mt = metricas_tipologia(lab, X, pid, y1)
                az = [metricas_tipologia(rng_perm.permutation(lab), X, pid, y1) for _ in range(20)]
                rows.append(base | {"algoritmo": alg, "k": k} | mt | {f"azar_{c}": float(np.mean([a[c] for a in az])) for c in mt}
                            | tamanos(lab) | extra)
        print(f"{base['espacio']:18s} d={d:<2} s={s}  {time.time() - t0:5.0f}s", flush=True)
    sem = []
    for met in ("ae", "lin"):
        for d in D_TIPOLOGIA:
            for k in K_TIPOLOGIA:
                aris = [adjusted_rand_score(mejores[(met, d, a, k)], mejores[(met, d, b, k)]) for a, b in combinations(SEEDS, 2)]
                sem.append({"espacio": NOMBRE[met], "d": d, "k": k, "ari_entre_semillas_media": float(np.mean(aris)),
                            "ari_min": float(np.min(aris)), "ari_max": float(np.max(aris))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "p12_tipologias.csv", index=False)
    pd.DataFrame(sem).to_csv(OUT / "p12_semillas.csv", index=False)
    print("->", P.rel(OUT / "p12_tipologias.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("no_vistos", cmd_no_vistos), ("codificar", cmd_codificar), ("reconstruccion", cmd_reconstruccion),
                   ("accesibilidad", cmd_accesibilidad), ("tipologias", cmd_tipologias)):
        sp = sub.add_parser(nm)
        sp.set_defaults(fn=fn)
        if nm in ("codificar", "reconstruccion", "accesibilidad"):
            sp.add_argument("--metodos", nargs="+", default=list(METODOS), choices=METODOS)
            sp.add_argument("--dims", nargs="+", type=int, default=None, help="default: p1_compresion.DIMS")
        if nm == "codificar":
            sp.add_argument("--threads", type=int, default=6)
            sp.add_argument("--rehacer", action="store_true")
        if nm == "accesibilidad":
            sp.add_argument("--jobs", type=int, default=4)
        if nm == "tipologias":
            sp.add_argument("--arranques", type=int, default=10)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
