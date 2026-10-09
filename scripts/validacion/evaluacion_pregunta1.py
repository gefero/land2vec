"""Evaluación de la Pregunta 1: capacidad de compresión (protocolo: docs/autoencoder_v3/protocolo_evaluacion.md, Parte I).

Subcomandos, en orden (desde la raíz del repo; salidas en data/autoencoder_v3/pregunta1/):
    python scripts/validacion/evaluacion_pregunta1.py no_vistos       # trayectorias del mundo que no están en Argentina y sus estratos A y B (§3)
    python scripts/validacion/evaluacion_pregunta1.py codificar       # z y reconstrucción de cada modelo, para el mundo y para Argentina (caché)
    python scripts/validacion/evaluacion_pregunta1.py reconstruccion  # reconstrucción (§4) -> p11_reconstruccion.csv, p11_parametros.csv

Requiere los modelos de models/autoencoder_v3/p1/: autoencoders (`train`), autoencoders lineales con pesos (`linae`)
y el ajuste de PCA y MCA (`linear`), todos de scripts/modelo/p1_compresion.py.
"""
import argparse
import sys
import time
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("no_vistos", cmd_no_vistos), ("codificar", cmd_codificar), ("reconstruccion", cmd_reconstruccion)):
        sp = sub.add_parser(nm)
        sp.set_defaults(fn=fn)
        if nm in ("codificar", "reconstruccion"):
            sp.add_argument("--metodos", nargs="+", default=list(METODOS), choices=METODOS)
            sp.add_argument("--dims", nargs="+", type=int, default=None, help="default: p1_compresion.DIMS")
        if nm == "codificar":
            sp.add_argument("--threads", type=int, default=6)
            sp.add_argument("--rehacer", action="store_true")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
