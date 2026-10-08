"""Pruebas AE contra OM sin usar OM como juez (protocolo: docs/autoencoder_v3/pruebas_ae_vs_om.md).

Subcomandos (desde la raíz del repo; cada uno escribe en data/autoencoder_v3/pruebas/):
    python scripts/validacion/pruebas_ae_vs_om.py mds          # MDS clásico de OM (cache om_mds.npy)
    python scripts/validacion/pruebas_ae_vs_om.py desfase      # 4.1 pares con desfase
    python scripts/validacion/pruebas_ae_vs_om.py semillas     # 4.2 estabilidad entre semillas
    python scripts/validacion/pruebas_ae_vs_om.py sondas       # 4.3 sondas sobre z
    python scripts/validacion/pruebas_ae_vs_om.py desacuerdos  # 4.4 desacuerdos z contra OM
    python scripts/validacion/pruebas_ae_vs_om.py adyacentes   # 4.5 (1/2) extrae pares de píxeles adyacentes del .nc
    python scripts/validacion/pruebas_ae_vs_om.py espacial     # 4.5 (2/2) coherencia espacial
"""
import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo", _ROOT / "scripts" / "datos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

import p1_compresion as p1  # noqa: E402

DS = (1, 4, 7, 31)
SEEDS = (0, 1, 2)
KS = (1, 2, 5, 10)
OUT = P.DATA / "autoencoder_v3" / "pruebas"
MOD = p1.OUT_MODELS
OM_PATH = p1.OUT_DATA / "om_trate.npy"
YEAR0 = 1992


# ---------------------------------------------------------------------------
# Universo y espacios
# ---------------------------------------------------------------------------
class Universo:
    def __init__(self):
        self.u, self.X = p1.load_universe()
        self.N = len(self.u)
        self.px = self.u.n_px.values.astype(float)
        self.nch = self.u.n_cambios.values
        self.dyn = np.flatnonzero(self.nch > 0)
        toks = [s.split("-") for s in self.u.seqs]
        self.toks = toks
        st, ys = [], []
        for t in toks:
            s, y = [t[0]], []
            for i in range(1, len(t)):
                if t[i] != t[i - 1]:
                    s.append(t[i])
                    y.append(YEAR0 + i)
            st.append(tuple(s))
            ys.append(tuple(y))
        self.states, self.years = st, ys
        codes = {s: i for i, s in enumerate(dict.fromkeys(st))}
        self.proc = np.array([codes[s] for s in st])
        self.year1 = np.array([y[0] if y else 0 for y in ys])
        self.Y = np.zeros((self.N, 4), int)
        for i, y in enumerate(ys):
            self.Y[i, :len(y)] = y


def load_om():
    return np.load(OM_PATH)


def mds_coords(D, kmax=31):
    cache = OUT / "om_mds.npy"
    if cache.exists():
        return np.load(cache)
    from scipy.sparse.linalg import eigsh
    D2 = D.astype(np.float64) ** 2
    r = D2.mean(1)
    B = -0.5 * (D2 - r[:, None] - r[None, :] + r.mean())
    del D2
    w, V = eigsh(B, k=kmax, which="LA")
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    Z = V * np.sqrt(np.clip(w, 0, None))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, Z.astype(np.float32))
    print(f"MDS de OM: autovalores positivos {int((w > 0).sum())} de {kmax}; primeros {np.round(w[:6], 1)}", flush=True)
    return Z.astype(np.float32)


class Esp:
    "Un espacio: embedding Z (N, d) o la matriz OM completa."
    def __init__(self, name, d, seed, Z=None, D=None):
        self.name, self.d, self.seed, self.Z, self.D = name, d, seed, Z, D

    def rows(self, ids):
        if self.D is not None:
            return self.D[ids].astype(np.float32)
        return cdist(self.Z[ids], self.Z).astype(np.float32)

    def pair(self, i, j):
        if self.D is not None:
            return self.D[i, j].astype(np.float32)
        return np.linalg.norm(self.Z[i] - self.Z[j], axis=1).astype(np.float32)

    @property
    def label(self):
        return f"{self.name} d={self.d}" + (f" s{self.seed}" if self.seed is not None else "")


def espacios(U, D, ds=DS, con_om=True, con_onehot=True):
    mds = mds_coords(D)
    out = []
    for d in ds:
        for s in SEEDS:
            out.append(Esp("AE", d, s, np.load(MOD / f"ae_d{d}_s{s}.npz")["z"]))
            out.append(Esp("AE lineal", d, s, np.load(MOD / f"lin_d{d}_s{s}.npz")["z"]))
        out.append(Esp("PCA", d, None, np.load(MOD / f"pca_d{d}.npz")["z"]))
        out.append(Esp("MCA", d, None, np.load(MOD / f"mca_d{d}.npz")["z"]))
        out.append(Esp("OM-MDS", d, None, mds[:, :d].copy()))
    if con_onehot:
        out.append(Esp("One-hot", 341, None, p1.onehot(U.X).astype(np.float32)))
    if con_om:
        out.append(Esp("OM", 0, None, D=D))
    return out


def knn_sets(E, k=10, chunk=512):
    N = E.D.shape[0] if E.D is not None else len(E.Z)
    nn = np.empty((N, k), np.int32)
    for a in range(0, N, chunk):
        ids = np.arange(a, min(N, a + chunk))
        R = E.rows(ids)
        R[np.arange(len(ids)), ids] = np.inf
        nn[ids] = np.argpartition(R, k, axis=1)[:, :k]
    return nn


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / a.shape[1] for x, y in zip(a, b)]))


# ---------------------------------------------------------------------------
# 4.1 Pares con desfase
# ---------------------------------------------------------------------------
def auc_vs(dp, ns):
    left, right = np.searchsorted(ns, dp, "left"), np.searchsorted(ns, dp, "right")
    return float(np.mean((len(ns) - right + 0.5 * (right - left)) / len(ns)))


def cmd_desfase(args):
    U = Universo()
    D = load_om()
    OUT.mkdir(parents=True, exist_ok=True)
    dyn = U.dyn
    groups = {}
    for i in dyn:
        groups.setdefault(U.proc[i], []).append(i)
    pos = {}
    for g in groups.values():
        g = np.array(g)
        Yg = U.Y[g][:, :U.nch[g[0]]]
        K = np.abs(Yg[:, None, :] - Yg[None, :, :]).max(-1)
        for ai, a in enumerate(g):
            for k in KS:
                idx = g[K[ai] == k]
                if len(idx):
                    pos.setdefault(a, {})[k] = idx
    anclas = np.array(sorted(pos))
    print(f"{len(anclas)} anclas con algún positivo (k en {KS}); procesos dinámicos {len(groups)}", flush=True)
    nch_ok = {n: (U.nch == n) for n in (1, 2, 3)}
    rng = np.random.default_rng(0)
    ra, rb = rng.choice(dyn, 20000), rng.choice(dyn, 20000)
    keep = ra != rb
    ra, rb = ra[keep], rb[keep]
    rows = []
    for E in espacios(U, D):
        t0 = time.time()
        med = float(np.median(E.pair(ra, rb)))
        acc = {}   # (nch, k, kind) -> (aucs, pesos, dists)
        for c0 in range(0, len(anclas), 256):
            ids = anclas[c0:c0 + 256]
            R = E.rows(ids)
            for r, a in zip(R, ids):
                n = int(U.nch[a])
                base = nch_ok[n] if n in nch_ok else (U.nch == n)
                gen = np.sort(r[base & (U.proc != U.proc[a])])
                if n == 1:
                    ysame = np.sort(r[base & (U.proc != U.proc[a]) & (U.year1 == U.year1[a])])
                for k, idx in pos[a].items():
                    dp = r[idx]
                    for kind, ns in (("general", gen),) + ((("mismo_anio", ysame),) if n == 1 and len(ysame) else ()):
                        acc.setdefault((n, k, kind), ([], [], []))
                        a_, w_, d_ = acc[(n, k, kind)]
                        a_.append(auc_vs(dp, ns))
                        w_.append(U.px[a])
                        if kind == "general":
                            d_.append(float(np.median(dp)))
        for (n, k, kind), (a_, w_, d_) in acc.items():
            a_, w_ = np.array(a_), np.array(w_)
            rows.append({"espacio": E.name, "d": E.d, "semilla": E.seed, "n_cambios": n, "k": k, "negativos": kind,
                         "auc_tipo": a_.mean(), "auc_px": float((a_ * w_).sum() / w_.sum()), "n_anclas": len(a_),
                         "dist_rel_mediana": (float(np.median(d_)) / med) if d_ else np.nan})
        print(f"  {E.label:22s} {time.time() - t0:5.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_desfase.csv", index=False)
    print("->", P.rel(OUT / "t1_desfase.csv"))


# ---------------------------------------------------------------------------
# 4.2 Estabilidad entre semillas
# ---------------------------------------------------------------------------
def cmd_semillas(args):
    from scipy.spatial import procrustes
    U = Universo()
    D = load_om()
    om = Esp("OM", 0, None, D=D)
    nn_om = knn_sets(om)
    rows = []
    for d in DS:
        for name, f in (("AE", "ae"), ("AE lineal", "lin")):
            Zs = [np.load(MOD / f"{f}_d{d}_s{s}.npz")["z"] for s in SEEDS]
            nns = [knn_sets(Esp(name, d, s, Z)) for s, Z in zip(SEEDS, Zs)]
            for i in range(3):
                for j in range(i + 1, 3):
                    rows.append({"espacio": name, "d": d, "comparacion": f"s{i}-s{j}", "tipo": "entre_semillas",
                                 "knn10": overlap(nns[i], nns[j]), "procrustes": float(procrustes(Zs[i], Zs[j])[2])})
            for i in range(3):
                rows.append({"espacio": name, "d": d, "comparacion": f"s{i}-OM", "tipo": "con_OM", "knn10": overlap(nns[i], nn_om),
                             "procrustes": np.nan})
        # referencias sin semillas contra OM
        for E in espacios(U, D, ds=(d,), con_om=False, con_onehot=False):
            if E.name in ("PCA", "MCA", "OM-MDS"):
                rows.append({"espacio": E.name, "d": d, "comparacion": "-OM", "tipo": "con_OM", "knn10": overlap(knn_sets(E), nn_om),
                             "procrustes": np.nan})
        print("d =", d, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "t2_semillas.csv", index=False)
    print("->", P.rel(OUT / "t2_semillas.csv"))


# ---------------------------------------------------------------------------
# 4.3 Sondas
# ---------------------------------------------------------------------------
def cmd_sondas(args):
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    U = Universo()
    D = load_om()
    dyn = U.dyn
    uno = dyn[U.nch[dyn] == 1]
    ini = np.array([U.states[i][0] for i in range(U.N)])
    fin = np.array([U.states[i][-1] for i in range(U.N)])
    objetivos = [("estado_inicial", dyn, ini, "clf"), ("estado_final", dyn, fin, "clf"), ("n_cambios", dyn, U.nch, "clf"),
                 ("proceso_1_cambio", uno, U.proc, "clf"), ("anio_primer_cambio", dyn, U.year1, "reg")]
    kf = KFold(5, shuffle=True, random_state=0)

    def correr(E):
        Z = E.Z
        out = []
        for nom, idx, y, tipo in objetivos:
            Xf = StandardScaler().fit_transform(Z[idx])
            yy, w = y[idx], U.px[idx]
            for sonda, mdl in (("lineal", LogisticRegression(max_iter=500) if tipo == "clf" else Ridge(1.0)),
                               ("vecinos", KNeighborsClassifier(5) if tipo == "clf" else KNeighborsRegressor(5))):
                pr = cross_val_predict(mdl, Xf, yy, cv=kf)
                if tipo == "clf":
                    ok = (pr == yy).astype(float)
                    out.append({"espacio": E.name, "d": E.d, "semilla": E.seed, "objetivo": nom, "sonda": sonda,
                                "metrica": "exactitud", "tipo": ok.mean(), "px": float((ok * w).sum() / w.sum())})
                else:
                    e = np.abs(pr - yy)
                    out.append({"espacio": E.name, "d": E.d, "semilla": E.seed, "objetivo": nom, "sonda": sonda,
                                "metrica": "mae_anios", "tipo": e.mean(), "px": float((e * w).sum() / w.sum())})
        return out

    esp = [E for E in espacios(U, D, con_om=False) if E.Z is not None]
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(correr)(E) for E in esp)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(OUT / "t3_sondas.csv", index=False)
    print("->", P.rel(OUT / "t3_sondas.csv"))


# ---------------------------------------------------------------------------
# 4.4 Desacuerdos
# ---------------------------------------------------------------------------
def categoria(U, a, b):
    sa, sb = U.states[a], U.states[b]
    if sa == sb:
        return "mismo_proceso"
    mi, mf = sa[0] == sb[0], sa[-1] == sb[-1]
    if mi and mf:
        return "mismo_inicio_y_final_distinta_ruta"
    if mf:
        return "mismo_final_distinto_inicio"
    if mi:
        return "mismo_inicio_distinto_final"
    return "distinto_inicio_y_final"


def desc(U, i):
    return " ".join(f"{U.states[i][k]}>{U.states[i][k + 1]}@{U.years[i][k]}" for k in range(len(U.years[i]))) or "constante"


def cmd_desacuerdos(args):
    U = Universo()
    D = load_om()
    om = Esp("OM", 0, None, D=D)
    dyn = U.dyn
    rows, muestra = [], []
    rng = np.random.default_rng(0)
    for name, f in (("AE", "ae"), ("AE lineal", "lin")):
        E = Esp(name, 4, 0, np.load(MOD / f"{f}_d4_s0.npz")["z"])
        pares = {"z_cerca_OM_lejos": [], "OM_cerca_z_lejos": []}
        for c0 in range(0, len(dyn), 256):
            ids = dyn[c0:c0 + 256]
            Rz, Ro = E.rows(ids), om.rows(ids)
            for r, (rz, ro) in zip(ids, zip(Rz, Ro)):
                rz[r] = np.inf
                ro[r] = np.inf
                for a_row, b_row, key in ((rz, ro, "z_cerca_OM_lejos"), (ro, rz, "OM_cerca_z_lejos")):
                    top = np.argpartition(a_row, 10)[:10]
                    rank_other = (b_row[None, :] < b_row[top][:, None]).sum(1)
                    for b, rk in zip(top, rank_other):
                        if rk > 100 and U.nch[b] > 0:
                            pares[key].append((r, b))
        # línea de base: pares al azar de dinámicas
        az = [(a, b) for a, b in zip(rng.choice(dyn, 20000), rng.choice(dyn, 20000)) if a != b]
        for key, lst in list(pares.items()) + [("azar", az)]:
            cats = pd.Series([categoria(U, a, b) for a, b in lst]).value_counts()
            for c, n in cats.items():
                rows.append({"espacio": name, "conjunto": key, "categoria": c, "n": int(n), "prop": n / len(lst), "total": len(lst)})
            if key != "azar" and lst:
                for j in rng.choice(len(lst), min(20, len(lst)), replace=False):
                    a, b = lst[j]
                    muestra.append({"espacio": name, "conjunto": key, "a": desc(U, a), "b": desc(U, b), "categoria": categoria(U, a, b)})
        print(name, {k: len(v) for k, v in pares.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "t4_desacuerdos.csv", index=False)
    pd.DataFrame(muestra).to_csv(OUT / "t4_muestra.csv", index=False)
    print("->", P.rel(OUT / "t4_desacuerdos.csv"))


# ---------------------------------------------------------------------------
# 4.5 Coherencia espacial
# ---------------------------------------------------------------------------
def cmd_adyacentes(args):
    import xarray as xr
    import censo_trayectorias as ct
    from land2vec.extract import LCCS_CODE_TO_TOKEN
    U = Universo()
    tok2code = {v: k for k, v in LCCS_CODE_TO_TOKEN.items()}
    A = np.array([[tok2code[t] for t in tk] for tk in U.toks], dtype=np.uint8)
    hi, lo = ct.encode(A)
    mult = np.uint64(0x9E3779B97F4A7C15)
    key = (hi * mult) ^ lo
    order = np.argsort(key)
    ks = key[order]
    assert len(np.unique(ks)) == U.N, "colisión de claves"
    ds = xr.open_dataset(ct.P.NC_V3, mask_and_scale=False)
    var = ds["lccs_class"]
    T, NY, NX = var.shape
    mask = ct.argentina_mask(ds["lat"].values.astype(float), ds["lon"].values.astype(float))
    ids = np.full((NY, NX), -1, np.int16)
    for y0 in range(0, NY, ct.BAND):
        a = var[:, y0:y0 + ct.BAND, :].values.reshape(T, -1).T
        m = mask[y0:y0 + ct.BAND, :].ravel()
        h, l = ct.encode(a[m])
        k = (h * mult) ^ l
        pos = np.searchsorted(ks, k)
        assert (ks[np.minimum(pos, len(ks) - 1)] == k).all(), "píxel con trayectoria fuera del universo"
        blk = np.full(m.shape, -1, np.int16)
        blk[m] = order[pos].astype(np.int16)
        ids[y0:y0 + ct.BAND, :] = blk.reshape(-1, NX)
        print(f"  filas {y0}-{min(NY, y0 + ct.BAND)}", flush=True)
    din = U.nch > 0
    tot = ident = 0
    acc = {}
    for A_, B_ in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])):
        ok = (A_ >= 0) & (B_ >= 0)
        a, b = A_[ok].astype(np.int32), B_[ok].astype(np.int32)
        tot += len(a)
        ident += int((a == b).sum())
        m = (a != b) & din[a] & din[b]
        a, b = a[m], b[m]
        lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
        u_, c_ = np.unique(lo_.astype(np.int64) * 8192 + hi_, return_counts=True)
        for x, c in zip(u_, c_):
            acc[x] = acc.get(x, 0) + int(c)
    pk = np.array(list(acc), dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pares_adyacentes.npz", a=(pk // 8192).astype(np.int32), b=(pk % 8192).astype(np.int32),
                        n=np.array(list(acc.values()), dtype=np.int64), total=tot, identicos=ident)
    print(f"pares adyacentes dentro de Argentina: {tot:,}; idénticos {ident:,} ({ident / tot:.2%}); "
          f"distintos y dinámicos: {sum(acc.values()):,} en {len(acc):,} parejas de trayectorias")


def cmd_espacial(args):
    U = Universo()
    D = load_om()
    z = np.load(OUT / "pares_adyacentes.npz")
    a, b, n = z["a"], z["b"], z["n"].astype(float)
    rng = np.random.default_rng(0)
    p = U.px * (U.nch > 0)
    p = p / p.sum()
    ra, rb = rng.choice(U.N, 2_000_000, p=p), rng.choice(U.N, 2_000_000, p=p)
    k = ra != rb
    ra, rb = ra[k], rb[k]
    rows = []
    for E in espacios(U, D):
        nul = np.sort(E.pair(ra, rb))
        d = E.pair(a, b)
        pct = (np.searchsorted(nul, d, "left") + np.searchsorted(nul, d, "right")) / (2 * len(nul))
        rows.append({"espacio": E.name, "d": E.d, "semilla": E.seed, "percentil_medio": float((pct * n).sum() / n.sum()),
                     "frac_bajo_p10": float(((pct < 0.10) * n).sum() / n.sum()),
                     "percentil_mediano": float(np.median(np.repeat(pct, np.minimum(n, 50).astype(int))))})
        print(f"  {E.label:22s} percentil medio {rows[-1]['percentil_medio']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "t5_espacial.csv", index=False)
    print("->", P.rel(OUT / "t5_espacial.csv"))


def cmd_mds(args):
    mds_coords(load_om())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("mds", cmd_mds), ("desfase", cmd_desfase), ("semillas", cmd_semillas), ("sondas", cmd_sondas),
                   ("desacuerdos", cmd_desacuerdos), ("adyacentes", cmd_adyacentes), ("espacial", cmd_espacial)):
        sp = sub.add_parser(nm)
        sp.set_defaults(fn=fn)
        if nm == "sondas":
            sp.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
