"""Control externo de P2 (plan autoencoder_v3 §4.2): ¿cuánta de la información sobre desmonte que
tienen las secuencias sobrevive al agrupar las trayectorias en k tipos, y en qué espacio se pierde menos?

No es una competencia de detección: es una prueba de validez de la descripción. Cada píxel de la
referencia (Colección 13.0, ya cruzada con la grilla por scripts/datos/build_desmonte_labels.py) toma
el tipo de su trayectoria en la tipología de P2 (data/autoencoder_v3/p2/p2_labels.csv). **Las
constantes van a su propia clase estable `const_<estado>`, nunca a un cluster de cambio** (bloqueante B
de la auditoría). Se evalúa sobre positivo / negativo_limpio (tau 0,5, dilate 1, como v2), ponderado por
`weight` (IPW de zonas submuestreadas).

Métricas por tipología:
    u_ratio       U(D|T) ajustado por permutación / U(D|secuencia exacta): fracción de la información
                  de la secuencia que conserva la partición. Sin umbral. Es la métrica principal.
    mcc_cv        MCC de la tabla por tipo con validación cruzada espacial (5 folds, bloques de 0,2°;
                  el umbral se elige en train). Mismo procedimiento que el techo de REF-07 (tabla por
                  secuencia exacta), así que mcc_cv / mcc_cv_seq es la fracción del techo que se conserva.
    mcc_sem       MCC de la regla semántica "el cluster es desmonte si su medoide cumple R0'". No usa la referencia.
IC por bootstrap de bloques (0,05°) sobre sumas de confusión por bloque; la diferencia con R0' es pareada.

Líneas de base y techo (p2_desmonte_bases.csv): R0 (original), R0' (F → {A,G,Sh,Sp,B} tras la última F),
F2000∧¬F2022, "alguna transición" y la tabla por secuencia exacta con CV espacial.

Uso, desde la raíz del repo:
    python scripts/validacion/p2_desmonte.py [--zone chaco_santiago_frontier] [--spaces om ae_d8] [--ks 12] [--n-boot 1000]
    python scripts/validacion/p2_desmonte.py --cv-seeds 20      # ruido de la CV espacial
"""
import argparse
import importlib.util
import time
from pathlib import Path
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec import geo as G  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_desmonte", ROOT / "scripts" / "validacion" / "eval_desmonte.py")
ED = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ED)

OUT = P.DATA / "autoencoder_v3" / "p2"
ZONES = ["chaco_santiago_frontier", "yungas", "periurbano_cordoba"]
DEFOR_TARGET = {"A", "G", "Sh", "Sp", "B"}


# ---------------------------------------------------------------------------
# Reglas por secuencia (se aplican a las trayectorias distintas y se mapean a los píxeles)
# ---------------------------------------------------------------------------
def rule_r0(toks): return not np.isnan(ED._r0_year(toks))


def rule_r0p(toks):
    "F → {A,G,Sh,Sp,B} en algún año posterior a la última F."
    f = [i for i, t in enumerate(toks) if t == "F"]
    return bool(f) and any(t in DEFOR_TARGET for t in toks[f[-1] + 1:])


def rule_f2000(toks): return toks[0] == "F" and toks[-1] != "F"
def rule_cambio(toks): return any(a != b for a, b in zip(toks, toks[1:]))


RULES = {"r0": rule_r0, "r0p": rule_r0p, "f2000_no_f2022": rule_f2000, "alguna_transicion": rule_cambio}


# ---------------------------------------------------------------------------
# Métricas sobre arrays
# ---------------------------------------------------------------------------
def conf_mcc(y, yhat, w) -> float:
    vp, vn = w[y & yhat].sum(), w[~y & ~yhat].sum()
    fp, fn = w[~y & yhat].sum(), w[y & ~yhat].sum()
    return ED._mcc(vp, vn, fp, fn)


def block_conf(y, yhat, w, bidx, nb):
    "Matriz (nb, 4) de vp, vn, fp, fn ponderados por bloque."
    out = np.zeros((nb, 4))
    for j, m in enumerate((y & yhat, ~y & ~yhat, ~y & yhat, y & ~yhat)):
        out[:, j] = np.bincount(bidx[m], weights=w[m], minlength=nb)
    return out


def mcc_from_conf(c) -> np.ndarray:
    vp, vn, fp, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))  # negativo sólo por error de redondeo en los extremos
        return (vp * vn - fp * fn) / den


def boot_ci(conf_a, conf_b=None, n_boot=1000, seed=42):
    "IC 95 % del MCC (o de la diferencia pareada a - b) remuestreando bloques con reposición."
    rng = np.random.default_rng(seed)
    nb = conf_a.shape[0]
    cnt = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(float)
    est = mcc_from_conf(cnt @ conf_a)
    if conf_b is not None:
        est = est - mcc_from_conf(cnt @ conf_b)
    est = est[np.isfinite(est)]
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))) if len(est) else (np.nan, np.nan)


def _h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def u_adjusted(codes, y, w, n_perm=49, seed=42) -> float:
    "U(D|C) = [H(D) − H(D|C)] / H(D), ponderado, menos la media de n_perm permutaciones de la etiqueta."
    wy = w * y
    p0 = wy.sum() / w.sum()
    hd = float(_h2(p0))
    if hd == 0:
        return 0.0
    ncls = codes.max() + 1

    def h_cond(c):
        ws, wys = np.bincount(c, weights=w, minlength=ncls), np.bincount(c, weights=wy, minlength=ncls)
        m = ws > 0
        return float((ws[m] / ws.sum() * _h2(wys[m] / ws[m])).sum())

    rng = np.random.default_rng(seed)
    perm = np.mean([(hd - h_cond(rng.permutation(codes))) / hd for _ in range(n_perm)])
    return float((hd - h_cond(codes)) / hd - perm)


def cv_table_mcc(codes, y, w, block, folds=5, seed=42):
    """MCC out-of-fold de la tabla por clase: p_c por train, umbral que maximiza el MCC en train, se
    aplica en eval. Folds por bloques espaciales. Devuelve (mcc, yhat OOF)."""
    ub = np.unique(block)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(ub), np.arange(len(ub)) % folds))
    fold = np.array([fold_of[b] for b in ub])[np.searchsorted(ub, block)]
    ncls = codes.max() + 1
    yhat = np.zeros(len(y), bool)
    wy = w * y
    for k in range(folds):
        tr, ev = fold != k, fold == k
        ws = np.bincount(codes[tr], weights=w[tr], minlength=ncls)
        wys = np.bincount(codes[tr], weights=wy[tr], minlength=ncls)
        seen = ws > 0
        pc = np.where(seen, wys / np.maximum(ws, 1e-12), wys.sum() / ws.sum())
        order = np.argsort(-pc[seen])
        cls = np.flatnonzero(seen)[order]
        tp, fp = np.cumsum(wys[cls]), np.cumsum(ws[cls] - wys[cls])
        P_, N_ = wys.sum(), ws.sum() - wys.sum()
        mcc = mcc_from_conf(np.stack([tp, N_ - fp, fp, P_ - tp], axis=1))
        t = pc[cls][np.nanargmax(mcc)]
        yhat[ev] = pc[codes[ev]] >= t
    return conf_mcc(y, yhat, w), yhat


# ---------------------------------------------------------------------------
# Datos por zona
# ---------------------------------------------------------------------------
def load_zone(zone: str, uni: pd.DataFrame) -> pd.DataFrame:
    px = ED.load_desmonte_px(zone, ROOT / "data")
    coords, seqs = ED.load_coords(zone, ROOT / "data"), ED.load_seqs(zone, ROOT / "data")
    grid = G.ZoneGrid.from_coords(coords, zone)
    base = ED.build_base(px, coords, ED.baseline_table(seqs), ED.compute_weights(seqs, grid))
    base = base.merge(seqs[["ID", "seqs"]], on="ID", how="left")
    base = base[base["referencia"].isin(["positivo", "negativo_limpio"])].copy()
    base = base.merge(uni[["seqs", "traj_id", "constante", "costura"]], on="seqs", how="left")
    assert base.traj_id.notna().all(), "hay secuencias de la zona que no están en el universo"
    base["traj_id"] = base.traj_id.astype(int)
    base["block20"] = ED.spatial_blocks(base.latitude.to_numpy(), base.longitude.to_numpy(), 0.2)
    for name, fn in RULES.items():
        flag = {s: fn(s.split("-")) for s in base.seqs.unique()}
        base[name] = base.seqs.map(flag).astype(bool)
    return base


def factorize(a):
    return pd.factorize(a)[0]


def evaluate(zone: str, uni_name: str, df: pd.DataFrame, labels: pd.DataFrame, uni_tab: pd.DataFrame, n_boot: int):
    "df: píxeles evaluables de la zona (ya filtrados por universo). Devuelve (filas de bases, filas de tipologías)."
    y, w = (df.referencia == "positivo").to_numpy(), df.weight.to_numpy()
    b05, nb = pd.factorize(df.block)[0], df.block.nunique()
    out_b, out_t = [], []

    def base_row(name, yhat=None, conf=None):
        c = block_conf(y, yhat, w, b05, nb)
        lo, hi = boot_ci(c, n_boot=n_boot)
        return {"zona": zone, "universo": uni_name, "linea": name, "mcc": float(mcc_from_conf(c.sum(0))), "ic_lo": lo, "ic_hi": hi}, c

    confs = {}
    for name in RULES:
        r, confs[name] = base_row(name, df[name].to_numpy())
        out_b.append(r)
    codes_seq = factorize(df.seqs)
    u_seq = u_adjusted(codes_seq, y, w)
    mcc_seq, yh_seq = cv_table_mcc(codes_seq, y, w, df.block20.to_numpy())
    r, _ = base_row("techo_tabla_secuencia_cv", yh_seq)
    out_b.append(r | {"u_adj": u_seq, "n_eval": len(df), "prevalencia": float(w[y].sum() / w.sum())})

    const_state = df.seqs.str.split("-").str[0]
    gcols = (["metodo"] if "metodo" in labels else []) + ["espacio", "peso", "k"]
    for key, lab in labels.groupby(gcols):
        met, (esp, pw, k) = (key[0], key[1:]) if "metodo" in labels else ("kmedoides", key)
        lm = lab.set_index("traj_id")
        et = df.traj_id.map(lm.etiqueta)
        part = np.where(df.constante.to_numpy(), "const_" + const_state.to_numpy(), "c" + et.fillna(-1).astype(int).astype(str).to_numpy())
        codes = factorize(part)
        mcc_t, _ = cv_table_mcc(codes, y, w, df.block20.to_numpy())
        u_t = u_adjusted(codes, y, w)
        # regla semántica: el cluster es desmonte si el medoide de su tipo cumple R0'
        med = df.traj_id.map(lm.medoide)
        flag_med = med.map(uni_tab.set_index("traj_id").seqs).map(lambda s: rule_r0p(s.split("-") if isinstance(s, str) else ["x"]))
        yhat = np.where(df.constante.to_numpy(), False, flag_med.fillna(False).to_numpy().astype(bool))
        c_sem = block_conf(y, yhat, w, b05, nb)
        lo, hi = boot_ci(c_sem, n_boot=n_boot)
        dlo, dhi = boot_ci(c_sem, confs["r0p"], n_boot=n_boot)
        out_t.append({"zona": zone, "universo": uni_name, "metodo": met, "espacio": esp, "peso": pw, "k": k, "n_clases": len(np.unique(codes)),
                      "u_adj": u_t, "u_ratio": u_t / u_seq if u_seq else np.nan,
                      "mcc_cv": mcc_t, "mcc_cv_ratio_techo": mcc_t / mcc_seq,
                      "mcc_sem": float(mcc_from_conf(c_sem.sum(0))), "mcc_sem_lo": lo, "mcc_sem_hi": hi,
                      "mcc_sem_menos_r0p": float(mcc_from_conf(c_sem.sum(0)) - mcc_from_conf(confs["r0p"].sum(0))),
                      "dif_lo": dlo, "dif_hi": dhi})
    return out_b, out_t


def cv_noise(zone: str, df: pd.DataFrame, labels: pd.DataFrame, n_seeds: int) -> list[dict]:
    """Ruido de la CV espacial: repite `cv_table_mcc` con n_seeds particiones distintas de bloques en folds
    (mismos folds para la secuencia exacta y para todas las tipologías en cada semilla, así que la razón
    mcc_T / mcc_seq y las diferencias entre espacios son pareadas). Sólo universo completo y peso `px`."""
    y, w = (df.referencia == "positivo").to_numpy(), df.weight.to_numpy()
    codes_seq = factorize(df.seqs)
    const_state = df.seqs.str.split("-").str[0].to_numpy()
    parts = {}
    for (esp, k), lab in labels[(labels.universo == "completo") & (labels.peso == "px")].groupby(["espacio", "k"]):
        et = df.traj_id.map(lab.set_index("traj_id").etiqueta)
        parts[(esp, k)] = factorize(np.where(df.constante.to_numpy(), "const_" + const_state, "c" + et.fillna(-1).astype(int).astype(str).to_numpy()))
    rows = []
    for sd in range(n_seeds):
        m_seq, _ = cv_table_mcc(codes_seq, y, w, df.block20.to_numpy(), seed=sd)
        for (esp, k), codes in parts.items():
            m_t, _ = cv_table_mcc(codes, y, w, df.block20.to_numpy(), seed=sd)
            rows.append({"zona": zone, "semilla": sd, "espacio": esp, "k": k, "mcc_seq": m_seq, "mcc_cv": m_t, "ratio": m_t / m_seq})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", action="append", default=None, help="repetible; default: chaco_santiago_frontier y yungas")
    ap.add_argument("--spaces", nargs="+", default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--labels", type=Path, default=OUT / "p2_labels.csv", help="etiquetas a evaluar (p2_labels.csv o p2_labels_metodos.csv, con columna `metodo`)")
    ap.add_argument("--solo-completo", action="store_true", help="sólo el universo completo")
    ap.add_argument("--tag", default="", help="sufijo de los CSV de salida (p. ej. _metodos)")
    ap.add_argument("--cv-seeds", type=int, default=0, help="si > 0: sólo mide el ruido de la CV con N semillas de folds (p2_desmonte_cv.csv)")
    args = ap.parse_args()

    uni = pd.read_csv(P.DATA / "autoencoder_v3" / "universo_argentina.csv")
    labels = pd.read_csv(args.labels)
    if args.spaces:
        labels = labels[labels.espacio.isin(args.spaces)]
    if args.ks:
        labels = labels[labels.k.isin(args.ks)]

    if args.cv_seeds:
        rows = []
        for zone in args.zone or ZONES[:2]:
            rows += cv_noise(zone, load_zone(zone, uni), labels, args.cv_seeds)
            print(zone, "listo", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "p2_desmonte_cv.csv", index=False)
        print("->", P.rel(OUT / "p2_desmonte_cv.csv"))
        return

    rows_b, rows_t = [], []
    for zone in args.zone or ZONES[:2]:
        t0 = time.time()
        dfz = load_zone(zone, uni)
        print(f"{zone}: {len(dfz):,} px evaluables, prevalencia {dfz.weight[dfz.referencia == 'positivo'].sum() / dfz.weight.sum():.4f}", flush=True)
        for uni_name in (("completo",) if args.solo_completo else ("completo", "sin_costura")):
            df = dfz if uni_name == "completo" else dfz[~dfz.costura.astype(bool)]
            lab = labels[labels.universo == uni_name]
            b, t = evaluate(zone, uni_name, df.reset_index(drop=True), lab, uni, args.n_boot)
            rows_b += b
            rows_t += t
            print(f"  {uni_name}: {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_b).to_csv(OUT / f"p2_desmonte_bases{args.tag}.csv", index=False)
    pd.DataFrame(rows_t).to_csv(OUT / f"p2_desmonte{args.tag}.csv", index=False)
    print("->", P.rel(OUT))


if __name__ == "__main__":
    main()
