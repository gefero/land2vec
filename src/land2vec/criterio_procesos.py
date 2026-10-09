"""Criterio de selección del tuneo del autoencoder (docs/autoencoder_v3/tuneo.md, §3 y §4).

    from land2vec import criterio_procesos as C
    et = C.etiquetas(X)                                  # {proceso: (positivo bool, año del primer evento)}
    val = C.particion(X, area_km2, semilla=0)            # bool: tipos de validación (§3)
    m = C.agrupamiento(z, et, val, area_km2)             # criterio principal: {"S", "S_sup", "grupos<k>_...", ...} (§4.1, §4.2)
    m |= C.sonda(C.vecinos(z[~val], z[val]), et, val, area_km2)   # secundario: {"S_vecinos", "fechado_<proceso>", ...} (§4.3)
"""
import numpy as np

from land2vec.procesos import PROCESOS, eventos_de, procesos_por_evento

FAMILIAS = {
    "deforestacion": ["deforestacion_D1", "deforestacion_D2", "deforestacion_D3"],
    "degradacion": ["degradacion"],
    "regeneracion": ["regeneracion"],
    "expansion_urbana": ["expansion_urbana"],
}
K_VECINOS = 10
K_GRUPOS = (8, 12, 16, 24, 32, 48, 64)
N_INIT = 10   # arranques de k-medias


# ---------------------------------------------------------------------------
# Etiquetas y partición (§2, §3)
# ---------------------------------------------------------------------------
def etiquetas(X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    "Para cada proceso: (positivo, año del primer evento del proceso o nan)."
    pe = procesos_por_evento(eventos_de(X)[0])
    out = {}
    for p in PROCESOS:
        a = pe[pe.proceso == p].groupby("i").anio.min()
        anio = np.full(len(X), np.nan)
        anio[a.index.values] = a.values
        out[p] = (~np.isnan(anio), anio)
    return out


def particion(X: np.ndarray, area: np.ndarray, et: dict, semilla: int = 0, frac: float = 0.2) -> np.ndarray:
    """Máscara de validación: estratos = combinación de familias presentes x decil de superficie entre los
    dinámicos; en cada estrato, el round(frac * n) al azar. Constantes y estratos de un tipo van a entrenamiento."""
    din = (X != X[:, :1]).any(1)
    fam = np.zeros(len(X), np.int64)
    for b, ps in enumerate(("deforestacion_D3", "degradacion", "regeneracion", "expansion_urbana")):
        fam |= et[ps][0].astype(np.int64) << b
    cortes = np.quantile(area[din], np.linspace(0, 1, 11)[1:-1])
    decil = np.searchsorted(cortes, area, side="right")
    estrato = fam * 10 + decil
    rng = np.random.default_rng(semilla)
    val = np.zeros(len(X), bool)
    for e in np.unique(estrato[din]):
        idx = np.flatnonzero(din & (estrato == e))
        if len(idx) < 2:
            continue
        val[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return val


# ---------------------------------------------------------------------------
# Sonda de vecinos (secundaria, §4.3)
# ---------------------------------------------------------------------------
def vecinos(A_tr: np.ndarray, A_va: np.ndarray, k: int = K_VECINOS, metric: str = "euclidean") -> np.ndarray:
    "(n_va, k) índices (en A_tr) de los k vecinos más cercanos."
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute" if metric == "hamming" else "auto")
    return nn.fit(A_tr).kneighbors(A_va, return_distance=False)


def _f1(pred, pos, w):
    tp = (w * (pred & pos)).sum()
    den = (w * pred).sum() + (w * pos).sum()
    return float(2 * tp / den) if den > 0 else np.nan


def sonda(idx: np.ndarray, et: dict, val: np.ndarray, area: np.ndarray) -> dict:
    """Métricas de la sonda de vecinos. idx: vecinos (índices dentro de los tipos de entrenamiento, ~val) de
    cada tipo de validación, en el orden de np.flatnonzero(val)."""
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    m = {}
    for p, (pos, anio) in et.items():
        vp = pos[tr][idx]                                  # (n_va, k) vecinos positivos
        pred = vp.sum(1) * 2 >= idx.shape[1]                # al menos la mitad
        y = pos[va]
        m[f"f1_tipo_{p}"] = _f1(pred, y, np.ones(len(va)))
        m[f"f1_sup_{p}"] = _f1(pred, y, area[va])
        # fechado: positivos predichos positivos; mediana del año entre los vecinos positivos
        sel = np.flatnonzero(pred & y)
        if len(sel):
            an = np.where(vp[sel], anio[tr][idx[sel]], np.nan)
            m[f"fechado_{p}"] = float(np.mean(np.abs(np.nanmedian(an, 1) - anio[va][sel])))
        else:
            m[f"fechado_{p}"] = np.nan
    return m | {"S_vecinos": familias(m, "f1_tipo"), "S_vecinos_sup": familias(m, "f1_sup")}


def familias(m: dict, pref: str) -> float:
    "Promedio de las 4 familias; deforestación = media de D1-D3 (§4.1)."
    return float(np.mean([np.mean([m[f"{pref}_{p}"] for p in ps]) for ps in FAMILIAS.values()]))


# ---------------------------------------------------------------------------
# Criterio principal: agrupamiento con k-medias (§4.1, §4.2)
# ---------------------------------------------------------------------------
def agrupamiento(z: np.ndarray, et: dict, val: np.ndarray, area: np.ndarray, ks=K_GRUPOS, semilla: int = 0) -> dict:
    """k-medias sobre z de todos los tipos (sin ponderar), para cada k. Un grupo se asigna a un proceso si al menos el
    50 % de sus tipos de ENTRENAMIENTO son positivos; F1 en VALIDACIÓN por tipo y por superficie.
    S / S_sup = promedio sobre los k del promedio de las 4 familias."""
    from sklearn.cluster import KMeans
    m, uno = {}, np.ones(val.sum())
    for k in ks:
        lab = KMeans(k, n_init=N_INIT, random_state=semilla).fit_predict(z)
        n = np.bincount(lab[~val], minlength=k)
        for p, (pos, _) in et.items():
            npos = np.bincount(lab[~val], weights=pos[~val], minlength=k)
            pred = ((n > 0) & (npos * 2 >= n))[lab[val]]
            m[f"grupos{k}_f1_tipo_{p}"] = _f1(pred, pos[val], uno)
            m[f"grupos{k}_f1_sup_{p}"] = _f1(pred, pos[val], area[val])
        m[f"grupos{k}_S"] = familias(m, f"grupos{k}_f1_tipo")
        m[f"grupos{k}_S_sup"] = familias(m, f"grupos{k}_f1_sup")
    m["S"] = float(np.mean([m[f"grupos{k}_S"] for k in ks]))
    m["S_sup"] = float(np.mean([m[f"grupos{k}_S_sup"] for k in ks]))
    return m


def reconstruccion(X: np.ndarray, R: np.ndarray, sel: np.ndarray) -> dict:
    return {"rec_anio": float((R[sel] == X[sel]).mean()), "rec_exacta": float((R[sel] == X[sel]).all(1).mean())}
