"""Evaluación de la Pregunta 2: detección de procesos (protocolo: docs/autoencoder_v3/protocolo_evaluacion.md, Parte II).

Subcomandos (desde la raíz del repo; salidas en data/autoencoder_v3/pregunta2/):
    python scripts/validacion/evaluacion_pregunta2.py catalogo    # catálogo descriptivo de procesos (§7.6), Argentina y mundo

Los eventos y procesos se definen en src/land2vec/procesos.py.
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo", _ROOT / "scripts" / "datos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import p1_compresion as p1  # noqa: E402
from land2vec.procesos import PROCESOS, eventos_de, procesos_por_evento  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402

OUT = P.DATA / "autoencoder_v3" / "pregunta2"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("catalogo")
    sp.set_defaults(fn=cmd_catalogo)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
