"""Catálogo de procesos por universo (autoencoder_v3, plan de tuneo, Etapa 0).

Para cada universo dinámico (Argentina, América Latina, mundo y mundo sin América Latina): número de tipos y
superficie con cada proceso de src/land2vec/procesos.py, y si pasa el umbral de evaluabilidad
(>= 1 % de la superficie dinámica y >= 100 tipos). La superficie es km², salvo en Argentina (píxeles: su censo
no guarda el área).

    python scripts/validacion/catalogo_procesos.py   # -> data/autoencoder_v3/latam/catalogo_procesos.csv
"""
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from land2vec.procesos import PROCESOS, eventos_de, procesos_por_evento  # noqa: E402
from land2vec.universos import cargar, no_vistos  # noqa: E402

UMBRAL_FRAC, UMBRAL_N = 0.01, 100
OUT = P.DATA / "autoencoder_v3" / "latam" / "catalogo_procesos.csv"


def catalogo(u):
    d = u.subconjunto(u.dinamica)
    w = d.n_px.astype(float) if np.isnan(d.area_km2).all() else d.area_km2
    pe = procesos_por_evento(eventos_de(d.X)[0])
    filas = [{"universo": u.nombre, "proceso": "(dinámicas)", "n_tipos": len(d), "superficie": w.sum(), "frac": 1.0}]
    for p in PROCESOS:
        idx = np.unique(pe.i[pe.proceso == p].values)
        filas.append({"universo": u.nombre, "proceso": p, "n_tipos": len(idx), "superficie": w[idx].sum(),
                      "frac": w[idx].sum() / w.sum()})
    df = pd.DataFrame(filas)
    df["unidad"] = "px" if np.isnan(d.area_km2).all() else "km2"
    df["evaluable"] = (df.frac >= UMBRAL_FRAC) & (df.n_tipos >= UMBRAL_N)
    return df


def main():
    df = pd.concat([catalogo(u) for u in (cargar("argentina"), cargar("latam"), cargar("mundo"), no_vistos("latam"))])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={"superficie": "{:,.0f}".format, "frac": "{:.1%}".format}))
    print("->", P.rel(OUT))


if __name__ == "__main__":
    main()
