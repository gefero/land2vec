"""¿Por qué variables se organizan los grupos de k-medias sobre z? (docs/autoencoder_v3/tuneo.md §8.2)

Agrupa con k-medias (10 arranques, semilla 0, cada tipo pesa 1, como el criterio S) los z de un modelo del tuneo
y, como comparación, PCA d = 8 (ajustado con los tipos de entrenamiento) y el one-hot. Sobre los tipos dinámicos
de América Latina mide la información mutua normalizada (NMI) entre el grupo y:
  - los procesos presentes (combinación de las 4 familias),
  - el estado inicial, el estado final y el par inicial-final,
  - el año del primer cambio y el número de cambios.
Y, para los positivos de cada familia, cuántos grupos hacen falta para cubrir el 80 % (fragmentación).

    python scripts/validacion/diagnostico_z.py                                   # el piloto
    python scripts/validacion/diagnostico_z.py --corrida busqueda/c07_d8_s0 --k 8 24 64
"""
import argparse
from pathlib import Path
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score as nmi  # noqa: E402

import p1_compresion as p1  # noqa: E402
import tuneo_ae as T  # noqa: E402

FAMILIAS = {"deforestacion": "deforestacion_D3", "degradacion": "degradacion", "regeneracion": "regeneracion",
            "expansion_urbana": "expansion_urbana"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrida", default="piloto/base_d8_s0", help="<etapa>/<corrida> en models/autoencoder_v3/tuneo")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 24])
    a = ap.parse_args()

    u, val, et = T.datos()
    X, din = u.X, u.dinamica
    ini, fin = X[:, 0], X[:, -1]
    cambia = X[:, 1:] != X[:, :-1]
    anio1 = np.where(cambia.any(1), cambia.argmax(1) + 1 + P.V3_YEARS[0], 0)
    combo = sum(et[p][0].astype(int) << b for b, p in enumerate(FAMILIAS.values()))
    ajuste = p1.pca_fit(X[~val], T.pesos_ajuste(u.n_px[~val], T.BASE["tope"]), 8)
    espacios = {a.corrida: np.load(T.MODELS / a.corrida / "codigos.npz")["z"],
                "PCA d=8": p1.pca_apply(ajuste, X, 8)[0], "one-hot": p1.onehot(X).astype(np.float32)}

    filas = []
    for k in a.k:
        for nombre, z in espacios.items():
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(z)
            g = lab[din]
            f = {"k": k, "espacio": nombre, "procesos": nmi(combo[din], g), "estado inicial": nmi(ini[din], g),
                 "estado final": nmi(fin[din], g), "par inicial-final": nmi((ini * 20 + fin)[din], g),
                 "año del 1er cambio": nmi(anio1[din], g), "n.º de cambios": nmi(cambia.sum(1)[din], g)}
            for fam, p in FAMILIAS.items():
                pos = et[p][0]
                c = np.sort(np.bincount(lab[pos]))[::-1].cumsum() / pos.sum()
                f[f"grupos para el 80 % de {fam}"] = int(np.searchsorted(c, 0.8) + 1)
            filas.append(f)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(filas).set_index(["k", "espacio"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
