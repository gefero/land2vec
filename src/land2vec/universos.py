"""Universos de trayectorias de autoencoder_v3 (1992-2022) con una interfaz común.

    from land2vec.universos import cargar, no_vistos
    u = cargar("latam")          # "argentina" | "latam" | "mundo"
    u.X                          # (n, T) tokens de Tokenizer.VOCAB, un año por columna desde P.V3_YEARS[0]
    u.n_px, u.area_km2           # superficie de cada trayectoria (area_km2 = nan en Argentina: su censo no la guarda)
    u.dinamica                   # bool: tiene al menos un cambio
    nv = no_vistos()             # trayectorias del mundo que no aparecen en América Latina

Argentina viene de censo_trayectorias.py (secuencias como texto); América Latina y el mundo, de censo_mundial.py
(4 bits por año en dos uint64, `hi` y `lo`, con los códigos de preprocess.token_table).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from land2vec import paths as P
from land2vec import preprocess as pp
from land2vec.tokenizer import Tokenizer

T = P.V3_YEARS[1] - P.V3_YEARS[0] + 1
ARCHIVOS = {
    "argentina": P.DATA / "autoencoder_v3" / "universo_argentina.csv",
    "latam": P.DATA / "autoencoder_v3" / "latam" / "universo_latam.csv.gz",
    "mundo": P.DATA / "autoencoder_v3" / "mundo" / "universo_mundo.csv.gz",
}
# código de 4 bits del censo -> token del vocabulario (-1: código sin token)
_COD2TOK = np.full(16, -1, np.int64)
for _c, _t in enumerate(pp.token_table(pp.LCCS_IPCC)):
    _COD2TOK[_c] = Tokenizer.VOCAB[_t]


@dataclass
class Universo:
    nombre: str
    X: np.ndarray
    n_px: np.ndarray
    area_km2: np.ndarray

    @property
    def dinamica(self) -> np.ndarray:
        return (self.X != self.X[:, :1]).any(1)

    def __len__(self):
        return len(self.X)

    def subconjunto(self, sel, nombre=None) -> "Universo":
        return Universo(nombre or self.nombre, self.X[sel], self.n_px[sel], self.area_km2[sel])


def decodificar(hi: np.ndarray, lo: np.ndarray, T: int = T) -> np.ndarray:
    "(n,) uint64 x 2 -> (n, T) códigos de 4 bits (misma codificación que censo_trayectorias.encode / decode)."
    k = min(T, 12 if T <= 23 else 16)   # = censo_trayectorias.split_years
    hi, lo = np.asarray(hi, np.uint64), np.asarray(lo, np.uint64)
    a = [(hi >> np.uint64(4 * (k - 1 - i))) & np.uint64(15) for i in range(k)]
    b = [(lo >> np.uint64(4 * (T - k - 1 - i))) & np.uint64(15) for i in range(T - k)]
    return np.stack(a + b, 1).astype(np.int64)


def cargar(nombre: str) -> Universo:
    f = ARCHIVOS[nombre]
    if nombre == "argentina":
        u = pd.read_csv(f)
        X = np.array([Tokenizer.encode(s) for s in u.seqs], dtype=np.int64)
        area = np.full(len(u), np.nan)
    else:
        u = pd.read_csv(f, dtype={"hi": "uint64", "lo": "uint64"})
        X = _COD2TOK[decodificar(u.hi.values, u.lo.values)]
        area = u.area_km2.values.astype(float)
    assert X.shape[1] == T and (X > 0).all(), f"{nombre}: estados fuera del vocabulario o largo distinto de {T}"
    return Universo(nombre, X, u.n_px.values.astype(np.int64), area)


def claves(X: np.ndarray) -> np.ndarray:
    "Una clave hashable por fila (bytes de la secuencia), para cruzar universos."
    Xb = np.ascontiguousarray(X.astype(np.uint8))
    return Xb.view(np.dtype((np.void, Xb.shape[1]))).ravel()


def no_vistos(referencia: str = "latam") -> Universo:
    "Trayectorias del mundo que no aparecen en `referencia`."
    m, r = cargar("mundo"), cargar(referencia)
    return m.subconjunto(~np.isin(claves(m.X), claves(r.X)), f"mundo sin {referencia}")
