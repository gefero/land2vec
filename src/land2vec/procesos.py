"""Eventos y procesos de las trayectorias (protocolo de evaluación v3, Parte II, §7).

Una trayectoria es una fila de tokens (Tokenizer.VOCAB), un año por columna desde Y0.

- Tramo: racha máxima de años con el mismo estado.
- Persistente: dura al menos PERSISTENCIA años; el tramo que empieza en el primer año (censura a la izquierda)
  y el que empieza en uno de los dos últimos años y llega al final (censura a la derecha) cuentan como persistentes.
- Evento: paso entre dos tramos persistentes consecutivos con estados distintos (los transitorios intermedios
  se ignoran). Año = primer año del tramo de destino.
- Oscilación: dos tramos persistentes consecutivos con el mismo estado y transitorios entre ellos.

    from land2vec.procesos import eventos_de
    ev, osc = eventos_de(X)          # DataFrames: un evento / una oscilación por fila, con la fila de X en `i`
"""
import numpy as np
import pandas as pd

from land2vec import paths as P
from land2vec.tokenizer import Tokenizer

Y0 = P.V3_YEARS[0]
PERSISTENCIA = 3
COSTURAS = (1995, 2016)   # = censo_trayectorias.COSTURAS (año de llegada del cambio)
SENSOR = (1999, 2000)

_T = Tokenizer.VOCAB
# proceso -> pares (origen, destino); "*" = cualquier estado
PROCESOS = {
    "deforestacion_D1": {("F", "A")},
    "deforestacion_D2": {("F", "A"), ("Sh", "A")},
    "deforestacion_D3": {("F", "A"), ("Sh", "A"), ("G", "A")},
    "expansion_urbana": {("*", "U")},
    "degradacion": {("F", "G"), ("F", "Sh"), ("F", "B")},
    "regeneracion": {("A", "F"), ("Sh", "F"), ("G", "F")},
}
NOMBRE = {"deforestacion_D1": "Deforestación D1 (F→A)", "deforestacion_D2": "Deforestación D2 (F, Sh→A)",
          "deforestacion_D3": "Deforestación D3 (F, Sh, G→A)", "expansion_urbana": "Expansión urbana (*→U)",
          "degradacion": "Degradación (F→G, Sh, B)", "regeneracion": "Regeneración (A, Sh, G→F)"}


def tramos(row: np.ndarray) -> list[tuple[int, int, int]]:
    "(estado, inicio, largo) de cada tramo; inicio = índice de columna."
    ch = np.flatnonzero(row[1:] != row[:-1]) + 1
    ini = np.r_[0, ch]
    fin = np.r_[ch, len(row)]
    return [(int(row[a]), int(a), int(b - a)) for a, b in zip(ini, fin)]


def persistente(inicio: int, largo: int, T: int, minimo: int = PERSISTENCIA) -> tuple[bool, str]:
    "(es persistente, censura: '' | 'izquierda' | 'derecha')."
    if largo >= minimo:
        return True, ""
    if inicio == 0:
        return True, "izquierda"
    if inicio + largo == T and inicio >= T - (minimo - 1):
        return True, "derecha"
    return False, ""


def procesos_del_evento(origen: str, destino: str) -> list[str]:
    return [p for p, pares in PROCESOS.items() if (origen, destino) in pares or ("*", destino) in pares and origen != destino]


def eventos_de(X: np.ndarray, y0: int = Y0, minimo: int = PERSISTENCIA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eventos y oscilaciones de cada fila de X.

    eventos: i, origen, destino, anio, censura, costura, sensor, transitorios (estados intermedios ignorados)
    oscilaciones: i, base, transitorio (un registro por estado transitorio distinto), anio (primer año del transitorio)
    """
    rev = Tokenizer.REVERSE_VOCAB
    T = X.shape[1]
    ev, osc = [], []
    for i, row in enumerate(X):
        prev = None            # (estado, censura) del último tramo persistente
        trans = []             # transitorios desde ese tramo
        for est, ini, lar in tramos(row):
            ok, cens = persistente(ini, lar, T, minimo)
            if not ok:
                trans.append((est, ini))
                continue
            if prev is not None:
                if est != prev[0]:
                    anio = y0 + ini
                    ev.append((i, rev[prev[0]], rev[est], anio, ",".join(c for c in (prev[1] and "origen_izquierda",
                               cens and "destino_derecha") if c), anio in COSTURAS, anio in SENSOR,
                               "-".join(rev[t] for t, _ in trans)))
                else:
                    for t in dict.fromkeys(t for t, _ in trans):
                        osc.append((i, rev[est], rev[t], y0 + next(a for s, a in trans if s == t)))
            prev, trans = (est, "izquierda" if cens == "izquierda" else ""), []
    ev = pd.DataFrame(ev, columns=["i", "origen", "destino", "anio", "censura", "costura", "sensor", "transitorios"])
    ev["censurado"] = ev.censura != ""
    osc = pd.DataFrame(osc, columns=["i", "base", "transitorio", "anio"])
    return ev, osc


def procesos_por_evento(ev: pd.DataFrame) -> pd.DataFrame:
    "Una fila por (evento, proceso). Los eventos que no son de ningún proceso no aparecen."
    filas = []
    for k, r in enumerate(ev.itertuples(index=False)):
        for p in procesos_del_evento(r.origen, r.destino):
            filas.append((k, p))
    pe = pd.DataFrame(filas, columns=["evento", "proceso"])
    return pe.join(ev, on="evento")
