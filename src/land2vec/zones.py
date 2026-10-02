"""Lista canónica de zonas de land2vec v2 -- **solo librería estándar**, para que
scripts sin torch/pandas (`scripts/viz/build_cluster_map.py`) la puedan importar
igual que `scripts/clustering/tune_clustering.py`/`scripts/clustering/describe_clusters.py` importan
`land2vec.cluster` (con el mismo preámbulo `sys.path.insert`).

Dos grupos, disjuntos por diseño (ver `scripts/datos/build_eval_zones.py`):

- **eval**: las 7 zonas out-of-domain, benchmark held-out. Nunca vistas por el
  autoencoder ni por el clustering.
- **train**: `chaco_santiago_frontier` (la zona original, usada para entrenar
  el autoencoder desde la v1) + las 7 zonas nuevas de la v2, una por
  ecorregión de "eval" (`PAIRING`). El autoencoder las vio en entrenamiento
  -- son in-sample para los embeddings, aunque out-of-sample para el
  clustering en sí, que solo se ajustó sobre "eval". Ver
  `docs/v2/v2_autoencoder_training.md` §4.1.

Antes esta lista estaba duplicada en `land2vec.cluster`, `build_cluster_map.py`,
`build_eval_zones.py` y cuatro `EXTRA_ZONE_LABELS` en los scripts de imagery.
"""

from __future__ import annotations

# bbox = (minx, miny, maxx, maxy) en lon/lat -- mismo formato que
# `land2vec.extract.crop_bbox` y `scripts/datos/build_eval_zones.py`.

EVAL_ZONES: dict[str, tuple[float, float, float, float]] = {
    "puna_noa": (-67.0, -23.5, -65.0, -22.0),
    "patagonia_estepa": (-70.0, -43.0, -66.0, -40.0),
    "periurbano_cordoba": (-64.4, -31.6, -64.0, -31.2),
    "ibera": (-58.0, -29.0, -56.5, -27.5),
    "delta_parana": (-59.6, -33.8, -58.6, -32.4),
    "pampa_nucleo": (-62.0, -35.0, -60.0, -33.0),
    "misiones_selva": (-55.5, -27.5, -54.0, -25.5),
}

TRAIN_ZONES: dict[str, tuple[float, float, float, float]] = {
    "puna_salta_catamarca": (-68.5, -26.5, -66.5, -24.5),
    "patagonia_santacruz": (-71.0, -49.0, -67.0, -46.0),
    "periurbano_gba": (-58.8, -34.9, -58.3, -34.4),
    "corrientes_humedal": (-58.9, -27.4, -58.1, -26.3),
    "delta_oeste": (-59.95, -33.8, -59.65, -32.6),
    "pampa_deprimida": (-60.0, -37.5, -58.0, -36.0),
    "yungas": (-64.8, -25.5, -64.0, -24.0),
}

# Zona original de entrenamiento del autoencoder (desde la v1), sin par de
# evaluación propio -- bbox de scripts/datos/build_eval_zones.py:TRAIN_BBOX.
CHACO_ZONE: dict[str, tuple[float, float, float, float]] = {
    "chaco_santiago_frontier": (
        -63.44994621163554, -28.12819902009702, -59.37401847726054, -25.431378332142593,
    ),
}

ZONE_LABELS: dict[str, str] = {
    # eval (out-of-domain)
    "puna_noa": "Puna (Jujuy)",
    "patagonia_estepa": "Estepa patagónica",
    "periurbano_cordoba": "Periurbano (Córdoba)",
    "ibera": "Iberá",
    "delta_parana": "Delta del Paraná",
    "pampa_nucleo": "Pampa núcleo",
    "misiones_selva": "Selva misionera",
    # train (in-sample para el encoder)
    "chaco_santiago_frontier": "Chaco-Santiago (frontera, train)",
    "puna_salta_catamarca": "Puna (Salta-Catamarca, train)",
    "patagonia_santacruz": "Patagonia (Santa Cruz, train)",
    "periurbano_gba": "Periurbano (GBA, train)",
    "corrientes_humedal": "Humedal (Corrientes, train)",
    "delta_oeste": "Delta oeste (train)",
    "pampa_deprimida": "Pampa deprimida (train)",
    "yungas": "Yungas (train)",
}

ZONES_BY_GROUP: dict[str, list[str]] = {
    "eval": list(EVAL_ZONES),
    "train": list(CHACO_ZONE) + list(TRAIN_ZONES),
}

GROUP_OF: dict[str, str] = {
    z: g for g, zs in ZONES_BY_GROUP.items() for z in zs
}

# zona de entrenamiento -> su par de evaluación, misma ecorregión, bbox
# disjunto -- docs/v2/v2_autoencoder_training.md §4.1.
PAIRING: dict[str, str] = {
    "puna_salta_catamarca": "puna_noa",
    "patagonia_santacruz": "patagonia_estepa",
    "periurbano_gba": "periurbano_cordoba",
    "corrientes_humedal": "ibera",
    "delta_oeste": "delta_parana",
    "pampa_deprimida": "pampa_nucleo",
    "yungas": "misiones_selva",
}
