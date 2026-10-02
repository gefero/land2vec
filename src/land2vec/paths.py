"""Rutas del repo en un solo lugar."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path | str) -> str:
    "Ruta relativa al repo para mensajes; absoluta si cae fuera (p. ej. un --out-dir en /tmp)."
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
