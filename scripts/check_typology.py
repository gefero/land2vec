"""Smoke test de `land2vec.typology` y `land2vec.seqdist` sobre datos sintéticos.

Corré directo desde el repo, sin `pip install` ni PYTHONPATH:

    python scripts/check_typology.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from land2vec.seqdist import _smoke as seqdist_smoke  # noqa: E402
from land2vec.typology import _smoke as typology_smoke  # noqa: E402

if __name__ == "__main__":
    typology_smoke()
    seqdist_smoke()
