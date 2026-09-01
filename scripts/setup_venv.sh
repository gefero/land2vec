#!/usr/bin/env bash
# Crea un venv (.venv) e instala land2vec en modo editable.
#
# Uso:
#   bash scripts/setup_venv.sh
#   PYTHON=python3.12 VENV_DIR=.venv312 bash scripts/setup_venv.sh   # override opcional
#
# Después, en cada sesión nueva:
#   source .venv/bin/activate        # Linux/macOS
#   .venv\Scripts\activate           # Windows (cmd/PowerShell)

set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "No se encontró '$PYTHON' en PATH. Instalá Python 3.11+ o corré con PYTHON=<binario> bash scripts/setup_venv.sh" >&2
    exit 1
fi

echo "Creando entorno en $VENV_DIR con $("$PYTHON" --version) ..."
"$PYTHON" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "Listo. torch.cuda.is_available() ->"
python -c "import torch; print(' ', torch.cuda.is_available())"
echo ""
echo "Para activar este entorno en futuras sesiones:"
echo "  source $VENV_DIR/bin/activate"
