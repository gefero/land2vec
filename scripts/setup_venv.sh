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

# torch/torchvision: instalar desde el canal CUDA 12.6 ANTES que requirements.txt.
# El default de PyPI para torch 2.11 es cu130, que dropeó los kernels Pascal/Maxwell
# (sm_50-sm_72). GPUs Pascal (p.ej. GTX 10xx, sm_61) sólo andan con el wheel cu126.
# Override: TORCH_CUDA_CHANNEL=cpu bash scripts/setup_venv.sh  (o cu128, cu130, ...)
TORCH_CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-cu126}"
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"

pip install -r requirements.txt
pip install -e .

echo ""
echo "Listo. torch.cuda.is_available() ->"
python -c "import torch; print(' ', torch.cuda.is_available())"
echo ""
echo "Para activar este entorno en futuras sesiones:"
echo "  source $VENV_DIR/bin/activate"
