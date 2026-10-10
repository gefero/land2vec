#!/usr/bin/env bash
# Crea el entorno de Python para el tuneo en Mendieta. Se corre UNA vez en el nodo cabecera
# (los nodos de cómputo no tienen internet), desde ~/land2vec:
#
#   bash scripts/cluster/entorno_mendieta.sh
#   sbatch scripts/cluster/prueba_mendieta.sbatch      # después: verificar en un nodo de cómputo
#
# Al final arma entorno/py313_tuneo.tar (empaquetar_entorno.sh), que los sbatch descomprimen en el /scratch del nodo.
#
# Notas:
# - uv en vez de Miniconda: un solo binario en ~/.local/bin, mismo resultado (entorno propio en el home).
# - Python: se pide la variante x86-64 base. Los nodos de Mendieta son Xeon E5-2680 v2 (Ivy Bridge, sin AVX2);
#   un intérprete compilado para x86-64-v3 daría "Illegal instruction" en el nodo aunque ande en la cabecera.
# - torch cu126: las A30 son sm_80, soportadas por cu126 y cu128. Override: TORCH_CUDA_CHANNEL=cu128.
# - Sólo las dependencias que importa el tuneo (la cadena universos -> preprocess arrastra geopandas, xarray y rasterio).
set -euo pipefail
cd "$(dirname "$0")/../.."

TORCH_CUDA_CHANNEL=${TORCH_CUDA_CHANNEL:-cu126}
PY=${PY:-cpython-3.13-linux-x86_64-gnu}

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install "$PY"
# only-managed: el Python de uv (relocalizable), no el de Miniconda que esté primero en el PATH
uv venv --python-preference only-managed --python "$PY" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install torch==2.11.0 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"
uv pip install $(grep -E '^(numpy|pandas|scikit-learn|scipy|joblib|threadpoolctl|geopandas|shapely|pyproj|xarray|rasterio|pyarrow)==' requirements.txt)
uv pip install -e .

python -c "import torch, numpy, pandas, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# los jobs no usan .venv desde el /home (lento por NFS): lo empaquetan al /scratch del nodo
bash scripts/cluster/empaquetar_entorno.sh
echo "Entorno listo. Falta probarlo en un nodo con GPU: sbatch scripts/cluster/prueba_mendieta.sbatch"
