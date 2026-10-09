#!/bin/bash
# Prueba del entorno en un nodo de cómputo de Mendieta (partición short, < 5 min).
#   sbatch scripts/cluster/prueba_mendieta.sbatch && tail -f logs/prueba-*.out
#SBATCH --job-name=prueba_entorno
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:10
#SBATCH --output=logs/prueba-%j.out

. /etc/profile
source .venv/bin/activate
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python - <<'EOF'
import time, torch, numpy as np
import sys; sys.path.insert(0, "src")
from land2vec.universos import cargar
u = cargar("latam")
print("universo latam:", u.X.shape)
print("torch", torch.__version__, "cuda", torch.version.cuda, "disponible", torch.cuda.is_available())
d = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(d), "capacidad", torch.cuda.get_device_capability(d))
a = torch.randn(4096, 4096, device=d); torch.cuda.synchronize(); t = time.time()
for _ in range(20): a @ a
torch.cuda.synchronize(); print(f"matmul fp32: {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s")
EOF
