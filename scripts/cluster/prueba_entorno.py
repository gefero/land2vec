"""Prueba del entorno del tuneo en un nodo con GPU (la lanza scripts/cluster/prueba_mendieta.sbatch).

Recorre, con la hora de cada etapa, lo mismo que hace una corrida de tuneo_ae.py: importar, cargar el universo, iniciar
CUDA, entrenar unos pasos del autoencoder y evaluar con k-medias. Si algo se cuelga, faulthandler vuelca la pila cada 60 s.
"""
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.dump_traceback_later(60, repeat=True)
T0 = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def paso(msg):
    print(f"{time.time() - T0:7.1f}s  {msg}", flush=True)


paso("inicio")
import numpy as np  # noqa: E402
import torch  # noqa: E402
paso(f"torch {torch.__version__}, cuda {torch.version.cuda}")
from land2vec.universos import cargar  # noqa: E402
from land2vec.model import TrajectoryAutoencoder  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402
u = cargar("latam")
paso(f"universo latam {u.X.shape}")

paso(f"cuda disponible: {torch.cuda.is_available()}")
dev = torch.device("cuda")
torch.cuda.init()
paso(f"GPU {torch.cuda.get_device_name(dev)}, capacidad {torch.cuda.get_device_capability(dev)}")
a = torch.randn(4096, 4096, device=dev)
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
paso(f"matmul fp32: {20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12:.1f} TFLOP/s")

V = len(Tokenizer.VOCAB)
X = torch.tensor(u.X, device=dev)
model = TrajectoryAutoencoder(V, u.X.shape[1], 8, n_embd=128, n_head=4, n_layer=2, dropout=0.1, pooling="query").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
paso("modelo base (n_embd 128, 2 capas) en la GPU")
for n_pasos in (20, 500):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n_pasos):
        b = torch.randint(0, len(X), (128,), device=dev)
        loss = torch.nn.functional.cross_entropy(model(X[b]).reshape(-1, V), X[b].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    paso(f"{n_pasos} pasos de entrenamiento, lote 128: {n_pasos / (time.time() - t):.1f} pasos/s")

from sklearn.cluster import KMeans  # noqa: E402
with torch.no_grad():
    z = torch.cat([model.encode(X[i:i + 1024]) for i in range(0, len(X), 1024)]).cpu().numpy()
t = time.time()
for k in (8, 12, 16, 24, 32, 48, 64):
    KMeans(k, n_init=10, random_state=0).fit_predict(z)
paso(f"k-medias de la evaluación (7 k x 10 arranques): {time.time() - t:.1f}s")
faulthandler.cancel_dump_traceback_later()
paso("PRUEBA OK")
