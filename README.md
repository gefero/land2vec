<picture>
  <source media="(prefers-color-scheme: dark)" srcset="imgs/logo-dark.png">
  <img alt="land2vec" src="imgs/logo-light.png" width="360">
</picture>

`land2vec` entrena un modelo de lenguaje tipo GPT (transformer decoder-only,
causal self-attention) sobre **secuencias temporales de uso/cobertura del
suelo** de parcelas entre 2000 y 2022 (zonas de Chaco, Santiago del Estero y
frontera agrícola). La idea es análoga a *word2vec*, pero en vez de predecir
palabras a partir de su contexto, el modelo predice el próximo estado de uso
del suelo de una parcela a partir de su historial de estados anteriores.

## Estructura del repo

```
src/land2vec/
  config.py     # dataclass Config con hiperparámetros del modelo/entrenamiento
  tokenizer.py  # Tokenizer estático: vocabulario fijo de estados de uso del suelo
  dataset.py    # Datasets de PyTorch (ventaneado y no ventaneado) + carga de CSV/zip
  model.py      # GPTDecoder (transformer causal) + loop de entrenamiento/eval
  utils.py      # Guardado/carga de config, modelo y métricas
data/           # Secuencias de entrenamiento y de test (CSV/zip)
models/         # Checkpoints entrenados (config.json + model.pt + train_data.csv)
notebooks/      # Notebooks de experimentación ("pruebas") en Google Colab
```

## Instalación

```bash
pip install -r requirements.txt
pip install -e .
```

Requiere Python 3.11+ (usa `dataclass(slots=True)` y sintaxis de tipos
moderna) y, para entrenar en GPU, una instalación de PyTorch con soporte
CUDA.

## Datos

Cada fila de un CSV de secuencias representa una parcela, identificada por
`ID`, con una columna `seqs` que contiene su trayectoria anual de estados
separados por `-`, por ejemplo:

```
ID,seqs
0,F-Sh-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-Sh-Sh-Sh-Sh
```

El vocabulario de estados (`land2vec.tokenizer.Tokenizer.VOCAB`) es:

| Token   | Significado (código) |
|---------|----|
| `[UNK]` | desconocido / relleno, ignorado en la loss |
| `A`     | estado A |
| `F`     | forestal |
| `G`     | pastizal/grassland |
| `Wt`    | humedal (wetland) |
| `U`     | urbano |
| `Sh`    | arbustal (shrub) |
| `Sp`    | estado Sp |
| `B`     | estado B |
| `Wa`    | agua (water) |
| `Nd`    | sin dato (no data) |

Archivos en `data/`:

- `id_seqs_text_2000_2022_chaco_santiago_frontier.zip` — dataset principal de entrenamiento (Chaco, Santiago del Estero, frontera).
- `id_seqs_text_2000_2022_test_set.zip` — set de test held-out, usado en la evaluación final.
- `test_sample_0/1/2.zip` — muestras adicionales de test.
- `seqs_short.csv` — muestra chica (10 filas) usada para pruebas rápidas/debug.

`land2vec.dataset.load_data()` carga cualquiera de estos archivos y
devuelve:

- `SequenceDataset` (si se pasa `window=`) — ejemplos de next-token
  prediction con ventana deslizante de tamaño fijo.
- `SequenceDatasetNonWindow` (si no se pasa `window`) — usa la secuencia
  completa de cada parcela como un solo ejemplo (padding/batching lo maneja
  el `DataLoader`).

## Modelo

`GPTDecoder` (`src/land2vec/model.py`) es un transformer decoder causal
"desde cero": embeddings de token + posición, bloques de
self-attention causal (`F.scaled_dot_product_attention`) + feed-forward con
GELU, weight tying entre el embedding de entrada y la capa de salida
(`lm_head`), y un método `generate()` con muestreo por temperatura, top-k y
top-p.

Los hiperparámetros por defecto están en `land2vec.config.Config`
(`block_size`, `n_embd`, `n_head`, `n_layer`, `dropout`, `epochs`, `lr`,
`patience`, `batch_size`, `device`, `seed`, ...).

`land2vec.model.run_epoch()` corre una época de entrenamiento o evaluación
(según si se pasa un `optimizer`), con soporte de AMP (`torch.autocast` +
`GradScaler`) y cross-entropy ponderada que ignora el token `[UNK]`.

## Entrenamiento y evaluación (uso típico)

```python
from land2vec.config import Config
from land2vec.dataset import load_data
from land2vec.model import GPTDecoder, run_epoch
from land2vec.tokenizer import Tokenizer
from land2vec.utils import save_config, save_model, collect_predictions, compute_metrics

config = Config(block_size=22, batch_size=1024)
dataset = load_data(file_path="data/seqs_short.csv")  # o el dataset completo

model = GPTDecoder(
    vocab_size=len(Tokenizer.VOCAB),
    block_size=config.block_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,
    dropout=config.dropout,
).to(config.device)

# ... entrenar con run_epoch() en un loop con early stopping por patience ...

preds, targets, loss = collect_predictions(model, val_loader, config.device, weights)
metrics = compute_metrics(targets, preds)  # accuracy, macro F1
```

Los checkpoints se guardan/cargan con `save_model`/`load_model` y
`save_config`/`load_config` de `land2vec.utils`, en una carpeta por modelo
dentro de `models/` (`config.json` + `model.pt` + `train_data.csv` con el
historial de entrenamiento).

## Ejemplo de uso: inferencia con un modelo entrenado

Este ejemplo carga el modelo final (`models/full_model`) y predice el
próximo estado de uso del suelo a partir de una secuencia histórica:

```python
from pathlib import Path
import torch

from land2vec.tokenizer import Tokenizer
from land2vec.utils import load_config, load_model

target_folder = Path("models/full_model")
config = load_config(target_folder)
model = load_model(config, target_folder)  # ya queda en eval() y en config.device

# Secuencia histórica de una parcela (estados separados por "-")
seq = "F-Sh-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-F-Sh-Sh-Sh"
tokens = torch.tensor([Tokenizer.encode(seq)], device=config.device)  # (1, T)

# Predecir el próximo estado más probable
with torch.inference_mode():
    logits = model(tokens[:, -config.block_size:])
next_state_id = logits[0, -1].argmax().item()
print(Tokenizer.decode(torch.tensor([next_state_id])))  # p.ej. "Sh"

# Generar varios pasos hacia adelante muestreando (autoregresivo)
generated = model.generate(tokens, max_new_tokens=5, temperature=0.8, top_k=5)
print(Tokenizer.decode(generated[0]))
```

## Notebooks / pruebas experimentales

Los notebooks en `notebooks/` documentan las corridas de experimentación
(diseñados para correr en Google Colab, con carga de datos y de módulos
adaptada a ese entorno). Cada uno sigue el flujo *cargar datos → crear/cargar
modelo → entrenar → evaluar → predecir*:

| Notebook | Qué prueba | Config relevante | Modelo resultante |
|---|---|---|---|
| `prueba_1.ipynb` | Exploración inicial de datos y primer entrenamiento con dataset ventaneado (`window=block_size`) | `patience=6` | `models/patience_6` (no incluido en el repo) |
| `prueba_2.ipynb` | Segunda iteración de entrenamiento, mismo esquema ventaneado | `patience=4` | `models/2026-05-20` |
| `prueba_3.ipynb` | Cambia a dataset **no ventaneado** (secuencia completa por parcela) y a un dataset balanceado; agrega matriz de confusión | `patience=4, batch_size=1024, block_size=22` | `models/balanced_1` |
| `test_1.ipynb` | Retoma el modelo `balanced_1`, continúa el entrenamiento y evalúa contra el set de test held-out (`id_seqs_text_2000_2022_test_set.zip`) | Config heredada de `balanced_1`, `patience=6` | `models/full_model` (modelo final) |

No hay tests automatizados (`pytest` u otro framework); la validación del
proyecto se hace de forma exploratoria en estos notebooks, evaluando
accuracy y F1 macro sobre datos de validación/test y revisando matrices de
confusión.

## Resultados preliminares

> ⚠️ **Resultados preliminares**, obtenidos en `test_2.ipynb` evaluando el
> modelo final (`models/full_model`) sobre el set de test held-out
> (`data/id_seqs_text_2000_2022_test_set.zip`). Sujetos a revisión con más
> datos y validaciones adicionales.

- Parámetros del modelo: 795,520
- Mejor F1 macro en validación (durante entrenamiento, época 1): 0.9886
- **Accuracy (test set)**: 0.9929
- **Macro F1 (test set)**: 0.9005

## Modelos entrenados incluidos

- `models/full_model/` — modelo final, entrenado sobre el dataset completo y evaluado en el test set held-out (ver `test_1.ipynb`).
- `models/balanced_1/` — modelo entrenado sobre un dataset balanceado, con secuencias completas sin ventaneo (ver `prueba_3.ipynb`).
- `models/2026-05-20/` — checkpoint intermedio de una corrida anterior (ver `prueba_2.ipynb`).
- `models/first-test.pt` — checkpoint suelto de una prueba temprana.

Cada carpeta de modelo incluye `config.json` (hiperparámetros usados),
`model.pt` (pesos) y `train_data.csv` (historial de loss/métricas por época).

## Créditos

- **Coordinación**: Germán Rosati (CONICET-EIDAES/UNSAM)
- **Colaboración**: Gonzalo Jara (Lic. en Ciencia de Datos - ECyT/UNSAM)
