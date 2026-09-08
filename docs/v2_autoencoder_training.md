# land2vec v2 — autoencoder de trayectorias

Documento de referencia del entrenamiento de la segunda versión de land2vec:
un autoencoder secuencial que comprime la trayectoria anual de uso del suelo
de una parcela (2000-2022, 23 estados) en un vector de baja dimensión (`z`),
pensado como insumo para clustering/tipología de trayectorias, como variable
de entrada de otros modelos, y para visualización/exploración espacial.

Ver el plan original en `/root/.claude/plans/1-inspeccion-el-repo-scalable-scroll.md`
(sesión de diseño) y el código real en `src/land2vec/model.py`,
`src/land2vec/config.py`, `src/land2vec/utils.py`, `src/land2vec/extract.py`
y `scripts/train_autoencoder.py`.

## 1. Por qué una v2

La v1 (`models/full_model`, `GPTDecoder`) es un transformer causal
decoder-only entrenado para predecir el próximo estado de uso del suelo.
Cumple ese objetivo, pero **no produce un embedding de trayectoria**: nunca
está obligada a resumir la secuencia completa en un vector (su estado
oculto de 128 dims por posición ni siquiera queda expuesto fuera de
`forward()`), y `notebooks/eval_ood_zones.ipynb` mostró que además
generaliza mal fuera de la región de entrenamiento original
(Chaco-Santiago-frontera): macro F1 pooled 0.7507 en 7 zonas out-of-domain,
contra 0.9005 in-domain, con un colapso a 0.4682 en la zona dominada por
clases casi ausentes en entrenamiento (`puna_noa`, con `B`/`Sp`).

La v2 ataca dos problemas a la vez:

1. **Arquitectura con cuello de botella explícito**, que fuerza a comprimir
   toda la trayectoria en un vector `z` de pocas dimensiones (en vez de
   nunca tener que hacerlo, como la v1).
2. **Datos de entrenamiento geográficamente más diversos**: además de
   Chaco-Santiago, se suman 7 zonas nuevas repartidas por el país, elegidas
   en las mismas ecorregiones que las 7 zonas de evaluación out-of-domain
   pero con huella geográfica disjunta de ellas — así el benchmark de la v1
   se puede reutilizar para medir la v2 sin haberlo "quemado" durante el
   entrenamiento.

## 2. Arquitectura: `TrajectoryAutoencoder`

Clase real en `src/land2vec/model.py:149`. Flujo:

```
secuencia (23 años) → ENCODER (bidireccional) → pooling → z (d dims) → DECODER (bidireccional) → secuencia reconstruida (23 años)
```

### 2.1 Encoder

1. **Embeddings de entrada**: cada uno de los 23 tokens se convierte en un
   vector de `n_embd=128` dims (`token_embedding`, tabla 11×128) y se le
   suma el embedding de su posición (`position_embedding`, tabla 23×128).
2. **`n_layer` bloques transformer bidireccionales** (`Block` con
   `is_causal=False`): igual estructura que los bloques de la v1
   (autoatención + feed-forward con GELU), pero cada posición puede atender
   a **todas** las demás, no solo a las anteriores — no hay nada que
   predecir "hacia adelante", el objetivo es resumir toda la secuencia.
3. **Pooling**: colapsa las 23 salidas del encoder (una por año) a un solo
   vector de 128 dims. Dos variantes, seleccionables por `pooling`:
   - `"mean"`: promedio simple de las 23 posiciones.
   - `"query"`: atención de una cabeza contra un vector de consulta
     *aprendido* (`pool_query`), que le puede dar más peso a ciertos años
     que a otros en vez de pesarlos todos igual.
4. **Proyección al cuello de botella** (`to_latent`, `Linear(128 → d)`): de
   los 128 dims contextuales resultantes del pooling a `z`, el embedding
   final de `d` dimensiones.

### 2.2 Decoder

5. **Expansión desde `z`** (`from_latent`, `Linear(d → 128)`): el vector
   `z` se proyecta de vuelta a 128 dims y se **difunde (broadcast)** por
   igual a las 23 posiciones — arrancan siendo idénticas entre sí, solo
   diferenciadas después por el embedding de posición que se suma a
   continuación.
6. **`n_layer` bloques transformer bidireccionales más** (pesos propios,
   no compartidos con el encoder), que despliegan esa información repetida
   en algo que se parezca a 23 años distintos entre sí.
7. **Capa de salida** (`lm_head`, con los mismos pesos que
   `token_embedding` — weight tying, igual que en la v1): logits sobre las
   11 clases del vocabulario, para cada una de las 23 posiciones.

### 2.3 La decisión de diseño central: decoder no autorregresivo

El decoder **nunca ve la secuencia de entrada**, solo `z`. No hay forma de
que "copie" el año 2015 mirando el año 2015 real, porque esa información
no está disponible del otro lado del cuello de botella — todo lo que el
decoder sabe sobre las 23 posiciones tiene que venir comprimido en esos `d`
números. Esto es intencional: un decoder autorregresivo (que sí viera los
tokens ya generados) podría aprender a reconstruir usando contexto local y
aprender a ignorar `z` casi por completo — el modo de falla clásico de los
autoencoders de secuencias.

### 2.4 Pérdida y entrenamiento

Cross-entropy entre la secuencia reconstruida y la original, en las 23
posiciones a la vez (no una a la vez como en la v1), con pesos por clase
inversos a la frecuencia y `ignore_index` en `[UNK]` — reutiliza
`run_epoch()` de `model.py`, sin cambios respecto a la v1.

### 2.5 Tamaño del modelo

| Config | n_layer | pooling | embed_dim (`d`) | Parámetros |
|---|---|---|---|---|
| Barrido primario / secundario (baseline) | 4 | mean | 8 | 1,589,128 |
| **Elegida para el modelo final** | **2** | **query** | **8** | **798,216** |

(Para referencia: la v1 tiene 795,520 parámetros — la v2 con `n_layer=2`
queda en un orden de magnitud comparable, casi la mitad de una v2 con
`n_layer=4`.)

## 3. Estrategia de tuneo: dos barridos

Con un modelo de este tamaño, cada corrida completa (hasta 25 épocas, con
`early stopping` de `patience=5`) tarda entre ~30 minutos y ~2 horas en una
GPU modesta (medido en una GTX 1060 de 6GB: ~145s/época con `n_layer=2`,
~285s/época con `n_layer=4`, sobre las 400,460 secuencias de entrenamiento).
Correr una grilla completa de todos los hiperparámetros a la vez no era
viable en ese presupuesto, así que el tuneo se separó en dos etapas
secuenciales.

### 3.1 Barrido primario: dimensión del embedding (`d`)

Pregunta: ¿cuántas dimensiones hacen falta para comprimir la trayectoria
sin perder demasiada fidelidad de reconstrucción? El resto de los
hiperparámetros se deja fijo (`n_layer=4`, `pooling=mean`, `lr=1e-3`, con
pesos por clase) y se varía solo `d`:

```bash
python scripts/train_autoencoder.py --sweep dim --batch-size 2048 --out-dir models/sweep_dim
```

`d ∈ {4, 8, 12, 16, 32}` — los primeros cuatro son valores comprimidos
(menores a las 23 posiciones de la secuencia); `d=32` es un control
**no compresivo**, para poder medir cuánto se pierde específicamente por
imponer el cuello de botella, comparando contra un `d` que ya no comprime
nada.

**Resultado real** (`models/sweep_dim/summary.csv`, 5 corridas, ~6.8h de
GPU en total):

| d | macro F1 (reconstrucción) | épocas hasta converger |
|---|---:|---:|
| 4 | 0.8919 | 22 |
| 8 | 0.8939 | 13 |
| 12 | 0.8971 | 17 |
| 16 | 0.8921 | 11 |
| 32 (control) | 0.8983 | 23 |

La diferencia entre el `d` más chico (4) y el control no-compresivo (32)
es de solo 0.006 — con 4 números ya se reconstruye casi tan bien como con
32, señal de que la dimensionalidad intrínseca de estas trayectorias es
baja (la mayoría de los píxeles tienen pocas transiciones reales en 23
años). La curva no es perfectamente monótona (`d=16` da menos que `d=8` y
`d=12`); con una sola corrida por valor de `d` no se puede descartar que
sea ruido de entrenamiento antes que una diferencia real.

**Elegido**: `d=8` — el codo más claro de la curva (a solo 0.0044 del
techo con muchas menos dimensiones que 12/16/32).

### 3.2 Barrido secundario: hiperparámetros de entrenamiento, con `d` fijo

Con `d=8` fijo, se prueban otros aspectos del entrenamiento (no la
dimensión del embedding):

```bash
python scripts/train_autoencoder.py --sweep secondary --embed-dim 8 --batch-size 2048 --out-dir models/sweep_secondary
```

8 corridas (`SECONDARY_SWEEP` en `scripts/train_autoencoder.py`), variando
una o dos cosas por vez respecto de un `baseline`:

| Corrida | learning rate | capas (enc/dec) | pooling | pesos por clase |
|---|---|---|---|---|
| `baseline` | 1e-3 | 4 | mean | sí |
| `lr_bajo` | 3e-4 | 4 | mean | sí |
| `n_layer_2` | 1e-3 | 2 | mean | sí |
| `pooling_query` | 1e-3 | 4 | query | sí |
| `sin_pesos` | 1e-3 | 4 | mean | **no** |
| `lr_bajo_query` | 3e-4 | 4 | query | sí |
| `n_layer_2_query` | 1e-3 | 2 | query | sí |
| `lr_bajo_n_layer_2` | 3e-4 | 2 | mean | sí |

Cuatro ejes: **learning rate** (paso de ajuste de pesos), **profundidad**
(2 vs. 4 bloques transformer en encoder y decoder), **pooling** (media vs.
atención con consulta aprendida) y **pesos por clase** (si la loss
pondera más las clases raras como `B`/`Sp`, o no).

**Resultado real** (`models/sweep_secondary/summary.csv`, 8 corridas):

| Corrida | macro F1 | épocas | seg/época |
|---|---:|---:|---:|
| `lr_bajo` | 0.8989 | 25 (tope) | 285 |
| `n_layer_2_query` | 0.8985 | 25 (tope) | 145 |
| `lr_bajo_query` | 0.8985 | 25 (tope) | 285 |
| `baseline` | 0.8977 | 25 (tope) | 285 |
| `lr_bajo_n_layer_2` | 0.8977 | 24 | 145 |
| `n_layer_2` | 0.8966 | 16 | 145 |
| `sin_pesos` | 0.8959 | 12 | 285 |
| `pooling_query` | 0.8953 | 16 | 285 |

**Nota metodológica importante**: `best_val_macro_f1` es un máximo sobre
todas las épocas efectivamente entrenadas. Comparar el máximo de 25 épocas
contra el máximo de 12-16 épocas (las que sí activaron `early stopping`)
no es estrictamente una comparación justa — con más épocas hay más chances
de toparse con un pico alto por ruido, independientemente de si la
configuración es mejor. Mirando las curvas época a época de las dos
mejores corridas, ambas oscilan en una banda de ±0.002-0.003 sobre el
final sin tendencia ascendente clara, así que no parece que llegar al tope
de épocas sea señal de que "les faltaba entrenar" — pero sí hace que el
ranking fino entre las 5 corridas que llegaron a 25 épocas y las 3 que no
no sea del todo comparable. Con esa salvedad, las 4-5 corridas mejores
están esencialmente empatadas (spread de solo 0.0012 entre las 4 mejores).

**Elegida**: `n_layer_2_query` (`lr=1e-3`, `n_layer=2`, `pooling=query`,
con pesos por clase) — empata en la práctica con la mejor corrida
(diferencia de 0.0004, muy por debajo del ruido observado) pero con la
mitad de las capas y la mitad del tiempo por época; relevante porque el
paso siguiente es correr el encoder sobre potencialmente millones de
píxeles de Argentina para extraer embeddings.

## 4. Datos de entrenamiento y validación

### 4.1 Zonas usadas

![Zonas de entrenamiento (Chaco-Santiago + 7 nuevas) y de evaluación out-of-domain (7, held-out)](../imgs/v2_train_eval_zones.png)

*Mapa generado con `scripts/plot_v2_zones.py`, a partir de las coordenadas
reales por píxel (`data/lat_long_df_*.zip`). Las 7 zonas de evaluación
(azul) son exactamente las mismas que ya se usaban como benchmark
out-of-domain de la v1 (`notebooks/eval_ood_zones.ipynb`) — **nunca se
tocan para entrenar la v2**. Las 7 zonas nuevas de entrenamiento (verde)
están en las mismas ecorregiones que sus contrapartes de evaluación, pero
con bboxes distintos y geográficamente disjuntos, verificado
automáticamente en `scripts/build_eval_zones.py` (`forbidden_bboxes()`)
antes de generarlas — así no se repite el problema de fuga de datos que
tenía el test set original de la v1.*

Correspondencia entre zona de entrenamiento y su par de evaluación (misma
ecorregión, bbox distinto):

| Zona de entrenamiento (v2) | Zona de evaluación (benchmark, sin tocar) | Ecorregión |
|---|---|---|
| `puna_salta_catamarca` | `puna_noa` | Puna / árido de altura |
| `patagonia_santacruz` | `patagonia_estepa` | Estepa patagónica |
| `periurbano_gba` | `periurbano_cordoba` | Periurbano |
| `corrientes_humedal` | `ibera` | Humedal |
| `delta_oeste` | `delta_parana` | Humedal fluvial (Delta) |
| `pampa_deprimida` | `pampa_nucleo` | Agricultura extensiva |
| `yungas` | `misiones_selva` | Bosque húmedo subtropical |

### 4.2 Composición del dataset de entrenamiento

`load_training_sequences()` en `scripts/train_autoencoder.py` combina:

1. **Chaco-Santiago-frontera** (`data/id_seqs_text_2000_2022_chaco_santiago_frontier.zip`),
   la base heredada de la v1 (1,424,457 píxeles), **submuestreada en el
   momento de entrenar** con `subsample_constant_sequences(max_fraction=0.15)`:
   como la enorme mayoría de los píxeles de cualquier zona no cambia nunca
   en 23 años, sin este submuestreo el modelo aprendería poco más que
   reconstruir "23 años de lo mismo". Se recorta a que las secuencias
   constantes sean como mucho el 15% del subconjunto de Chaco → queda en
   **302,034** filas.
2. **Las 7 zonas nuevas**, ya submuestreadas de la misma forma al momento
   de construirlas (`scripts/build_eval_zones.py --zone-set train
   --max-constant-fraction 0.15`) → **98,426** filas en total.

Total combinado: **400,460 secuencias** (número real impreso al arrancar
cada corrida de entrenamiento).

*Nota sobre el mapa de la sección 4.1*: el polígono gris de Chaco-Santiago
muestra su huella **completa** (los 1.4M píxeles), porque el submuestreo
de esa zona se aplica de forma aleatoria sobre las filas en el momento de
entrenar (no es un recorte espacial, así que no se puede visualizar como
un subconjunto geográfico limpio). Las 7 zonas nuevas, en cambio, ya
vienen submuestreadas en el archivo — su patrón disperso en el mapa
(puntos salteados en vez de un bloque sólido) refleja exactamente qué
píxeles quedaron después de aplicar el mismo criterio.

### 4.3 Composición por clase de cada zona nueva de entrenamiento

Distribución de clases (sobre el total de tokens, 23 años × N píxeles) tras
el submuestreo — es decir, la mezcla real que ve el modelo:

| Zona | n (filas) | Clases dominantes |
|---|---:|---|
| `puna_salta_catamarca` | 31,518 | B=65.2%, Sp=25.0%, Sh=6.4%, F=1.5% |
| `patagonia_santacruz` | 20,821 | Sp=56.6%, G=24.5%, Sh=11.7%, B=5.1% |
| `periurbano_gba` | 4,003 | U=48.2%, A=29.9%, F=10.6%, Sh=6.9% |
| `corrientes_humedal` | 17,314 | F=35.3%, Wt=32.3%, Sh=15.3%, A=13.4% |
| `delta_oeste` | 3,561 | Wt=56.0%, F=18.9%, A=17.2%, Wa=4.3% |
| `pampa_deprimida` | 704 | A=58.2%, U=13.5%, Sh=10.2%, Wt=8.1% |
| `yungas` | 20,505 | F=40.6%, A=33.4%, Sh=24.8%, G=1.0% |

Para contraste, la composición de las 7 zonas de evaluación (sin
submuestrear, tal como se usan en `eval_ood_zones.ipynb`) está documentada
en el README (sección "Evaluación out-of-domain") y en
`scripts/build_eval_zones.py`.

### 4.4 Split de entrenamiento/validación

Dentro de las 400,460 secuencias combinadas, `train_one()`
(`scripts/train_autoencoder.py`) hace un `random_split` 90/10 (semilla
fija, `config.seed=42`) para separar entrenamiento de validación — la
validación usada para el `early stopping` y para elegir la mejor época
(`best_val_macro_f1`) es una muestra aleatoria *dentro* de las mismas
zonas de entrenamiento, no las zonas de evaluación out-of-domain. Las 7
zonas de evaluación quedan completamente afuera de este proceso y solo se
usan después, con el modelo ya entrenado y fijo, para medir generalización
real (igual que se hizo con la v1).

## 5. Configuración final del modelo

```bash
python scripts/train_autoencoder.py --embed-dim 8 --n-layer 2 --pooling query --lr 1e-3 --batch-size 2048
```

| Hiperparámetro | Valor |
|---|---|
| `embed_dim` (`d`) | 8 |
| `n_layer` (encoder y decoder) | 2 |
| `pooling` | query |
| `n_embd` | 128 |
| `n_head` | 4 |
| `lr` | 1e-3 |
| `dropout` | 0.1 |
| `batch_size` | 2048 |
| `epochs` (máximo) | 25 |
| `patience` | 5 |
| pesos por clase | sí (inverso a frecuencia) |
| Parámetros totales | 798,216 |
| Datos de entrenamiento | 400,460 secuencias (Chaco-Santiago submuestreado + 7 zonas nuevas) |

Guarda en `models/autoencoder_v2/` (`config.json` + `model.pt` +
`train_data.csv`), cargable con `land2vec.utils.load_config`/`load_model`
sin pasos adicionales.

## 6. Resultados de la corrida final

Modelo entrenado con la config de la sección 5
(`models/autoencoder_v2/config.json` + `model.pt` + `train_data.csv`).

- **Macro F1 de reconstrucción (mejor época)**: **0.8971** (época 17).
- **Accuracy de reconstrucción (mejor época)**: **0.9993**.
- **Épocas**: corrió **23** (0 a 22) y activó `early stopping` en la
  época 22 (`patience=5`: la 17 fue la mejor, y de la 18 a la 22 —5
  épocas— no volvió a superarla). No llegó al tope de 25.
- **Tiempo total de entrenamiento**: no quedó un log con el tiempo por
  época de esta corrida puntual (`train_data.csv` no lo registra); usando
  el ritmo medido para esta misma configuración en el barrido secundario
  (`n_layer_2_query`, ~145s/época), **23 épocas ≈ 56 minutos** — es una
  estimación por analogía, no una medición directa de esta corrida.
- **Parámetros**: 798,216 (coincide con `sweep_secondary/n_layer_2_query`,
  misma arquitectura).

![Curvas de loss, accuracy y macro F1 por época del modelo final](../imgs/v2_final_model_training_curve.png)

*Generado a partir de `models/autoencoder_v2/train_data.csv`. El panel
izquierdo (loss, escala log) y el derecho (accuracy/macro F1) muestran el
mismo patrón que ya se había visto en el barrido secundario: converge
rápido en las primeras ~10 épocas y después entra en una meseta ruidosa
(picos de loss en las épocas 7, 12 y 21, que se recuperan al toque) sin
una tendencia de mejora sostenida — consistente con la lectura de la
sección 3.2 de que el `early stopping` tardío no es señal de que "le
faltaba entrenar".*

### Comparación contra el resto del barrido

| Corrida | macro F1 | Diferencia vs. este modelo |
|---|---:|---:|
| `sweep_secondary/lr_bajo` (la mejor del barrido, 4 capas) | 0.8989 | +0.0018 |
| `sweep_secondary/n_layer_2_query` (misma config, corrida del barrido) | 0.8985 | +0.0014 |
| **`autoencoder_v2` (este modelo, corrida final)** | **0.8971** | — |
| `sweep_dim/d8` (mismo `d`, config del barrido primario: 4 capas, mean) | 0.8939 | -0.0032 |
| `sweep_dim/d32` (control no-compresivo) | 0.8983 | +0.0012 |

El modelo final queda 0.0014 por debajo de su propia corrida gemela en el
barrido secundario (misma config exacta: `d=8, n_layer=2, pooling=query`)
— una diferencia bien adentro del ruido de ±0.002-0.003 que se observó en
todas las curvas de esta etapa, no una señal de que algo haya cambiado
entre una corrida y la otra (incluso con la misma semilla, hay fuentes de
no-determinismo en GPU — orden de reducción en operaciones paralelas,
kernels no deterministas de cuDNN/CUDA — que pueden mover el resultado en
ese rango). Confirma lo esperado: reconstruye casi tan bien como el
control no-compresivo (`d=32`, -0.0012) con 8 dimensiones y menos de la
mitad de los parámetros de las corridas de 4 capas.

## 7. Resultados de la evaluación de embeddings (`eval_embeddings_v2.ipynb`)

Con el modelo final entrenado, se extrajeron embeddings sobre las 7 zonas
de evaluación out-of-domain (`scripts/extract_embeddings.py`) y se corrieron
los dos notebooks de evaluación: `notebooks/eval_embeddings_v2.ipynb`
(fidelidad de reconstrucción por zona, probing contra la v1, visualización
2D) y `notebooks/cluster_evaluation.ipynb` (clustering/tipología, §7.2 --
aislado en su propio notebook porque consume la salida de
`scripts/tune_clustering.py` en vez de re-generarla).

### 7.1 Reconstrucción por zona -- y un hallazgo metodológico

| Zona | Accuracy | Macro F1 |
|---|---:|---:|
| `puna_noa` | 0.9996 | 0.8985 |
| `patagonia_estepa` | 0.9999 | 0.8577 |
| `periurbano_cordoba` | 0.9995 | 0.8949 |
| `ibera` | 0.9998 | 0.8941 |
| `delta_parana` | 0.9998 | 0.8926 |
| `pampa_nucleo` | 0.9999 | 0.8976 |
| `misiones_selva` | 0.9999 | 0.6992 |

![Matrices de confusión de reconstrucción por zona](../imgs/v2_eval_confusion_zones.png)

A primera vista, `misiones_selva` parece la zona más difícil por mucho
margen. Mirando su matriz de confusión, sin embargo, la diagonal está en
~1.00 en **todas** las clases que efectivamente aparecen ahí -- la
reconstrucción es prácticamente perfecta también en esa zona. Lo que pasa
es que `compute_metrics` (fix de una sesión anterior, para que las
comparaciones entre corridas del barrido con distinta composición de
clases fueran justas) promedia el F1 sobre las **10 clases del vocabulario
completo**, y una clase sin ninguna muestra en el ground truth de una zona
cuenta como F1=0 en ese promedio (`zero_division=0`). `misiones_selva`
tiene 3 de 10 clases con soporte cero (`Sp`, `B`, `Nd`):
`(7 clases×1.0 + 3 clases×0) / 10 = 0.70`, exactamente el número
reportado. El mismo patrón explica el resto de las zonas (todas con 9/10
clases presentes, así que el descuento es de solo una clase ausente
-- macro F1 ≈0.86-0.90).

**Conclusión corregida**: la varianza de macro F1 entre zonas refleja casi
enteramente cuántas de las 10 clases del vocabulario aparecen en cada
zona, no una diferencia real en la calidad de reconstrucción -- el modelo
generaliza de forma consistentemente buena en las 7 zonas out-of-domain.
Esto no invalida las comparaciones *entre modelos* del barrido (sección 3),
que usan siempre el mismo `pooled_seqs` para todas las corridas y por lo
tanto no están afectadas por este problema -- solo invalida comparar el
macro F1 de una zona *contra otra*. El mismo artefacto contamina el
desglose constante/transición: la reconstrucción en secuencias con
transición tiene accuracy 0.986-0.998 (alta) pero macro F1 más bajo
(0.69-0.89) por el mismo motivo, no porque el modelo falle
específicamente en las transiciones.

### 7.2 Clustering / tipología de trayectorias

*Notebook: `notebooks/cluster_evaluation.ipynb` (aislado de
`eval_embeddings_v2.ipynb`). Consume `models/cluster_v2/` y
`data/clusters_*.zip`; no re-corre el barrido.*

Filtrado a las secuencias con al menos una transición (3.2% del pool,
107,362 de 3,344,976 -- el resto son trayectorias constantes, triviales de
agrupar y que antes de este filtro ahogaban cualquier tipología real, ver
nota metodológica de una sesión anterior). El `k=8` de la ronda anterior
era arbitrario (silhouette 0.43 se calculó *después*, como reporte, no
como criterio de selección). Esta ronda lo reemplaza por un barrido
explícito -- `scripts/tune_clustering.py` (`src/land2vec/cluster.py`) -- de
4 familias (KMeans, GaussianMixture, HDBSCAN, jerárquico/aglomerativo) x
preprocesado de `z` (crudo / estandarizado / L2) x `k` de 2 a 120, 380
configuraciones en total (198 jerárquico, 92 GMM, 66 KMeans, 24 HDBSCAN),
evaluadas con:

1. **Métricas internas**: silhouette (promediada sobre 5 semillas, no un
   solo número), Calinski-Harabasz, Davies-Bouldin.
2. **Estabilidad**: ARI entre reajustes sobre pares de submuestras al 80%
   (bootstrap) -- la ronda anterior no tenía ninguna defensa contra un `k`
   que solo se sostiene por el azar de una muestra puntual.
3. **Fidelidad del prototipo**: macro F1 (restringido a las clases con
   soporte>0 en cada cluster) de comparar cada secuencia real contra la
   trayectoria prototípica decodificada de su cluster -- un cluster sirve
   si su prototipo es una descripción honesta de sus miembros, no solo si
   está bien separado de los demás.
4. **Coherencia espacial**: fracción de vecinos geográficos (excluyendo
   pares donde alguno de los dos quedó como ruido de HDBSCAN) que comparten
   cluster, contra una línea de base de etiquetas permutadas -- sin excluir
   el ruido, dos vecinos "sin asignar" contarían como si compartieran un
   cluster real, inflando la métrica en las configs con más ruido por la
   sola contigüidad espacial de las zonas ambiguas sin tipificar, no por una
   tipología genuina.

El jerárquico no pudo ajustarse directamente sobre las 107k filas: la
matriz condensada de distancias de scipy es O(n²) (inviable), y restringir
la conectividad a un grafo k-NN -- la mitigación habitual -- resultó igual
de costosa en este entorno, porque las muchísimas trayectorias idénticas
de esta base fragmentan ese grafo en cientos de componentes que
`AgglomerativeClustering` no logra reconectar en tiempo razonable. Se
ajusta en cambio sobre una submuestra estratificada por zona
(`--hier-sample`, 5.000 filas -- 25.000 midió memoria estable en una
corrida aislada pero la acumuló sin liberarla a través de las ~6 reajustes
de estabilidad por config del barrido completo, hasta un OOM-kill real) y
se extiende al resto por centroide más cercano.

**Criterio de decisión** (`select_winner()` en `scripts/tune_clustering.py`):
entre las configuraciones con `stability_ari >= 0.75` y `noise_frac` por
debajo de un tope, la de mejor `prototype_fidelity`; desempate por
`silhouette_mean` y luego por menor `k`. El tope de `noise_frac` existe
porque `prototype_fidelity` solo se calcula sobre los miembros no-ruido de
cada cluster -- sin él, el criterio recompensa mecánicamente a HDBSCAN por
descartar como ruido los puntos difíciles (hasta 45% de las filas en configs
con `min_cluster_size` alto), no solo por tener una estructura de cluster
genuinamente mejor. El ranking usa las métricas del barrido (`n_boot=3`); la
ganadora de cada celda se re-ajusta después con `n_boot=10` y *ese* número de
estabilidad es el que va a `chosen*.json`.

**La matriz 3 x 2**: `--select` no elige una config sino seis -- 3 niveles de
granularidad {**fina** (sin tope de `k`, `noise_frac <= 0.5`) / **media**
(`k <= 40`, ruido `<= 0.25`) / **gruesa** (`k <= 20`, ruido `<= 0.25`)} x
2 familias {**HDBSCAN** / **no-HDBSCAN** (lo mejor de KMeans/GMM/jerárquico)}.
La segunda columna existe porque, al extender el barrido de `k` más allá de
20, las familias paramétricas dejaron de ser un también-corrió: **GMM `diag`
sobre `z` L2 sube `prototype_fidelity` de forma monótona con `k`** -- 0.61 a
`k=20`, 0.90 a `k=120`, con 0% de ruido -- y su `stability_ari` acompaña
(0.67 -> 0.92). KMeans y jerárquico se estancan en ~0.60 aun a `k=120` pese a
estabilidad altísima (0.94-0.95): es la varianza por componente de GMM `diag`
la que le permite aislar en clusters puros los bloques de trayectorias
idénticas, muy frecuentes en esta base. Queda abierto si eso es una tipología
genuina o una tabla de patrones frecuentes -- los prototipos decodificados
ayudan a discriminar. Una celda se omite si su ganadora coincide con otra ya
guardada.

| nivel / familia | algo (espacio) | `k` | silh. | `stab_ari` | `proto_fid` | ruido |
|---|---|---:|---:|---:|---:|---:|
| fina / HDBSCAN | HDBSCAN L2, `min_cluster_size=250` | 118 | 0.91 | 0.91 | **0.96** | 9.8% |
| fina / no-HDBSCAN | GMM `diag` L2, `k=120` | 120 | 0.74 | 0.92 | **0.90** | 0% |
| media / HDBSCAN | HDBSCAN `standard`, `min_cluster_size=1000` | 31 | 0.80 | 0.91 | 0.87 | 24% |
| media / no-HDBSCAN | GMM `diag` L2, `k=40` | 40 | 0.55 | 0.82 | 0.72 | 0% |
| gruesa / HDBSCAN | HDBSCAN L2, `min_cluster_size=2500, min_samples=25` | 17 | 0.58 | 0.72¹ | 0.74 | 20% |
| gruesa / no-HDBSCAN | GMM `full` L2, `k=18` | 18 | 0.47 | 0.78 | 0.58 | 0% |

¹ `stability_ari` 0.77 en el barrido (`n_boot=3`), **0.72 al re-verificar con
`n_boot=10`** -- por debajo del umbral de 0.75 del propio criterio
(`chosen_coarse.json` queda con `fallback=false` porque esa bandera refleja el
criterio con las métricas del barrido, no el reajuste). Es exactamente el
sobreajuste al ruido de una sola corrida que la estabilidad por bootstrap
está pensada para exponer; el tope de `noise_frac` ya bajó el ruido de esta
celda del 44,6% (config que ganaba sin el tope, `min_samples` por defecto) al
20,4%, pero no rescató la estabilidad. Se documenta igual como tipología
*exploratoria*.

**Lecturas**:

- **Fina / HDBSCAN (k=118)** es la config de mayor fidelidad de todo el
  barrido, pero 118 tipos no es una tipología legible para un mapa o una
  narrativa, y descarta ~10% de las filas como ruido. El espacio L2 desplazó
  por poco a `standard` (que ganaba antes de barrer `raw`/`l2` en todas las
  familias): mejora marginal en los cuatro números.
- **Fina / no-HDBSCAN (GMM `diag`, k=120)**: casi la misma fidelidad (0.90)
  sin descartar un solo punto. La comparación honesta contra el 0.96 de
  HDBSCAN tiene que descontar que HDBSCAN mide su fidelidad solo sobre el 90%
  que no marcó como ruido.
- **Media / HDBSCAN (k~31)**: el sweet spot -- `prototype_fidelity` 0.87 con
  `stability_ari` 0.91, en 31 tipos manejables y espacialmente coherentes
  (`spatial_coherence` 0.63, el más alto de los seis). GMM al mismo techo de
  `k` se queda en 0.72. **Es la tipología recomendada para análisis y mapas.**
- **Grueso (k~17-18)**: HDBSCAN llega a 0.74 de fidelidad pero con la
  estabilidad marginal ya señalada; GMM `full` k=18 es estable (0.78) pero de
  fidelidad baja (0.58). Ninguna es una tipología firme a esta granularidad.

![Clusters de trayectoria -- fina / HDBSCAN (k=118), por zona](../imgs/v2_eval_cluster_map_fina_hdbscan.png)

![Clusters de trayectoria -- fina / no-HDBSCAN (GMM diag, k=120), por zona](../imgs/v2_eval_cluster_map_fina_nohdbscan.png)

![Clusters de trayectoria -- media / HDBSCAN (k=31), por zona](../imgs/v2_eval_cluster_map_media_hdbscan.png)

Los seis niveles forman parches espacialmente coherentes dentro de cada zona
(frontera este-oeste nítida en `puna_noa`, banda diagonal en `ibera`) --
`spatial_coherence` 0.56-0.63 -- evidencia de que el embedding captura
tipologías de trayectoria geográficamente reales, no ruido. Las trayectorias
prototípicas (centroide de cada cluster en `z` crudo, decodificado) son
variadas e interpretables -- deforestación a humedal, oscilación en la
frontera agrícola, humedal a pastizal/estepa, etc. -- no clases constantes
triviales. El detalle completo del barrido (`models/cluster_v2/summary.csv`),
las curvas de las métricas vs. `k` (`imgs/v2_cluster_selection.png`) y el
dendrograma de la mejor corrida jerárquica
(`imgs/v2_cluster_dendrogram.png`) quedan versionados junto con las seis
configs elegidas
(`models/cluster_v2/chosen{,_medium,_coarse}{,_parametric}.json`) y sus
etiquetas
(`data/clusters_{dynamic,pooled_subsampled}{,_medium,_coarse}{,_parametric}.zip`
-- el archivo `pooled_subsampled` de cada celda lleva las secuencias
constantes submuestreadas al 15% en vez de excluidas, para que el mapa cubra
el pool completo y no solo el 3,2% con transición).

**Nota metodológica -- ruido honesto vs. mapa completo**: los dos archivos
de etiquetas de cada celda de la matriz no tratan el `-1` de la misma forma,
a propósito. `clusters_dynamic*.zip` preserva la etiqueta `-1` donde HDBSCAN
la asignó (0% para las familias no-HDBSCAN, que asignan todos los puntos) --
es la lectura "honesta" del clustering, la que efectivamente se evaluó con
`prototype_fidelity`/`stability_ari`. `clusters_pooled_subsampled*.zip`, en
cambio, asigna los puntos (incluidas las constantes submuestreadas al 15%)
por Voronoi a su centroide más cercano vía `assign_pool()` -- HDBSCAN no
tiene centroides ni clase de ruido propia sobre el pool completo -- para que
el mapa cubra el pool en vez de dejar el 96,8% del territorio sin colorear.
Para que ese segundo archivo no sobrestime la cobertura, la asignación por
Voronoi lleva un tope de distancia: los puntos que caen más lejos de todo
centroide que el percentil `--untyped-dist-pct` (default P95) de la distancia
al centroide entre los puntos *no-ruido* del pool dinámico salen como `-1`
("sin tipificar"), calibrado a la geometría real de cada config (el umbral
queda versionado en cada `chosen*.json` como `untyped_dist_threshold`).
Sigue siendo una lectura distinta de la de `clusters_dynamic*.zip` -- el `-1`
de acá es "lejos de todo centroide", no "HDBSCAN lo marcó como difícil" --
pero ya no asigna ciegamente cada punto a algún cluster: un punto
genuinamente atípico del pool aparece como no tipificado en las dos lecturas,
y las seis celdas quedan comparables entre sí.

### 7.3 Probing: `z` vs. secuencia cruda vs. estado oculto de la v1

| Tarea | `z` (v2, 8 dims) | one-hot crudo (253 dims) | hidden v1 pooled (128 dims) |
|---|---:|---:|---:|
| Clase dominante | 0.9998 | 0.9999 | 0.9995 |
| Hubo transición | **0.9665** | 0.9926 | 0.9889 |
| Ecorregión | 0.7918 | 0.7992 | 0.7953 |

En 2 de las 3 tareas, el embedding de 8 dimensiones empata en la práctica
con representaciones 16-30x más grandes -- señal fuerte de que la
compresión no pierde la información relevante para identificar la clase
dominante ni la ecorregión de origen (las tres representaciones tocan el
mismo techo de ~80% en ecorregión, probablemente un límite intrínseco de
la tarea -- hay trayectorias legítimamente ambiguas entre zonas -- más que
una limitación de `z`). La brecha real está en "hubo transición" (-2.6
puntos contra el one-hot crudo): es la señal más rara del dataset (3.2%
positivos) y la más fácil de atenuar en un cuello de botella optimizado
sobre todo para reconstruir la secuencia completa -- el costo de
compresión más honesto que aparece en todo el análisis.

### 7.4 Visualización 2D (PCA)

![PCA de z a 2D, coloreado por zona y por clase dominante](../imgs/v2_eval_pca.png)

2 componentes principales capturan 75.8% de la varianza de `z`. Coloreado
por clase dominante, el plano separa grupos con bastante claridad;
coloreado por zona se mezcla mucho más -- consistente con el probing
(clase dominante ~100% de accuracy, ecorregión ~80%).

### 7.5 Interpretación de las tipologías (batería estilo TraMineR)

*Módulos: `land2vec.typology` (descriptivo) + `land2vec.seqdist` (disimilitud
entre secuencias). Productor: `scripts/describe_clusters.py`. Notebook:
`notebooks/cluster_evaluation.ipynb` §5 (consume, no produce) +
`viz/typology/index.html` (navegador estático, **solo local por ahora** -- sus
datos traen trayectorias textuales; ver `viz/typology/README.md` y la nota al
final de esta sección).*

7.2 *valida* las seis particiones (silhouette, estabilidad, fidelidad del
prototipo); nada de eso *lee* los clusters. `scripts/tune_clustering.py --select`
deja 118+120+31+40+17+18 = **344 clusters**, cada uno descrito solo por su
prototipo decodificado, su `prototype_fidelity` y sus vecinos reales más
cercanos al centroide.

`land2vec.typology` agrega el paso siguiente al clustering en análisis de
secuencias: describir cada grupo en el espacio de las secuencias mismas, con la
batería de TraMineR reimplementada sobre las matrices `(n, 23)` de tokens de
cada cluster (todo O(n·T); la disimilitud entre pares va aparte, en
`land2vec.seqdist`, más abajo):

- **cronograma** (`seqdplot`): proporción de cada estado por año -- el gráfico
  ancla, el que muestra cuándo y hacia qué se dan las transiciones;
- **secuencia modal** (`seqmsplot`), libre de modelo -- se compara posición a
  posición con el prototipo decodificado; una discrepancia sistemática sería un
  hallazgo sobre el decoder, no sobre el clustering (`modal_vs_proto` en el CSV);
- **entropía transversal** (`seqHtplot`), **tiempo medio por estado**
  (`seqmtplot`), **tasas de transición** (`seqtrate`);
- **secuencias distintas más frecuentes** (`seqfplot`) y su cobertura -- responde
  directo la pregunta abierta de 7.2 (¿GMM `diag` a `k=120` es tipología o tabla
  de patrones frecuentes?): si la secuencia distinta más frecuente de un cluster
  ya lo cubre casi entero, es lo segundo;
- índices longitudinales por secuencia (nº de transiciones, índice de
  complejidad de Gabadinho, entropía longitudinal, duración de tramos);
- la suma de la diversidad de Gini por posición, que **es** la distancia de
  Hamming media entre pares del cluster -- compacidad en el espacio de secuencias
  sin salir del presupuesto O(n·T).

**Etiquetado automático** (`auto_label`): de la secuencia modal y su cronograma,
determinista, sale una etiqueta legible --
`F»A · monotónica · ~2008 · deforestación para agricultura`,
`A»Sh»A · oscilante · ~2011 · abandono agrícola con arbustización`,
`Wt · estable · humedal sin cambio neto`. La `forma` (estable / monotónica /
oscilante / múltiple) sale del DSS de la modal; el año, del salto de mayor
variación total del cronograma; la glosa, de un diccionario a mano de los pares
(estado inicial, estado final) del área de estudio (`land2vec.typology.GLOSSES`).
Convierte los 344 ids numéricos en leyendas de mapa.

**Comparación entre corridas**: los seis `data/clusters_dynamic*.zip` tienen las
mismas 107.362 filas en el mismo orden, así que el acuerdo entre particiones
(ARI, NMI) es columna contra columna. Se reporta como **medida primaria** el ARI
sobre las filas que *ninguna* de las dos corridas dejó como ruido, con la
cobertura de esa intersección al lado (las fracciones de ruido van de 0% a 24%
entre corridas -- el ARI no se lee sin ella); y como secundaria, el ARI tratando
`-1` como una etiqueta más. Es la misma doble lectura del `-1` que la Nota
metodológica de 7.2. Los diagramas aluviales (`plot_alluvial`) muestran cómo se
parten los clusters al bajar la granularidad y dónde HDBSCAN y GMM cortan
distinto; `nesting_table` da, por cluster grueso, cuántos clusters medios lo
componen y su pureza.

**Disimilitud entre secuencias** (`land2vec.seqdist`): el pool dinámico tiene
solo **1.128 trayectorias distintas** de 107.362, así que colapsado a ese
conjunto con pesos (idéntico a operar sobre las 107k filas), toda la maquinaria
de disimilitud de TraMineR es una matriz 1128x1128 y unos segundos de cómputo.
`build_distance` ofrece tres métricas sobre secuencias de largo fijo 23 (años
calendario alineados): `HAM` (Hamming), `DHD` (Hamming dinámico, costo de
sustitución por frecuencia posicional -- simplificación del DHD de Lesnard), y
`OM` (Optimal Matching, alineación global con indels; por defecto costos TRATE
`c(i,j) = 2 - p(i|j) - p(j|i)` e `indel = max(c)/2`). El default de
`describe_clusters.py` es `OM`; los indels reintroducen desalineación temporal,
informativa acá, así que `DHD` es una alternativa razonable
(`--seqdist-method dhd`). Con esa matriz:

- **pseudo-R² de discrepancia** (`pseudo_r2`, equivalente a `TraMineR::dissassoc`):
  `1 - SS_within/SS_total` con la suma de cuadrados de discrepancia ponderada de
  Studer/Ritschard -- qué fracción de la variación real de trayectoria explica
  cada partición, medida **fuera del espacio `z`** que la produjo. Es la
  comparación en igualdad de condiciones entre las 6 corridas
  (`models/cluster_v2/typology_seqdist.csv`, gráfico
  `imgs/v2_typology_seqdist.png`).
- **ASW en espacio de secuencias** (`asw`): silueta ponderada por conteo usando
  la disimilitud entre trayectorias, no la distancia euclídea en `z` --
  complementa el silhouette de 7.2, que se mide en `z`.
- **secuencias representativas** (`representative_sequences`, criterio de
  densidad de vecindad de `TraMineR::seqrplot`): por cluster, un conjunto
  codicioso de trayectorias reales no redundantes dentro de un radio del 10% de
  la distancia máxima, con su cobertura y la distancia media al representante más
  cercano. Van al CSV por cluster y al navegador.

El navegador (`viz/typology/index.html`) muestra además, por cluster, las
**trayectorias distintas más frecuentes que acumulan ≥80%** de sus miembros --
la lectura directa de si el cluster es un patrón o una nube.

**Qué se versiona y qué no.** Los `imgs/v2_typology_*.png` (atlas de cronogramas,
heatmaps ARI/NMI, aluviales, scatter pseudo-R²) sí -- son agregados, sin
trayectorias verbatim. **No** se versionan (gitignoreados, `models/cluster_v2/typology*`
y `viz/typology/typology_browser.json`): los CSV por cluster y el payload del
navegador **traen trayectorias de uso del suelo textuales con sus conteos** --
p. ej. el JSON incluye ~566 de las 1.128 trayectorias dinámicas distintas,
verbatim, con frecuencia por cluster y zona dominante (sin coordenadas). Eso es
más que "solo agregados", así que por ahora se genera y se mira en local
(`python -m http.server -d viz/typology`); publicarlo requeriría recortar
`top_seqs_80`/`representantes`/`vecinos` a prototipos, o cifrar el payload.

## 8. Próximos pasos

1. Calcular una versión de macro F1 restringida a las clases con soporte
   real en cada subconjunto (por zona, o constante/transición) -- ver
   7.1 -- para tener una comparación de fidelidad de reconstrucción entre
   zonas que no esté sesgada por cuántas clases del vocabulario aparecen
   en cada una. (`land2vec.cluster.prototype_fidelity` ya implementa esta
   misma corrección de soporte para el clustering -- reusar el patrón.)
2. La celda gruesa / HDBSCAN (k=17) quedó documentada con una estabilidad
   por debajo del umbral propio del criterio (`stability_ari` 0.72 vs. 0.75
   al re-verificar con `n_boot=10`, ver 7.2) -- el tope de `noise_frac` ya
   mejoró esto (era 0.69 con la config anterior, 44,6% de ruido) pero no lo
   resolvió. Si se va a usar una tipología a esta granularidad, la celda
   gruesa / no-HDBSCAN (GMM `full` k=18) es estable (0.78) pero de fidelidad
   baja (0.58); vale la pena re-barrer `min_cluster_size` alrededor de 2500
   con más bootstraps desde el inicio (no solo al reajustar la ganadora)
   para ver si hay una config HDBSCAN vecina genuinamente estable con `k`
   igual de chico y ruido igual de bajo. Para la mayoría de los usos, el
   nivel medio (k~31) es preferible.
3. Prototipos por cluster desagregados por zona (hoy el centroide se
   decodifica una sola vez de forma global) -- útil sobre todo para la
   tipología fina (k=118), donde un mismo cluster puede tener composición
   de zona mixta.
