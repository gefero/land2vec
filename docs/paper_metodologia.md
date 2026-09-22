# Metodología

> Borrador para el apartado metodológico del paper. Todos los valores numéricos
> provienen del código y de los artefactos versionados del repositorio
> (`models/autoencoder_v2/config.json`, `models/sweep_*/summary.csv`,
> `models/cluster_v2/{summary.csv,chosen*.json,typology_seqdist.csv}`).
> Los puntos marcados con ⚠ requieren una decisión o un dato del autor antes de
> enviar.

---

## 1. Datos

### 1.1 Fuente

El insumo primario es el producto de cobertura del suelo **ESA CCI Land Cover**
(*ESACCI-LC-L4-LCCS-Map-300m-P1Y*, versión 2.0.7cds, UCLouvain / European Space
Agency Climate Change Initiative), distribuido como mapas anuales globales de
300 m de resolución espacial (0,002778° en latitud y longitud) derivados de
series temporales MERIS FR/RR y SPOT-VGT. Se utilizó la variable `lccs_class`
para los **23 años consecutivos del período 2000–2022**.

El producto global fue recortado a una ventana sudamericana que cubre la
totalidad del territorio continental argentino —latitud [−55,0°, −20,0°],
longitud [−75,0°, −53,0°]— y almacenado como un único archivo netCDF
tridimensional `(tiempo, latitud, longitud)`.

### 1.2 Construcción de las trayectorias

La unidad de análisis es el **píxel de 300 m**, tratado como una parcela cuya
historia se representa como una **secuencia de estados de longitud fija
T = 23** (un estado por año calendario, alineados entre sí). Formalmente, cada
parcela *i* queda descrita por

$$x_i = (x_{i,2000}, x_{i,2001}, \dots, x_{i,2022}) \in \mathcal{V}^{23},$$

donde $\mathcal{V}$ es un alfabeto de estados de cobertura/uso del suelo. El
procedimiento de extracción (implementado en `land2vec.extract`) recorta el
netCDF a un *bounding box*, aplana la grilla resultante a un vector de píxeles
con identificador posicional, y traduce los códigos numéricos de `lccs_class` a
los símbolos del alfabeto, conservando para cada parcela sus coordenadas
geográficas en un archivo paralelo.

El alfabeto consta de **10 estados**, obtenidos por agregación de la leyenda
LCCS original del producto —22 clases— según la correspondencia siguiente, más
un símbolo de relleno:

| Símbolo | Código | Estado | Clases LCCS agregadas |
|---|:---:|---|---|
| `Nd` | 0 | Sin datos | *No data* |
| `A` | 1 | Agricultura | 10 *Cropland, rainfed*; 20 *Cropland, irrigated or post-flooding*; 30 *Mosaic cropland (>50 %) / natural vegetation (<50 %)*; 40 *Mosaic natural vegetation (>50 %) / cropland (<50 %)* |
| `F` | 2 | Forestal | 50, 60, 70, 80, 90 *Tree cover* (latifoliado/aciculifoliado, perenne/caducifolio, mixto; cobertura > 15 %); 100 *Mosaic tree and shrub (>50 %) / herbaceous cover (<50 %)*; 110 *Mosaic herbaceous cover (>50 %) / tree and shrub (<50 %)* |
| `G` | 3 | Pastizal | 130 *Grassland* |
| `Wt` | 4 | Humedal | 180 *Shrub or herbaceous cover, flooded, fresh/saline/brackish water* |
| `U` | 5 | Urbano | 190 *Urban areas* |
| `Sh` | 6 | Arbustal | 120 *Shrubland* |
| `Sp` | 7 | Vegetación esparsa | 140 *Lichens and mosses*; 150 *Sparse vegetation (<15 %)*; 160 *Tree cover, flooded, fresh or brackish water*; 170 *Tree cover, flooded, saline water* |
| `B` | 8 | Suelo desnudo | 200 *Bare areas* |
| `Wa` | 9 | Agua | 210 *Water bodies*; 220 *Permanent snow and ice* |
| `[UNK]` | — | Relleno | Excluido de la función de pérdida y de todas las métricas |

Tres decisiones de esta agregación condicionan la interpretación de los
procesos identificados más adelante y conviene explicitarlas. **Primero**, las
coberturas arbóreas inundadas (160 y 170) se agregan a `Sp` —*vegetación
esparsa*— y no a `F` ni a `Wt`: una transición desde bosque inundado hacia otra
cobertura no se contabiliza, por lo tanto, como pérdida forestal, y la clase
`Wt` queda restringida a la cobertura arbustiva o herbácea anegada.
**Segundo**, la clase `F` es deliberadamente inclusiva: incorpora los mosaicos
de árboles y arbustos (100) y también los mosaicos de predominio herbáceo con
árboles y arbustos subordinados (110), mientras que arbustal (120) y pastizal
(130) puros constituyen clases propias; en consecuencia, las transiciones
`F → Sh` y `F → G` pueden corresponder tanto a una pérdida de cobertura arbórea
como al pasaje de un mosaico a una cobertura homogénea. **Tercero**, la nieve y
el hielo permanentes (220) se agregan a `Wa`, lo que resulta pertinente para las
zonas de altura y australes del área de estudio.

El producto se obtuvo agregado al primer nivel de la leyenda LCCS —sin los
códigos de subnivel (11 y 12 bajo 10, 61 y 62 bajo 60, 151 a 153 bajo 150,
etcétera)—, de modo que la correspondencia de la tabla anterior es exhaustiva.
La agregación a los diez estados es, a su vez, previa a la construcción del
netCDF empleado: el archivo almacena directamente los códigos finales 0–9, y la
rutina de extracción sólo los traduce a los símbolos del alfabeto.

### 1.3 Diseño espacial: zonas de ajuste y zonas de evaluación

Para poder evaluar la capacidad de generalización del modelo fuera de su región
de ajuste, se definieron dos conjuntos de zonas de estudio **geográficamente
disjuntos**, emparejados uno a uno por ecorregión. Cada zona es un *bounding
box* rectangular; la disyunción entre todos los pares de cajas (y respecto del
área de entrenamiento original y del conjunto de prueba heredado) se verifica
programáticamente antes de generar los datos, y el procedimiento aborta ante
cualquier superposición (`scripts/build_eval_zones.py`).

| Ecorregión | Zona de ajuste (*train*) | *bbox* (minx, miny, maxx, maxy) | Zona de evaluación (*held-out*) | *bbox* |
|---|---|---|---|---|
| Frontera agrícola chaqueña | `chaco_santiago_frontier` | (−63,45, −28,13, −59,37, −25,43) | — | — |
| Puna / árido de altura | `puna_salta_catamarca` | (−68,5, −26,5, −66,5, −24,5) | `puna_noa` | (−67,0, −23,5, −65,0, −22,0) |
| Estepa patagónica | `patagonia_santacruz` | (−71,0, −49,0, −67,0, −46,0) | `patagonia_estepa` | (−70,0, −43,0, −66,0, −40,0) |
| Periurbano | `periurbano_gba` | (−58,8, −34,9, −58,3, −34,4) | `periurbano_cordoba` | (−64,4, −31,6, −64,0, −31,2) |
| Humedal | `corrientes_humedal` | (−58,9, −27,4, −58,1, −26,3) | `ibera` | (−58,0, −29,0, −56,5, −27,5) |
| Humedal fluvial (Delta) | `delta_oeste` | (−59,95, −33,8, −59,65, −32,6) | `delta_parana` | (−59,6, −33,8, −58,6, −32,4) |
| Agricultura extensiva | `pampa_deprimida` | (−60,0, −37,5, −58,0, −36,0) | `pampa_nucleo` | (−62,0, −35,0, −60,0, −33,0) |
| Bosque húmedo subtropical | `yungas` | (−64,8, −25,5, −64,0, −24,0) | `misiones_selva` | (−55,5, −27,5, −54,0, −25,5) |

El emparejamiento por ecorregión es deliberado: garantiza que las modalidades de
uso del suelo presentes en la evaluación estén *representadas* en el ajuste
—evitando penalizar al modelo por clases que nunca vio— sin que ninguna
observación de evaluación participe del entrenamiento. Las siete zonas de
evaluación **no intervienen en ninguna etapa del ajuste**: ni en el
entrenamiento del autoencoder, ni en la selección de hiperparámetros, ni en la
elección del modelo final.

### 1.4 Submuestreo de trayectorias constantes

En cualquier ventana territorial de este producto, la abrumadora mayoría de los
píxeles no registra ningún cambio de estado en 23 años (bosque continuo, agua
permanente, cultivo estable). Entrenado sobre esa distribución, un autoencoder
aprende poco más que reproducir "23 años del mismo estado". Para evitarlo se
aplicó un **submuestreo de las secuencias constantes**
(`extract.subsample_constant_sequences`): las trayectorias con al menos una
transición se conservan íntegramente y las constantes se submuestrean al azar
—semilla fija— hasta representar como máximo una fracción $\alpha$ del conjunto
resultante. Con $\alpha = 0{,}15$, el número de secuencias constantes retenidas
es $n_{\text{cte}} = \lfloor \alpha\, n_{\text{var}} / (1-\alpha) \rfloor$.

Las trayectorias constantes se submuestrean pero no se eliminan: la estabilidad
es una forma legítima de dinámica territorial y debe seguir representada en el
espacio latente.

### 1.5 Conjunto de entrenamiento resultante

| Componente | Filas |
|---|---:|
| `chaco_santiago_frontier` (1.424.457 píxeles, submuestreado a $\alpha=0{,}15$) | 302.034 |
| 7 zonas nuevas de ajuste (submuestreadas a $\alpha=0{,}15$) | 98.426 |
| **Total** | **400.460** |

Composición por clase de las siete zonas nuevas, computada sobre el total de
tokens (23 años × N píxeles) *posterior* al submuestreo, es decir la mezcla que
efectivamente observa el modelo:

| Zona | n | Clases dominantes |
|---|---:|---|
| `puna_salta_catamarca` | 31.518 | B=65,2 %; Sp=25,0 %; Sh=6,4 %; F=1,5 % |
| `patagonia_santacruz` | 20.821 | Sp=56,6 %; G=24,5 %; Sh=11,7 %; B=5,1 % |
| `corrientes_humedal` | 17.314 | F=35,3 %; Wt=32,3 %; Sh=15,3 %; A=13,4 % |
| `yungas` | 20.505 | F=40,6 %; A=33,4 %; Sh=24,8 %; G=1,0 % |
| `periurbano_gba` | 4.003 | U=48,2 %; A=29,9 %; F=10,6 %; Sh=6,9 % |
| `delta_oeste` | 3.561 | Wt=56,0 %; F=18,9 %; A=17,2 %; Wa=4,3 % |
| `pampa_deprimida` | 704 | A=58,2 %; U=13,5 %; Sh=10,2 %; Wt=8,1 % |

---

## 2. El modelo: autoencoder secuencial de trayectorias

### 2.1 Planteo

El objetivo es aprender una representación vectorial densa y de baja
dimensionalidad de la trayectoria completa de una parcela, análoga en espíritu a
los *embeddings* distribucionales del procesamiento de lenguaje natural pero
definida sobre la secuencia entera y no sobre estados individuales. Se emplea un
**autoencoder secuencial con cuello de botella explícito**: un codificador
$f_\theta:\mathcal{V}^{23}\to\mathbb{R}^d$ que comprime la trayectoria en un
vector $z$ de $d$ dimensiones, y un decodificador $g_\phi:\mathbb{R}^{d}\to
\mathbb{R}^{23\times|\mathcal{V}|}$ que reconstruye la secuencia completa a
partir *exclusivamente* de ese vector.

$$z_i = f_\theta(x_i), \qquad \hat{x}_i = \arg\max g_\phi(z_i).$$

La representación de interés es $z$; la reconstrucción es solamente la tarea
auxiliar que la induce.

### 2.2 Arquitectura

Ambos módulos son pilas de bloques *transformer* con **auto-atención
bidireccional** (sin enmascaramiento causal): a diferencia de un modelo de
lenguaje autorregresivo, aquí no hay nada que predecir "hacia adelante", sino
que se busca resumir la secuencia completa, de modo que toda posición debe poder
atender a todas las demás.

**Codificador.** (i) Cada uno de los 23 tokens se proyecta a un espacio de
$n_{\text{embd}}=128$ dimensiones mediante una tabla de *embeddings*
$|\mathcal{V}|\times 128$, a la que se suma un *embedding* posicional aprendido
($23\times 128$). (ii) $L$ bloques transformer con normalización previa
(*pre-LayerNorm*), auto-atención multi-cabeza de 4 cabezas y red *feed-forward*
de expansión 4× con activación GELU, ambos con conexión residual y *dropout*
0,1. (iii) Un operador de **agregación temporal** (*pooling*) colapsa las 23
representaciones contextuales en un único vector de 128 dimensiones; se
evaluaron dos variantes:

- `mean`: promedio no ponderado de las 23 posiciones;
- `query`: atención de una cabeza contra un vector de consulta aprendido
  $q\in\mathbb{R}^{128}$, que permite ponderar diferencialmente los años,
  $\;\text{pool}(h) = \sum_t \text{softmax}_t\!\left(h_t^\top q/\sqrt{128}\right) h_t$.

(iv) Una proyección lineal $\mathbb{R}^{128}\to\mathbb{R}^{d}$ produce el
*embedding* final $z$.

**Decodificador.** (v) Una proyección lineal $\mathbb{R}^{d}\to\mathbb{R}^{128}$
expande $z$, que se **difunde idénticamente a las 23 posiciones**; las
posiciones se diferencian únicamente por el *embedding* posicional que se les
suma a continuación. (vi) $L$ bloques transformer bidireccionales adicionales
con parámetros propios (no compartidos con el codificador). (vii) Una capa de
salida lineal produce *logits* sobre las $|\mathcal{V}|$ clases en cada una de
las 23 posiciones; sus pesos están **atados** (*weight tying*) a la tabla de
*embeddings* de entrada.

### 2.3 Decisión de diseño: decodificador no autorregresivo

El decodificador **nunca observa la secuencia de entrada**: recibe únicamente
$z$. Esta restricción es deliberada y central al diseño. Un decodificador
autorregresivo —que condicionara cada posición sobre los tokens ya generados—
podría reconstruir la secuencia apoyándose en el contexto local e ignorar el
cuello de botella casi por completo, el modo de falla clásico de los
autoencoders de secuencias. Al privarlo de esa información, toda la señal de
reconstrucción está forzada a atravesar los $d$ números de $z$, lo que garantiza
que el *embedding* sea una descripción suficiente de la trayectoria.

### 2.4 Función de pérdida

Entropía cruzada categórica evaluada **simultáneamente en las 23 posiciones**,
con ponderación por clase inversa a la frecuencia y exclusión del token de
relleno:

$$\mathcal{L}(\theta,\phi) = -\frac{1}{|B|}\sum_{i \in B} \sum_{t=1}^{23}
w_{x_{i,t}} \, \log p_\phi\!\left(x_{i,t} \mid z_i, t\right),
\qquad w_c \propto 1/n_c,$$

donde $n_c$ es la frecuencia del estado $c$ en el corpus de entrenamiento y los
pesos se normalizan a media unitaria entre las clases presentes; $w_{[\text{UNK}]}=0$.
La ponderación busca que estados poco frecuentes pero sustantivamente relevantes
(suelo desnudo, vegetación esparsa, urbano) no queden absorbidos por las clases
mayoritarias.

---

## 3. Entrenamiento y selección de hiperparámetros

### 3.1 Protocolo de ajuste

Las 400.460 secuencias de las zonas de ajuste se particionaron aleatoriamente en
**90 % entrenamiento / 10 % validación** (semilla fija = 42). El conjunto de
validación es, por lo tanto, una muestra aleatoria *dentro de las mismas zonas
de entrenamiento*; su función es exclusivamente operativa —detención temprana y
selección de la mejor época— y no constituye una prueba de generalización. Esa
función la cumplen las siete zonas de evaluación, espacialmente disjuntas y
reservadas por completo (§ 3.5).

| Elemento | Valor |
|---|---|
| Optimizador | AdamW ($\beta$ por defecto), *weight decay* = 1×10⁻² |
| Tasa de aprendizaje | 1×10⁻³ (constante) |
| Tamaño de lote | 2.048 |
| Precisión | mixta (FP16 con *gradient scaling*) |
| Épocas máximas | 25 |
| Detención temprana | paciencia = 5 épocas sin mejora del macro F1 de validación |
| Selección de pesos | se restauran los de la época con mejor macro F1 de validación |
| Semilla | 42 (PyTorch y partición) |
| *Hardware* | GPU NVIDIA GTX 1060 6 GB |

Cada época demandó aproximadamente 145 s con $L=2$ bloques y 285 s con $L=4$
sobre las 400.460 secuencias.

### 3.2 Estrategia de tuneo: dos barridos secuenciales

Un barrido factorial completo sobre todos los hiperparámetros era inviable
dentro del presupuesto de cómputo disponible (cada corrida insume entre ~30
minutos y ~2 horas). La búsqueda se organizó, en cambio, en **dos etapas
secuenciales**, la primera sobre el parámetro de mayor interés sustantivo —la
dimensión del cuello de botella— y la segunda sobre los hiperparámetros de
optimización con la dimensión ya fijada.

**Barrido primario — dimensión del *embedding*.** Con el resto de la
configuración fija ($L=4$, *pooling* `mean`, lr = 1×10⁻³, pérdida ponderada) se
evaluó $d \in \{4, 8, 12, 16, 32\}$. El valor $d=32$ opera como **control no
compresivo**: al superar la longitud de la secuencia (23), permite cuantificar
cuánta fidelidad se pierde específicamente por imponer el cuello de botella.

| $d$ | Macro F1 de reconstrucción (validación) | Épocas |
|---:|---:|---:|
| 4 | 0,8919 | 22 |
| 8 | 0,8939 | 13 |
| 12 | 0,8971 | 17 |
| 16 | 0,8921 | 11 |
| 32 (control) | 0,8983 | 23 |

La diferencia entre la dimensión más pequeña y el control no compresivo es de
apenas 0,006: con cuatro números se reconstruye casi tan bien como con
treinta y dos, lo que sugiere que la **dimensionalidad intrínseca** de estas
trayectorias es baja. Se adoptó $d = 8$ como codo de la curva (a 0,0044 del
techo). La curva no es estrictamente monótona ($d=16$ rinde por debajo de $d=8$
y $d=12$); con una sola corrida por valor no puede descartarse que se trate de
ruido de entrenamiento antes que de una diferencia real.

**Barrido secundario — hiperparámetros de optimización.** Con $d=8$ fijo se
corrieron ocho configuraciones siguiendo un esquema de **variación de un factor
por vez** (OFAT) respecto de una línea de base, sobre cuatro ejes: tasa de
aprendizaje, profundidad, operador de agregación temporal y ponderación por
clase.

| Corrida | lr | $L$ | *pooling* | Pesos por clase | Macro F1 | Épocas | s/época |
|---|---:|---:|---|---|---:|---:|---:|
| `lr_bajo` | 3×10⁻⁴ | 4 | mean | sí | 0,8989 | 25 (tope) | 285 |
| `n_layer_2_query` | 1×10⁻³ | 2 | query | sí | 0,8985 | 25 (tope) | 145 |
| `lr_bajo_query` | 3×10⁻⁴ | 4 | query | sí | 0,8985 | 25 (tope) | 285 |
| `baseline` | 1×10⁻³ | 4 | mean | sí | 0,8977 | 25 (tope) | 285 |
| `lr_bajo_n_layer_2` | 3×10⁻⁴ | 2 | mean | sí | 0,8977 | 24 | 145 |
| `n_layer_2` | 1×10⁻³ | 2 | mean | sí | 0,8966 | 16 | 145 |
| `sin_pesos` | 1×10⁻³ | 4 | mean | **no** | 0,8959 | 12 | 285 |
| `pooling_query` | 1×10⁻³ | 4 | query | sí | 0,8953 | 16 | 285 |

Las cuatro o cinco mejores configuraciones están empatadas en la práctica
(amplitud de 0,0012 entre las cuatro primeras), muy por debajo de la oscilación
época a época observada (±0,002–0,003). Se seleccionó `n_layer_2_query` —a
0,0004 de la mejor corrida— por **paridad de desempeño a la mitad del costo
computacional** (la mitad de capas y la mitad del tiempo por época), criterio
relevante dado que la etapa siguiente requiere aplicar el codificador sobre
millones de píxeles.

*Nota metodológica.* La métrica reportada, `best_val_macro_f1`, es un máximo
sobre las épocas efectivamente entrenadas. Comparar el máximo de 25 épocas
contra el de 12–16 (las que activaron la detención temprana) no es una
comparación estrictamente equitativa: a mayor número de épocas, mayor
probabilidad de encontrar un pico alto por azar. La inspección de las curvas
época a época de las dos mejores corridas no muestra tendencia ascendente al
llegar al tope, sino oscilación en una banda estrecha, por lo que la
detención tardía no parece indicar entrenamiento insuficiente; aun así, el
ordenamiento fino entre corridas con distinto número de épocas debe leerse con
cautela.

### 3.3 Configuración final

| Hiperparámetro | Valor |
|---|---|
| Dimensión del *embedding* ($d$) | 8 |
| Bloques por módulo ($L$) | 2 (codificador) + 2 (decodificador) |
| Dimensión del modelo ($n_{\text{embd}}$) | 128 |
| Cabezas de atención | 4 |
| Agregación temporal | `query` (atención con consulta aprendida) |
| *Dropout* | 0,1 |
| Longitud de secuencia | 23 |
| Tamaño de vocabulario | 11 |
| Ponderación por clase | sí (inversa a la frecuencia) |
| **Parámetros entrenables** | **798.216** |

### 3.4 Desempeño de la corrida final

El modelo final alcanzó un **macro F1 de reconstrucción de 0,8971** y una
**exactitud de 0,9993** en validación (mejor época: 17). La detención temprana
se activó en la época 22, sin alcanzar el tope de 25. La configuración
reconstruye prácticamente tan bien como el control no compresivo ($d=32$,
−0,0012) con ocho dimensiones y menos de la mitad de los parámetros de las
variantes de cuatro capas.

### 3.5 Evaluación

La evaluación se organizó en cuatro frentes, todos ejecutados sobre el modelo
final ya fijo y sobre las **siete zonas de evaluación no utilizadas en ninguna
etapa del ajuste**.

**(a) Fidelidad de reconstrucción fuera de dominio.** Exactitud y macro F1 por
zona, sobre la totalidad de los píxeles de cada una.

**(b) Corrección por soporte de clase.** El macro F1 se computa sobre el
vocabulario completo de 10 clases, con `zero_division=0`, a fin de que las
comparaciones *entre corridas* con distinta composición de clases sean
equitativas. Ello implica que una clase sin ninguna observación en una zona
contribuye con F1 = 0 al promedio de esa zona, de modo que **el macro F1 no es
directamente comparable entre zonas**: su varianza refleja casi enteramente
cuántas de las 10 clases están presentes en cada una, y no diferencias reales de
calidad de reconstrucción. El diagnóstico se apoya, en consecuencia, en las
matrices de confusión por zona (diagonal por clase efectivamente presente) además
del agregado.

**(c) *Probing* del contenido informativo del *embedding*.** Se entrenaron
clasificadores sobre tres representaciones alternativas de la misma trayectoria
—el *embedding* $z$ de 8 dimensiones; la codificación *one-hot* cruda de
23 × 11 = 253 dimensiones; y el estado oculto promediado (128 dimensiones) de un
modelo autorregresivo previo entrenado sobre los mismos datos— para tres tareas:
clase dominante de la trayectoria, presencia de al menos una transición, y
ecorregión de origen. La comparación permite cuantificar qué información
sobrevive a la compresión.

**(d) Estructura del espacio latente.** Análisis de componentes principales
sobre $z$, con proyección bidimensional coloreada por zona y por clase
dominante.

---

## 4. Extracción de los *embeddings* y construcción del *pool*

### 4.1 Extracción

Con el modelo final en modo inferencia, se aplicó el codificador
$z = f_\theta(x)$ —sin el decodificador— a la totalidad de los píxeles de las
siete zonas de evaluación, preservando el orden posicional del archivo de
secuencias para mantener la alineación con las coordenadas geográficas. La
salida es una matriz de $N \times 8$ por zona. El emparejamiento entre
secuencias, *embeddings* y coordenadas se valida explícitamente por
identificador y por longitud antes de cualquier análisis posterior.

### 4.2 Conjuntos de análisis

Se definieron dos conjuntos, con propósitos distintos:

- ***Pool* dinámico** (*n* = 107.362): las trayectorias con **al menos una
  transición** de estado en los 23 años, el 3,2 % de los 3.344.976 píxeles de
  las siete zonas. Es el conjunto sobre el que se **ajusta y evalúa** el
  clustering. El filtro es necesario: las trayectorias constantes son triviales
  de agrupar y, sin excluirlas, dominan cualquier partición y anulan la
  tipología de interés.
- ***Pool* submuestreado** (≈126.000 parcelas): las trayectorias dinámicas más
  las constantes submuestreadas al 15 %, empleado exclusivamente para la
  **representación cartográfica**, de modo que el mapa cubra el territorio y no
  únicamente el 3,2 % dinámico.

### 4.3 Preprocesamiento del espacio latente

Dado que las métricas de distancia empleadas por los algoritmos de agrupamiento
son sensibles a la escala y a la geometría del espacio, se trataron **tres
transformaciones de $z$ como un hiperparámetro más** del barrido:

- `raw`: el espacio latente sin transformar;
- `standard`: estandarización por dimensión (media 0, desvío 1);
- `l2`: normalización a norma unitaria, que descarta la magnitud de $z$ y
  conserva sólo su dirección.

Toda métrica interna se computa en el mismo espacio en que se ajustó la
partición. La única excepción es la fidelidad del prototipo (§ 5.3), que
requiere decodificar centroides y por lo tanto opera necesariamente sobre el
espacio $z$ crudo: el decodificador sólo observó esa escala durante el
entrenamiento, de modo que un centroide en espacio estandarizado o normalizado
no constituye una entrada válida para $g_\phi$.

---

## 5. Agrupamiento (*clustering*) y construcción de la tipología

### 5.1 Familias de algoritmos

Se evaluaron **cuatro familias** que descansan sobre supuestos distintos acerca
de la forma de los grupos, de manera de no condicionar el resultado a una única
concepción de "cluster":

1. **k-medias** (particional, grupos esféricos y equiescalares; `n_init = 10`).
2. **Mezclas gaussianas** (modelo probabilístico generativo; matrices de
   covarianza `full` y `diag`; `n_init = 10` para paridad con k-medias).
3. **HDBSCAN** (basado en densidad, jerárquico; no requiere fijar el número de
   grupos, admite grupos de densidad y forma heterogéneas y **asigna a una clase
   de ruido** las observaciones que no pertenecen a ningún grupo denso;
   hiperparámetros `min_cluster_size` ∈ {250, 500, 1000, 2500} y `min_samples`
   ∈ {ninguno, 25}).
4. **Aglomerativo jerárquico** (enlaces de Ward, promedio y completo).

*Tratamiento del método jerárquico.* El aglomerativo no pudo ajustarse
directamente sobre las 107.362 observaciones: la matriz condensada de distancias
es O(*n*²), computacional y espacialmente inviable a esa escala. La mitigación
habitual —restringir la conectividad a un grafo de *k* vecinos más próximos—
resultó igualmente impracticable en este caso, porque la enorme cantidad de
trayectorias idénticas presentes en estos datos fragmenta el grafo en centenares
de componentes que el algoritmo no logra reconectar en tiempo razonable. Se
adoptó en cambio el procedimiento estándar para esta escala: **ajuste sobre una
submuestra estratificada por zona** (5.000 observaciones) y **extensión de las
etiquetas al conjunto completo por centroide más próximo**. La partición
resultante se evalúa luego sobre las 107.362 filas, en igualdad de condiciones
con las otras tres familias.

### 5.2 Grilla de búsqueda

El número de grupos se barrió en
$k \in \{2,\dots,10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 65, 80, 100, 120\}$
(22 valores), en cruce con las tres transformaciones del espacio latente y con
los hiperparámetros propios de cada familia. La banda alta ($k > 20$) se incluyó
deliberadamente para que las familias paramétricas pudieran compararse contra el
$k$ efectivo que alcanza HDBSCAN con `min_cluster_size` pequeño (~120).

| Familia | Configuraciones |
|---|---:|
| Aglomerativo jerárquico | 198 |
| Mezclas gaussianas | 92 |
| k-medias | 66 |
| HDBSCAN | 24 |
| **Total** | **380** |

### 5.3 Criterios de evaluación de las particiones

Cada configuración se evaluó con cuatro grupos de indicadores, escogidos para
cubrir dimensiones complementarias de la calidad de una tipología y para no
depender exclusivamente de medidas de separación geométrica:

**(i) Validez interna.** Coeficiente de silueta —promediado sobre cinco
submuestras aleatorias de hasta 20.000 observaciones, para reportar una
estimación y no un valor puntual—, índice de Calinski-Harabasz e índice de
Davies-Bouldin. Las observaciones clasificadas como ruido se excluyen de las
tres.

**(ii) Estabilidad.** Índice de Rand ajustado (ARI) entre dos reajustes
independientes del mismo procedimiento sobre pares de submuestras del 80 %,
comparados sobre la intersección de sus índices. Protege contra un $k$ que se
sostenga sólo por el azar de una muestra particular. Se emplearon 3 repeticiones
durante el barrido (por costo computacional) y **10 repeticiones al reajustar la
configuración ganadora**; el valor reportado para cada tipología final proviene
de este segundo cómputo.

**(iii) Fidelidad del prototipo.** Indicador propio de este diseño, que
aprovecha la naturaleza generativa del modelo: el centroide de cada grupo en el
espacio $z$ crudo se **decodifica** mediante $g_\phi$, obteniendo una
*trayectoria prototípica* de 23 años, y se compara posición a posición contra la
secuencia real de cada miembro del grupo. Se reporta el macro F1 resultante
—restringido a las clases con soporte positivo en cada grupo y ponderado por
tamaño de grupo. Responde a una pregunta que las métricas geométricas no
responden: *si el prototipo del grupo es una descripción honesta de sus
miembros*, y no meramente si el grupo está bien separado de los demás.

**(iv) Coherencia espacial.** Fracción de los 8 vecinos geográficos más próximos
—dentro de la misma zona— que comparten grupo, menos esa misma fracción bajo
permutación aleatoria de las etiquetas (línea de base que descuenta la
coincidencia esperable por el mero tamaño relativo de los grupos). Las
observaciones de ruido se excluyen **antes** de la búsqueda de vecinos: sin ese
filtro, dos vecinos ambos "no asignados" contarían como si compartieran un grupo
real, inflando el indicador en las configuraciones más ruidosas por la simple
contigüidad espacial de las zonas ambiguas.

### 5.4 Regla de decisión

Entre las configuraciones con **estabilidad ARI ≥ 0,75** y con fracción de ruido
por debajo de un tope, se selecciona la de mayor **fidelidad del prototipo**;
los empates se dirimen por coeficiente de silueta y, en segundo término, por
menor $k$.

El tope sobre la fracción de ruido es necesario porque la fidelidad del
prototipo se computa únicamente sobre los miembros no-ruido: sin esa
restricción, el criterio recompensaría mecánicamente a HDBSCAN por descartar las
observaciones difíciles (hasta el 45 % de las filas en configuraciones con
`min_cluster_size` elevado) y no por poseer una estructura de grupos
genuinamente mejor.

### 5.5 Tipologías seleccionadas: una matriz de 3 × 2

La regla no se aplicó una sola vez sino sobre una **matriz de tres niveles de
granularidad × dos familias**, por dos razones. Primero, porque la granularidad
óptima depende del uso: una tipología de 118 tipos maximiza la fidelidad pero no
es legible como leyenda de un mapa ni como narrativa analítica. Segundo, porque
HDBSCAN y las familias paramétricas resuelven el problema de manera
cualitativamente distinta —el primero admite dejar observaciones sin clasificar;
las segundas asignan la totalidad de los puntos— y la comparación honesta entre
ambas exige tratarlas como columnas separadas a granularidad pareja.

- **Fina**: sin tope de $k$; ruido ≤ 0,50.
- **Media**: $k \le 40$; ruido ≤ 0,25.
- **Gruesa**: $k \le 20$; ruido ≤ 0,25.

| Nivel / familia | Algoritmo (espacio) | $k$ | Silueta | ARI estab. | Fid. prototipo | Coh. espacial | Ruido |
|---|---|---:|---:|---:|---:|---:|---:|
| Fina / HDBSCAN | HDBSCAN L2, `min_cluster_size=250` | 118 | 0,913 | 0,905 | **0,955** | 0,602 | 9,8 % |
| Fina / no-HDBSCAN | GMM `diag` L2 | 120 | 0,735 | 0,916 | 0,905 | 0,585 | 0 % |
| **Media / HDBSCAN** | **HDBSCAN estandarizado, `min_cluster_size=1000`** | **31** | **0,797** | **0,914** | **0,866** | **0,631** | **24,2 %** |
| Media / no-HDBSCAN | GMM `diag` L2 | 40 | 0,547 | 0,824 | 0,719 | 0,600 | 0 % |
| Gruesa / HDBSCAN | HDBSCAN L2, `min_cluster_size=2500`, `min_samples=25` | 17 | 0,580 | 0,719¹ | 0,740 | 0,595 | 20,5 % |
| Gruesa / no-HDBSCAN | GMM `full` L2 | 18 | 0,470 | 0,782 | 0,578 | 0,562 | 0 % |

¹ Esta configuración alcanzaba 0,77 de estabilidad con 3 repeticiones de
*bootstrap* durante el barrido, pero **0,719 al reverificarla con 10
repeticiones**, por debajo del umbral de 0,75 del propio criterio. Es
precisamente el sobreajuste al azar de una muestra que el indicador de
estabilidad está diseñado para exponer, y se documenta como tipología
*exploratoria*.

El nivel **medio con HDBSCAN (k = 31)** constituye la tipología de referencia
para el análisis sustantivo: combina la fidelidad de prototipo más alta entre
las configuraciones de granularidad manejable (0,866) con estabilidad elevada
(0,914) y la mayor coherencia espacial de las seis (0,631).

### 5.6 Asignación del *pool* completo y tratamiento del ruido

Las seis particiones se aplicaron al *pool* submuestreado (§ 4.2) mediante
asignación al centroide más próximo. Dado que HDBSCAN carece de centroides y de
una clase de ruido definida fuera de la muestra de ajuste, una asignación
irrestricta clasificaría cada punto en *algún* grupo y sobrestimaría la
cobertura efectiva de la tipología. Se introdujo por ello un **umbral de
distancia**: los puntos cuya distancia al centroide más próximo excede el
percentil 95 de esa misma distancia entre los puntos **no-ruido** del *pool*
dinámico se etiquetan como *sin tipificar*. El umbral queda así calibrado sobre
la geometría real de cada configuración y se versiona junto con ella.

En consecuencia, la etiqueta −1 admite dos lecturas distintas y explícitas
según el conjunto: en el *pool* dinámico significa "HDBSCAN lo identificó como
ruido"; en el *pool* cartográfico significa "queda lejos de todo centroide". Las
comparaciones entre particiones reportan siempre la cobertura junto al
estadístico (§ 5.7).

### 5.7 Validación externa e interpretación de las tipologías

Los indicadores de § 5.3 validan las particiones pero se computan, en su mayor
parte, dentro del mismo espacio $z$ que las generó. Para evaluarlas en un
espacio independiente, y para hacerlas interpretables, se incorporó el
instrumental del **análisis de secuencias** (tradición TraMineR), reimplementado
sobre las matrices de tokens de cada grupo.

**Disimilitud entre trayectorias.** El *pool* dinámico contiene solamente
**1.128 secuencias distintas** entre sus 107.362 filas; colapsado a ese conjunto
con pesos iguales a sus frecuencias —operación exactamente equivalente a
trabajar sobre las filas completas—, el cómputo de matrices de disimilitud es
trivial. Se implementaron tres métricas sobre secuencias de longitud fija 23 con
años calendario alineados: Hamming; Hamming dinámico (costos de sustitución
dependientes de la posición, $c_t(i,j) = 2 - f_t(i) - f_t(j)$, simplificación
por frecuencia del DHD de Lesnard); y **Optimal Matching** (alineamiento global
con inserciones/eliminaciones, costos de sustitución TRATE
$c(i,j) = 2 - p(i\mid j) - p(j\mid i)$ derivados de las tasas de transición
globales, e *indel* $= \max(c)/2$), adoptada por defecto.

Sobre esa matriz se computaron:

- **Pseudo-R² de discrepancia** (análisis de discrepancia de Studer y
  Ritschard): $1 - SS_{\text{dentro}}/SS_{\text{total}}$ con la suma de
  cuadrados de discrepancia ponderada. Cuantifica qué fracción de la variación
  real de las trayectorias explica cada partición, **medida fuera del espacio
  latente que la produjo**, y constituye la comparación en igualdad de
  condiciones entre las seis tipologías.
- **Silueta en el espacio de secuencias** (ASW ponderada), que complementa la
  silueta euclídea computada sobre $z$.
- **Secuencias representativas** (criterio de densidad de vecindad): por grupo,
  un conjunto voraz de trayectorias reales no redundantes dentro de un radio del
  10 % de la distancia máxima, con su cobertura asociada.

| Tipología | $k$ | Pseudo-R² | ASW (secuencias) | Cobertura |
|---|---:|---:|---:|---:|
| Fina / HDBSCAN | 118 | 0,981 | 0,683 | 0,307 |
| Fina / no-HDBSCAN | 120 | 0,934 | 0,522 | 1,000 |
| Media / HDBSCAN | 31 | 0,944 | 0,583 | 0,187 |
| Media / no-HDBSCAN | 40 | 0,823 | 0,329 | 1,000 |
| Gruesa / HDBSCAN | 17 | 0,808 | 0,424 | 0,366 |
| Gruesa / no-HDBSCAN | 18 | 0,710 | 0,260 | 1,000 |

(La cobertura corresponde a la fracción de *secuencias distintas* efectivamente
clasificadas, y debe leerse conjuntamente con el pseudo-R²: las configuraciones
HDBSCAN explican más varianza, pero sobre un subconjunto menor de trayectorias.)

**Descripción de los grupos.** Cada grupo se caracterizó con la batería
descriptiva estándar del análisis de secuencias —cronograma de proporción de
estados por año, secuencia modal, entropía transversal, tiempo medio por estado,
tasas de transición, secuencias distintas más frecuentes y su cobertura, e
índices longitudinales por secuencia (número de transiciones, índice de
complejidad, entropía longitudinal, duración de tramos)—, todo con un costo
computacional O(*n·T*). La comparación entre la **secuencia modal** (libre de
modelo) y la **trayectoria prototípica decodificada** (dependiente del
decodificador) funciona además como diagnóstico: una discrepancia sistemática
sería un hallazgo sobre el decodificador y no sobre la partición.

**Etiquetado.** A partir de la secuencia modal y del cronograma se deriva de
manera determinista una etiqueta legible que combina la forma colapsada de la
trayectoria, su patrón temporal (estable / monotónica / oscilante / múltiple), el
año de mayor variación y una glosa sustantiva —por ejemplo,
`F»A · monotónica · ~2008 · deforestación para agricultura`. Los grupos se
agrupan luego en procesos territoriales conceptuales (deforestación, degradación
forestal, expansión agrícola sobre pastizal, pérdida de vegetación/aridización,
revegetación, regeneración de bosque, dinámica de humedal, urbanización,
oscilante, otros) según los estados inicial y final de su secuencia modal.

**Acuerdo entre particiones.** Las seis particiones comparten filas e
indexación, de modo que su acuerdo se computa columna contra columna mediante
ARI y NMI. Se reporta como medida primaria el ARI sobre las filas que **ninguna**
de las dos particiones dejó como ruido, acompañado de la cobertura de esa
intersección —las fracciones de ruido varían entre 0 % y 24 %, por lo que el ARI
no es interpretable sin ella—, y como medida secundaria el ARI tratando −1 como
una etiqueta más.

---

## 6. Implementación y reproducibilidad

El modelo se implementó en PyTorch 2.11; los procedimientos de agrupamiento y
las métricas de validación, en scikit-learn 1.8 y SciPy 1.17; el manejo de datos
geoespaciales, en xarray y netCDF4. El instrumental de análisis de secuencias
(disimilitudes, análisis de discrepancia, descriptores de grupo) fue
reimplementado en NumPy siguiendo las definiciones de TraMineR.

Todas las etapas estocásticas —partición entrenamiento/validación,
inicialización de pesos, submuestreo de secuencias constantes, inicializaciones
de k-medias y mezclas gaussianas, submuestras de *bootstrap*— utilizan semillas
fijas. Se deja constancia de que, aun con semilla idéntica, el entrenamiento en
GPU presenta fuentes de no-determinismo (orden de reducción en operaciones
paralelas, *kernels* no deterministas) que desplazan el macro F1 de
reconstrucción dentro de un rango de ±0,002–0,003.

### Disponibilidad de código y datos

El código fuente del modelo, de los procedimientos de agrupamiento y de los
análisis reportados se encuentra disponible para los revisores a pedido durante
el proceso de evaluación. El repositorio se hará público una vez publicado el
artículo.

Los datos primarios son de acceso libre y gratuito: el producto ESA CCI Land
Cover se distribuye bajo la política de datos de la ESA CCI y puede obtenerse
del Climate Data Store de Copernicus.

---

## 7. Herramienta de inspección visual: imágenes satelitales de referencia

> Esta sección documenta un instrumento de **control de calidad cualitativo**,
> no un componente de la validación cuantitativa de §5. Su función es permitir
> la inspección visual del paisaje real detrás de una trayectoria o un cluster;
> no interviene en el ajuste del modelo, en la selección de hiperparámetros ni
> en ninguna métrica reportada en este documento. ⚠ A criterio del autor,
> podría quedar fuera del paper final y documentarse solo en el repositorio.

### 7.1 Propósito

El visor interactivo (`viz/clusters/`) superpone los puntos clasificados por
cada tipología (§5.5) sobre un mosaico satelital de **inicio y fin de
período**, para las siete zonas de evaluación y para `chaco_santiago_frontier`
—la zona base del conjunto de ajuste (§1.3)—, a la que no se le calculó ninguna
partición por no formar parte de la evaluación. El objetivo es exclusivamente
exploratorio: identificar a simple vista si un cluster corresponde a un patrón
territorial reconocible (frente de deforestación, urbanización, humedal) antes
o en paralelo a la caracterización estadística de §5.7.

### 7.2 Fuente de las imágenes

Se evaluaron dos vías antes de adoptar la definitiva. Un primer intento generó
los mosaicos con Landsat y Sentinel-2 a través del catálogo STAC de Microsoft
Planetary Computer, componiendo la mediana temporal "a mano" (lectura y
reproyección independiente de cada escena); el resultado mostraba un
desalineamiento visible entre escenas de distinta órbita en las zonas más
extensas, por lo que se descartó. La vía adoptada delega el mosaico en
**Google Earth Engine**, que resuelve la reproyección y la reducción temporal
de manera consistente. Es importante una precisión terminológica: Earth Engine
expone los mismos catálogos públicos (Landsat, Sentinel), **no** la capa de
imagen propietaria de la aplicación Google Earth (la composición Maxar /
DigitalGlobe con buscador de fechas históricas), que no tiene una vía de acceso
programático.

- **Inicio de período (2000):** Landsat 5 TM, colección
  `LANDSAT/LT05/C02/T1_L2` (reflectancia de superficie, Collection 2 Level 2).
- **Fin de período (2022):** Sentinel-2 SR armonizado,
  `COPERNICUS/S2_SR_HARMONIZED`.
- **Compuesto:** mediana por píxel sobre una ventana de ±5 meses alrededor del
  año objetivo, con enmascarado de nubes y sombra (bits de `QA_PIXEL` en
  Landsat; clases de `SCL` en Sentinel-2) previo a la reducción.
- **Muestreo de escenas:** por costo computacional del lado del servidor
  (Earth Engine impone un límite de memoria por solicitud), el número de
  escenas que entran a cada mediana se acota y se reparte **por tile/órbita**
  —no globalmente por nubosidad total de la colección— para que ninguna
  sub-región de una zona extensa quede sin representación en el compuesto.

### 7.3 Limitación conocida: costuras en el límite de huso UTM

Las tres zonas cuyo *bounding box* cruza un límite de huso UTM
(`patagonia_estepa`, en el límite 19/20; `chaco_santiago_frontier` y
`pampa_nucleo`, ambas en el límite 20/21) presentaban una franja sin dato en el
compuesto de Sentinel-2, de forma y ubicación idénticas independientemente del
número de escenas o del umbral de nubosidad admitido. La causa es estructural,
no un déficit de datos: la grilla de *tiles* MGRS de Sentinel-2 está definida
por huso, y la costura entre husos no queda cubierta por ningún *tile* para esa
franja exacta. Se mitigó componiendo, solo para esas tres zonas, un mosaico de
respaldo con Landsat 8/9 (`LANDSAT/LC08|LC09/C02/T1_L2`) —cuya grilla de
órbitas *path/row* no comparte esa discontinuidad— y utilizándolo únicamente
donde Sentinel-2 no aporta dato. El parche es visualmente distinguible (30 m de
resolución nativa frente a 10 m) y se limita al área puntual de la costura.

### 7.4 Reproducibilidad

El procedimiento está implementado en `scripts/fetch_zone_imagery_gee.py`
(requiere una cuenta de Google Earth Engine, gratuita para uso no comercial) y
es idempotente: cada corrida verifica qué combinaciones zona/año ya están
generadas y solo completa lo faltante. Un intento previo basado en capturas
manuales de Google Earth Pro (`scripts/make_zone_kml.py` +
`scripts/import_manual_imagery.py`) queda documentado en
`viz/clusters/README.md` como alternativa no automatizable a escala.

---

## Referencias metodológicas sugeridas

> ⚠ Verificar y completar según el estilo de la revista.

- Vaswani, A. *et al.* (2017). Attention is all you need. *NeurIPS*.
- Hinton, G. E. y Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786).
- Loshchilov, I. y Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Press, O. y Wolf, L. (2017). Using the output embedding to improve language models. *EACL*.
- Campello, R. J. G. B., Moulavi, D. y Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*.
- McInnes, L., Healy, J. y Astels, S. (2017). hdbscan: hierarchical density based clustering. *JOSS*, 2(11).
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301).
- Rousseeuw, P. J. (1987). Silhouettes. *Journal of Computational and Applied Mathematics*, 20.
- Caliński, T. y Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1).
- Davies, D. L. y Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2).
- Hubert, L. y Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1).
- Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data Analysis*, 52(1).
- Gabadinho, A., Ritschard, G., Müller, N. S. y Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. *Journal of Statistical Software*, 40(4).
- Studer, M., Ritschard, G., Gabadinho, A. y Müller, N. S. (2011). Discrepancy analysis of state sequences. *Sociological Methods & Research*, 40(3).
- Lesnard, L. (2010). Setting cost in optimal matching to uncover contemporaneous socio-temporal patterns. *Sociological Methods & Research*, 38(3).
- Abbott, A. y Tsay, A. (2000). Sequence analysis and optimal matching methods in sociology. *Sociological Methods & Research*, 29(1).
- ESA (2017). Land Cover CCI Product User Guide, Version 2. UCLouvain / European Space Agency.
