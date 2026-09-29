# Visor de concordancia entre clusterizaciones (local)

Visor estático (SVG a mano, sin build, sin CDN, sin librerías) para ver **cómo cambia
de grupo cada píxel según la clusterización**. Las 6 corridas de la v2 (granularidad
fina/media/gruesa × familia HDBSCAN/paramétrico) etiquetan los mismos píxeles en el
mismo orden, así que comparar dos particiones es columna contra columna.

Cubre los **dos conjuntos** de etiquetas de las zonas de evaluación, con un selector:

- `dynamic` (107.362 px) — el pool sobre el que se **ajustó** el clustering.
- `pooled_subsampled` (126.306 px) — todo el pool **aplicado** por centroide más
  cercano, con las constantes submuestreadas al 15 %.

**El `-1` no significa lo mismo en los dos**, y esto cambia las lecturas: en `dynamic`
es el ruido del ajuste de HDBSCAN, y por eso las corridas paramétricas no tienen
ninguno (0 %); en `pooled_subsampled` es el corte por distancia al centroide, que se
aplica a las seis (7–33 %). Una comparación entre familias en `dynamic` es por lo
tanto despareja —HDBSCAN se mide sobre menos masa— y en `pooled` está emparejada.

Complementa a los otros dos: `viz/typology/` responde *cómo es* cada cluster y
`viz/clusters/` *dónde está*; éste, **si sobrevive al cambio de corrida**.

## Qué muestra

**Flujo (aluvial).** Selectores independientes de **origen** y **destino** (cualquiera
de las 6 corridas, más un botón `⇄` para invertir), en dos niveles:

- **proceso** — las 10 categorías conceptuales (deforestación, urbanización, …) más
  "sin tipificar". Las dos columnas usan el **mismo orden fijo**, así que el acuerdo se
  lee como bandas horizontales y el desacuerdo como diagonales.
- **cluster** — los clusters reales (hasta 120 por lado). Se ordenan agrupados por
  proceso y después con un barrido baricéntrico que minimiza cruces ponderados por masa.

Controles: color de banda por **origen** o por **destino** (por destino se ve mucho
mejor a dónde va el `-1`), **resaltar desacuerdo** (apaga la diagonal), y un umbral de
**flujo mínimo** con lectura en vivo de qué fracción de la masa esconde.

Hover sobre una banda: `n`, % del origen, % del destino, % del total. Sobre un nodo:
`n`, cantidad de destinos, pureza y destino modal — la pureza reproduce la columna
homónima de `land2vec.typology.nesting_table`. Click en un nodo entra en **modo foco**
(el resto baja a opacidad 8%) y abre el panel con el reparto completo.

A nivel cluster, el tooltip y el panel muestran además la **trayectoria más frecuente**
del cluster como una tira de 23 celdas coloreadas por estado (2000 → 2022, misma paleta
que el resto del proyecto), con su peso: qué fracción de los miembros del cluster sigue
esa secuencia *exacta*. Es la lectura más concreta de qué es cada grupo, y varía mucho —
hay clusters donde la modal cubre el 100% y otros donde apenas el 25%, que es en sí un
indicador de lo compacto que es el cluster en el espacio de secuencias.

**Matriz 6×6.** Acuerdo entre todas las corridas. Click en una celda abre su flujo.

Incluye una medida **direccional**, `anidamiento`: qué fracción de la masa de cada
cluster de la **fila** cae en un único cluster de la **columna**. Es la única medida
del visor que **no es simétrica**, y responde a si la escalera de granularidad es una
jerarquía real. Resultado principal: en `dynamic`, `fina/HDBSCAN → gruesa/HDBSCAN` da
**1,0000 con 0 de 84 clusters partidos** —cada cluster fino vive entero dentro de uno
grueso—, mientras que `media/paramétrico → gruesa/paramétrico` da 0,897 con 19 de 40
clusters partidos. La familia HDBSCAN produce una jerarquía anidada; la paramétrica,
tres particiones distintas que solo difieren en *k*.

Ojo con el denominador: el anidamiento se mide siempre sobre la intersección no-ruido,
así que en `dynamic` ese 1,0000 vale sobre el 74 % de los píxeles (34 de los 118
clusters finos quedan enteros dentro del `-1` del nivel grueso). En `pooled`, donde
las seis corridas tienen `-1`, la ventaja de HDBSCAN se mantiene pero se achica
(0,996 vs 0,959). La barra del borde inferior de cada celda es esa cobertura.

## El `-1` no se descarta

Es la diferencia principal con los PNG estáticos de `imgs/v2_typology_alluvial_*`:
`land2vec.typology.plot_alluvial` saca el `-1` de los dos lados y normaliza por lo que
queda, así que esos gráficos están calculados sobre entre el 69% y el 90% de los datos
sin decirlo. Acá el `-1` es un nodo más, **tramado a 45°** para que se lea como "no es
una categoría" incluso impreso o en escala de grises.

Importa sobre todo en `dynamic`, donde HDBSCAN deja sin tipificar entre el 9,8% y el
24,2% de los píxeles según la granularidad y el paramétrico **0%**: a dónde va esa masa
es justamente la pregunta. (En `pooled_subsampled` las seis tienen `-1`, entre 7% y 33%,
porque ahí el criterio es la distancia al centroide.)
Respuesta (`dynamic`, media/HDBSCAN → media/paramétrico, 25.961 píxeles): `oscilante` 26,7%,
`regeneración de bosque` 15,5%, `pérdida de vegetación` 13,5%, `dinámica de agua` 9,6%.
El ruido de HDBSCAN no es un residuo amorfo — son mayormente trayectorias oscilantes que
GMM sí tipifica.

## La medida de la matriz

Por defecto, **acuerdo de proceso**: la fracción de píxeles a los que las dos corridas
asignan la misma categoría conceptual. ARI y NMI existen porque las etiquetas de cluster
son *arbitrarias* (el cluster 7 de una corrida no tiene nada que ver con el 7 de otra);
las de proceso **no lo son**, son una nomenclatura común a las 6. Con nomenclatura común
el acuerdo simple está bien definido, y además **es literalmente lo que muestra el
aluvial**: es la masa en la diagonal. Las dos vistas se explican mutuamente.

También están `ARI de proceso` y `ARI de cluster` (este último reproduce los números
publicados en `models/cluster_v2/typology_crossrun.csv`).

**Tratamiento del `-1`**, con dos lecturas explícitas:

- `excluir` (default) — sobre la intersección no-ruido. Es la lectura primaria, la misma
  de `TY.crossrun_agreement`. La cobertura de esa intersección va de 0,69 a 1,00 según el
  par, así que **las celdas no son comparables entre sí** sin mirarla: por eso cada una
  lleva una barra de cobertura en el borde inferior.
- `11ª categoría` — "sin tipificar" cuenta como una categoría más, denominador = N. Ojo
  en `dynamic`: ahí el paramétrico no deja ruido, así que entre familias el `-1` **nunca**
  puede coincidir y el número es un piso duro (entre las dos corridas HDBSCAN sí es
  informativo). En `pooled_subsampled` las seis tienen `-1`, así que la lectura es válida
  para todos los pares.

La medida `anidamiento` ignora este toggle: se calcula siempre sobre la intersección
no-ruido.

## Cómo levantarlo

```bash
python scripts/build_crossrun.py                    # los dos sets -> viz/crossrun/crossrun.json
python scripts/build_crossrun.py --sets dynamic     # solo uno
python -m http.server -d viz/crossrun 8002          # -> http://localhost:8002
```

Con `file://` el navegador bloquea el `fetch` del JSON; el visor lo detecta y lo dice.
La vista activa queda en el hash (`#2-3/proceso`), así que se puede compartir un enlace
a un par concreto.

`build_crossrun.py` es **solo-stdlib**: el ARI y el NMI se calculan a mano desde la tabla
de contingencia, no con sklearn (coinciden a ~1e-7, que es el redondeo del payload; si
está `models/cluster_v2/typology_crossrun.csv` el script contrasta los 60 valores y
reporta el delta máximo). Importa de `scripts/build_cluster_map.py` los helpers de color
y la clasificación en procesos, igual que `plot_process_maps.py` y `eval_desmonte.py`.

## Qué lleva el payload

`crossrun.json` (~196 KB con los dos sets): tablas de contingencia, etiquetas de cluster, el `inicio»fin`
colapsado y —para el tooltip— **la trayectoria modal de cada cluster con su peso**.

Eso último es una secuencia verbatim de 23 años por cluster: **344** sobre las 1.128
trayectorias distintas del pool. Es bastante menos que las 1.286 de
`viz/typology/typology_browser.json`, y son justamente las modales (las más frecuentes,
las menos identificatorias), pero conviene tenerlo presente: el payload **no** es
estrictamente solo agregado. Coordenadas por parcela no lleva ninguna, a diferencia de
`viz/clusters/data/`.

Si hace falta un payload sin ninguna secuencia —por ejemplo para publicarlo—:

```bash
python scripts/build_crossrun.py --no-modal-seq     # estrictamente agregado
```

El visor lo detecta y simplemente no muestra la tira; todo lo demás funciona igual.

Gitignoreado en los dos casos, por consistencia con los otros dos visores y porque se
regenera en un par de segundos.

## Dos cosas que conviene saber al leer el diagrama

- **El grosor de banda es vertical, no perpendicular.** Donde una banda es muy empinada
  se ve más fina de lo que vale. Es el compromiso estándar de cualquier Sankey; el dato
  correcto está siempre en el hover.
- **A nivel cluster no entran todas las etiquetas.** Con k=120 el lienzo llega al tope de
  1500 px y los nodos más chicos de `fina/paramétrico` (26 de 120, que juntos son el 3,4%
  de la masa) quedan por debajo de 3 px. Se dibujan a su tamaño real —no se colapsan ni se
  descartan— y siguen siendo hovereables y clickeables gracias a un área de click de 7 px
  desacoplada del dibujo. La forma de leer esa granularidad es el modo foco, no la vista
  completa.
