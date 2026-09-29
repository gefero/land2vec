# Estabilidad de las tipologías entre clusterizaciones

Análisis de concordancia entre las seis corridas de clustering de land2vec v2: cuánto
cambia la asignación de un píxel —a un cluster y a un proceso conceptual— según la
granularidad y la familia de algoritmos elegidas.

**Fecha**: 2026-09-29 · **Datos**: `viz/crossrun/crossrun.json` (`scripts/build_crossrun.py`)
y `models/cluster_v2/desmonte_eval*.csv` · **Números**: todos reproducibles con
`python scripts/analisis_estabilidad.py` · **Herramienta interactiva**: `viz/crossrun/`.

---

## Resumen

1. **La lectura por proceso es robusta a la elección de la corrida, con una excepción.**
   Entre corridas, el acuerdo de proceso va de 0,80 a 1,00. La excepción es
   `gruesa/paramétrico`, que se aparta de todas las demás (0,80–0,90 en `dynamic`).
2. **`gruesa/paramétrico` no tiene ningún cluster de urbanización.** La categoría
   desaparece del vocabulario de esa corrida, lo que genera la confusión más grande del
   sistema (urbanización ↔ regeneración de bosque, 17 % de todo el desacuerdo). Excluida
   esa corrida, la urbanización es 95 % estable.
3. **La etiqueta de un cluster describe peor a sus miembros cuanto más grueso es y, a
   igual granularidad, peor en la familia paramétrica.** La cobertura modal ponderada va
   de 0,88 (`fina/HDBSCAN`) a 0,35 (`gruesa/paramétrico`); en esta última, el 80 % de la
   masa está en clusters cuya trayectoria modal describe a menos de la mitad de sus
   miembros.
4. **La familia HDBSCAN forma una jerarquía anidada; la paramétrica, no.** En `dynamic`,
   ningún cluster fino de HDBSCAN se reparte entre dos gruesos (0 de 84; pureza 1,0000).
   En la paramétrica, 19 de 40 clusters medios se reparten entre gruesos. Con la
   comparación emparejada de `pooled_subsampled` la ventaja se mantiene en los tres
   peldaños pero se achica mucho en pureza (0,987 contra 0,959 en el peor caso, 0,996
   contra 0,990 en el mejor); donde sigue siendo clara es en cuántos clusters se
   reparten (2, 4 y 6 contra 16, 13 y 20).
5. **El acuerdo entre familias cae al engrosar**: ARI 0,93 en fina, 0,86 en media, 0,64
   en gruesa, en los dos conjuntos. A granularidad fina la elección de familia casi no
   importa; a granularidad gruesa, sí.
6. **Cobertura modal y desempeño externo (§5.8) se asocian a lo largo de la
   granularidad pero no de la familia** (Spearman 0,71, n = 6, no significativo). En
   media y gruesa la corrida paramétrica tiene peor cobertura modal y *mejor* MCC externo.

Todo lo anterior es la misma disyuntiva vista desde ángulos distintos: HDBSCAN produce
etiquetas más representativas, jerarquías limpias y conserva categorías minoritarias,
al precio de dejar sin tipificar entre el 10 % y el 33 % de los píxeles.

---

## 1. Objeto y datos

Seis corridas seleccionadas por `tune_clustering.py --select` (§5.5 de
`paper_metodologia.md`), combinando tres granularidades con dos familias:

| corrida | *k* ¹ | sin tipificar (`dynamic`) | sin tipificar (`pooled`) |
|---|---:|---:|---:|
| fina / HDBSCAN | 118 | 9,8 % | 26,4 % |
| fina / paramétrico | 120 | 0,0 % | 17,5 % |
| media / HDBSCAN | 31 | 24,2 % | 32,5 % |
| media / paramétrico | 40 | 0,0 % | 7,9 % |
| gruesa / HDBSCAN | 17 | 20,4 % | 16,7 % |
| gruesa / paramétrico | 18 | 0,0 % | 7,2 % |

¹ Clusters presentes en `dynamic`; en `pooled` aparecen 117, 116, 31, 39, 17 y 18.

En lo que sigue se abrevian `fina/H`, `fina/P`, etc.

Se analizan los dos conjuntos de etiquetas de las siete zonas de evaluación:

- **`dynamic`** (107.362 píxeles): las trayectorias con al menos una transición, sobre
  las que se *ajustó* el clustering.
- **`pooled_subsampled`** (126.306 píxeles): el *pool* completo asignado por centroide
  más cercano, con las trayectorias constantes submuestreadas al 15 %.

Dentro de cada conjunto las seis corridas etiquetan las mismas filas en el mismo orden
(se verifica elemento a elemento al construir el payload), así que comparar dos
particiones es columna contra columna.

**El −1 no significa lo mismo en los dos conjuntos**, y esto condiciona varias lecturas:

- En `dynamic` es el **ruido del ajuste** de HDBSCAN. Las corridas paramétricas no
  tienen ninguno, porque un modelo de mezcla asigna todo.
- En `pooled_subsampled` es el **corte por distancia al centroide** (umbral persistido
  en cada `chosen*.json`), que se aplica a las seis corridas por igual.

En consecuencia, toda comparación entre familias sobre `dynamic` es despareja —HDBSCAN
se mide sobre menos masa que el paramétrico— y sobre `pooled_subsampled` queda
emparejada.

## 2. Medidas

Todas se calculan desde la tabla de contingencia de cada par de corridas.

**Acuerdo de proceso.** Cada cluster lleva uno de diez procesos conceptuales
(deforestación, urbanización, …), asignado por una regla determinista sobre el inicio y
el fin de su trayectoria modal (`classify_process`, en `scripts/build_cluster_map.py`).
Todo píxel hereda el proceso de su cluster. El acuerdo de proceso entre dos corridas es
la fracción de píxeles a los que ambas asignan el mismo proceso. A diferencia de los ids
de cluster, que son arbitrarios en cada corrida, los procesos son una nomenclatura común,
así que el acuerdo simple está bien definido.

**ARI de cluster.** El índice de Rand ajustado entre particiones, invariante a la
permutación de ids. Reproduce los valores de `models/cluster_v2/typology_crossrun.csv`
(diferencia máxima 4,9 × 10⁻⁷, el redondeo del payload).

**Anidamiento.** Para un par ordenado A → B, la fracción de la masa de cada cluster de A
que cae en un único cluster de B, ponderada por tamaño. Es asimétrica: de fino a grueso
vale 1 si la partición fina es un refinamiento exacto de la gruesa. Coincide con la
pureza de `land2vec.typology.nesting_table`, agregada.

**Cobertura modal.** Para cada cluster, la fracción de sus miembros que sigue
*exactamente* su trayectoria modal. Como el proceso se asigna por la modal, mide qué tan
representativa es la etiqueta. Es una propiedad del cluster, medida sobre sus miembros
del pool dinámico; en `pooled_subsampled` solo cambian los pesos.

**Tratamiento del −1.** Salvo indicación, las medidas se calculan sobre la
**intersección no-ruido**: las filas que ninguna de las dos corridas dejó en −1. La
fracción de N que sobrevive a esa exclusión (la *cobertura* del par) se reporta junto a
cada medida, porque va de 0,62 a 1,00 según el par y sin ella las celdas no son
comparables. Las medidas agregadas por proceso se suman sobre los 30 pares ordenados.

## 3. Resultados

### 3.1 Acuerdo de proceso entre corridas

Acuerdo de proceso sobre la intersección no-ruido:

| `dynamic` | fina/P | media/H | media/P | gruesa/H | gruesa/P |
|---|---:|---:|---:|---:|---:|
| **fina/H** | 0,994 | 0,996 | 0,932 | 0,995 | 0,821 |
| **fina/P** | | 0,985 | 0,892 | 0,976 | 0,797 |
| **media/H** | | | 0,978 | 0,998 | 0,897 |
| **media/P** | | | | 0,929 | 0,820 |
| **gruesa/H** | | | | | 0,864 |

| `pooled` | fina/P | media/H | media/P | gruesa/H | gruesa/P |
|---|---:|---:|---:|---:|---:|
| **fina/H** | 1,000 | 0,997 | 0,963 | 0,905 | 0,867 |
| **fina/P** | | 0,998 | 0,951 | 0,900 | 0,849 |
| **media/H** | | | 0,991 | 0,943 | 0,928 |
| **media/P** | | | | 0,885 | 0,858 |
| **gruesa/H** | | | | | 0,868 |

Contando "sin tipificar" como una categoría más (denominador N), el rango cae a
0,680–0,896 en `dynamic` y 0,675–0,909 en `pooled`. En `dynamic` esa lectura es un piso
duro entre familias, porque el −1 del paramétrico no existe y nunca puede coincidir.

Dos lecturas:

- **En `dynamic`, la escalera de granularidad de HDBSCAN casi no cambia la lectura por
  proceso** (0,995–0,998): partir un cluster en varios más finos casi siempre los parte
  *dentro* del mismo proceso. En `pooled` esto se debilita: `fina/H ↔ gruesa/H` cae a
  0,905, lo que indica que la estabilidad observada en `dynamic` depende en parte de qué
  píxeles cada corrida deja fuera.
- **`gruesa/P` es la corrida que más se aparta** en los dos conjuntos. En `pooled`
  `gruesa/H` también baja (0,885–0,943 frente a las cuatro corridas no gruesas), pero
  `gruesa/P` sigue siendo la fila más baja en promedio.

### 3.2 Dónde se concentra el desacuerdo

Fracción de la masa de cada proceso que la otra corrida asigna al mismo proceso (30
pares ordenados, sin −1):

| proceso | `dynamic` | `pooled` |
|---|---:|---:|
| Revegetación de suelo árido | 0,964 | 0,961 |
| Dinámica de agua / humedal | 0,952 | 0,972 |
| Degradación forestal | 0,938 | 0,919 |
| Expansión agrícola | 0,936 | 0,817 |
| Deforestación | 0,929 | 0,853 |
| Pérdida de vegetación / aridización | 0,897 | 0,877 |
| Regeneración de bosque | 0,894 | 0,883 |
| Urbanización | 0,740 | 0,737 |
| Otro | 0,622 | 0,964 ¹ |
| Oscilante / múltiple | 0,390 | 0,134 ¹ |

¹ Masa muy chica en `pooled` (4.813 y 3.335 píxel-pares, contra cientos de miles en los
demás procesos): tasas poco informativas. "Otro" también es chica en `dynamic` (7.433).

Los siete procesos sustantivos mayoritarios quedan entre 0,82 y 0,97. Los frágiles son
`oscilante` —esperable: es la categoría de lo que no tiene dirección clara— y
**urbanización**, que no es residual y se analiza en §3.3.

Confusiones dominantes (pares de procesos, fracción de la masa fuera de la diagonal):

| `dynamic` | | `pooled` | |
|---|---:|---|---:|
| regeneración de bosque ↔ urbanización | 16,7 % | regeneración de bosque ↔ urbanización | 17,8 % |
| dinámica de agua ↔ oscilante | 14,7 % | deforestación ↔ expansión agrícola | 15,5 % |
| deforestación ↔ dinámica de agua | 8,8 % | degradación forestal ↔ revegetación | 13,8 % |
| regeneración de bosque ↔ dinámica de agua | 8,0 % | pérdida de vegetación ↔ revegetación | 10,3 % |
| regeneración de bosque ↔ oscilante | 7,9 % | deforestación ↔ dinámica de agua | 8,7 % |

La confusión **deforestación ↔ expansión agrícola** en `pooled` merece atención porque
toca la validación externa (§3.8): la regla de `classify_process` separa ambos procesos
según si la trayectoria modal arranca en bosque, y en §5.8 solo el primero cuenta como
predicción de desmonte.

### 3.3 Urbanización: una categoría que una corrida no tiene

Clusters cuyo proceso es urbanización (trayectoria modal que termina en `U`):

| corrida | clusters | píxeles (`dynamic`) |
|---|---:|---:|
| fina/H | 9 | 3.649 |
| fina/P | 9 | 4.730 |
| media/H | 1 | 2.879 |
| media/P | 3 | 4.631 |
| gruesa/H | 1 | 4.848 |
| **gruesa/P** | **0** | **0** |

`gruesa/paramétrico` **no tiene ningún cluster de urbanización**, en ninguno de los dos
conjuntos. Los píxeles que las otras corridas llaman urbanización caen, en esa corrida,
dentro de clusters grandes dominados por otras trayectorias, y heredan su etiqueta.

La estabilidad global de 0,740 mezcla dos situaciones muy distintas:

| pares | urbanización estable |
|---|---:|
| que involucran a `gruesa/P` | 0 de 20.737 (0,0 %) |
| los otros 20 pares ordenados | 68.928 de 72.465 (**95,1 %**) |

(En `pooled`: 0 % y 93,9 %.) Fuera de esa corrida, la urbanización es tan estable como
los procesos mayoritarios.

El mecanismo, en un caso concreto: el cluster `fina/H #38` es urbanización pura —su
modal `A-…-A-U-…-U` la siguen *todos* sus miembros (cobertura 1,00)—, y 759 de sus
píxeles caen en `gruesa/P #1`, cuya modal `A-…-A-F-F-F-F` describe apenas al 16,4 % de
sus miembros y lo etiqueta como abandono agrícola / reforestación.

**Por qué ocurre.** La urbanización es un fenómeno chico (3–4 % de los píxeles). Con 18
componentes para todo el conjunto, ningún componente queda con mayoría urbana, así que
lo urbano nunca gana la modal. No es solo una cuestión de *k*: `gruesa/HDBSCAN`, con 17
clusters, sí conserva uno de urbanización (4.848 píxeles). HDBSCAN puede aislar un
núcleo chico y denso como cluster propio y mandar el resto a −1; un modelo de mezcla
tiene que repartir todos los píxeles entre sus componentes, y en ese reparto lo
minoritario se diluye.

### 3.4 Representatividad de las etiquetas

| corrida | cobertura modal ponderada | masa en clusters con cobertura < 0,5 |
|---|---:|---:|
| fina/H | **0,881** | 8,8 % |
| fina/P | 0,693 | 28,1 % |
| media/H | 0,689 | 27,0 % |
| media/P | 0,499 | 63,3 % |
| gruesa/H | 0,492 | 57,2 % |
| gruesa/P | **0,351** | **80,0 %** |

(Pesos de `dynamic`; con los de `pooled` el orden es idéntico: 0,904 … 0,380.)

Dos gradientes nítidos:

- **Granularidad**: más grueso, menos representativo. Esperable —un cluster más grande
  es más heterogéneo— pero acá queda cuantificado.
- **Familia**: a igual granularidad, HDBSCAN gana siempre (0,881 contra 0,693; 0,689
  contra 0,499; 0,492 contra 0,351). La razón es el −1: HDBSCAN deja fuera los casos
  que no encajan y sus clusters quedan más homogéneos; el paramétrico debe asignar todo
  y absorbe esa heterogeneidad dentro de cada componente.

Casos extremos: `media/P #23` tiene cobertura modal 0,028 y reúne 7.879 píxeles (7,3 %
de N) bajo la etiqueta "oscilación F↔B"; `gruesa/P #8`, cobertura 0,068 con 9.053
píxeles (8,4 %). En esos clusters la etiqueta no describe a casi ninguno de sus miembros.

Esta medida explica el mecanismo de §3.3: **el proceso es una propiedad del cluster,
vía su modal, no del píxel**. Cuando la cobertura modal es baja, el proceso que heredan
los píxeles es poco representativo, y cualquier cambio de partición lo altera.

### 3.5 Anidamiento entre granularidades

¿Son las tres granularidades niveles de una misma jerarquía, o tres particiones
distintas que solo difieren en *k*?

| escalera | `dynamic`: pureza · se reparten · cobertura | `pooled`: pureza · se reparten · cobertura |
|---|---|---|
| fina/H → media/H | 0,9997 · 2/84 · 0,74 | 0,9954 · 2/80 · 0,62 |
| fina/H → gruesa/H | **1,0000 · 0/84** · 0,74 | 0,9956 · 4/98 · 0,67 |
| media/H → gruesa/H | 0,9986 · 3/28 · 0,69 | 0,9865 · 6/30 · 0,65 |
| fina/P → media/P | 0,9562 · 27/120 · 1,00 | 0,9782 · 16/99 · 0,79 |
| fina/P → gruesa/P | 0,9677 · 39/120 · 1,00 | 0,9900 · 13/103 · 0,79 |
| media/P → gruesa/P | 0,8972 · 19/40 · 1,00 | 0,9591 · 20/39 · 0,90 |

("Se reparten" cuenta los clusters de origen que caen en dos o más clusters de destino,
sobre los que tienen masa en la intersección.)

**En `dynamic`, HDBSCAN es una jerarquía estricta**: de fina a gruesa ningún cluster se
reparte. Decir que un cluster fino "es parte" de uno grueso es literalmente cierto. **La
familia paramétrica no lo es**: de media a gruesa se reparten 19 de 40 clusters y el
10,3 % de la masa cae fuera de su destino principal.

Dos advertencias de denominador, ambas necesarias para no sobreestimar el resultado:

1. **El anidamiento perfecto vale sobre el 74 % de los píxeles.** En `dynamic`, 34 de
   los 118 clusters finos de HDBSCAN quedan enteros dentro del −1 del nivel grueso: la
   jerarquía es exacta *entre lo que ambos niveles tipifican*.
2. **En `dynamic` la comparación está sesgada a favor de HDBSCAN**, que se mide sobre el
   74 % contra el 100 % del paramétrico. En `pooled`, donde las seis corridas tienen −1,
   la ventaja se mantiene en los tres peldaños pero se achica mucho. Peldaño por peldaño,
   la pureza pasa a 0,995 contra 0,978 (fina → media), 0,996 contra 0,990 (fina → gruesa)
   y 0,987 contra 0,959 (media → gruesa); la diferencia más nítida pasa a ser la cantidad
   de clusters que se reparten: 2, 4 y 6 en HDBSCAN contra 16, 13 y 20 en el paramétrico.
   Parte de la mejora del paramétrico en `pooled` es un efecto del denominador: los
   píxeles difíciles, que causaban los repartos, ahora quedan excluidos como −1.

La cifra honesta para citar es la de `pooled`, o ambas con la aclaración.

### 3.6 Misma granularidad, distinta familia

| par | ARI | HDBSCAN → paramétrico | paramétrico → HDBSCAN | cobertura |
|---|---:|---|---|---:|
| **`dynamic`** | | | | |
| fina | 0,934 | 0,955 · 40/118 | 0,833 · 44/105 | 0,90 |
| media | 0,864 | 0,905 · 16/31 | 0,870 · 9/33 | 0,76 |
| gruesa | **0,641** | 0,849 · 11/17 | 0,718 · 8/17 | 0,80 |
| **`pooled`** | | | | |
| fina | 0,942 | 0,987 · 12/117 | 0,835 · 34/88 | 0,74 |
| media | 0,859 | 0,965 · 9/31 | 0,870 · 9/30 | 0,67 |
| gruesa | **0,648** | 0,839 · 16/17 | 0,745 · 13/18 | 0,82 |

**El acuerdo entre familias cae al engrosar**, igual en los dos conjuntos. A *k* alto
los grupos son chicos y densos y cualquier criterio razonable los encuentra; a *k* bajo
hay que decidir dónde trazar fronteras entre regiones grandes, y ahí el criterio de
cada algoritmo —densidad frente a verosimilitud gaussiana— pesa de verdad. En gruesa, con
coberturas de 0,80–0,82, la discrepancia es estructural y no un efecto del −1.

**La relación es asimétrica en los seis casos**: un cluster de HDBSCAN suele caer entero
dentro de uno paramétrico (0,84–0,99), un cluster paramétrico suele repartirse entre
varios de HDBSCAN (0,72–0,87). Los clusters de HDBSCAN son subconjuntos más limpios, no
solo más numerosos, coherente con §3.4.

**Implicación práctica**: a granularidad fina la elección de familia casi no cambia la
partición (ARI 0,93–0,94); a granularidad gruesa la cambia mucho.

### 3.7 A dónde va el −1

En `dynamic`, el −1 de `media/HDBSCAN` son 25.961 píxeles (24,2 %). `media/paramétrico`,
que no tiene −1, los reparte así:

| proceso asignado por media/P | fracción |
|---|---:|
| oscilante / múltiple | 26,7 % |
| regeneración de bosque | 15,5 % |
| pérdida de vegetación / aridización | 13,5 % |
| dinámica de agua / humedal | 9,6 % |
| revegetación de suelo árido | 8,6 % |
| expansión agrícola | 7,5 % |

El ruido de HDBSCAN no es un residuo amorfo: su categoría principal bajo el modelo de
mezcla es la de trayectorias sin dirección clara.

En `pooled` el cuadro cambia. Allí el −1 de `media/H` es mayor (41.079 píxeles, 32,5 %)
y `media/P` deja **el 23,7 % también sin tipificar**; el resto va a dinámica de agua
(15,9 %), regeneración de bosque (15,9 %) y deforestación (14,4 %), y "oscilante" casi
desaparece como categoría en ese conjunto (§3.2). Es compatible con leer el −1 como
"trayectoria sin forma estable" —que en `dynamic` el modelo de mezcla está obligado a
absorber y en `pooled` el corte por distancia rechaza en ambas familias—, pero esa
lectura no se verificó píxel a píxel y queda como hipótesis.

### 3.8 Relación con la validación externa (§5.8)

Cobertura modal frente al MCC contra polígonos de desmonte en Chaco Seco
(`chaco_santiago_frontier`, −1 = negativo, IC 95 % por *bootstrap* de bloques):

| corrida | cobertura modal | MCC externo [IC 95 %] |
|---|---:|---|
| fina/H | 0,881 | **0,553** [0,538; 0,568] |
| fina/P | 0,693 | 0,534 [0,521; 0,546] |
| media/H | 0,689 | 0,450 [0,435; 0,464] |
| media/P | 0,499 | 0,514 [0,498; 0,529] |
| gruesa/H | 0,492 | 0,415 [0,400; 0,428] |
| gruesa/P | 0,351 | 0,466 [0,450; 0,482] |

Spearman = 0,714. Con n = 6 **no es significativo** (valor crítico 0,829 a α = 0,05
unilateral; 0,886 bilateral).

Más importante que el coeficiente es su estructura: **la asociación viene de la
granularidad, no de la familia.** A lo largo de la escalera fina → media → gruesa, menos
cobertura modal acompaña menos MCC dentro de cada familia. Pero a igual granularidad,
en media y en gruesa la corrida paramétrica tiene *peor* cobertura modal y *mejor* MCC
(0,514 contra 0,450; 0,466 contra 0,415), con intervalos que no se solapan.

Una explicación consistente con el resto del informe: el MCC de §5.8 cuenta el −1 como
negativo, así que la parte del territorio que HDBSCAN deja sin tipificar pesa como
desmonte no detectado. Es la misma disyuntiva cobertura/precisión: HDBSCAN clasifica
mejor lo que clasifica, y el paramétrico gana cuando la métrica premia clasificar más.
En fina, donde HDBSCAN deja poco −1 (9,8 %), la ventaja de representatividad se impone.

## 4. Síntesis

| dimensión | HDBSCAN | paramétrico |
|---|---|---|
| Cobertura | deja 10–33 % sin tipificar | clasifica todo en `dynamic`; 7–17 % sin tipificar en `pooled` |
| Representatividad de la etiqueta | cobertura modal 0,49–0,88 | 0,35–0,69 |
| Estructura entre granularidades | jerarquía anidada (exacta en `dynamic`) | tres particiones distintas |
| Categorías minoritarias | conserva urbanización con *k* = 17 | la pierde con *k* = 18 |
| Acuerdo con la otra familia | alto en fina (ARI 0,93), bajo en gruesa (0,64) | ídem |
| MCC externo (Chaco) | mejor en fina | mejor en media y gruesa |

No es un empate ambiguo sino una disyuntiva con forma definida. HDBSCAN gana en todas
las dimensiones que miden la calidad de *lo que clasifica*; el paramétrico, en las que
premian *cuánto clasifica*. Cuál conviene depende del uso: la interpretación tipológica
favorece a HDBSCAN; la cobertura cartográfica completa, al paramétrico.

## 5. Implicaciones para el paper

1. **`gruesa/paramétrico` no sostiene afirmaciones sobre procesos minoritarios**: no
   tiene el concepto de urbanización, tiene el 80 % de su masa en clusters con etiquetas
   poco representativas, y es la corrida que más se aparta de las demás.
2. **Reportar la cobertura modal ponderada en la tabla de §5.5.** Es barata,
   interpretable y responde la pregunta que un revisor va a hacer sobre una tipología
   construida por trayectorias modales: qué tan bien describe la etiqueta a los miembros
   del grupo.
3. **La afirmación "HDBSCAN produce una jerarquía anidada" es estructural y verificable**,
   pero debe citarse con el denominador (74 % de los píxeles en `dynamic`) o con las
   cifras emparejadas de `pooled`, donde la ventaja en pureza es chica (0,987 contra 0,959
   en media → gruesa, el peldaño más separado) y la señal clara es la cantidad de
   clusters repartidos.
4. **La discrepancia de §5.5/§5.8 —Fina/HDBSCAN gana externamente, la recomendación
   interna es Media/HDBSCAN— tiene una explicación parcial**: Fina/HDBSCAN combina la
   mejor cobertura modal con poco −1. No alcanza como evidencia (n = 6, asociación solo a
   lo largo de la granularidad), pero ordena la discusión.
5. **La frontera deforestación / expansión agrícola es sensible a la corrida en el pool
   aplicado** (deforestación 0,853 de estabilidad en `pooled`, confusión principal con
   expansión agrícola, 15,5 % del desacuerdo). Como solo la primera cuenta como
   predicción de desmonte en §5.8, parte de la variación del MCC entre corridas puede
   venir de esta frontera y no de la calidad del agrupamiento. Vale la pena mencionarlo
   entre las limitaciones de §5.8.6.
6. **El −1 de HDBSCAN admite una lectura sustantiva** ("trayectoria sin forma estable")
   en `dynamic`; en `pooled` la evidencia es más débil y la lectura queda como hipótesis.

## 6. Limitaciones

- **Seis corridas no son una muestra.** Son las seleccionadas por `--select`, no un
  muestreo del espacio de clusterizaciones posibles; los patrones entre familias y
  granularidades son descriptivos de estas seis.
- **El proceso es una etiqueta de cluster, no de píxel.** El acuerdo de proceso mide el
  acuerdo entre etiquetas heredadas por la regla de `classify_process` sobre la
  trayectoria modal; no dice si el proceso es correcto para cada píxel.
- **La cobertura modal se mide sobre el pool dinámico** (los miembros sobre los que se
  describió la tipología). En `pooled` solo cambian los pesos.
- **El anidamiento y las medidas primarias excluyen el −1**, y su magnitud depende de la
  cobertura de cada par, que varía entre 0,62 y 1,00.
- **Solo las siete zonas de evaluación.** Las ocho de entrenamiento (`train_pooled`) no
  se compararon.
- **Las medidas agregadas por proceso ponderan igual cada par**; `gruesa/P` interviene en
  10 de los 30 pares ordenados y domina varias de las confusiones.
- **La asociación con §5.8 usa una sola zona** y seis puntos.

## 7. Reproducibilidad

```bash
python scripts/build_crossrun.py         # contingencias y medidas -> viz/crossrun/crossrun.json
python scripts/analisis_estabilidad.py   # todas las tablas de este informe
python -m http.server -d viz/crossrun 8000   # visor interactivo -> http://localhost:8000
```

`build_crossrun.py` es solo-stdlib y verifica al construir: alineamiento fila a fila de
las seis corridas, conservación de masa en cada contingencia, consistencia de
marginales, y contraste del ARI/NMI contra los valores de sklearn de
`typology_crossrun.csv`. Los valores de anidamiento se contrastaron además contra un
cálculo independiente sobre los `data/clusters_*.zip`.

**Nota sobre cifras preliminares.** Durante la exploración se manejaron números que este
informe corrige: la robustez por proceso se había agregado sobre el lado de origen de 15
pares no ordenados, lo que dejaba a `gruesa/P` fuera como origen y daba, por ejemplo,
0,973 para deforestación (acá 0,929) y 0,610 para urbanización (acá 0,740). Las cifras
válidas son las de este documento.
