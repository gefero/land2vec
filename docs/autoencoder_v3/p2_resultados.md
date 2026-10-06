# P2: descripción de las dinámicas (primer pase)

**Fecha:** 2026-10-05 (primer pase) y 2026-10-06 (control externo con desmonte, sección final). Ejecuta plan §4.2; mapas, interpretación de los tipos y ruido de la CV, 2026-10-06 (secciones finales). Código: `scripts/clustering/p2_tipologias.py`. Salidas: `data/autoencoder_v3/p2/` (`p2_metricas.csv`, `p2_acuerdo.csv`, `p2_labels.csv`).

## Montaje
- **Qué se tipifica:** las 4.545 trayectorias dinámicas (1,68 M px). Las 9 constantes (95 % de la superficie) quedan fuera, como clases estables.
- **Espacios** (mismo algoritmo en todos: k-medoides ponderado sobre la matriz de distancias del espacio, k-medoides++, 10 reinicios): `om` (OM con costos TRATE), `onehot`, `pca8` (PCA del one-hot, 8 comp.), `ae_d3`, `ae_d8`, `ae_d16` (z del autoencoder de P1, semilla 0). Se llevan varias d porque P1 no define una (ver `p1_resultados.md`).
- **Granularidad:** k = 6, 12, 24. Peso = superficie (`px`); la variante `tipo` pesa 1 por trayectoria.
- **Calidad:** pseudo-R² y ASW **corregida** (denominador W_k − 1, `seqdist.asw(correct=True)`), siempre medidas con la distancia OM entre dinámicas.
- **Estabilidad:** 20 submuestras reales del 80 % de los píxeles (adelgazamiento binomial de `n_px`), ARI ponderado contra el ajuste completo.
- **Qué agrupa:** NMI ponderada de las etiquetas con (a) la secuencia de estados sin años (p. ej. F→A) y (b) el año del primer cambio.
- **Costura:** todo se repite excluyendo las 655 trayectorias con cambio en 2015→16 (los z no se reentrenaron).

## Resultados (universo completo, peso por superficie)

| | k | om | onehot | pca8 | ae_d3 | ae_d8 | ae_d16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| pseudo-R² | 6 | 0,61 | 0,57 | 0,61 | 0,53 | 0,50 | 0,46 |
| | 12 | 0,73 | 0,73 | 0,72 | 0,66 | 0,66 | 0,62 |
| | 24 | 0,80 | 0,79 | 0,81 | 0,77 | 0,76 | 0,76 |
| ASW corr. | 12 | 0,49 | 0,45 | 0,41 | 0,34 | 0,33 | 0,28 |
| NMI con estados | 12 | 0,59 | 0,62 | 0,60 | 0,63 | 0,59 | 0,57 |
| NMI con año 1.er cambio | 12 | 0,20 | 0,23 | 0,23 | 0,19 | 0,26 | 0,31 |
| Estabilidad (ARI 80 %) | 6 / 12 / 24 | .98/.80/.67 | .80/.62/.67 | 1.0/.81/.71 | .97/.82/.74 | .74/.75/.73 | .55/.76/.73 |

ARI entre espacios (k = 12, superficie): om–pca8 0,64; om–onehot 0,73; om–ae 0,56–0,67; ae–ae 0,47–0,63. Entre semillas del mismo z: 0,38–0,71. Con pesos por tipo en vez de superficie, la ARI contra la de superficie baja a 0,47–0,66 (0,88–0,90 en om y pca8 con k = 6).

## Lectura
1. **Las tipologías agrupan sobre todo por proceso (secuencia de estados) y poco por el año.** La NMI con la secuencia de estados es 0,5–0,7 en todos los espacios; con el año del primer cambio, 0,05–0,37. La excepción es z de d=16 (0,31 con k = 12, contra 0,20 de OM), que separa más por año; z de d=3 es el menos sensible al año (0,08–0,19 en k = 6–12).
2. **Medida en el espacio de secuencias, las tipologías de z no superan a OM/one-hot/PCA.** Pseudo-R² y ASW son menores en z (ASW 0,28–0,34 contra 0,41–0,49 con k = 12). Cautela: la vara es la distancia OM, así que favorece a quien agrupa con OM (es circular para `om` y no es neutral para el resto). Lo único que se puede decir es que z no recupera la geometría OM, coherente con P1.
3. **Estabilidad.** Con submuestras del 80 %, om y pca8 son casi idénticas con k = 6 (ARI ≈ 0,98–1,0); ae_d16 es inestable ahí (0,55). A k = 12 y 24 todo ronda 0,62–0,82. **El ruido entre semillas del autoencoder (ARI 0,38–0,71) es del mismo orden o mayor que el de la submuestra**: la tipología en z depende de la réplica de entrenamiento, y las de OM/PCA no tienen esa fuente de ruido. No se midió el ruido de los reinicios del k-medoides por separado.
4. **La d importa poco para la tipología** en calidad (pseudo-R² 0,46–0,53 con k = 6, converge en k = 24) y bastante en cómo agrupa: d=16 es más sensible al año, d=3 más al proceso. Los z de distinta d coinciden entre sí poco (ARI 0,47–0,63), tanto como con OM.
5. **Costura:** ningún cluster está definido por trayectorias de la costura (máximo de cluster con ≥ 80 % de superficie de costura: 0 en todas las configuraciones). Pero con k = 24 hay clusters donde ≥ 50 % de la superficie son trayectorias cuyo único cambio es 2015→16 (one-hot 0,61; ae_d3 0,51; ae_d8 0,49): ahí la tipología concentra la costura sin llegar al umbral. Excluir esas trayectorias mueve poco la calidad (pseudo-R² ± 0,03) pero reordena los clusters (ARI completo vs. sin costura 0,59–1,00; menor con k altos y en z).
6. **Peso.** Pesar por superficie o por tipo cambia mucho la tipología (ARI 0,47–0,66 en z y one-hot): con superficie, unas pocas trayectorias dominan cada cluster. Hay que reportar las dos.

## Pendiente y límites
- Actualizar el visor (`viz/`) y el atlas con la tipología de v3: no hecho (sólo hay mapas estáticos).
- Representativas e interpretación de los tipos como procesos (`describe`): no hecho; las NMI son un resumen grueso.
- Un solo algoritmo (k-medoides ponderado, 10 reinicios); la tipología de la v2 era HDBSCAN. Sin barrido de k más allá de 6/12/24.
- Los z son de la semilla 0 de P1; no se reentrenó sin las trayectorias de la costura.
- Calidad medida sólo con OM (ver lectura 2); falta una vara neutral (p. ej. la distancia de Hamming).

## Control externo con la referencia de desmonte (2026-10-06)

Código: `scripts/validacion/p2_desmonte.py`; salidas `data/autoencoder_v3/p2/p2_desmonte.csv` y `p2_desmonte_bases.csv`. Es una prueba de validez, no una competencia de detección: se mide cuánta información sobre desmonte conserva cada tipología respecto de la secuencia exacta.

**Diseño.** Cada píxel de la referencia (Colección 13.0; positivo / negativo_limpio, τ 0,5, dilate 1, pesos IPW) toma el tipo de su trayectoria. **Las constantes van a 9 clases estables propias** (`const_<estado>`) y nunca a un cluster de cambio (bloqueante B de la auditoría). Métricas: (1) `u_ratio` = U(D|tipo)/U(D|secuencia exacta), ambos ajustados por permutación; (2) MCC de la tabla por tipo con CV espacial (5 folds, bloques de 0,2°, umbral elegido en train), comparado con el techo de la tabla por secuencia exacta; (3) MCC de una regla semántica que no usa la referencia: el cluster es desmonte si su medoide cumple R0'.

**Las líneas de base reproducen la auditoría** (Chaco, 793.112 evaluables, prevalencia 0,2604): R0 0,399, **R0' 0,638 [0,625;0,650]**, F2000∧¬F2022 0,637, alguna transición 0,616, techo (tabla por secuencia, CV) **0,640**. En yungas R0' = 0,670 y techo 0,649.

**Chaco, universo completo, peso por superficie:**

| espacio | u_ratio k=6 / 12 / 24 | MCC_cv / techo k=12 | MCC semántico k=12 | − R0' (IC 95 %) |
|---|---|---|---|---|
| om | 0,91 / 0,94 / 0,96 | 0,980 | 0,622 | −0,016 [−0,019;−0,012] |
| onehot | 0,94 / 0,96 / 0,98 | 0,988 | 0,617 | −0,021 [−0,025;−0,018] |
| pca8 | 0,91 / 0,96 / 0,97 | 0,983 | 0,617 | −0,021 [−0,025;−0,018] |
| ae_d3 | 0,91 / 0,96 / 0,98 | 0,971 | 0,595 | −0,043 [−0,048;−0,038] |
| ae_d8 | 0,93 / 0,95 / 0,97 | 0,985 | 0,627 | −0,011 [−0,014;−0,008] |
| ae_d16 | 0,94 / 0,96 / 0,98 | 0,990 | 0,633 | −0,005 [−0,007;−0,002] |

**Lectura**
1. **La compresión en tipos casi no pierde información sobre desmonte.** Con 12 clusters dinámicos (más las 9 constantes) se conserva el 94–96 % de la información de la secuencia exacta y 97–99 % del techo de MCC (0,64). Con 6 clusters, 91–94 % y 97–98 %. Las diferencias entre espacios son chicas (≤ 0,01 de MCC en Chaco) pero consistentes entre particiones de la CV (ver la subsección "Ruido de la CV").
2. **La regla semántica (medoide cumple R0') queda 0,005–0,02 por debajo de R0', con IC que no cruzan 0** en todos los espacios (la menor diferencia es la de ae_d16, −0,005); ninguna tipología supera la regla trivial, coherente con el techo. Los tipos son legibles como "desmonte / no desmonte" casi tan bien como la regla explícita. La excepción es `ae_d3` (−0,043): con d = 3 los clusters mezclan procesos.
3. **El peso importa para la lectura semántica.** Con tipologías ajustadas con peso por tipo (no por superficie), el MCC semántico de k = 12 cae a 0,22 (ae_d16), 0,41 (ae_d3), 0,44 (om) y −0,02 (pca8), mientras que one-hot y ae_d8 se mantienen en 0,62, porque el medoide de un cluster dominado por trayectorias raras no representa la superficie. `mcc_cv` (que no depende del medoide) no cambia: 0,63.
4. **Costura.** Excluir las trayectorias de la costura cambia poco (`u_ratio` ± 0,01–0,02; MCC de las líneas de base ± 0,003): la costura no está dirigiendo estos resultados.
5. **Yungas** (11.777 evaluables, prevalencia 0,179, muy pocos píxeles con desmonte) da lo mismo: R0' 0,670 y MCC semántico 0,63–0,68. Los cocientes salen levemente > 1 (hasta 1,03) porque el ajuste por permutación penaliza más la partición fina (muchas clases con pocos píxeles); no hay que leerlos como que el tipo supera a la secuencia.

**Qué no dice.** Que las tipologías conserven la información de la secuencia no demuestra que describan bien la dinámica: la secuencia ya está limitada por el techo de ESA CCI (0,64), y la compresión hasta 12 tipos casi no cuesta nada porque R0' ya captura casi todo. Es una prueba de que no se pierde la señal de desmonte, no una ventaja de un espacio sobre otro. Periurbano Córdoba no se evaluó (la referencia no es visible en ESA CCI, VE-9).

### Ruido de la CV (2026-10-06)
`p2_desmonte.py --cv-seeds 20` repite la CV espacial con 20 asignaciones distintas de bloques a folds (los mismos folds para la secuencia exacta y para todas las tipologías en cada semilla, así que las diferencias son pareadas). Salida: `p2_desmonte_cv.csv`.

- **Chaco:** el techo da 0,6403 con desvío 0,0006 entre semillas, y el MCC de las tipologías tiene desvío 0,0005–0,0016. La asignación de folds casi no mueve nada. Con k = 12 el orden por MCC es ae_d16 (0,634) > onehot (0,632) > ae_d8 (0,630) > pca8 (0,629) > om (0,627) > ae_d3 (0,624), y **cada diferencia pareada tiene el mismo signo en las 20 semillas** (p. ej. ae_d16 − om = +0,0068, desvío 0,0006). El cociente con el techo va de 0,975 (ae_d3) a 0,990 (ae_d16).
- **Alcance de esa conclusión:** el rango entero entre espacios es de ~0,01 de MCC (1,5 % del techo), menor que el IC del MCC por bloques espaciales (±0,013), y esta CV no incluye el ruido del propio clustering (reinicios del k-medoides y semillas del autoencoder, no medido). Las diferencias son sistemáticas respecto de cómo se reparten los folds, pero no demuestran que un espacio sea mejor: sólo ae_d3 queda claramente por debajo, también en la regla semántica.
- **Yungas:** el desvío de la CV es 0,013 (pocos píxeles con desmonte) y todos los cocientes con el techo dan 1,02–1,04: la tabla por secuencia exacta sobreajusta con tan pocos píxeles y el techo no es una referencia confiable ahí.

## Interpretación de los tipos (2026-10-06)
Código: `scripts/clustering/p2_interpretacion.py`. Tablas por cluster en `data/autoencoder_v3/p2/p2_clusters_<espacio>_k12.csv` (superficie, etiqueta automática, secuencia dominante y su fracción, año del primer cambio, % de costura, tres trayectorias principales, medoide) y cruces en `p2_cruce_om_ae_d8_k12.csv`. Se interpretan om y ae_d8 con k = 12, peso por superficie.

**Los dos espacios encuentran los mismos procesos grandes** (porcentajes de la superficie dinámica):
- deforestación F»A temprana (~2001–2003: 17 %) y más tardía (~2005–2008: 7–8 %);
- degradación F»Sh en dos olas (~2004: 12–16 %; ~2008–2009: 12–14 %);
- anegamiento F»Wt (7 %), recuperación Sh»F ~2016 (8–9 %), revegetación B»Sp ~2019 (7–8 %, la Puna) y G»Sp (5–6 %).
El cruce om → ae_d8 es nítido: 7 de los 12 clusters de OM mandan ≥ 76 % de su superficie a un solo cluster de z (rango 0,36–0,94).

**Dónde difieren.** z separa más por año: el F»Sh de ~2019 tiene cluster propio (8 %), la recuperación Sh»F se parte en ~2004 y ~2017, y aparece un cluster A»F de abandono agrícola (~2010, 6 %). OM agrupa en cambio un cluster "F estable" (13 % de la superficie dinámica), formado por trayectorias con cambios tardíos desde F (primer cambio mediano 2016), y un cluster "G estable" (3 %); tiene además los dos clusters chicos A»U (urbanización, 1,4 %) y Wa (1,1 %), que z no separa. Es coherente con la NMI del primer pase: z es más sensible al año y OM al proceso.

**Dos advertencias.**
1. **Las etiquetas automáticas engañan en los clusters de mezcla.** `auto_label` usa la secuencia modal: si el cluster mezcla cambios en años distintos, la moda queda constante y la etiqueta dice "estable" (OM 8: la secuencia más frecuente sólo cubre el 33 % de su superficie). Para esos clusters hay que leer las trayectorias principales y no la etiqueta.
2. **La costura sigue visible.** En los dos espacios el cluster Sh»F ~2016 tiene 27–30 % de su superficie en trayectorias de la costura (y el A»F de z, 17 %; el Wa de OM, 17 %). Es una recuperación forestal real mezclada con la reclasificación del producto, y no se puede separar con estos datos.

## Mapas (2026-10-06)
Código: `scripts/viz/p2_mapa_pais.py`. Cada píxel de Argentina (36.084.989) toma el tipo de su trayectoria; todas las trayectorias están en el universo (comprobado), las constantes quedan como clase aparte. Salidas: `docs/autoencoder_v3/mapas/{om,ae_d8}_k12_{pais,chaco}.png` y los rásters de etiquetas en `data/autoencoder_v3/p2/mapas/` (uint8, grilla completa, 11 MB en total). El mapa nacional agrega por bloques de 6 × 6 píxeles (gana el cluster dinámico más frecuente), el del Chaco está a resolución completa. Lo que se ve: la Puna dominada por B»Sp, los lotes de deforestación y degradación del Chaco (F»Sh, F»A), el mosaico F»A de la pampa y el litoral, las áreas anegadas del Paraná y G»Sp en Patagonia. Falta incorporar esta tipología al visor.

## Otros métodos de clustering (2026-10-06)

### Qué se compara
Se probaron **seis métodos de clustering** (filas) sobre **seis representaciones** de las mismas 4.545 trayectorias dinámicas (columnas). Cada celda de cada tabla es un clustering distinto: un método aplicado a una representación, con 12 clusters (k = 12; ver "Elección de k" abajo: es un valor razonable pero no fue elegido por un criterio), pesando cada trayectoria por su superficie. "—" significa que el método no se puede aplicar a esa representación. Código: `scripts/clustering/p2_metodos.py` (clustering) y `scripts/clustering/p2_tablas_metodos.py` (genera las tablas de abajo).

**Columnas: representaciones**
- **OM**: la distancia de Optimal Matching entre secuencias (no hay coordenadas, sólo distancias).
- **one-hot**: cada año codificado como un vector de 10 estados (230 números por trayectoria).
- **PCA-8**: ese one-hot proyectado a 8 dimensiones.
- **AE d=3 / d=8 / d=16**: el vector z que el autoencoder asigna a cada trayectoria, con 3, 8 y 16 dimensiones.

**Filas: métodos**
- **kmedoides**: k-medoides ponderado (el de los pases anteriores).
- **kmeans**: k-means ponderado. Necesita coordenadas, por eso no aplica a OM.
- **ward**: jerárquico de Ward ponderado. También necesita coordenadas.
- **avg**: jerárquico de enlace promedio ponderado. Sólo necesita distancias, aplica a todo.
- **gmm**: mezcla gaussiana. No admite pesos: se ajustó a una muestra de píxeles proporcional a la superficie. No se corrió con OM ni con one-hot (230 dimensiones).
- **hdbscan**: por densidad, no fija k: se eligió la configuración cuyo n.º de clusters se acercara a 12. No admite pesos: agrupó los tipos sin ponderar. Deja trayectorias sin cluster (ruido).

**Cómo leer las tablas.** En las métricas de "más es mejor" la **negrita** marca el mejor valor de cada columna entre los métodos con partición comparable (todos menos hdbscan). Las métricas se agrupan en cinco preguntas: ¿los clusters son buenos? (A), ¿se repiten si cambian los datos? (B), ¿qué agrupan? (C), ¿qué tan balanceados son? (D) y ¿conservan la señal de desmonte? (E).

---

### Elección de k
**k = 12 no se eligió con ningún criterio**: es el nivel intermedio de las tres granularidades fijas del plan (6 / 12 / 24) y lo usé como cifra de cabecera. Para elegir con datos se hizo un barrido de k = 2 a 60 con k-medoides, k-means y Ward en los seis espacios (`p2_metodos.py --ks … --tag _barrido`, con el control con desmonte en Chaco; avg, GMM y hdbscan quedaron afuera por lo visto más abajo). Las tablas siguientes son el promedio entre esos tres métodos y los espacios; se regeneran con `p2_tablas_metodos.py --barrido`.

|   k |   pseudo-R² |   pseudo-F (miles) |   ASW |   estabilidad |   NMI estados |   NMI año |   mayor cluster |   U ratio |   MCC/techo |
|----:|------------:|-------------------:|------:|--------------:|--------------:|----------:|----------------:|----------:|------------:|
|   2 |       0,183 |            379,089 | 0,296 |         0,9   |         0,343 |     0,022 |           0,647 |     0,908 |       0,965 |
|   3 |       0,304 |            375,754 | 0,351 |         0,915 |         0,412 |     0,043 |           0,469 |     0,91  |       0,965 |
|   4 |       0,416 |            409,16  | 0,425 |         0,862 |         0,494 |     0,05  |           0,377 |     0,912 |       0,967 |
|   5 |       0,471 |            389,471 | 0,416 |         0,874 |         0,509 |     0,092 |           0,312 |     0,918 |       0,97  |
|   6 |       0,543 |            410,375 | 0,449 |         0,883 |         0,552 |     0,102 |           0,291 |     0,921 |       0,971 |
|   8 |       0,595 |            364,3   | 0,385 |         0,793 |         0,581 |     0,159 |           0,232 |     0,939 |       0,979 |
|  10 |       0,643 |            346,179 | 0,383 |         0,825 |         0,601 |     0,197 |           0,195 |     0,947 |       0,982 |
|  12 |       0,681 |            332,484 | 0,375 |         0,81  |         0,622 |     0,226 |           0,176 |     0,957 |       0,985 |
|  16 |       0,729 |            306,337 | 0,384 |         0,803 |         0,652 |     0,269 |           0,143 |     0,965 |       0,989 |
|  20 |       0,757 |            280,56  | 0,368 |         0,789 |         0,663 |     0,305 |           0,127 |     0,969 |       0,99  |
|  24 |       0,785 |            269,962 | 0,377 |         0,806 |         0,673 |     0,334 |           0,113 |     0,973 |       0,99  |
|  32 |       0,82  |            249,539 | 0,392 |         0,8   |         0,686 |     0,381 |           0,088 |     0,977 |       0,991 |
|  40 |       0,842 |            231,874 | 0,418 |         0,814 |         0,698 |     0,408 |           0,083 |     0,98  |       0,992 |
|  60 |       0,879 |            208,504 | 0,466 |         0,811 |         0,711 |     0,461 |           0,065 |     0,985 |       0,996 |

**Qué dice cada criterio**
- **No hay un k natural.** La calidad (pseudo-R²) crece sin parar con k, sin codo claro: su codo es k = 6 en escala logarítmica y k = 12 en escala lineal. Las dinámicas forman un continuo y cada k es un nivel de resolución, no una estructura que se descubre.
- **Criterios internos: k chico (4–6).** El ASW y el pseudo-F tienen su máximo en k = 4–6 en one-hot, PCA-8 y AE d=3 (ASW 0,42–0,45 promedio); en AE d=8 y d=16 el máximo no es estable (entre 2 y 40, según el método). Después de k = 6 el ASW baja a 0,37–0,39 y el pseudo-F cae de forma continua. Con pocos clusters también es máxima la estabilidad (0,86–0,92 en k ≤ 6) pero la mitad de la superficie cae en 1 o 2 clusters (el mayor es 29–65 %).
- **Criterios externos: se saturan entre k = 10 y 16.** La información retenida (U) y el cociente con el techo de MCC suben rápido hasta k ≈ 12 y después casi no (U 0,957 → 0,965 → 0,973 de k = 12 a 16 a 24; MCC/techo 0,985 → 0,989 → 0,990). Su codo cae en k = 16. Con una regla fijada *después* de ver las curvas (MCC/techo ≥ 0,98 y U ≥ 0,95), el menor k que la cumple es 12 en one-hot y PCA-8 con k-means y Ward, 12–20 en AE d=3 y d=8, 8 en AE d=16 y 16 con k-medoides en casi todos los espacios.
- **Criterios descriptivos: siguen subiendo.** La NMI con el proceso sube de 0,55 (k = 6) a 0,67 (k = 24) y a 0,71 (k = 60); la NMI con el año del primer cambio, de 0,10 (k = 6) a 0,23 (k = 12), 0,33 (k = 24) y 0,46 (k = 60). Pedir que los clusters distingan el momento de cambio exige k alto.
- **Estabilidad:** en k-means y k-medoides baja con k (k-means 0,92 en k = 6; 0,78 en k = 24); Ward, en cambio, se recupera para k ≥ 20 (0,90–0,96).

**Decisión.** No hay un k óptimo único; depende del objetivo:
- **k ≈ 6** para una descripción gruesa: es donde la separación interna es máxima (ASW, pseudo-F) y donde cada cluster todavía es interpretable de un vistazo, a costa de perder resolución temporal y de retener sólo el 97 % del techo de MCC.
- **k ≈ 12–16** para una tipología que conserve la señal de desmonte (≥ 98 % del techo, U ≥ 0,95) y empiece a distinguir el momento del cambio. Es el rango donde se saturan los criterios externos. Los umbrales de la regla los fijé mirando las curvas, así que es una regla razonable pero no a priori.
- **k ≥ 24** sólo si se necesita resolver el momento del cambio; lo externo casi no mejora.
Por eso las tablas de abajo, a k = 12, caen dentro del rango recomendado (y el k = 6 y 24 se pueden regenerar con `p2_tablas_metodos.py --k 6`). Las conclusiones de cada tabla no dependen del k elegido en el rango 6–24: el orden de los métodos (Ward ≈ k-means ≈ k-medoides, por encima de avg y hdbscan) se mantiene en k = 6, 12 y 24 en las comparaciones por k de la síntesis; sólo GMM iguala a los mejores en el control externo con k = 24.

**Pendiente.** Los criterios se aplicaron a k-medoides, k-means y Ward promediados; no se hizo un barrido de k para avg, GMM y hdbscan, ni se probó un criterio basado en brecha (gap statistic) ni en estabilidad entre semillas del autoencoder.

---

### A. Calidad de la partición
Se mide con la distancia OM entre las trayectorias, para todos los espacios por igual. **Ojo:** como la vara es OM, favorece a quien agrupa con OM (circular); aun así, no se nota en los resultados (one-hot y PCA-8 igualan o superan a OM).

#### A1. Pseudo-R² (más es mejor)
Fracción de la dispersión total entre trayectorias que explican los clusters (0 = nada, 1 = todo).

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | **0,72** | **0,72** | **0,73** | 0,66 | 0,68 | 0,64 |
| kmeans | — | 0,72 | 0,73 | **0,66** | **0,69** | **0,70** |
| ward | — | 0,71 | 0,71 | 0,64 | 0,67 | 0,67 |
| avg | 0,64 | 0,65 | 0,66 | 0,55 | 0,52 | 0,62 |
| gmm | — | — | 0,63 | 0,60 | 0,59 | 0,56 |
| hdbscan | 0,01 | 0,03 | 0,76 | 0,03 | 0,20 | 0,72 |

**Resumen.** k-means, k-medoides y Ward quedan juntos (promedio 0,70 / 0,69 / 0,68); avg y GMM, unos 0,08–0,09 por debajo (0,61 / 0,60). Por representación, one-hot, PCA-8 y OM (≈0,72) quedan por encima del autoencoder (0,64–0,70).

**Lectura.**
- Con cualquier método válido, las tres representaciones "directas" (OM, one-hot, PCA-8) dan clusters que explican entre el 71 y el 73 % de la dispersión, y las del autoencoder entre el 64 y el 70 %. La diferencia entre representaciones es más grande que la que hay entre k-medoides, k-means y Ward.
- Entre los z del autoencoder no hay orden por dimensión: d=3 y d=16 dan resultados parecidos y d=8 queda en medio.
- avg pierde sobre todo en el autoencoder (0,52–0,62), porque encadena (ver D1).
- **Los valores de hdbscan engañan.** El 0,76 de PCA-8 y el 0,72 de AE d=16 se calculan *sin* los píxeles de ruido (15 % y 69 % de la superficie), y los 0,01–0,03 de OM, one-hot y AE d=3 son un cluster que contiene casi todo (ver D). No se pueden comparar con los demás.
- Al subir k, todos suben (k = 6 → 24: kmedoides 0,53 → 0,78); no hay un k donde un método se despegue.

#### A2. ASW corregida (más es mejor)
Silhouette promedio: qué tan cerca está cada trayectoria de las de su cluster comparada con el cluster vecino (−1 a 1; cerca de 0 = los clusters se superponen).

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,48 | 0,47 | 0,44 | 0,34 | **0,38** | 0,32 |
| kmeans | — | 0,45 | 0,44 | 0,34 | 0,36 | **0,41** |
| ward | — | 0,41 | 0,43 | **0,35** | 0,33 | 0,33 |
| avg | **0,49** | **0,52** | **0,52** | 0,25 | 0,21 | 0,26 |
| gmm | — | — | 0,35 | 0,28 | 0,25 | 0,24 |
| hdbscan | -0,43 | -0,22 | 0,55 | -0,19 | 0,25 | 0,39 |

**Resumen.** Ningún método pasa de 0,52 (salvo hdbscan en PCA-8, con 15 % de ruido): los clusters no están bien separados. El mejor valor es avg con OM / one-hot / PCA-8 (0,49–0,52) y el peor, avg con el autoencoder (0,21–0,26).

**Lectura.**
- Valores de 0,3–0,5 indican que las dinámicas forman un continuo y que cada corte es una partición de conveniencia, no una separación natural. Esto vale para todos los métodos.
- avg es el método más sensible a la representación: el mejor con OM, one-hot y PCA-8, el peor con el autoencoder. kmedoides y kmeans son más parejos (0,33–0,48).
- Las representaciones directas separan mejor que el autoencoder (OM 0,48 y one-hot 0,45 contra 0,34–0,36 del AE, promedio de los tres métodos válidos).
- hdbscan con OM da −0,43: una silhouette negativa porque casi todo cae en un solo cluster.

---

### B. Estabilidad

#### B1. Estabilidad con submuestras (ARI, más es mejor)
Se repite el clustering con el 80 % de los píxeles (20 veces; 10 para GMM) y se compara con el clustering completo con el ARI ponderado (1 = idéntico, 0 = al azar), sobre las trayectorias presentes en cada submuestra.

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,82 | 0,78 | 0,72 | 0,78 | 0,74 | 0,79 |
| kmeans | — | 0,84 | 0,92 | 0,80 | 0,82 | 0,78 |
| ward | — | 0,89 | 0,88 | 0,79 | 0,83 | 0,93 |
| avg | **0,96** | **0,98** | **0,97** | **1,00** | **0,99** | **0,97** |
| gmm | — | — | 0,73 | 0,68 | 0,73 | 0,64 |
| hdbscan | 0,03 | 0,41 | 0,51 | 0,02 | 0,03 | 0,62 |

**Resumen.** avg es casi perfecto (0,96–1,00), seguido de Ward (0,79–0,93), k-means (0,78–0,92), k-medoides (0,72–0,82), GMM (0,64–0,73) y hdbscan (0,02–0,62).

**Lectura.**
- **La estabilidad de avg no es un mérito.** Es estable porque un cluster gigante absorbe un tercio de la superficie (D1) y los demás cortes son fáciles de reproducir. Hay que leer esta tabla junto con la D1.
- Entre los métodos con clusters balanceados, Ward es el más estable (0,86 promedio, hasta 0,93 con AE d=16 y 0,89 con one-hot), seguido de k-means (0,83) y k-medoides (0,77).
- Con k más alto, Ward se mantiene o mejora (0,90 → 0,93 de k = 6 a 24) y k-medoides cae (0,82 → 0,69): el árbol de Ward es determinista, y k-medoides depende de la inicialización.
- GMM es el menos estable de los válidos (0,64–0,73), en parte porque su ajuste ya incluye el ruido del muestreo de píxeles.
- Por representación no hay un patrón claro: las diferencias (0,79–0,84 de promedio) son menores que entre métodos.
- Esta estabilidad sólo mide el ruido de los datos (submuestras): no incluye el ruido de las semillas del autoencoder, que en el primer pase era comparable o mayor (ARI entre semillas 0,38–0,71).

---

### C. Qué agrupa cada clustering
Dos NMI ponderadas (0 = el clustering no tiene que ver con eso, 1 = lo reproduce): contra la **secuencia de estados** sin años (p. ej. F→A, A→Sh→A: el *proceso*) y contra el **año del primer cambio** (el *momento*).

#### C1. NMI con la secuencia de estados (el proceso)
| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,61 | 0,63 | 0,61 | 0,64 | 0,58 | 0,60 |
| kmeans | — | 0,64 | 0,61 | 0,63 | 0,59 | 0,59 |
| ward | — | 0,67 | 0,62 | 0,66 | 0,60 | 0,66 |
| avg | 0,59 | 0,59 | 0,61 | 0,69 | 0,54 | 0,61 |
| gmm | — | — | 0,71 | 0,67 | 0,63 | 0,69 |
| hdbscan | 0,17 | 0,26 | 0,57 | 0,26 | 0,43 | 0,53 |

#### C2. NMI con el año del primer cambio (el momento)
| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,18 | 0,23 | 0,22 | 0,20 | 0,26 | 0,29 |
| kmeans | — | 0,23 | 0,21 | 0,19 | 0,25 | 0,26 |
| ward | — | 0,22 | 0,21 | 0,20 | 0,26 | 0,24 |
| avg | 0,06 | 0,06 | 0,10 | 0,08 | 0,14 | 0,12 |
| gmm | — | — | 0,18 | 0,19 | 0,23 | 0,24 |
| hdbscan | 0,03 | 0,04 | 0,20 | 0,05 | 0,05 | 0,11 |

**Resumen.** Todos los métodos válidos agrupan sobre todo por proceso (NMI 0,55–0,71) y poco por momento (0,06–0,29). El momento lo capturan más el autoencoder (AE d=8 y d=16: 0,24–0,29) y menos OM (0,18 con k-medoides) y avg (0,06–0,14).

**Lectura.**
- **Proceso (C1):** GMM tiene los valores más altos (0,63–0,71), y avg con AE d=3 (0,69); el resto está entre 0,55 y 0,67 sin diferencias de peso. Que la NMI con el proceso sea alta en todos es esperable: las secuencias sólo difieren en estados y años.
- **Momento (C2):** hay dos efectos separados. El de la *representación* es el mayor: dentro de k-medoides, OM 0,18, one-hot 0,23, PCA-8 0,22, AE d=3 0,20, AE d=8 0,26 y AE d=16 0,29. El de la *método* es la separación entre avg (0,06–0,14, no separa por momento) y los demás (0,18–0,29).
- Lectura conjunta: para obtener tipos que distingan *cuándo* cambió el uso del suelo conviene el autoencoder de dimensión alta; para tipos que distingan sólo *qué* proceso, da igual la representación.
- hdbscan tiene NMI bajas (0,03–0,2 con el momento, 0,17–0,57 con el proceso), consistente con que no arma una partición real.

---

### D. Balance de los clusters

#### D1. Tamaño del cluster más grande (fracción de la superficie dinámica; menos es más equilibrado)
| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,22 | 0,15 | 0,13 | 0,18 | 0,16 | 0,15 |
| kmeans | — | 0,20 | 0,17 | 0,18 | 0,15 | 0,17 |
| ward | — | 0,20 | 0,16 | 0,18 | 0,17 | 0,15 |
| avg | 0,34 | 0,34 | 0,33 | 0,41 | 0,36 | 0,34 |
| gmm | — | — | 0,15 | 0,18 | 0,20 | 0,21 |
| hdbscan | 0,98 | 0,95 | 0,23 | 0,89 | 0,73 | 0,08 |

#### D2. Píxeles sin cluster (ruido; fracción de la superficie, sólo hdbscan)
| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| kmeans | — | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| ward | — | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| avg | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| gmm | — | — | 0,00 | 0,00 | 0,00 | 0,00 |
| hdbscan | 0,01 | 0,00 | 0,15 | 0,09 | 0,13 | 0,69 |

#### D3. Clusters realmente obtenidos (objetivo: 12)
| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | 12 | 12 | 12 | 12 | 12 | 12 |
| kmeans | — | 12 | 12 | 12 | 12 | 12 |
| ward | — | 12 | 12 | 12 | 12 | 12 |
| avg | 12 | 12 | 12 | 12 | 12 | 12 |
| gmm | — | — | 12 | 12 | 12 | 12 |
| hdbscan | 17 | 12 | 14 | 5 | 5 | 20 |

**Resumen.** Con k-means, k-medoides, Ward y GMM el cluster mayor ronda el 13–22 % y no hay ruido. Con avg sube a 33–41 %. hdbscan es un caso aparte: o un cluster casi único (OM 98 %, one-hot 95 %, AE d=3 89 %, AE d=8 73 %), o muchísimo ruido (AE d=16: 69 %), con 5 a 20 clusters en lugar de 12.

**Lectura.**
- **avg** encadena: siempre un cluster de ~1/3 de la superficie, en las seis representaciones. Es la explicación de su estabilidad y de su baja sensibilidad al momento.
- **hdbscan no entrega una partición utilizable en este universo**, ni ajustando min_cluster_size, min_samples y eom/leaf (se probaron 36 configuraciones por espacio y k). No es un fallo de implementación: es la estructura de los datos (un continuo sin zonas de densidad separadas). Lo mismo había concluido la auditoría para v2.
- Los métodos válidos reparten bien, sin que ninguno gane claramente. PCA-8 es el que arma clusters más parejos con k-medoides (12,8 %).

---

### E. Control externo con la referencia de desmonte (Chaco)
Cada píxel de la referencia toma el cluster de su trayectoria (las constantes van a clases propias).

#### E1. Información retenida (U ratio, más es mejor)
Cuánto de la información sobre desmonte de la secuencia exacta conserva el clustering (1 = toda).

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | **0,945** | 0,955 | **0,964** | 0,956 | 0,944 | 0,958 |
| kmeans | — | 0,957 | 0,952 | 0,947 | **0,958** | 0,956 |
| ward | — | **0,959** | 0,962 | **0,957** | 0,956 | **0,971** |
| avg | 0,914 | 0,915 | 0,926 | 0,914 | 0,940 | 0,924 |
| gmm | — | — | 0,960 | 0,944 | 0,949 | 0,961 |
| hdbscan | 0,909 | 0,913 | 0,935 | 0,917 | 0,920 | 0,931 |

#### E2. MCC de la tabla por cluster con validación cruzada espacial, dividido por el techo (más es mejor)
El techo es el MCC de la tabla por secuencia exacta (0,640 en Chaco). 1 = el clustering conserva todo lo que la secuencia permite.

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | **0,985** | 0,990 | 0,985 | 0,972 | 0,981 | 0,989 |
| kmeans | — | 0,991 | 0,983 | 0,977 | 0,981 | 0,988 |
| ward | — | **0,991** | **0,985** | **0,978** | **0,989** | **0,992** |
| avg | 0,969 | 0,970 | 0,977 | 0,969 | 0,982 | 0,975 |
| gmm | — | — | 0,985 | 0,972 | 0,978 | 0,988 |
| hdbscan | 0,966 | 0,968 | 0,975 | 0,970 | 0,970 | 0,976 |

#### E3. Regla semántica menos R0' (más cerca de 0 es mejor)
Un cluster se marca "desmonte" si su medoide cumple R0'. Se resta el MCC de R0' (0,638): negativo = peor que la regla trivial.

| método | OM | one-hot | PCA-8 | AE d=3 | AE d=8 | AE d=16 |
|---|---:|---:|---:|---:|---:|---:|
| kmedoides | **-0,013** | -0,013 | -0,023 | -0,053 | -0,017 | **-0,005** |
| kmeans | — | -0,008 | -0,049 | -0,058 | -0,015 | -0,011 |
| ward | — | **-0,008** | -0,023 | -0,060 | **-0,009** | -0,007 |
| avg | -0,023 | -0,023 | **-0,018** | -0,025 | -0,013 | -0,018 |
| gmm | — | — | -0,079 | **-0,018** | -0,017 | -0,008 |
| hdbscan | -0,021 | -0,019 | -0,121 | -0,018 | -0,018 | -0,340 |

**Resumen.** Todos los métodos (menos hdbscan) conservan el 91–97 % de la información y el 97–99 % del techo de MCC. Ward con one-hot o AE d=16 y k-means con one-hot son los mejores (U 0,96–0,97; MCC 0,99). Ninguna combinación supera a R0': la regla semántica queda entre −0,005 y −0,06, salvo casos degenerados.

**Lectura.**
- **El control externo casi no discrimina entre métodos válidos:** el rango de E2 entre k-medoides, k-means, Ward y GMM es 0,972–0,992 (≈ 0,013 de MCC), menor que el IC del MCC por bloques espaciales (±0,013). Con k=24 todos llegan a 0,990–0,992.
- avg retiene menos (U 0,91–0,94; MCC 0,97–0,98) y hdbscan también (0,91–0,94 y 0,97–0,98): coherente con sus clusters desbalanceados o con ruido.
- **E3 muestra que el autoencoder de d=3 es el peor** con k-medoides, k-means y Ward (−0,05 a −0,06): con 3 dimensiones, los clusters mezclan procesos. Con avg y GMM la caída es menor (−0,018 a −0,025). El mejor de los válidos es AE d=16 con Ward (−0,007), con k-means (−0,011) o con GMM (−0,008).
- Los valores extremos de E3 son de hdbscan (AE d=16: −0,34, porque el 69 % ruidoso no se marca como desmonte; PCA-8: −0,12) y de GMM con PCA-8 (−0,08).
- E3 depende de que el medoide represente al cluster: puede cambiar mucho entre métodos sin que cambie la información retenida (E1 y E2). Por eso E1 y E2 son más confiables como medida de "se conserva la señal".

---

### Síntesis

**Por método (k = 12)**

| método | A. calidad | B. estabilidad | C. qué agrupa | D. balance | E. control externo | veredicto |
|---|---|---|---|---|---|---|
| ward | alta (0,68) | alta (0,86) | proceso 0,64 · momento 0,22 | bueno (17 %) | el mejor (U 0,96; MCC 0,987) | **candidato principal**: determinista, estable, da k = 6/12/24 de un solo árbol |
| kmeans | alta (0,70) | media-alta (0,83) | proceso 0,61 · momento 0,23 | bueno (17 %) | muy bueno (U 0,95; MCC 0,984) | equivalente a k-medoides; el más simple |
| kmedoides | alta (0,69) | media (0,77, cae con k) | proceso 0,61 · momento 0,23 | bueno (17 %) | muy bueno (U 0,95; MCC 0,984) | usa cualquier distancia (también OM), medoides interpretables |
| gmm | media (0,60) | baja (0,70) | proceso 0,68 · momento 0,21 | bueno (19 %) | bueno (U 0,95; MCC 0,981) | no justifica su costo |
| avg | media (0,61) | muy alta (0,98), por encadenar | proceso 0,61 · momento 0,09 | **malo** (35 %) | más bajo (U 0,92; MCC 0,973) | descartable salvo como contraste |
| hdbscan | no interpretable | muy baja (0,27) | — | **inservible** (1 cluster o 69 % ruido) | el más bajo (U 0,92; MCC 0,971) | descartar en este universo |

**Por representación (promedio de k-medoides, k-means y Ward, k = 12)**

|                        |    om |   onehot |   pca8 |   ae_d3 |   ae_d8 |   ae_d16 |
|:-----------------------|------:|---------:|-------:|--------:|--------:|---------:|
| pseudo_r2              | 0,717 |    0,718 |  0,722 |   0,652 |   0,68  |    0,669 |
| asw_corr               | 0,481 |    0,446 |  0,433 |   0,344 |   0,356 |    0,354 |
| estabilidad_ari_80     | 0,822 |    0,838 |  0,838 |   0,789 |   0,797 |    0,834 |
| nmi_estados            | 0,614 |    0,647 |  0,612 |   0,641 |   0,588 |    0,617 |
| nmi_anio_primer_cambio | 0,179 |    0,224 |  0,21  |   0,194 |   0,255 |    0,264 |
| u_ratio                | 0,945 |    0,957 |  0,959 |   0,953 |   0,953 |    0,961 |
| mcc_cv_ratio_techo     | 0,985 |    0,991 |  0,984 |   0,976 |   0,984 |    0,99  |

Las filas, en orden: pseudo-R², ASW, estabilidad, NMI con el proceso, NMI con el momento, U ratio y MCC/techo. Las representaciones directas (OM, one-hot, PCA-8) dan mejor calidad y separación (A) que el autoencoder; el autoencoder de d alta (8 y 16) separa más por momento (C2: 0,26–0,29 contra 0,18–0,22); en el control externo (E) todos quedan entre 0,976 y 0,991 del techo, con AE d=3 en el piso.

**Cómo cambian los métodos con k** (promedio entre representaciones)

pseudo-R²

|    |   kmedoides |   kmeans |   ward |   avg |   gmm |   hdbscan |
|---:|------------:|---------:|-------:|------:|------:|----------:|
|  6 |       0,529 |    0,54  |  0,538 | 0,448 | 0,459 |     0,203 |
| 12 |       0,691 |    0,698 |  0,681 | 0,605 | 0,595 |     0,29  |
| 24 |       0,778 |    0,791 |  0,781 | 0,675 | 0,72  |     0,706 |

estabilidad

|    |   kmedoides |   kmeans |   ward |   avg |   gmm |   hdbscan |
|---:|------------:|---------:|-------:|------:|------:|----------:|
|  6 |       0,822 |    0,891 |  0,897 | 0,978 | 0,635 |     0,196 |
| 12 |       0,772 |    0,831 |  0,864 | 0,976 | 0,696 |     0,271 |
| 24 |       0,686 |    0,793 |  0,927 | 0,971 | 0,729 |     0,395 |


### Conclusiones y límites
1. **No hay un ganador claro entre k-medoides, k-means y Ward**: dan calidad, contenido y control externo casi iguales (ARI entre ellos 0,71–0,76 con el mismo espacio). Ward es el más estable y determinista, y por eso es el candidato práctico.
2. **El método importa tanto como la representación**: el ARI entre métodos con la misma representación es 0,44–0,76 (k = 12), del mismo orden que el ARI entre representaciones con el mismo método (0,5–0,7, primer pase). Cualquier lectura de "qué agrupa z frente a OM" debe tener ese margen.
3. **El control con desmonte no sirve para elegir método**: todos los válidos retienen el 97–99 % del techo.
4. **Límites.** Pseudo-R² y ASW usan OM como vara (circular). GMM y hdbscan no admiten pesos y lo resuelven de formas distintas (muestreo / sin ponderar), así que no son estrictamente comparables con los demás. La estabilidad sólo se mide sobre las trayectorias presentes en cada submuestra. Los k-medoides de este pase difieren levemente de los de `p2_metricas.csv` (otra semilla y protocolo de estabilidad). No se midió el ruido de los reinicios dentro de cada método. Los jerárquicos son implementaciones propias (verificadas contra scipy).
