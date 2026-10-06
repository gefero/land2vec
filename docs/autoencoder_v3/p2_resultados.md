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
Código: `scripts/clustering/p2_metodos.py` (y `p2_desmonte.py --labels … --tag …` para el control externo). Salidas: `p2_metricas_metodos.csv`, `p2_acuerdo_metodos.csv`, `p2_labels_metodos.csv.gz`, `p2_desmonte_metodos.csv` y los equivalentes `_hdbscan`. Mismos seis espacios, k = 6/12/24, peso por superficie, universos completo y sin costura.

**Métodos.** `kmedoides` (el de los pases anteriores, vuelto a correr con el protocolo de este script), `avg` (jerárquico de enlace promedio, ponderado por superficie), `ward` (jerárquico de Ward ponderado; sólo espacios con coordenadas), `kmeans` (ponderado, sklearn), `gmm` (covarianza completa; sklearn no pondera, así que se ajusta a una muestra de 20.000 píxeles proporcional a la superficie; pca8 y z) y `hdbscan` (sobre la matriz de distancias, **sin ponderar**, sin k: se busca la configuración cuyo n.º de clusters sea el más cercano al objetivo). Los dos jerárquicos son implementaciones propias (cadena de vecinos + Lance-Williams con masas), verificadas contra scipy: con masas 1 dan el mismo árbol, y con masas enteras coinciden con repetir las filas.

**Resumen (k = 12, universo completo; promedio entre los espacios aplicables y [mín–máx]).** Estabilidad = ARI con submuestras del 80 % de los píxeles, sobre las trayectorias presentes en cada submuestra. U y MCC_cv son del control con desmonte en Chaco.

| método | pseudo-R² (en OM) | ASW corr. | estabilidad | NMI estados | NMI año 1.er cambio | mayor cluster (% sup.) | U ratio | MCC_cv / techo |
|---|---|---|---|---|---|---|---|---|
| kmedoides | 0,69 [0,64–0,73] | 0,41 | 0,77 [0,72–0,82] | 0,61 | 0,23 | 17 % | 0,954 | 0,984 |
| kmeans | 0,70 [0,66–0,73] | 0,40 | 0,83 [0,78–0,92] | 0,61 | 0,23 | 17 % | 0,954 | 0,984 |
| ward | 0,68 [0,64–0,71] | 0,37 | 0,86 [0,79–0,93] | 0,64 | 0,22 | 17 % | 0,961 | 0,987 |
| gmm | 0,60 [0,56–0,63] | 0,28 | 0,70 [0,64–0,73] | 0,68 | 0,21 | 19 % | 0,953 | 0,981 |
| avg | 0,61 [0,52–0,66] | 0,38 | 0,98 [0,96–1,00] | 0,61 | 0,09 | **35 %** | 0,922 | 0,973 |
| hdbscan | 0,29 [0,01–0,76] | 0,06 | 0,27 [0,02–0,62] | 0,37 | 0,08 | 64 % | 0,921 | 0,971 |

**Lectura**
1. **k-medoides, k-means ponderado y Ward dan tipologías parecidas** (ARI entre ellos 0,71–0,76 en k = 12, para el mismo espacio), con calidad, NMI y control externo casi iguales. Ward es el más estable (0,86) y retiene lo más del techo (0,987), pero la diferencia con k-medoides es ≤ 0,005 de cociente, dentro del ruido que ya se vio en la CV. No hay un motivo para preferir uno por la fidelidad a desmonte; Ward tiene la ventaja práctica de ser determinista y dar la jerarquía k = 6/12/24 de un solo árbol.
2. **El enlace promedio es el más estable (0,98) por una mala razón: encadena.** Un solo cluster concentra el 35 % de la superficie dinámica (17 % en los demás) y la NMI con el año del primer cambio cae a 0,09: no separa por momento. Su estabilidad alta refleja un reparto desbalanceado, no una tipología mejor. Con OM, en cambio, avg tiene la mejor ASW (0,49).
3. **GMM** agrupa algo más por proceso (NMI con estados 0,68, la más alta) pero tiene menor calidad en OM (0,60), menor estabilidad (0,70, incluye el ruido del muestreo) y es el más caro.
4. **HDBSCAN no produce una partición utilizable en este universo.** Para k = 12 devuelve, según el espacio, un cluster gigante con casi todo (OM 98 %, one-hot 95 %) o muchísimo ruido (ae_d16: 69 % de la superficie sin tipo; con k = 24, hasta 84 %), y la estabilidad es 0,02–0,62. Las métricas de calidad excluyen el ruido y quedan sin sentido (pseudo-R² 0,01–0,76). Coincide con lo que había visto la auditoría en v2. En el control con desmonte retiene lo menos (U 0,92) y la regla semántica llega a −0,34 respecto de R0' en ae_d16 porque el ruido queda sin marcar. No se evaluó con ponderación por superficie (sklearn no la admite), que probablemente es parte del problema.
5. **El método importa tanto como el espacio.** El ARI entre métodos con el mismo espacio es 0,44–0,76 (k = 12), del mismo orden que el ARI entre espacios con el mismo método (0,5–0,7, primer pase). Cualquier conclusión sobre "qué agrupa z frente a OM" debe leerse con ese margen.
6. **Sobre el control externo, nada cambia:** todas las combinaciones (salvo HDBSCAN) retienen el 97–99 % del techo de MCC y el 91–97 % de la información; ninguna regla semántica supera a R0'. Con ae_d3 la regla semántica sigue siendo la peor en todos los métodos.

**Límites.** Pseudo-R² y ASW están medidas con OM, que favorece a los métodos que agrupan con OM (circular, como antes). Las diferencias de estabilidad mezclan el algoritmo con el protocolo (sólo se compara en las trayectorias presentes en la submuestra). GMM y HDBSCAN no admiten pesos, y cada uno lo resuelve de una forma distinta (muestreo, sin ponderar), así que no son estrictamente comparables con los otros. Los k-medoides de este pase difieren levemente de los de `p2_metricas.csv` (otra semilla y protocolo de estabilidad). No se midió el ruido de los reinicios dentro de cada método.
