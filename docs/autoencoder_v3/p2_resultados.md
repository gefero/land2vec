# P2: descripción de las dinámicas (primer pase)

**Fecha:** 2026-10-05 (primer pase) y 2026-10-06 (control externo con desmonte, sección final). Ejecuta plan §4.2; los mapas siguen pendientes. Código: `scripts/clustering/p2_tipologias.py`. Salidas: `data/autoencoder_v3/p2/` (`p2_metricas.csv`, `p2_acuerdo.csv`, `p2_labels.csv`).

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
- Mapas del país: no hecho.
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
1. **La compresión en tipos casi no pierde información sobre desmonte.** Con 12 clusters dinámicos (más las 9 constantes) se conserva el 94–96 % de la información de la secuencia exacta y 97–99 % del techo de MCC (0,64). Con 6 clusters, 91–94 % y 97–98 %. En esto no hay diferencias claras entre espacios: las diferencias (≤ 0,03) son del orden del ruido de la CV, que no se cuantificó para `mcc_cv`.
2. **La regla semántica (medoide cumple R0') queda 0,005–0,02 por debajo de R0', con IC que no cruzan 0** en todos los espacios salvo la diferencia mínima de ae_d16; ninguna tipología supera la regla trivial, coherente con el techo. Los tipos son legibles como "desmonte / no desmonte" casi tan bien como la regla explícita. La excepción es `ae_d3` (−0,043): con d = 3 los clusters mezclan procesos.
3. **El peso importa para la lectura semántica.** Con tipologías ajustadas con peso por tipo (no por superficie), el MCC semántico de k = 12 cae a 0,22–0,44 en ae_d16, om y pca8 (−0,02 en pca8), porque el medoide de un cluster dominado por trayectorias raras no representa la superficie. `mcc_cv` (que no depende del medoide) no cambia: 0,63.
4. **Costura.** Excluir las trayectorias de la costura cambia poco (`u_ratio` ± 0,01–0,02; MCC de las líneas de base ± 0,003): la costura no está dirigiendo estos resultados.
5. **Yungas** (11.777 evaluables, prevalencia 0,179, muy pocos píxeles con desmonte) da lo mismo: R0' 0,670 y MCC semántico 0,63–0,68. Los cocientes salen levemente > 1 (hasta 1,03) porque el ajuste por permutación penaliza más la partición fina (muchas clases con pocos píxeles); no hay que leerlos como que el tipo supera a la secuencia.

**Qué no dice.** Que las tipologías conserven la información de la secuencia no demuestra que describan bien la dinámica: la secuencia ya está limitada por el techo de ESA CCI (0,64), y la compresión hasta 12 tipos casi no cuesta nada porque R0' ya captura casi todo. Es una prueba de que no se pierde la señal de desmonte, no una ventaja de un espacio sobre otro. Periurbano Córdoba no se evaluó (la referencia no es visible en ESA CCI, VE-9). Falta cuantificar el ruido de la CV (`mcc_cv`) para poder afirmar diferencias entre espacios.
