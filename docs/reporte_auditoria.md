# Reporte de auditoría de land2vec v2

**Fecha:** 2026-10-01
**Alcance:** pipeline de embeddings, clustering y validación, visualizaciones, estrategia de validación y resultados del borrador [`docs/paper_metodologia.md`](paper_metodologia.md).
**Proceso:** cinco auditores trabajaron en modo solo lectura y cada uno tuvo un verificador independiente. Un supervisor integró los resultados y pidió dos seguimientos: la referencia de desmonte (evaluador-validacion, REF-01..07) y la tabla canónica de validación externa (analista-resultados, VE-1..9). **Los seguimientos no pasaron por el verificador**: sus hallazgos llevan la marca *[seguimiento, no verificado]*. El redactor revalidó a mano sus cifras críticas (ver §6.3).

> **Convenciones.** La severidad es la que ajustó el verificador. Si varias áreas reportaron el mismo problema, aparece una sola vez, con una severidad y todos los ID. Los hallazgos con veredicto *plausible* llevan **[no confirmado]**. Abreviaturas: "Fina/H" es Fina/HDBSCAN y "Media/G" es Media/GMM (lo mismo para el resto). "const" quiere decir que las secuencias constantes se forzaron a `sin_cambio`. Salvo aviso, el MCC cuenta el −1 como negativo, τ = 0,5 y los IC son bootstrap por bloques de 0,05° (B=999, seed 42).

---

## 1. Resumen ejecutivo y bloqueantes

Las cifras del paper **están bien transcriptas**. Antes de calcular nada nuevo, las cinco áreas reprodujeron lo publicado:
- MCC de Chaco para Fina/H: 0,55298 [0,5383;0,5677] ([desmonte_eval.csv](../models/cluster_v2/desmonte_eval.csv)).
- Pseudo-R² y ASW de `typology_seqdist.csv`; `chosen*.json` contra la tabla de §5.5.
- 400.460 = 302.034 + 98.426 filas de entrenamiento y 798.216 parámetros.

La tabla canónica del seguimiento de analista-resultados (VE-1) reproduce los seis `desmonte_eval*.csv` con |Δ| ≤ 2,2·10⁻¹⁶ (B=999, seed 42, bloques de 0,05° y 0,2°). Es la fuente de todas las cifras de MCC, IC y diferencias pareadas de este reporte. El problema no son errores de copia: **tres conclusiones centrales del paper no se sostienen cuando se usan las comparaciones correctas**.

### Bloqueantes

**(A) La línea de base R0 está mal emparejada y una regla sin modelo supera a las seis tipologías en Chaco.** *(crítica: EXT-1, C1; VE-2, VE-3, REF-07 [seguimiento, no verificado])*
- **El problema.** Para los clusters, "desmonte" incluye `deforestacion` y `degradacion_forestal` ([eval_desmonte.py:60](../scripts/eval_desmonte.py#L60)). En cambio, `_r0_year` solo acepta A/G después de la **última** F ([eval_desmonte.py:157-164](../scripts/eval_desmonte.py#L157)). Además, el docstring ("Existe t con x_t=F y t'>t con x_t' en {A,G}") describe una regla distinta de la que implementa el código.
- **Lo publicado.** En Chaco: R0 = 0,39946 [0,3871;0,4119], P 0,873, R 0,254, prevalencia 0,26038, 4.510 bloques. En yungas: R0 = 0,4616 [0,4148;0,5103], prevalencia 0,1786.
- **Con el mismo positivo.** R0' (F→{A,G,Sh,Sp,B} después de la última F) da **0,638 [0,625;0,650]** con bloques de 0,05° y [0,611;0,664] con 0,2°. F2000∧¬F2022 da 0,637 [0,625;0,650] y "alguna transición", 0,616. La mejor tipología da 0,553.
- **Diferencia pareada.** Fina/H − R0' = **−0,085 [−0,093;−0,077]** (0,05°) y [−0,100;−0,071] (0,2°).
- **Gruesa/H frente a R0 publicado.** +0,015 [−0,000;0,031] con 0,05° y [−0,014;0,045] con 0,2°: **no es significativa** (VE-3).
- **Techo** *[seguimiento, no verificado]*. Una tabla por secuencia exacta, ajustada in-sample, llega a 0,644 (REF-07). R0' queda casi en ese techo.
- **Consecuencia.** No se sostiene "Las seis tipologías superan ampliamente ambas líneas de base triviales" ([paper_metodologia.md:809](paper_metodologia.md#L809)).

**(B) Las trayectorias constantes se asignan por centroide a clusters de cambio y deciden el ranking.** *(crítica: CL-01, EXT-2, H1; REF-03, VE-4, VE-9 [seguimiento, no verificado])*
- **Dónde pasa.** [tune_clustering.py:338-372](../scripts/tune_clustering.py#L338) y [assign_train_clusters.py:120-121](../scripts/assign_train_clusters.py#L120).
- **Peso en Chaco.** En las 5 corridas que no son Fina/H, los píxeles constantes son el 75,5-85,2 % de los FP y el 19,7-31,8 % de los TP.
- **Con constantes forzadas a `sin_cambio`.** Fina/H − Fina/G pasa de +0,019 [0,007;0,031] a **−0,064 [−0,072;−0,058]**: el ganador se invierte (VE-4).
- **Periurbano Córdoba** (VE-9, alta). Todo el MCC positivo de las corridas sale de píxeles constantes: con constantes forzadas, el MCC es ≤ 0 en las 6 corridas (−0,038 a −0,029).
- **Mapas.** Hay píxeles "siempre A" pintados como deforestación. Por eso §7.5 ([paper_metodologia.md:1029](paper_metodologia.md#L1029), "donde no hay clasificación se deja ver la imagen") hoy es falso.

**(C) Hay pocas trayectorias distintas: la generalización fuera de dominio es casi in-sample y las métricas internas están infladas.** *(alta: EMB-01, EMB-02, EMB-1, C2, CL-02, CLU-1, A4)*
- **Tamaño real.** Hay 1.372 secuencias distintas en 400.460 filas de entrenamiento y 1.128 en 107.362 filas dinámicas de evaluación.
- **Solapamiento.** El 99,9 % de las filas de validación tiene una copia exacta en entrenamiento (de ~760 secuencias de validación, solo ~30 son nuevas). En las zonas de evaluación, el 95,2 % de la masa dinámica ya está en entrenamiento.
- **Clusters.** En Fina/H, 56 de los 118 clusters tienen una sola secuencia.
- **Métricas.** La silueta cae de 0,913 a 0,377 sobre secuencias distintas. Una partición trivial, "las 118 secuencias más frecuentes y el resto en −1", da pseudo-R² y ASW de 1,000.

### Otros problemas de severidad alta
- **Estabilidad ARI (CL-06, CLU-2, A6).** `stability_ari` usa `boot_cap = 20000` ([cluster.py:436](../src/land2vec/cluster.py#L436), aplicado en :452), es decir, el 18,6 % de las filas. En cambio, [paper_metodologia.md:497-498](paper_metodologia.md#L497) dice "pares de submuestras del 80 %". Además, `min_cluster_size` no se reescala. El ARI contra la partición completa es ≈0,42 en fina y ≈0,17-0,23 en media.
- **Dilatación de la máscara (REF-01)** *[seguimiento, no verificado]*. Con `dilate=0`, la prevalencia es 0,313 (contra 0,260 con `dilate=1`) y gana Media/G con 0,561 frente a 0,555 de Fina/H. La elección de Fina/H en §5.8.5 no es robusta.
- **Positivos constantes (REF-02)** *[seguimiento, no verificado]*. El 36,7 % de los positivos es constante (75.695 de 206.508; Sh 38,8 %, F 32,1 %, A 27,7 %). Un oráculo de transiciones no puede pasar de una exhaustividad de 0,633 (MCC 0,749).
- **MCC sin −1 (A5, VE-7).** Al excluir el −1, el MCC de Chaco cae a 0,03-0,38 y el de Media/G en yungas, a 0,002. El paper no lo informa.
- **Faltan líneas de base sin embedding (CLU-3).** Con el mismo k, k-medias sobre one-hot supera a z en pseudo-R².

### Conteo de hallazgos de la fase de auditoría (sin fusionar duplicados)

| Área | Crítica | Alta | Media | Baja | Plausibles | Info sin verificar | Refutados |
|---|---:|---:|---:|---:|---:|---:|---:|
| Embeddings | 0 | 1 | 2 | 7 | 1 | 1 | 0 |
| Clustering (código) | 1 | 2 | 5 | 4 | 1 | 1 | 0 |
| Visualizaciones | 0 | 1 | 2 | 8 | 2 | 3 | 0 |
| Estrategia de validación | 1 | 5 | 8 | 2 | 1 | 0 | 0 |
| Resultados | 1 | 4 | 9 | 7 | 2 | 1 | 0 |
| **Total** | **3** | **13** | **26** | **28** | **7** | **6** | **0** |

Los seguimientos suman 16 hallazgos más, ninguno verificado: REF-01..07 (alta 3, media 1, baja 1, info 2) y VE-1..9 (alta 4, media 4, info 1).

---

## 2. Resultados de las validaciones

### 2.1 Embedding interno

| Medida | Valor | IDs |
|---|---:|---|
| Pool de ajuste: filas / secuencias distintas | 400.460 / 1.372 | EMB-01, C2 |
| Filas de validación con copia exacta en entrenamiento | 99,9 % (~30 de ~760 secuencias nuevas) | EMB-01, C2 |
| Masa dinámica de evaluación con secuencia ya vista | 95,2 % (458 tipos nuevos, 5.191 filas) | EMB-02, EMB-1 |
| Macro F1 final | 0,8971 sobre 10 clases = 0,9968 sobre 9 (techo 0,9 porque Nd no aparece) | EMB-03, A1 |
| Diferencia entre réplicas con configuración idéntica | 0,0038 (d8 frente a baseline); 0,0014 (barrido frente a modelo final) | A2 |
| Rango del barrido de d / del secundario | 0,0064 / 0,0036 | A2, EMB-2 |
| Mayor caída de F1 entre épocas | −0,224 (sin_pesos, época 7) | A3 |
| Probe "hubo transición": z / clase mayoritaria / one-hot | 0,9665 / 0,9679 / 0,9926 | EMB-3, M7 |
| Spearman de distancias z frente a OM (one-hot frente a OM) | 0,40 (0,84) | CLU-3 |

**Masa dinámica ya vista en entrenamiento, por zona:**

| Zona | % de dinámicas vistas | % de todas las filas vistas |
|---|---:|---:|
| puna_noa | 92,6 | 99,43 |
| patagonia_estepa | 88,8 | 99,92 |
| periurbano_cordoba | 98,6 | 99,86 |
| ibera | 98,1 | 99,80 |
| delta_parana | 97,9 | 99,87 |
| pampa_nucleo | 89,8 | 99,85 |
| misiones_selva | 98,9 | 99,95 |

**Interpretación.**
- La reconstrucción (exactitud 0,9993) muestra que el modelo codifica bien unos 1.400 patrones. No prueba "dimensionalidad intrínseca baja" ([paper_metodologia.md:289](paper_metodologia.md#L289)) ni que el modelo "generaliza de forma consistentemente buena" ([v2_autoencoder_training.md:442](v2_autoencoder_training.md#L442)).
- Las diferencias del barrido son menores que el ruido entre réplicas: elegir d=8 se justifica por costo, no por desempeño.
- Los picos de las curvas son inestabilidad real: aparecen también en val_loss y val_acc, que se calculan en FP32. No hay `clip_grad_norm` ni scheduler, y `min_lr` no se usa (A3). EMB-04 es solo una imprecisión de la curva de train_loss (FP16, fuera del autocast).
- El problema del probing es que las tareas son triviales y la métrica no está balanceada, no "memorización" (CL-13 **[no confirmado]**, M7).

### 2.2 Clustering interno

| Medida | Publicado | Corregido o control | IDs |
|---|---:|---:|---|
| Silueta de Fina/H (por filas → sobre secuencias distintas) | 0,913 | 0,377 | CLU-1 |
| Silueta de Media/H | 0,794-0,797 | 0,393 | CLU-1 |
| Pseudo-R² de "top-118 secuencias, resto −1" | — | 1,000 (80 % de cobertura) | CLU-1 |
| ASW seqdist de Fina/H (código → equivalente por filas) | 0,683 | 0,880 | EMB-05 |
| Cobertura de Fina/H (secuencias distintas → filas) | 0,307 | 0,902 | EMB-05 |
| ARI de Fina/H / Media/H (publicado → submuestra frente a partición completa) | 0,905 / 0,914 | ≈0,42 / ≈0,17-0,23 | CL-06, CLU-2, A6 |
| Fidelidad de GMM l2/diag según k | — | sube de 0,612 (k=20) a 0,905 (k=120) | CL-03, M4 |

**Pseudo-R² (distancia OM, ponderado por frecuencia): z frente a líneas de base sin embedding**

| k | z, cobertura completa | k-medias sobre one-hot | PAM-OM [no recalculado] |
|---|---:|---:|---:|
| 17-18 | 0,710 (Gruesa/G) | 0,822-0,827 | 0,815 |
| 40 | 0,823 (Media/G) | 0,899 | 0,888 |
| 118-120 | 0,934 (Fina/G) | 0,963-0,964 | 0,958 |

**Interpretación.**
- La regla de selección premia una partición que en la práctica cuenta las secuencias frecuentes.
- El k no lo deciden los datos sino los topes de la grilla (120/40/20) y el mínimo de `min_cluster_size` (250).
- En los hechos, la compuerta ARI ≥ 0,75 no se aplicó a las particiones HDBSCAN elegidas.
- Las particiones triviales tendrían una coherencia espacial del mismo orden que las elegidas **[no confirmado]** (CLU-5; no se recalculó).

### 2.3 Clustering externo (desmonte)

Fuente: tabla canónica de VE-1 *[seguimiento, no verificado; coincide con los CSV publicados]*.

**Chaco** (793.112 píxeles evaluables, prevalencia 0,260; 4.510 bloques de 0,05° y 308 de 0,2°)

| Fila | MCC | P | R | IC 0,05° | IC 0,2° | MCC sin −1 |
|---|---:|---:|---:|---:|---:|---:|
| Fina/H | 0,553 | 0,837 | 0,484 | [0,538;0,568] | [0,523;0,579] | 0,378 |
| Fina/G | 0,534 | 0,611 | 0,723 | [0,521;0,546] | [0,503;0,563] | 0,175 |
| Media/H | 0,450 | 0,582 | 0,613 | [0,435;0,464] | [0,419;0,482] | 0,137 |
| Media/G | 0,514 | 0,533 | 0,830 | [0,498;0,529] | [0,469;0,558] | 0,087 |
| Gruesa/H | 0,415 | 0,567 | 0,567 | [0,400;0,428] | [0,384;0,444] | 0,210 |
| Gruesa/G | 0,466 | 0,518 | 0,768 | [0,450;0,482] | [0,419;0,510] | 0,032 |
| Fina/G const | 0,617 | 0,838 | 0,581 | [0,605;0,630] | [0,590;0,641] | 0,559 |
| Media/H const | 0,542 | 0,833 | 0,471 | [0,527;0,557] | [0,511;0,569] | 0,553 |
| Media/G const | 0,618 | 0,833 | 0,586 | [0,605;0,630] | [0,589;0,642] | 0,533 |
| Gruesa/H const | 0,510 | 0,832 | 0,425 | [0,495;0,525] | [0,481;0,536] | 0,482 |
| Gruesa/G const | 0,577 | 0,832 | 0,524 | [0,564;0,590] | [0,547;0,602] | 0,504 |
| R0 publicado | 0,399 | 0,873 | 0,254 | [0,387;0,412] | [0,372;0,421] | — |
| **R0'** | **0,638** | 0,838 | 0,612 | [0,625;0,650] | [0,611;0,664] | — |
| F2000 ∧ ¬F2022 | 0,637 | 0,836 | 0,613 | [0,625;0,650] | [0,610;0,663] | — |
| Alguna transición | 0,616 | 0,785 | 0,633 | [0,604;0,630] | [0,587;0,645] | — |

En Fina/H, forzar las constantes no cambia nada (Δ = 0 exacto): esa corrida ya no predecía ninguna constante como desmonte. Techos *[seguimiento, no verificado]*: tabla por secuencia (in-sample) 0,644 (REF-07); oráculo de transiciones con R 0,633 y MCC 0,749 (REF-02).

**Diferencias pareadas en Chaco** (misma réplica de bloques)

| Comparación | ΔMCC [IC 0,05°] | IC 0,2° |
|---|---|---|
| Fina/H − R0' | −0,085 [−0,093;−0,077] | [−0,100;−0,071] |
| Media/G const − R0' | −0,020 [−0,024;−0,016] | [−0,025;−0,015] |
| Fina/H − Fina/G | +0,019 [0,007;0,031] | [−0,003;0,042] (cruza 0) |
| Fina/H − Media/G | +0,039 [0,024;0,055] | [0,002;0,077] |
| Fina/H const − Fina/G const | −0,064 [−0,072;−0,058] | [−0,077;−0,053] |
| Gruesa/H − R0 publicado | +0,015 [−0,000;0,031] | [−0,014;0,045] (no significativa) |

EXT-1 y su verificador calcularon además un IC pareado regla − Fina/H con bloques de 0,5°: [0,059;0,104] y [0,061;0,107], respectivamente. Ese tamaño de bloque no está en la tabla canónica, así que esas cifras se atribuyen a EXT-1 y no se presentan como canónicas. La cifra 0,634 (regla con `classify_process`) no se reproduce con ninguna variante (VE-6) y se descarta.

**Yungas** (con IPW; 480 bloques de 0,05°)
- MCC por corrida: Fina/H 0,598 [0,546;0,651], Fina/G 0,605 [0,552;0,656], Media/G 0,638 [0,577;0,693], R0 0,462 y **R0' 0,670 [0,620;0,719]**.
- Media/G − R0' = **−0,032 [−0,089;0,022]**: empate. Entre las corridas sin forzar y con bloques de 0,05°, es, en yungas, el único IC pareado contra R0' que cruza 0. Fina/H − R0' = −0,072 [−0,099;−0,046].
- Con constantes forzadas, Fina/G y Media/G quedan empatadas en 0,655.
- **No hay ganador en yungas.** Por eso no se sostiene que "la selección de granularidad no generaliza entre ecorregiones" ([paper_metodologia.md:852](paper_metodologia.md#L852)).

**Periurbano Córdoba** (VE-9)
- El MCC de las corridas va de −0,033 (Fina/H) a 0,139 (Media/H). Las cinco corridas con MCC positivo tienen IC que cruzan o rozan el 0.
- Con constantes forzadas, las 6 corridas quedan entre −0,038 y −0,029, es decir, ≤ 0.

**Chaco restringido (VE-8)** *[seguimiento, no verificado]*
- Sobre los 595.455 píxeles con F en 2000, R0' da 0,780, frente a 0,755 de la mejor corrida (Media/G) y 0,669 de Fina/H.
- Sobre los 166.664 dinámicos, R0' da 0,400, frente a 0,317 de Fina/G y 0,200 de Fina/H.

**Referencia de desmonte (REF)** *[seguimiento, no verificado]*
- Composición: negativo_limpio 586.604, excluido 437.556, positivo 206.508, control_pre 184.677 y control_post 9.112.
- Positivos constantes: 75.695 de 206.508 (36,7 %). Por estado: Sh 38,8 %, F 32,1 % y A 27,7 %. Las A se concentran en 2001-03 (desfase de fecha), las F en 2019-22 (ESA CCI todavía no registra el cambio) y las Sh en 2009-13 (ambigüedad de esa clase).
- REF-04 (info): los desmontes anteriores a 2001 se tratan bien y nunca quedan como negativo limpio ([eval_desmonte.py:131](../scripts/eval_desmonte.py#L131)). Esto descarta la hipótesis de que los A constantes con polígono previo castiguen injustamente a los modelos.
- REF-05 (media): lo excluido se reparte en 21,3 % fuera de la máscara y 9,4 % de bordes (porcentajes del total).
- REF-06 (baja): el docstring de `label_reference` ([eval_desmonte.py:122-126](../scripts/eval_desmonte.py#L122)) afirma que todo píxel tocado por un polígono cae dentro de la máscara. Es falso: 849 positivos quedan afuera.

**Sensibilidad a la dilatación de la máscara (REF-01)** *[seguimiento, no verificado]*

| | Fina/H | Fina/G | Media/H | Media/G | Gruesa/H | Gruesa/G | R0 | Prevalencia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dilate=1 | 0,553 | 0,534 | 0,450 | 0,514 | 0,415 | 0,466 | 0,399 | 0,260 |
| dilate=0 | 0,555 | 0,555 | 0,470 | 0,561 | 0,433 | 0,511 | 0,401 | 0,313 |

La fila `dilate=0` se calculó con un proxy de la máscara (que coincide al 100 % con `dilate=1`) y no tiene IC pareado.

**Interpretación.**
1. Ninguna tipología supera a una regla trivial aplicada a la secuencia. El aporte de la tipología es la interpretabilidad, no la detección.
2. El ranking entre tipologías depende de tres cosas: cómo se tratan las constantes, el parámetro `dilate` y el tamaño de bloque (con 0,2°, Fina/H − Fina/G deja de ser significativa).
3. La ganancia@10 % está saturada (techo 0,384; observado 0,284-0,336). La AUC-PR OOF está inflada por la integración trapezoidal (CL-11) y por las constantes, así que no sirve como argumento a favor de Media/G.
4. La exhaustividad de 0,830 de Media/G supera el techo de 0,633 solo porque marca constantes.

---

## 3. Evaluación de la estrategia de validación

**Fortalezas**
- La referencia de desmonte es independiente de ESA CCI.
- La lógica de etiquetas es cuidadosa: tiene controles pre y post y excluye los bordes.
- El bootstrap es por bloques espaciales y el IPW de yungas está bien aplicado para los estimadores puntuales (R0 se reproduce en 0,462).
- El análisis en espacio de secuencias (seqdist) complementa al del espacio z.
- Las tablas son reproducibles desde los artefactos.

**Debilidades**
- **Unidad de análisis.** Todo se calcula por fila, pero la unidad real es la secuencia distinta.
- **Líneas de base.** Las que hay son débiles (R0 está mal emparejada) y faltan otras (sin embedding).
- **Selección del clustering.** No hay conjunto reservado: se eligió entre 380 configuraciones sobre las mismas zonas de evaluación (CLU-4), aunque [paper_metodologia.md:109](paper_metodologia.md#L109) dice lo contrario.
- **Validación externa.** Cubre un solo proceso, y Chaco y yungas son in-sample para el encoder (EXT-6, M10).
- **Incertidumbre.** Los IC no son pareados, los bloques son chicos y una permutación i.i.d. se presenta como "espacial".
- **Sensibilidades anunciadas que no existen.** τ = 0,3/0,7 y el corte de longitud (CL-08, EXT-7, M8).
- **Representatividad.** Chaco aporta ~75 % del entrenamiento, ibera y puna_noa suman el 56 % del pool dinámico y no hay métricas por zona (VAL-1).

**Validaciones faltantes, priorizadas**

| # | Validación | Sección del paper |
|---:|---|---|
| 1 | Líneas de base por píxel emparejadas (R0', F2000∧¬F2022, "alguna transición") y diferencias pareadas | §5.8.4-5.8.5 |
| 2 | Separar constantes de dinámicas, declarar el techo de 0,633 y analizar la población "F en 2000" | §5.8.5-5.8.6, §7.5 |
| 3 | Rehacer la estabilidad HDBSCAN reescalando mcs o con bootstrap multinomial sobre secuencias distintas | §5.3(ii), §5.5 |
| 4 | Métricas internas sobre secuencias distintas, con controles "top-k" | §5.3-5.5, §5.7 |
| 5 | Líneas de base sin embedding con el mismo k | §5.7 |
| 6 | Reconstrucción sobre las 458 secuencias no vistas, con 3 a 5 semillas | §3.2-3.5 |
| 7 | Seleccionar en las zonas de entrenamiento y evaluar en las de evaluación, o dejar una zona afuera por vez | §1.3, §5.4-5.5 |
| 8 | Bloques de 0,2° o más, o elegidos por variograma; permutación toroidal o por bloques | §5.8.4, §5.8.6 |
| 9 | Probing con split agrupado por secuencia y métricas balanceadas | §3.5(c) |
| 10 | Sensibilidad a τ y a la dilatación (0, 1, 2) | §5.8.3, §5.8.6 |

---

## 4. Auditoría de código

Citas de línea corregidas según los verificadores: [build_eval_zones.py:53-65](../scripts/build_eval_zones.py#L53) y [:99-105](../scripts/build_eval_zones.py#L99); [extract.py:54-65](../src/land2vec/extract.py#L54). En `viz/clusters/index.html`, las citas originales estaban corridas entre 35 y 40 líneas. Las correctas son `SET_HINT` en ~:310-314, `colorFor` en :669 y `labelFor` en :913.

### 4.1 Críticos y altos

| Área | ID | Sev. | Hallazgo | Ubicación |
|---|---|---|---|---|
| Clustering | CL-01, EXT-2, H1; REF-03, VE-4 *[seg.]* | crítica | Las constantes se asignan por centroide a procesos de cambio y deciden el ranking | [tune_clustering.py:338](../scripts/tune_clustering.py#L338), [assign_train_clusters.py:120](../scripts/assign_train_clusters.py#L120), [eval_desmonte.py:369](../scripts/eval_desmonte.py#L369) |
| Clustering | EXT-1, C1; VE-2, VE-3 *[seg.]* | crítica | R0 está mal emparejado (`DEFOR_PROCESOS` frente a `_r0_year`) y su docstring no coincide con el código; R0' supera a todas las tipologías en Chaco | [eval_desmonte.py:60](../scripts/eval_desmonte.py#L60), [eval_desmonte.py:157-164](../scripts/eval_desmonte.py#L157), [paper_metodologia.md:809](paper_metodologia.md#L809) |
| Embeddings | EMB-02, EMB-1, C2 | alta | Las zonas "OOD" comparten con el entrenamiento el 95,2 % de su masa dinámica | [zones.py](../src/land2vec/zones.py), [v2_autoencoder_training.md:442](v2_autoencoder_training.md#L442), [paper_metodologia.md:109](paper_metodologia.md#L109) |
| Clustering | CL-02, CLU-1, A4 | alta | Las métricas internas están infladas por duplicados | [cluster.py:403](../src/land2vec/cluster.py#L403), [cluster.py:468](../src/land2vec/cluster.py#L468) |
| Clustering | CL-06, CLU-2, A6 | alta | ARI sobre submuestras de 20.000 filas sin reescalar mcs; el paper dice "80 %" | [cluster.py:436](../src/land2vec/cluster.py#L436), [cluster.py:452](../src/land2vec/cluster.py#L452), [tune_clustering.py:485](../scripts/tune_clustering.py#L485), [paper_metodologia.md:497-498](paper_metodologia.md#L497) |
| Clustering | CLU-3 | alta | Faltan líneas de base sin embedding | [seqdist.py](../src/land2vec/seqdist.py) |
| Clustering | A5; VE-7 *[seg.]* | alta | El paper no muestra el MCC sin −1 | [paper_metodologia.md:754](paper_metodologia.md#L754) |
| Validación | REF-01 *[seg., no verificado]* | alta | `dilate=1` decide la prevalencia (0,260 frente a 0,313) y el ganador | [geo.py:393-394](../src/land2vec/geo.py#L393) |
| Validación | REF-02 *[seg., no verificado]* | alta | El 36,7 % de los positivos es constante y el techo de exhaustividad de 0,633 no se declara | [eval_desmonte.py:134](../scripts/eval_desmonte.py#L134) |
| Resultados | VE-9 *[seg., no verificado]* | alta | En periurbano, todo el MCC positivo viene de constantes | [paper_metodologia.md:856](paper_metodologia.md#L856) |
| Viz | H1 | alta | Las constantes se pintan como procesos de cambio (en Media/G, 12,3 % de lo visible); por eso §7.5 es falso | [build_cluster_map.py:509](../scripts/build_cluster_map.py#L509), [plot_zone_atlas.py:130](../scripts/plot_zone_atlas.py#L130), [index.html:669](../viz/clusters/index.html#L669), [paper_metodologia.md:1029](paper_metodologia.md#L1029) |

### 4.2 Medios

| Área | ID | Hallazgo | Ubicación |
|---|---|---|---|
| Embeddings | EMB-01 | El split es por fila y la validación es casi igual al entrenamiento; el barrido de d se lee mal | [train_autoencoder.py:107](../scripts/train_autoencoder.py#L107), [paper_metodologia.md:289](paper_metodologia.md#L289) |
| Embeddings | EMB-03, A1 | El macro F1 tiene techo 0,9 porque Nd no aparece | [train_autoencoder.py:50](../scripts/train_autoencoder.py#L50), [utils.py:144](../src/land2vec/utils.py#L144) |
| Embeddings | EMB-05 | La ASW usa W_k−w_i y queda subestimada entre 0,11 y 0,20; la cobertura cuenta secuencias distintas | [seqdist.py:223](../src/land2vec/seqdist.py#L223), [seqdist.py:211](../src/land2vec/seqdist.py#L211) |
| Embeddings | EMB-2, A2, A3 | Hay una sola semilla; la diferencia entre réplicas supera el rango del barrido; hay colapsos porque no hay clip ni scheduler (min_lr sin usar) | [train_autoencoder.py](../scripts/train_autoencoder.py), [config.py:24](../src/land2vec/config.py#L24) |
| Embeddings | EMB-3, M7 | El probe de z queda debajo de la clase mayoritaria y §3.5 no da cifras | [paper_metodologia.md:375](paper_metodologia.md#L375) |
| Clustering | CL-04 | La cobertura y la detección de yungas no usan IPW (66,1 % sin ponderar frente a 8,7 % ponderada) | [eval_desmonte.py:414](../scripts/eval_desmonte.py#L414) |
| Clustering | CL-05, EXT-3, M1 **[no confirmado]**; VE-5 *[seg.]* | Los IC no son pareados y los bloques son de 0,05°; con 0,2°, los IC son ~2 veces más anchos | [eval_desmonte.py:549](../scripts/eval_desmonte.py#L549) |
| Clustering | CL-03, CL-12, M4 | El k queda en el borde de la grilla y GMM con k>20 solo se probó en l2/diag | [tune_clustering.py:248](../scripts/tune_clustering.py#L248) |
| Resultados | M5 | La regla de selección premia el ruido (Media/H tiene 24,2 % con tope de 25 %) | models/cluster_v2/summary.csv |
| Clustering | CL-07 | Las etiquetas del set dinámico no coinciden con las del pool (acuerdo 0,806-0,948) | [cluster.py:291](../src/land2vec/cluster.py#L291), [cluster.py:366](../src/land2vec/cluster.py#L366) |
| Validación | CL-08, EXT-7, M8 | Las sensibilidades a τ y al corte de longitud no existen | [paper_metodologia.md:893](paper_metodologia.md#L893) |
| Validación | CLU-4 | No hay conjunto reservado para la selección | [cluster.py:63](../src/land2vec/cluster.py#L63) |
| Validación | CLU-5 **[no confirmado]** | Las particiones triviales tendrían una coherencia espacial similar (no se recalculó) | [cluster.py:499](../src/land2vec/cluster.py#L499) |
| Resultados | EXT-5, M3 | La ganancia@10 % está saturada | desmonte_eval*.csv |
| Validación | EXT-6, VAL-1 | La validación externa es estrecha y las zonas están desbalanceadas | [paper_metodologia.md:852](paper_metodologia.md#L852) |
| Resultados | VE-6 *[seg., no verificado]* | Las cifras canónicas son R0' 0,638 y F2000∧¬F2022 0,637; el 0,634 no se reproduce | — |
| Resultados | VE-8 *[seg., no verificado]* | Restringido a F en 2000, R0' (0,780) supera a la mejor corrida (Media/G, 0,755) | — |
| Validación | REF-05 *[seg., no verificado]* | Lo excluido es 21,3 % fuera de la máscara y 9,4 % de bordes | [eval_desmonte.py:130](../scripts/eval_desmonte.py#L130) |
| Viz | H4 | El atlas calcula los porcentajes sobre lo que no es ruido: dice 55,3 % cuando el valor real es 49,9 % | [plot_paper_atlas.py:124](../scripts/plot_paper_atlas.py#L124), [plot_paper_atlas.py:217](../scripts/plot_paper_atlas.py#L217) |
| Viz | H5 | La corrida por defecto rompe los fondos de train | [build_cluster_map.py:374](../scripts/build_cluster_map.py#L374), [build_cluster_map.py:577](../scripts/build_cluster_map.py#L577) |

### 4.3 Bajos e info

| Área | ID | Sev. | Hallazgo | Ubicación |
|---|---|---|---|---|
| Embeddings | EMB-04 | baja | train_loss se calcula en FP16, fuera del autocast. Solo afecta la curva de train_loss; los picos se explican por A3 | [model.py:250](../src/land2vec/model.py#L250), [utils.py:117](../src/land2vec/utils.py#L117) |
| Embeddings | EMB-06 | baja | delta_oeste queda a 4,6-4,7 km de dos zonas de evaluación | [zones.py:41](../src/land2vec/zones.py#L41), [build_eval_zones.py:53-65](../scripts/build_eval_zones.py#L53) |
| Embeddings | EMB-07 | baja | La v1 no se entrenó "sobre los mismos datos" | [paper_metodologia.md:379](paper_metodologia.md#L379) |
| Embeddings | EMB-08 | baja | La fórmula de la pérdida no coincide con la normalización del código; los pesos se calculan con train+val | [model.py:253](../src/land2vec/model.py#L253), [train_autoencoder.py:113](../scripts/train_autoencoder.py#L113) |
| Embeddings | EMB-09 | baja | No se verifican "todos los pares de cajas" | [build_eval_zones.py:99-105](../scripts/build_eval_zones.py#L99), [paper_metodologia.md:89](paper_metodologia.md#L89) |
| Embeddings | EMB-11 | baja | El notebook de extracción de Chaco no es reproducible | src/3_concat_extract_nc_files.ipynb |
| Embeddings | B4 | baja | El config.json de sin_pesos no registra `weighted` | [train_autoencoder.py:56](../scripts/train_autoencoder.py#L56) |
| Embeddings | EMB-10 **[no confirmado]** | info | Los códigos LCCS mayores que 9 se convierten en Wa sin aviso | [extract.py:54-65](../src/land2vec/extract.py#L54) |
| Clustering | CL-09, M6, DOC-1 | baja | El rango "Entre 20,9 % y 35,1 % de las filas del pool dinámico de ajuste" (§5.8.6) corresponde en realidad a train_pooled | [paper_metodologia.md:897-898](paper_metodologia.md#L897) |
| Clustering | CL-10, EXT-4, M2 | baja | La permutación es i.i.d., no espacial | [eval_desmonte.py:682](../scripts/eval_desmonte.py#L682), [paper_metodologia.md:770](paper_metodologia.md#L770) |
| Clustering | CL-11 | baja | La AUC-PR por trapecios da 0,626 frente a 0,603 | [eval_desmonte.py:330](../scripts/eval_desmonte.py#L330) |
| Clustering | CL-13 **[no confirmado]** | baja | Probing con tareas triviales y métrica desbalanceada | eval_embeddings_v2.ipynb, celda 13 |
| Resultados | M9 | baja | El pseudo-R² no se compara "en igualdad de condiciones" | [paper_metodologia.md:616](paper_metodologia.md#L616) |
| Resultados | M10 | baja | Yungas se presenta como generalización del encoder, pero es in-sample | [paper_metodologia.md:835](paper_metodologia.md#L835) |
| Resultados | B1, B2, B5 | baja | Dice "mediana 0" donde es 1 y 12,5 % donde es 12,0 %; no informa la unidad mínima detectable | [paper_metodologia.md:826](paper_metodologia.md#L826), [paper_metodologia.md:885](paper_metodologia.md#L885), [paper_metodologia.md:761](paper_metodologia.md#L761) |
| Resultados | B3 **[no confirmado]** | info | La grilla de yungas está descrita de forma ambigua | [paper_metodologia.md:1055](paper_metodologia.md#L1055) |
| Validación | REF-06 *[seg., no verificado]* | baja | El docstring de `label_reference` es falso (849 positivos quedan fuera de la máscara) | [eval_desmonte.py:122-126](../scripts/eval_desmonte.py#L122) |
| Validación | REF-04, REF-07 *[seg., no verificado]* | info | Los desmontes previos a 2001 están bien tratados; el techo de la tabla por secuencia es 0,644 | [eval_desmonte.py:131](../scripts/eval_desmonte.py#L131) |
| Viz | H2 | baja | El "15 %" no es un submuestreo del visor: es un tope de constantes sobre el pool (≈0,59 % de las constantes reales) | [index.html:310-314](../viz/clusters/index.html#L310), [cluster.py:173](../src/land2vec/cluster.py#L173) |
| Viz | H3 | baja | El fondo y la imagen se estiran sobre Mercator (hasta 2,3 km) | [fetch_zone_imagery_gee.py:327](../scripts/fetch_zone_imagery_gee.py#L327) |
| Viz | H6 | baja | Aparecen 8 paneles vacíos | [plot_process_maps.py:103](../scripts/plot_process_maps.py#L103) |
| Viz | H7 | baja | La figura de zonas usa datos submuestreados | [plot_v2_zones.py:65](../scripts/plot_v2_zones.py#L65) |
| Viz | H8 | baja | El −1 se rotula siempre como "sin tipificar", aunque en el set dinámico es ruido | [index.html:913](../viz/clusters/index.html#L913), [build_crossrun.py:521](../scripts/build_crossrun.py#L521) |
| Viz | H9 | baja | Textos fijos en crossrun | [viz/crossrun/index.html:811](../viz/crossrun/index.html#L811) |
| Viz | H10 | baja | El hitTest no filtra los puntos ocultos | [index.html:481](../viz/clusters/index.html#L481) |
| Viz | H11 | baja | §7.2-7.3 no coinciden con el manifiesto | [fetch_zone_imagery_gee.py:79](../scripts/fetch_zone_imagery_gee.py#L79) |
| Viz | H12, H13 **[no confirmado]** | info | Rasterizado con (h−1) y paleta que se repite cada 40 | [plot_zone_atlas.py:147](../scripts/plot_zone_atlas.py#L147), [build_cluster_map.py:108](../scripts/build_cluster_map.py#L108) |

---

## 5. Plan de acción priorizado

| # | Acción | IDs | Cifras del paper que podrían cambiar |
|---:|---|---|---|
| 1 | Agregar R0', F2000∧¬F2022 y "alguna transición" con diferencias pareadas; corregir el docstring de `_r0_year`; reformular el aporte | EXT-1, C1, VE-2/3 | Tabla y conclusión de §5.8.5 (paper:809) |
| 2 | Tratar las constantes como sin_cambio o −2; rehacer clusters_*_full, pooled, eval_desmonte y el atlas | CL-01, EXT-2, H1, REF-03, VE-4/9 | MCC, P, R y detección de las 5 corridas (Media/G pasa de 0,514 a 0,618); el ganador; periurbano; §7.5; zone_atlas; cifras derivadas en analisis_estabilidad |
| 3 | Declarar las secuencias distintas y medir sobre las 458 no vistas | EMB-01/02, C2 | §1.3, §1.5, §3.2, §3.5 |
| 4 | Recalcular las métricas internas sobre secuencias distintas, con controles y líneas de base | CL-02, CLU-1/3, A4 | §5.5 (0,913) y §5.7 |
| 5 | Rehacer la estabilidad y corregir "80 %" | CL-06, CLU-2, A6 | Columna ARI de §5.5 y elegibilidad de las HDBSCAN |
| 6 | Sensibilidad a dilate, τ y bloque; declarar el techo de 0,633 | REF-01/02, CL-08, EXT-3, VE-5 | Prevalencia de 26 %, ranking, IC |
| 7 | Agregar la columna de MCC sin −1 con IC | A5, VE-7 | §5.8.5 |
| 8 | Declarar el techo de 0,9 y usar varias semillas o elegir por costo; agregar clip y scheduler | EMB-03, A1-A3, EMB-2 | §3.2-3.4 (0,8971 equivale a 0,997) |
| 9 | Corregir la ASW (W_k−1) y reportar la cobertura por filas | EMB-05, M9 | §5.7: ASW +0,11 a +0,20; cobertura de 0,19-0,37 a 0,76-0,90 |
| 10 | Usar IPW en la cobertura y la detección de yungas; no declarar ganador | CL-04, M10 | Yungas: cobertura de 66,1 % a 8,7 %; paper:852 |
| 11 | Correcciones de texto: §1.3, §5.3(ii), §5.8.6, mediana, 12,5 %, "permutación espacial", v1, fórmula, "todos los pares", §7.2-7.3 y docstring de `label_reference` | varios | Valores puntuales |
| 12 | Robustez de las visualizaciones antes de regenerar | H2-H10 | Porcentajes del atlas |

---

## 6. Alcance, limitaciones y proceso

### 6.1 Qué está bien
- La alineación ID/secuencia/coordenada/embedding está bien en las 15 zonas. Todas las secuencias tienen 23 tokens, no hay [UNK] y las grillas de evaluación están completas (EMB-12, H16).
- Los conteos de entrenamiento (400.460 = 302.034 + 98.426), los 798.216 parámetros y la mejor época coinciden con los documentos (I1).
- `chosen*.json`, `typology_seqdist.csv`, `desmonte_eval*.csv`, los deciles y `crossrun.json` coinciden con el paper.
- No hay lat/lon invertidas ni faltan filas en los joins, y `check_crossrun_viewer.js` da OK.
- Los desmontes previos a 2001 se tratan bien (REF-04, *seguimiento, no verificado*).

### 6.2 Limitaciones
- **Sin torch ni xarray.** No se midió la reconstrucción sobre las 458 secuencias no vistas, ni la fidelidad de los controles triviales, y no se abrió el netCDF.
- **CLU-5 sin recálculo.** La coherencia espacial de las particiones triviales queda como no confirmada. Tampoco se recalcularon PAM ni la AUC del probe.
- **Máscara de relevamiento por proxy** en REF-01, porque no hay geopandas.
- **Los seguimientos no pasaron por el verificador** (REF-01..07, VE-1..9). La tabla canónica (VE-1) coincide con los CSV publicados, lo que respalda sus cifras de bootstrap, pero sus hallazgos siguen sin un veredicto independiente.
- **Incidente de auditor-clustering, ya resuelto.** Un script escribió por error `master_chaco_santiago_frontier.pkl` en la raíz del repo. El auditor lo movió al scratchpad y `git status` quedó limpio.
- **Cifras descartadas.** No se citan los conteos de EXT-2 que no se reprodujeron, ni los ratios de picos de EMB-04 (son max/min de toda la corrida, no picos), ni el 0,634 de la regla con `classify_process` (VE-6).

### 6.3 Validación propia de los seguimientos
Script: `scratchpad/redactor/val.py`. Recalculado en Chaco desde los zips:
- 793.112 evaluables, prevalencia 0,2604 y 36,65 % de positivos constantes: coinciden.
- R0 0,3995, R0' 0,6380, F2000∧¬F2022 0,6375 y "alguna transición" 0,6165: coinciden.
- Corridas con constantes forzadas (0,553; 0,617; 0,542; 0,618; 0,510; 0,577), con TP constantes de 0 a 31,8 % y FP constantes de 0 a 85,2 %: coinciden.
- Chequeado contra las fuentes:
  - eval_desmonte.py: `DEFOR_PROCESOS` en :60 y `_r0_year` en :157-164.
  - cluster.py:436: `boot_cap = 20000`.
  - paper_metodologia.md: "80 %" en :497-498, y las frases de :809 y :897-898.
  - `desmonte_eval.csv`: R0, Fina/H, prevalencia y bloques de Chaco y yungas.
- Sin revalidar: el bootstrap, yungas y periurbano más allá de los CSV.

### 6.4 Meta-hallazgo (corregido)
La premisa "El visor submuestrea los puntos al 15%" era falsa (H2). El visor no submuestrea: el 15 % es un tope de constantes sobre el pool de cada zona (≈0,59 % de las constantes reales). La premisa estaba en las definiciones de los 8 agentes (`.claude/agents/`) y en la memoria del proyecto. Se corrigió en esos lugares el 2026-10-01, después de la auditoría.
