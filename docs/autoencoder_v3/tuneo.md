# Tuneo del autoencoder (autoencoder_v3, 1992-2022)

**Fecha:** 2026-10-09. **Estado:** protocolo escrito antes de entrenar. Todo lo que figura acá se fija antes de ver ningún resultado; cualquier cambio posterior se registra en §9 (desviaciones) con su motivo.

## 1. Para qué se tunea

El autoencoder (AE) se va a usar para **agrupar trayectorias de uso del suelo de modo que los grupos expresen procesos reales en el territorio**. Por eso la configuración no se elige por la reconstrucción sola, sino por **cuán bien el espacio z separa los procesos** en trayectorias que el modelo no vio.

El AE se entrena **sólo con reconstrucción** (no supervisado). Los procesos entran únicamente en el **criterio de selección**. Las fuentes externas (Monitor de Desmontes, Hansen, GHSL) **no se usan en el tuneo**: quedan reservadas para la evaluación.

Antecedente: el AE de la v3 anterior heredó la arquitectura del barrido de v2, que no discriminaba entre configuraciones (todas dentro de ±0,003 de macro F1, una semilla por configuración), y nunca se tuneó para su régimen de entrenamiento. Archivado en el tag `v3-1992-2022-previo`.

## 2. Procesos

Definidos en `src/land2vec/procesos.py`. Un **evento** es el paso entre dos tramos persistentes (≥ 3 años, con las excepciones por censura al principio y al final de la serie) de estados distintos. Una trayectoria es **positiva** para un proceso si tiene al menos un evento de ese proceso.

| Familia | Proceso | Eventos |
|---|---|---|
| Deforestación | D1, estricta | F → A |
| | D2, amplia | F, Sh → A |
| | D3, muy amplia | F, Sh, G → A |
| Degradación | | F → G, F → Sh, F → B |
| Regeneración | | A, Sh, G → F |
| Expansión urbana | | cualquier estado → U |

**Fuera:** la retracción urbana. El producto no la registra: en todo el mundo hay 2 tipos de trayectoria con U → otro estado, que suman 1 km², frente a 517.000 km² de expansión urbana.

Todos los procesos pasan el umbral de evaluabilidad (≥ 1 % de la superficie dinámica y ≥ 100 tipos) en América Latina (`data/autoencoder_v3/latam/catalogo_procesos.csv`).

## 3. Universo y partición

- **Universo:** América Latina y el Caribe, 22.299 tipos de trayectoria (22.290 dinámicos), del censo `data/autoencoder_v3/latam/universo_latam.csv.gz`. Máscara: Natural Earth v5.1.2, map units con `REGION_WB = "Latin America & Caribbean"` (`scripts/datos/mascara_latam.py`).
- **Partición por tipo:** 80 % entrenamiento, 20 % validación, con semilla fija (0).
  - **Estratos:** combinación de familias presentes (deforestación según D3, degradación, regeneración, urbana o ninguna; 16 combinaciones posibles) × decil de superficie (km²) entre los tipos dinámicos. En cada estrato se toma al azar el 20 % (redondeado; los estratos de un solo tipo van a entrenamiento).
  - **Las 9 trayectorias constantes van a entrenamiento.** No tienen procesos.
- **El resto del mundo no participa del tuneo.** Queda para la evaluación.
- Peso de ajuste en el entrenamiento: min(n_px, tope), con el tope como hiperparámetro (§5).

## 4. Criterio de selección

Se calcula sobre los tipos de **validación**, con los códigos z del modelo entrenado sólo con los tipos de entrenamiento.

### 4.1 Criterio principal S: agrupamiento con k-medias
1. **Agrupar:** k-medias sobre z de todos los tipos (entrenamiento y validación, cada tipo pesa 1), con 10 arranques y semilla fija, para cada **k ∈ {8, 12, 16, 24, 32, 48, 64}**.
2. **Asignar grupos a procesos:** para cada proceso, un grupo se asigna al proceso si al menos el 50 % de sus tipos de **entrenamiento** son positivos.
3. **Medir en validación:** un tipo de validación se predice positivo si cae en un grupo asignado al proceso. Para cada proceso: **F1 por tipo** (cada tipo de validación pesa 1).
4. **Promediar:**
   - para cada k, el promedio de las 4 familias: deforestación (como promedio de D1, D2 y D3), degradación, regeneración y expansión urbana. Así, la deforestación no pesa 3 de 6 por tener tres variantes;
   - **S = promedio sobre los 7 valores de k.**

Se usa k-medias porque es el algoritmo con el que se va a agrupar sobre z. Los k cubren desde una tipología gruesa hasta una fina, porque no hay un k natural. Con un solo k, la semilla de k-medias mueve el resultado hasta 0,07; promediado sobre los 7, S varía 0,013 (medido con PCA d = 8, 5 semillas de k-medias).

Se pondera por tipo y no por superficie porque, por superficie, cada proceso depende de pocas decenas de tipos efectivos. En el primer pase de la v3 anterior eso hacía saltar las métricas hasta 0,3 entre configuraciones vecinas.

### 4.2 Control por superficie
El mismo S con el F1 ponderado por km² (S_sup). Una configuración no se elige si su S_sup queda por debajo del de la mejor en más que el rango entre semillas de esa mejor.

### 4.3 Secundarios
Se reportan, pero no deciden:
- **Sonda de 10 vecinos (S_vecinos):** para cada tipo de validación, sus 10 vecinos más cercanos entre los de entrenamiento (euclídea en z); se predice positivo si al menos 5 lo son. F1 por tipo, promediado por familias como S.
- **Fechado:** en los tipos de validación positivos que la sonda predice positivos, error absoluto (en años) entre el año del primer evento del proceso y la mediana de ese año entre los vecinos positivos.
- **Reconstrucción** en validación y en entrenamiento: exactitud por año y fracción de secuencias exactas.
- **El F1 de cada proceso y cada k**, para ver qué procesos y qué escalas explican S.

### 4.4 Referencias del criterio
Se calcularon con la misma partición, antes de entrenar ningún modelo (`data/autoencoder_v3/tuneo/referencias.csv`):

| Referencia | S | S_sup | S_vecinos |
|---|--:|--:|--:|
| Piso: z al azar (gaussiano, d = 8) | 0,000 | 0,000 | 0,124 |
| Lineal: PCA del one-hot, d = 8 (ajustado con entrenamiento) | 0,415 | 0,540 | 0,900 |
| Sin compresión: one-hot de la secuencia | 0,529 | 0,622 | 0,947 |

Si el AE no supera al one-hot, comprimir no aporta para estas tareas, y eso es un resultado en sí.

## 5. Espacio de búsqueda

| Hiperparámetro | Valores | Base |
|---|---|---|
| Tope del peso de ajuste min(n_px, tope) | 1 (cada tipo pesa igual), 10, 100, sin tope | 100 |
| Dropout | 0; 0,1; 0,2 | 0,1 |
| Weight decay | 0; 1e-2; 1e-1 | 1e-2 |
| n_embd | 64, 128, 256 | 128 |
| n_layer (encoder y decoder) | 1, 2, 4 | 2 |
| Pooling | mean, query | query |
| Tasa de aprendizaje | 3e-4, 1e-3, 3e-3 | 1e-3 |
| Lote | 128, 512 | 128 |

- El calendario de la tasa de aprendizaje es igual para todas: calentamiento lineal durante el 5 % de S_max y después coseno hasta el 1 % de la tasa, definido sobre S_max desde el primer paso.
- El presupuesto se mide en **pasos**, no en épocas: con lote 512 cada paso ve cuatro veces más trayectorias.
- n_head queda fijo en 4.

## 6. Etapas

1. **Piloto.** Configuración base, d = 8, una semilla, 100.000 pasos, evaluación cada 5.000.
   - Mide la velocidad (pasos por segundo) en la GPU.
   - Fija **S_max**: el primer múltiplo de 10.000 pasos en el que la fracción de secuencias exactas en validación alcanza el 99 % de su máximo en el piloto, entre 20.000 y 100.000 (desviación 2). Resultado: **S_max = 80.000**.
   - Fija el intervalo de checkpoint: cada 5 minutos, aproximadamente.
2. **Búsqueda.** d = 8, una semilla, 24 configuraciones: la base más 23 sorteadas del espacio de §5 con semilla fija (sin repetir), guardadas en `data/autoencoder_v3/tuneo/plan_busqueda.json` la primera vez.
   - Las 24 se entrenan hasta S_max y se comparan por su S en S_max (sin rondas: desviación 3).
3. **Confirmación.** Las 4 de mayor S en la búsqueda, con 3 semillas (0, 1, 2) y d ∈ {4, 8, 16, 20}, hasta S_max (48 corridas).
4. **Elección.** Se elige por el S medio entre semillas, promediado sobre los cuatro d (§7).
5. **Modelos finales.** La configuración elegida, reentrenada con todo el universo de América Latina (entrenamiento y validación), con d ∈ {2, 3, 4, 6, 8, 12, 16, 20, 24, 31} × 3 semillas (30 corridas), hasta S_max. Son los modelos que entran a la evaluación.

## 7. Regla de decisión
- **Diferencias menores que el rango entre las 3 semillas no se interpretan.**
- Entre configuraciones empatadas, se elige la de menos parámetros. Si también empatan en eso, la de lote más chico y después la de menos pasos.
- El control por superficie (§4.2) puede descartar una configuración, pero no elegirla.
- d no se elige en el tuneo: se informa S contra d. La elección de d queda para la evaluación.

## 8. Ejecución
- Todo se puede interrumpir y retomar con el mismo comando:
  - checkpoint atómico por corrida, que incluye optimizador, calendario y estados de los generadores aleatorios;
  - parada ordenada con Ctrl+C, SIGTERM o un archivo `STOP`;
  - parada por horario con `--max-horas` o `--hasta`;
  - estado guardado en disco.
- **Código:** `scripts/modelo/tuneo_ae.py` (subcomandos `particion`, `referencias`, `piloto`, `busqueda`, `confirmar`, `final`, `estado`, `resumen`) y `src/land2vec/criterio_procesos.py`.
- **Salidas:** `models/autoencoder_v3/tuneo/` (modelos y checkpoints) y `data/autoencoder_v3/tuneo/` (partición, configuraciones, métricas).
- **Costo:** el piloto corrió a 41,9 pasos/s con un proceso en la GTX 1060. La búsqueda (24 × 80.000 = 1,92 millones de pasos) se estima en 6–9 horas en esa GPU con 3–4 corridas en paralelo; menos en Mendieta (`scripts/cluster/`). Las configuraciones grandes (n_embd 256, 4 capas, lote 512) son más lentas.

## 9. Desviaciones
1. **2026-10-09, antes del piloto: el criterio principal pasó de la sonda de 10 vecinos al agrupamiento con k-medias.**
   - **Motivo:** la sonda satura. Con la partición fijada dio 0,124 con z al azar, 0,900 con PCA d = 8 y 0,947 con el one-hot; una corrida de prueba del AE con sólo 100 pasos (reconstrucción por año 0,61) ya daba 0,852. Todas las configuraciones caerían en una franja de unas 0,1, con diferencias del orden del ruido entre semillas: el problema del barrido de v2. Los vecinos inmediatos de una trayectoria son casi copias suyas, así que casi cualquier representación los pone cerca. El agrupamiento mide el orden a mayor escala, que es lo que importa para agrupar, y va de 0 (azar) a 0,53 (one-hot).
   - **Qué se había visto de un AE al decidir:** sólo esa corrida de prueba de 100 pasos (S de la sonda 0,852), usada para probar el código.
   - **Cambios:** k-medias pasa de 4 a 10 arranques y de k ∈ {12, 24, 48} a k ∈ {8, 12, 16, 24, 32, 48, 64}; S pasa a ser el promedio sobre esos k; la sonda queda como secundaria (S_vecinos). El algoritmo (k-medias) y los k los eligió el usuario.
2. **2026-10-09, después del piloto: S_max se fija por la reconstrucción, no por S.**
   - **Motivo:** en el piloto (configuración base, d = 8), S bajó a medida que el AE aprendía a reconstruir: 0,429 a los 5.000 pasos, 0,354 a los 20.000 y ~0,30 desde los 40.000, mientras las secuencias exactas en validación subían de 0,29 a 0,935. El máximo de S está en la primera evaluación, así que la regla original diría "entrenar lo mínimo", y compararía AE que todavía no reconstruyen.
   - **Diagnóstico** (k-medias, información mutua normalizada con k = 24): el z del AE agrupa más por el año del primer cambio (0,15, contra 0,07 de PCA y 0,09 del one-hot) y menos por el estado final (0,12, contra 0,14 y 0,20) y por proceso (0,15, contra 0,24 y 0,25). Los procesos se definen por el estado de destino: la pérdida de reconstrucción organiza z en contra de esta tarea.
   - **Regla nueva:** primer múltiplo de 10.000 pasos con secuencias exactas ≥ 99 % de su máximo en el piloto → **S_max = 80.000** (0,930; máximo 0,935).
3. **2026-10-09, después del piloto: la búsqueda no usa rondas.**
   - **Motivo:** como S es más alto cuanto menos entrenado está el AE, cortar en S_max/4 y seguir con las de mayor S favorecería a las configuraciones que aprenden más despacio (tasa baja, modelos grandes), no a las que organizan mejor los procesos.
   - **Cambio:** las 24 configuraciones se entrenan hasta S_max y se comparan ahí. Costo: 1,92 millones de pasos en vez de 1,44 millones.
