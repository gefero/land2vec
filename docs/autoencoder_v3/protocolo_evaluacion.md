# Protocolo de evaluación de las representaciones (autoencoder_v3, 1992-2022)

**Fecha:** 2026-10-08; revisado el 2026-10-09. **Estado:** la Parte I está fijada y ejecutada ([`p1_resultados.md`](p1_resultados.md)); la Parte II está fijada **antes** de implementar y correr. Cualquier cambio posterior se registra en §14 (desviaciones) sin reescribir lo anterior.

**Revisión del 2026-10-09.** La Pregunta 1 se redujo a la capacidad de compresión (reconstrucción). La accesibilidad de la información en z (antes §5) y las tipologías evaluadas con métricas genéricas (antes §9, "Pregunta 1.2") se eliminaron, junto con su código y sus resultados (quedan en el historial, commit `c1f9365`). Se reemplazan por la Parte II: la evaluación de las tipologías según su capacidad de detectar los procesos territoriales que motivan el trabajo. Las decisiones de diseño del agrupamiento tomadas el 2026-10-08 (algoritmo, universo, ponderación, valores de k) se conservan en §8.

Antecedentes: reemplaza a las pruebas "AE contra OM" del 2026-10-08, descartadas (commits `69cc1e6` y `85ce841`, eliminadas en `7b78f76`).

---

## 1. Qué se evalúa

El objetivo es usar las representaciones para **agrupar secuencias de uso del suelo en tipologías que expresen procesos reales del territorio**. Se evalúan dos preguntas, por separado:

- **Pregunta 1 (Parte I): capacidad de compresión.** ¿Cuánta información de las trayectorias conserva cada método en d dimensiones, medida como reconstrucción de trayectorias que no vio?
- **Pregunta 2 (Parte II): detección de procesos.** ¿Las tipologías construidas sobre cada representación detectan los procesos de interés (deforestación, expansión urbana, degradación y regeneración de bosque)? En tres niveles:
  1. **Nivel 1:** ¿recuperan los procesos tal como los registra el producto (ESA CCI)?
  2. **Nivel 2:** ¿esos procesos coinciden con lo que registran fuentes independientes?
  3. **Nivel 3:** ¿el resultado es robusto a la semilla del modelo, al azar del agrupamiento y a trayectorias no vistas?

**Principio:** todos los métodos se comparan en pie de igualdad. Ninguno funciona como juez de los demás; en particular, **OM es un método más**, no una referencia de lo correcto. En la Parte I OM no participa porque no reconstruye trayectorias; participa en la Parte II.

---

# Parte I. Pregunta 1: capacidad de compresión

## 2. Métodos

| Método | Qué es | Semillas |
|---|---|---|
| **AE** | autoencoder transformer (configuración de P1: ancho 128, 2+2 bloques, 400 épocas, lote 128, lr 1e-3, peso min(n_px, 100)) | 0, 1, 2 |
| **AE lineal** | codificador y decodificador lineales sobre el one-hot, misma pérdida y pesos que el AE (Adam lr 1e-2, 3.000 pasos) | 0, 1, 2 |
| **PCA** | componentes principales ponderados del one-hot | — |
| **MCA** | análisis de correspondencias múltiples ponderado del one-hot | — |

- **Dimensiones:** d ∈ {1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31}. La zona de interés es d ≤ 7.
- **Ajuste:** todos los métodos se ajustan **sólo con el universo de Argentina** (7.827 trayectorias de `universo_argentina.csv`).
- **Tamaño de cada método:** se reporta el número de parámetros junto a los resultados (el AE tiene unos 800.000; PCA con d = 4, unos 1.700), como información, no como criterio.

---

## 3. Trayectorias de evaluación

### 3.1 Conjunto no visto
Las trayectorias **dinámicas del censo mundial que no existen en Argentina**: 62.121 tipos (`data/autoencoder_v3/mundo/universo_mundo.csv.gz`), 3,9 % del área dinámica mundial. Usan los mismos 9 estados que Argentina y no tienen valores sin dato. Ningún método las vio al ajustarse.

### 3.2 Estratos
Las trayectorias no vistas se clasifican con dos criterios, definidos sin usar ninguno de los métodos comparados:

- **A. Proceso visto o nuevo.** El *proceso* de una trayectoria es la secuencia de estados por la que pasa, en orden, sin las fechas (por ejemplo, bosque → agrícola → bosque). Es **visto** si alguna trayectoria de Argentina tiene el mismo proceso, y **nuevo** si no.
- **B. Años distintos (h).** Número de años, de 31, en que la trayectoria difiere de la trayectoria de Argentina más parecida (la que minimiza ese número entre las 7.827). Cortes: 1, 2, 3, 4–5, 6 o más.

| h | Proceso visto | Proceso nuevo |
|---|--:|--:|
| 1 | 11.617 | 2.031 |
| 2 | 8.927 | 1.841 |
| 3 | 8.267 | 2.583 |
| 4–5 | 9.786 | 5.487 |
| 6 o más | 4.489 | 7.093 |
| **Total** | **43.086** | **19.035** |

**Lectura principal por A; B como gradiente** dentro de cada valor de A: cómo cae la fidelidad a medida que la trayectoria se aleja de lo que el método vio.

**Decisiones registradas:** se descartó medir la novedad como el corrimiento de fechas respecto de la trayectoria argentina del mismo proceso, por ser una distancia de edición emparentada con OM. Se descartó también la validación cruzada dentro de Argentina, reemplazada por este conjunto.

### 3.3 Argentina como referencia
Las mismas métricas sobre las 7.818 trayectorias dinámicas de Argentina (las de ajuste) se reportan **sólo como referencia**, no para comparar métodos: ahí se mide capacidad de recordar, no de comprimir.

---

## 4. Reconstrucción

Cada trayectoria se codifica en d números y se decodifica; la reconstrucción toma, en cada año, el estado de mayor puntaje.

### 4.1 Métricas principales (fijadas)
| Métrica | Qué mide |
|---|---|
| **Exactitud por año** | Fracción de los 31 años con el estado correcto. Información alineada en el tiempo: el *cuándo*. |
| **Secuencia de estados correcta** | 1 si la reconstrucción pasa por los mismos estados, en el mismo orden, que la original, sin mirar las fechas; 0 en otro caso. Información de proceso: el *qué*. |

### 4.2 Métricas secundarias (para explicar las principales)
- **Reconstrucción exacta:** los 31 años correctos.
- **Número de cambios correcto.**
- **Error de fechado:** promedio del error absoluto, en años, del momento de cada cambio. Sólo se puede calcular donde la secuencia de estados es correcta; se reporta junto con la fracción de trayectorias en las que aplica (cobertura).
- **F1 macro por clase:** sobre los años de todas las trayectorias, para ver si las clases raras se pierden primero.

---

## 5. Ponderación y semillas

- **Ponderación principal: por tipo** (cada trayectoria distinta pesa 1). Las no vistas son casi todas raras (mediana de 2 a 9 píxeles por tipo), así que la superficie dice poco.
- **Secundaria: por superficie**, con el área en km² del censo mundial (el área del píxel varía con la latitud).
- **Semillas:** para el AE y el AE lineal se reporta la media de las tres semillas y el rango mínimo–máximo. **Una diferencia menor que ese rango no se interpreta.**

---

## 6. Cómo se leen los resultados

- No hay una puntuación única. Se reportan las curvas de cada métrica contra d, por estrato.
- **Interpretación fijada:** un método que comprime información general de las trayectorias debería mantener la fidelidad en el estrato de proceso visto y degradarse de forma gradual con h. Una caída fuerte ya en h = 1 indica que el método reproduce lo que vio más que una regla general.
- El estrato de proceso nuevo mide extrapolación. Un mal resultado ahí no se interpreta como falta de compresión, sino como falta de generalización a procesos ausentes del ajuste.

---

# Parte II. Pregunta 2: detección de procesos

## 7. Procesos y definiciones operativas (decididas 2026-10-09)

### 7.1 Estados
Los 9 estados del producto agrupado: agrícola (A), bosque (F), pastizal (G), humedal (Wt), urbano (U), arbustal (Sh), vegetación rala (Sp), suelo desnudo (B) y agua (Wa).

### 7.2 Tramos y persistencia
- Un **tramo** es una racha máxima de años consecutivos con el mismo estado.
- Un tramo es **persistente** si dura **al menos 3 años**. Dos excepciones, por censura:
  - **a la izquierda:** el tramo que empieza en 1992 es persistente aunque dure menos (no se sabe desde cuándo estaba);
  - **a la derecha:** un tramo que empieza en 2021 o 2022 y llega a 2022 es persistente (no hay años suficientes para comprobarlo).
- Los tramos no persistentes son **transitorios** y no definen procesos. Esto excluye, entre otros, los estados de un año.

### 7.3 Eventos
- Un **evento** es el paso entre dos tramos persistentes consecutivos (ignorando los transitorios que haya entre ellos) con estados distintos.
- **Origen:** el estado del tramo persistente anterior. **Destino:** el del siguiente. **Año:** el primer año del tramo de destino.
- Si dos tramos persistentes consecutivos tienen el mismo estado (por ejemplo, F – Sh de 2 años – F), no hay evento: los transitorios intermedios se registran como **oscilación**. Las oscilaciones entre F, Sh y G son una medida directa de la confusión entre esas clases.
- Una trayectoria puede tener varios eventos y, por lo tanto, **varios procesos** (por ejemplo, F → A → F es deforestación y después regeneración).

### 7.4 Procesos

| Proceso | Eventos (origen → destino) |
|---|---|
| **Deforestación, variante estricta (D1)** | F → A |
| **Deforestación, variante amplia (D2)** | F → A, Sh → A |
| **Deforestación, variante muy amplia (D3)** | F → A, Sh → A, G → A |
| **Expansión urbana** | cualquier estado → U |
| **Degradación de bosque** | F → G, F → Sh, F → B |
| **Regeneración de bosque** | A → F, Sh → F, G → F |

- **Las tres variantes de deforestación se evalúan en paralelo, sin elegir una.** Responden a la confusión entre F, Sh y G en el producto: cuanto más amplia, más robusta a esa confusión y menos específica de bosque. Con D3, el proceso es más bien "conversión de vegetación natural a agricultura" (G → A incluye la expansión agrícola sobre pastizal).
- Un mismo evento puede contar para más de un proceso sólo entre las variantes D1–D3. Entre procesos distintos, los eventos no se superponen; sí puede haber varios procesos en una trayectoria (por ejemplo, F → Sh → A con los dos tramos persistentes es degradación y después deforestación D2/D3).
- **Procesos excluidos y por qué:**
  - **Retracción urbana:** el producto no la registra (no hay ningún paso de U a otro estado en Argentina).
  - **Degradación dentro del bosque** (pérdida de dosel sin cambio de clase): no es observable, porque los subtipos de bosque están agrupados en F.

### 7.5 Marcas
Los eventos se **marcan, no se excluyen**:
- **costura de producto:** año del evento 1995 o 2016;
- **cambio de sensor:** año del evento 1999 o 2000;
- **censura:** el tramo de destino está censurado a la derecha (§7.2), o el de origen a la izquierda.

Todas las métricas se calculan con todos los eventos (principal) y sin los eventos marcados (secundario). En particular, la costura de 2016 reclasifica hacia bosque y **puede inflar la regeneración**. Se reporta aparte, aunque esa inflación juegue en contra de la hipótesis de trabajo.

### 7.6 Catálogo descriptivo (antes de cualquier método)
Para Argentina y para el mundo, y por proceso y variante:
- superficie (km²) y número de trayectorias con el proceso;
- distribución del año del evento;
- fracción de eventos con cada marca;
- superposición entre procesos (trayectorias con más de uno);
- oscilaciones por par de estados.

Es una descripción del producto, no una evaluación de métodos. Sirve para decidir si un proceso tiene superficie suficiente para evaluarse (umbral: ver §13).

---

## 8. Espacios y tipologías

### 8.1 Espacios
| Espacio | Distancia | d | Semillas |
|---|---|---|---|
| AE | euclídea en z | **4** (principal); 16 (sensibilidad) | 0, 1, 2 |
| AE lineal | euclídea en z | 4; 16 | 0, 1, 2 |
| PCA | euclídea en z | 4; 16 | — |
| MCA | euclídea en z | 4; 16 | — |
| OM | matriz OM completa (`om_trate.npy`) | — | — |
| One-hot | Hamming (años distintos) | — | — |

d = 16 se agrega como sensibilidad porque es donde la Parte I mostró que el AE empieza a generalizar. El one-hot con Hamming es la representación sin compresión: se compara como un método más.

### 8.2 Algoritmo de agrupamiento (decidido 2026-10-08)
- **Principal: k-medoides**, 10 arranques (inicio k-medoides++, iteraciones alternadas; implementación de `scripts/clustering/p2_tipologias.py`), conservando el de menor costo.
- **Sensibilidad: jerárquico de enlace completo**, cortado en exactamente k grupos.

**Justificación.** En la evaluación el algoritmo se mantiene fijo para que las diferencias entre tipologías se puedan atribuir al espacio y no al algoritmo. Se usa k-medoides porque es el único algoritmo de uso estándar que:
- opera sobre una matriz de distancias arbitraria (necesario para incluir OM);
- permite fijar el número de grupos;
- representa cada grupo con una trayectoria observada.

| Algoritmo descartado | Motivo |
|---|---|
| k-medias, mezclas gaussianas | requieren coordenadas; no se aplican a OM |
| Ward | requiere distancia euclídea; no se aplica a OM |
| HDBSCAN, DBSCAN | no permiten fijar k y dejan trayectorias sin grupo; en el ejercicio 2000-2022 fue inservible en este universo |
| Jerárquico de enlace promedio | encadena: en el ejercicio 2000-2022 reunió el 35 % de las trayectorias en un solo grupo |

**Límite.** El orden entre espacios vale para k-medoides. Lo atenúan el análisis de sensibilidad y el antecedente de 2000-2022, donde k-medoides, k-medias y Ward dieron resultados prácticamente iguales sobre los embeddings.

### 8.3 Universo agrupado y ponderación (decidido 2026-10-08)
- Las **7.818 trayectorias dinámicas de Argentina**. Las 9 constantes se excluyen: no tienen eventos.
- **El agrupamiento no se pondera**: cada trayectoria distinta pesa 1, porque la tipología describe la variedad de dinámicas. **La evaluación sí se pondera por superficie** (§12): detectar un proceso es una cuestión de hectáreas.

### 8.4 Número de grupos
**k ∈ {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48}.** Los espacios se comparan siempre a igual k. En esta evaluación **no se elige k**: se reportan las curvas.

### 8.5 Evaluación de los espacios y tipología final: dos etapas distintas

| | Evaluación (Parte II) | Tipología final |
|---|---|---|
| Pregunta | ¿qué espacio produce tipologías que detectan mejor los procesos? | ¿cuál es la mejor tipología sobre el espacio elegido? |
| Qué varía | el espacio | el algoritmo de agrupamiento y k |
| Qué se mantiene fijo | el algoritmo | el espacio |

Dos reglas para la etapa final, fijadas desde ahora:
1. **El espacio final no se presupone.** Se elige según los resultados de las Preguntas 1 y 2; no tiene por qué ser el autoencoder.
2. **Las referencias externas no se usan a la vez para elegir y para validar.** Si se las usa para elegir el algoritmo o k, se dividen en una parte de selección y otra de validación (por bloques espaciales).

---

## 9. Nivel 1: recuperación de los procesos definidos sobre el producto

**Pregunta.** Con las etiquetas de §7, que son reglas aplicadas a la propia secuencia, ¿la tipología no supervisada reúne en algunos de sus grupos las trayectorias de cada proceso?

Una regla acierta por construcción; no se pregunta si el agrupamiento es mejor detector que la regla, sino **si encuentra solo los procesos** y cómo los organiza.

### 9.1 Asignación de grupos a procesos
- Para cada proceso P, una trayectoria es **positiva** si tiene al menos un evento de P.
- Un grupo se asigna a P si **al menos el 50 % de su superficie** son trayectorias positivas de P. Un grupo puede quedar asignado a varios procesos, o a ninguno.
- **Ajuste cruzado:** las trayectorias se dividen al azar en dos mitades (semilla fija). La asignación se estima con una mitad y las métricas se calculan con la otra, y viceversa; se reporta el promedio. Así, con k alto, un grupo chico no "acierta" sólo porque su asignación se estimó con sus propias trayectorias.

### 9.2 Métricas (por proceso y variante, ponderadas por superficie)
- **Recall:** fracción de la superficie positiva de P que cae en grupos asignados a P.
- **Precisión:** fracción de la superficie de los grupos asignados a P que es positiva de P. Si ningún grupo queda asignado, la precisión no se define y se reporta "sin grupo".
- **F1:** media armónica de las dos.
- **Fechado:** dentro de los grupos asignados a P, error absoluto medio entre el año del primer evento de P de cada trayectoria positiva y el año mediano (ponderado) de su grupo.
- **Fragmentación** (descriptiva): número de grupos asignados a P.

### 9.3 Piso
**Partición al azar** con los mismos tamaños de grupo que la evaluada (20 permutaciones; se reporta la media), con la misma asignación por ajuste cruzado.

---

## 10. Nivel 2: correspondencia con fuentes independientes

**Pregunta.** Los píxeles que la tipología asigna a un proceso, ¿son los que una fuente independiente registra con ese proceso?

### 10.1 Referencias
| Proceso | Referencia | Cobertura | Estado |
|---|---|---|---|
| Deforestación (D1–D3) | Monitor de Desmontes, Colección 13.0 | Chaco Seco argentino; desmontes anuales 2001-2022, los previos agrupados | disponible (`data/geo/`) |
| Degradación y regeneración | Global Forest Change (Hansen et al.): pérdida y ganancia de cobertura arbórea | global; período según la versión | a obtener |
| Expansión urbana | GHSL, superficie construida (épocas quinquenales) | global | a obtener |

- Antes de correr se verifica, para cada referencia por obtener, su resolución, su período y su cobertura en Argentina. Si una referencia no es utilizable, ese proceso queda sólo en el Nivel 1 y se registra como desviación.
- Ninguna referencia separa exactamente nuestros procesos. La pérdida de cobertura arbórea de Hansen no distingue deforestación de degradación, así que se compara con la **unión** de los dos procesos (los píxeles asignados a deforestación o a degradación). No se filtra la referencia con información de ESA, para no quitarle independencia. La ganancia de Hansen se compara con la regeneración.

### 10.2 Unidad y ventana
- **Unidad:** el píxel ESA (300 m) dentro de la cobertura de cada referencia. La referencia se lleva a la grilla ESA; un píxel es **positivo** si al menos el 50 % de su área tiene el proceso en la ventana.
- **Ventana:** la intersección entre 1993-2022 y el período de la referencia. Para el Monitor, la ventana fechada es 2001-2022; los desmontes previos a 2001 se evalúan aparte, como un solo período (1993-2000), sin fechado.

### 10.3 Techo del producto
Las reglas de §7 aplicadas directamente a los píxeles ESA, comparadas con la referencia: precisión, recall y F1. Es lo máximo que puede alcanzar una tipología basada en este producto. Separa el error de ESA del error del método.

### 10.4 Métricas de los métodos
- Cada píxel hereda el grupo de su trayectoria. La asignación de grupos a procesos es la del Nivel 1, estimada con todas las trayectorias. **La referencia no se usa para ajustar nada**, así que no hace falta dividirla.
- **Precisión, recall y F1** por superficie, en absoluto y **relativos al techo** (F1 del método / F1 del techo).
- **Fechado:** en los píxeles positivos en ambas fuentes, error absoluto medio entre el año mediano del grupo y el año de la referencia.

---

## 11. Nivel 3: robustez

- **Semillas del modelo (AE y AE lineal):** el rango de las métricas de los Niveles 1 y 2 entre las tres semillas. Además, por proceso, el **índice de Jaccard ponderado por superficie** entre los conjuntos de trayectorias asignados a P en cada par de semillas. Esto reemplaza al índice de Rand ajustado global: una tipología puede variar mucho en general y conservar el grupo de un proceso.
- **Arranques de k-medoides (todos los espacios):** el mismo Jaccard por proceso entre los 10 arranques.
- **Trayectorias no vistas (secundaria):** las 62.121 trayectorias del conjunto no visto (§3.1) se asignan al medoide más cercano en cada espacio (en OM, con la distancia OM entre cada trayectoria y los medoides). Se calculan las métricas del Nivel 1 con la asignación de grupos a procesos estimada en Argentina, ponderando por km². Mide si la tipología reconoce los procesos en dinámicas que no participaron del agrupamiento.

---

## 12. Ponderación, semillas y lectura (Parte II)

- **Ponderación principal: por superficie** (píxeles en Argentina; km² en el mundo). **Secundaria: por tipo.**
- **Semillas:** media de las tres y rango mínimo–máximo; **una diferencia menor que ese rango no se interpreta**.
- No hay una puntuación única: cada proceso, variante y nivel se reporta por separado, contra k.
- **Lectura:**
  - El Nivel 1 dice si el espacio permite que una tipología reúna los procesos que el producto registra.
  - El Nivel 2, si eso se corresponde con el terreno, siempre leído relativo al techo del producto.
  - El Nivel 3, si el resultado depende del azar.
  - Un espacio que gana en el Nivel 1 y no en el 2 organiza bien el producto, pero no agrega correspondencia con el terreno.

---

## 13. Decisiones registradas y pendientes

**Registradas (2026-10-09):**
- Procesos: deforestación (D1–D3, en paralelo), expansión urbana, degradación y regeneración de bosque.
- Persistencia mínima de 3 años; los estados transitorios no definen procesos.
- Los eventos en costuras y cambios de sensor se marcan, no se excluyen.
- La retracción urbana y la degradación dentro del bosque quedan fuera: el producto no las registra.

**Pendientes (antes de correr):**
1. **Umbral de superficie mínima** para evaluar un proceso, a fijar con el catálogo (§7.6) antes de ver ningún resultado de métodos.
2. **Disponibilidad de Hansen y GHSL** (§10.1).
3. **d de trabajo para la tipología final** (abierto desde la Parte I): esta evaluación lo informa con d = 4 y 16.

---

## 14. Desviaciones

*(Ninguna en la Parte II por ahora. En la Parte I: el análisis de estados de un año y el reentrenamiento del AE lineal, registrados en `p1_resultados.md` §7.)*

---

## Requisitos de implementación (no cambian el protocolo)

**Parte I**
- El AE lineal se reentrena con la misma configuración, guardando los pesos, para codificar las trayectorias del mundo.
- PCA y MCA se reajustan con Argentina (son determinísticos) y se proyectan las trayectorias del mundo; para MCA, con la proyección de filas suplementarias.
- El AE codifica el mundo con los pesos guardados (`models/autoencoder_v3/p1/ae_d<d>_s<s>.pt`).

**Parte II**
- Etiquetado de eventos y procesos (§7) para los censos de Argentina y del mundo.
- Un mapa de píxel a trayectoria para Argentina, en la grilla ESA (el censo actual cuenta píxeles por trayectoria, pero no guarda dónde están).
- Los polígonos del Monitor de Desmontes llevados a la grilla ESA completa del Chaco Seco (el preprocesamiento de v2 lo hacía sólo por zonas).
- Distancias OM entre las trayectorias no vistas y los medoides (§11).
