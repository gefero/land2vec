# Protocolo de evaluación de las representaciones (autoencoder_v3, 1992-2022)

**Fecha:** 2026-10-08. **Estado:** fijado **antes** de implementar y correr. Cubre la Pregunta 1.1; la 1.2 y la Pregunta 2 se agregarán como secciones nuevas. Cualquier cambio posterior se registra en §8 (desviaciones) sin reescribir lo anterior.

Antecedentes: [`p1_resultados.md`](p1_resultados.md) (compresión dentro de Argentina, 1992-2022). Reemplaza a las pruebas "AE contra OM" del 2026-10-08, descartadas (commits `69cc1e6` y `85ce841`, eliminadas en `7b78f76`).

---

## 1. Qué significa "describir mejor"

Se evalúan dos dimensiones, por separado:

- **Pregunta 1, dimensión interna:** ¿qué capacidad de compresión de la información de las trayectorias tiene cada método?
  - **1.1** Sobre las representaciones continuas ("en crudo"): reconstrucción y accesibilidad de la información.
  - **1.2** Sobre las tipologías de k grupos que se obtienen de cada método *(pendiente de diseño)*.
- **Pregunta 2, dimensión externa:** ¿qué capacidad tiene cada método de detectar procesos registrados por fuentes independientes, como el Monitor de Desmontes? *(pendiente de diseño)*.

**Principio:** todos los métodos se comparan en pie de igualdad. Ninguno funciona como juez de los demás; en particular, **OM es un método más**, no una referencia de lo correcto. En la 1.1 OM no participa porque no produce una representación continua; participa en la 1.2.

---

## 2. Pregunta 1.1: métodos

| Método | Qué es | Semillas |
|---|---|---|
| **AE** | autoencoder transformer (configuración de P1: ancho 128, 2+2 bloques, 400 épocas, lote 128, lr 1e-3, peso min(n_px, 100)) | 0, 1, 2 |
| **AE lineal** | codificador y decodificador lineales sobre el one-hot, misma pérdida y pesos que el AE (Adam lr 1e-2, 3.000 pasos) | 0, 1, 2 |
| **PCA** | componentes principales ponderados del one-hot | — |
| **MCA** | análisis de correspondencias múltiples ponderado del one-hot | — |

- **Dimensiones:** d ∈ {1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31}. La zona de interés es d ≤ 7.
- **Ajuste:** todos los métodos se ajustan **sólo con el universo de Argentina** (7.827 trayectorias de `universo_argentina.csv`). Los AE de d ≠ 2, 3 ya están entrenados (P1); los de d = 2 y 3 se entrenan con la misma configuración.
- **Tamaño de cada método:** se reporta el número de parámetros junto a los resultados (el AE tiene unos 800.000; PCA con d = 4, unos 1.700), como información, no como criterio.

---

## 3. Pregunta 1.1: trayectorias de evaluación

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

## 4. Pregunta 1.1: reconstrucción

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

## 5. Pregunta 1.1: accesibilidad de la información en z

**Qué mide.** La reconstrucción mide si la información es *recuperable* con el decodificador. Aquí se mide si es **accesible** directamente en z, con una operación simple. Importa porque la Pregunta 1.2 agrupa sobre z, sin decodificador.

### 5.1 Tareas (etiquetas calculadas de la propia trayectoria)
| Tarea | Tipo | Dimensión |
|---|---|---|
| Estado inicial | clasificación (9 clases) | *qué* |
| Estado final | clasificación (9 clases) | *qué* |
| Número de cambios | clasificación | *qué* |
| Proceso | clasificación | *qué* |
| Año del primer cambio | regresión (años) | *cuándo* |

"Proceso" se evalúa **sólo en el estrato de proceso visto**: un proceso nuevo no está entre las clases con que se ajusta la sonda.

### 5.2 Sondas
- **Principal: vecinos.** 5 vecinos más cercanos, distancia euclídea **sobre z sin reescalar** (la misma geometría que usará el agrupamiento de la 1.2). Clasificación por voto; regresión por promedio.
- **Secundaria: lineal.** Regresión logística multinomial (clasificación) o regresión lineal con penalización L2 (año), con z estandarizado con la media y el desvío de Argentina. Regularización fija: C = 1 (logística) y α = 1 (lineal).
- No se usa una red neuronal como sonda: con capacidad suficiente recuperaría lo mismo que el decodificador.

### 5.3 Ajuste y evaluación
- Las sondas se **ajustan con las 7.818 trayectorias dinámicas de Argentina** (cada tipo pesa 1) y se **evalúan en las no vistas del mundo**, por estrato.
- **Métricas:** exactitud balanceada (promedio de la tasa de acierto de cada clase presente en el conjunto evaluado) para las clasificaciones; error absoluto medio en años para el año del primer cambio.
- **Líneas de base:** predecir siempre la clase más frecuente en Argentina y el año promedio de Argentina.

---

## 6. Ponderación y semillas

- **Ponderación principal: por tipo** (cada trayectoria distinta pesa 1). Las no vistas son casi todas raras (mediana de 2 a 9 píxeles por tipo), así que la superficie dice poco.
- **Secundaria: por superficie**, con el área en km² del censo mundial (el área del píxel varía con la latitud).
- **Semillas:** para el AE y el AE lineal se reporta la media de las tres semillas y el rango mínimo–máximo. **Una diferencia menor que ese rango no se interpreta.**

---

## 7. Cómo se leen los resultados

- No hay una puntuación única. Se reportan las curvas de cada métrica contra d, por estrato.
- **Interpretación fijada:** un método que comprime información general de las trayectorias debería mantener la fidelidad en el estrato de proceso visto y degradarse de forma gradual con h. Una caída fuerte ya en h = 1 indica que el método reproduce lo que vio más que una regla general.
- El estrato de proceso nuevo mide extrapolación. Un mal resultado ahí no se interpreta como falta de compresión, sino como falta de generalización a procesos ausentes del ajuste.

---

## 8. Desviaciones

*(Ninguna por ahora.)*

---

## Requisitos de implementación (no cambian el protocolo)
- El AE lineal no guardó sus pesos en P1, sólo sus códigos y reconstrucciones: hay que reentrenarlo con la misma configuración guardando los pesos para codificar las trayectorias del mundo.
- PCA y MCA se reajustan con Argentina (son determinísticos) y se proyectan las trayectorias del mundo; para MCA, con la proyección de filas suplementarias.
- El AE codifica el mundo con los pesos guardados (`models/autoencoder_v3/p1/ae_d<d>_s<s>.pt`).
