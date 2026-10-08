# Pruebas AE contra OM sin usar OM como juez (protocolo y resultados)

**Fecha:** 2026-10-08. Parte del plan de reformulación ([`plan.md`](plan.md)) y de los límites de P1 ([`p1_resultados.md`](p1_resultados.md) §2.9 y §7).
**Estado:** el §1 al §4 es el **protocolo, fijado antes de correr las pruebas**. Los resultados se agregan en el §5 sin modificar lo anterior; si algo del protocolo cambia, se anota como desviación.

## 1. Pregunta y principio

**Pregunta:** ¿qué aporta el autoencoder respecto de OM y de las alternativas lineales para describir las trayectorias de cobertura de Argentina 1992-2022?

**Por qué no basta la comparación con OM de P1.** Spearman y vecinos en común contra OM (P1 §2.9) usan OM como referencia: miden *parecido a OM*, no calidad. Aquí, cada prueba tiene una respuesta correcta definida por construcción o por los datos, y **OM entra como un competidor más**, evaluado con la misma vara que el autoencoder.

**Decisiones del usuario (2026-10-08):** (1) vale el criterio "el proceso importa más que la fecha"; (2) desfases de 1, 2, 5 y 10 años; (3) OM entra también como embedding, vía MDS clásico; (4) dimensiones d = 1, 4, 7 y 31.

## 2. Espacios que se comparan

Universo: las 7.827 trayectorias de `universo_argentina.csv` (7.818 dinámicas). Todas las distancias entre trayectorias son euclídeas en el espacio correspondiente, salvo OM.

| Espacio | Dimensión | Semillas | Notas |
|---|---|---|---|
| **AE** | d ∈ {1, 4, 7, 31} | 0, 1, 2 | `models/autoencoder_v3/p1/ae_d<d>_s<s>.npz`, clave `z` |
| **AE lineal** | d ∈ {1, 4, 7, 31} | 0, 1, 2 | `lin_d<d>_s<s>.npz` |
| **PCA**, **MCA** | d ∈ {1, 4, 7, 31} | — | determinísticos |
| **OM-MDS** | d ∈ {1, 4, 7, 31} | — | MDS clásico (doble centrado de D², primeros d autovalores positivos) de la matriz OM. Es la **mejor representación euclídea de OM con ese presupuesto de dimensiones**: da a OM las mismas d que a los demás |
| **One-hot crudo** | 341 | — | euclídea sobre el one-hot; referencia sin compresión |
| **OM** | — | — | la matriz completa `om_trate.npy`; referencia sin compresión |

Con semillas, se reporta la media y el rango (mín–máx). **Una diferencia menor que el rango entre semillas no se interpreta.** Ninguna prueba produce un puntaje único "ganador": se leen en conjunto.

## 3. Definiciones comunes

- **Proceso** de una trayectoria: la secuencia de estados sucesivos sin las fechas (por ejemplo, `F→A`). Dos trayectorias son del **mismo proceso** si tienen la misma secuencia.
- **Fechas de cambio:** el año desde el cual rige el nuevo estado (`F>A@1999` = agrícola desde 1999).
- **Desfase k** entre dos trayectorias del mismo proceso: el máximo, entre sus cambios, de la diferencia absoluta de fechas.
- **Ponderación:** se reporta con cada tipo pesando 1 (`tipo`, la principal) y con peso `n_px` (`px`). Los anclajes se ponderan, no los pares.
- **Empates:** una distancia igual cuenta como 0,5 donde se compara.

## 4. Las cinco pruebas

### 4.1 Pares con desfase (discriminación y tolerancia)
**Hipótesis normativa, declarada:** dos trayectorias del mismo proceso son más parecidas entre sí que dos de procesos distintos, **aunque el desfase sea grande**.

Para cada trayectoria dinámica `a` (ancla) y cada k ∈ {1, 2, 5, 10}:
- **Positivos:** las del mismo proceso con desfase exactamente k.
- **Negativos generales:** las de proceso distinto con el mismo número de cambios que `a`.
- **Negativos del mismo año** (sólo trayectorias de un cambio): las de proceso distinto cuyo cambio ocurre en el mismo año que el de `a`. Aíslan la identidad del proceso de la coincidencia de fechas.
- **AUC por ancla** = probabilidad de que un positivo esté más cerca de `a` que un negativo. Se promedia sobre anclas.

**Lectura principal (fijada):** AUC contra negativos del mismo año, k = 5 y k = 10, trayectorias de un cambio, peso `tipo`, d = 4. **Secundarias:** los demás k y d; trayectorias de dos cambios; los negativos generales; peso `px`; y la **tolerancia**: mediana de la distancia a positivos de desfase k, relativa a la mediana de la distancia entre pares al azar del mismo espacio (sin unidades), junto con su correlación con k.

### 4.2 Estabilidad entre semillas
Para AE y AE lineal, por d: proporción de los 10 vecinos más cercanos (en z) que coincide entre cada par de semillas (3 pares) y entre cada semilla y OM; y la disparidad de Procrustes entre semillas. **Criterio:** si la coincidencia entre semillas no supera claramente la coincidencia con OM, la geometría del espacio no es una propiedad reproducible del método.

### 4.3 Sondas lineales (y de vecinos) sobre z
Predecir desde z (estandarizado), sobre las trayectorias dinámicas, con validación cruzada de 5 particiones por tipo (semilla 0):
- estado inicial; estado final; número de cambios (1–4); proceso (para las de un cambio, 62 clases); año del primer cambio (regresión).
- Modelos: logística o Ridge (**sonda lineal**) y 5 vecinos más cercanos (**sonda local**).
- Métricas: exactitud (clasificación), error absoluto medio en años (regresión); tipo y `px`.
- Referencia superior: el one-hot crudo con la misma sonda lineal.

### 4.4 Análisis de desacuerdos
Entre AE (d = 4, semilla 0) y OM, y entre AE lineal (d = 4, semilla 0) y OM. Para cada trayectoria dinámica, sus 10 vecinos en z y sus 10 en OM:
- **z-cerca/OM-lejos:** vecinos en el top-10 de z con rango OM mayor que 100.
- **OM-cerca/z-lejos:** el inverso (rango en z mayor que 100).

Cada par se clasifica con un criterio fijado: *mismo proceso*; *mismos estados inicial y final pero distinta ruta*; *mismo final, distinto inicial*; *mismo inicial, distinto final*; *distinto inicial y final*. **Lectura principal:** la proporción de *mismo proceso* en cada conjunto. Si es alta en z-cerca/OM-lejos, z agrupa lo que OM separa por fechas (a favor de z según el criterio normativo). Si es alta en OM-cerca/z-lejos, z separa pares que OM junta correctamente. Se lee además una muestra de 20 pares por conjunto.

### 4.5 Coherencia espacial
Los píxeles adyacentes (vecinos de 4 direcciones) deberían tener trayectorias parecidas. Ningún espacio usa geografía, así que es un juez independiente.
- Se extraen del `.nc` los pares de píxeles adyacentes, dentro de Argentina, ambos dinámicos y con **trayectorias distintas** (los idénticos tienen distancia 0 en todos los espacios y no discriminan).
- Para cada espacio, se calcula el **percentil de cada par adyacente dentro de la distribución de distancias entre pares al azar** (2.000.000 de pares de píxeles dinámicos con trayectorias distintas, sorteados con probabilidad proporcional a `n_px`). Es adimensional y comparable entre espacios.
- **Lectura principal:** el percentil medio ponderado por número de pares (menor = más coherente) y la fracción de pares por debajo del percentil 10.

## 5. Resultados

*(Se agregan después de correr las pruebas. El §1 al §4 no se edita.)*
