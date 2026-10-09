# Pregunta 1: capacidad de compresión de los métodos (resultados)

**Fecha:** 2026-10-08 (reducido el 2026-10-09 a la capacidad de compresión: la accesibilidad en z y las tipologías se eliminaron y se reemplazan por la evaluación por procesos de la Pregunta 2). Ejecuta la Parte I del [protocolo de evaluación](protocolo_evaluacion.md), fijada antes de correr. Reemplaza al informe anterior de P1 (compresión medida sólo dentro de Argentina), que queda en el historial (`5fae580`).

**Código y salidas**
- Modelos: `scripts/modelo/p1_compresion.py` (`train`, `linear`, `linae`; `ae_apply`, `lin_apply`, `pca_apply` y `mca_apply` los aplican a trayectorias nuevas). Lanzador de los autoencoders: `scripts/modelo/p1_correr_ae.py`.
- Evaluación: `scripts/validacion/evaluacion_pregunta1.py` (`no_vistos`, `codificar`, `reconstruccion`).
- Resultados: `data/autoencoder_v3/pregunta1/` (`p11_reconstruccion.csv`, `p11_parametros.csv`, `no_vistos.npz`; los códigos por modelo en `codigos/`, no versionados).
- Figuras: `scripts/viz/pregunta1_figuras.py` → `docs/autoencoder_v3/figuras/p1/`. Informe: `scripts/viz/pregunta1_informe.py`, que rellena `docs/autoencoder_v3/plantillas/p1_resultados.plantilla.md`.

**Cómo leer las cifras.** Las tablas cubren todas las d (1, 2, 3, 4, 7, …, 31). Las que no entran en el texto (métricas ponderadas por superficie, gradiente de novedad para todas las d y métricas secundarias) están en el [anexo](#anexo-tablas-completas). Salvo indicación, cada trayectoria distinta (*tipo*) pesa 1. Para el autoencoder (AE) y el AE lineal se da la media de las tres semillas y, entre paréntesis, el rango mínimo–máximo; **una diferencia menor que ese rango no se interpreta** (protocolo §5).

---

## 1. Resumen

1. **En las trayectorias de ajuste (Argentina) el autoencoder reconstruye todo con d ≥ 3**, como ya se sabía. Pero **esa fidelidad no se traslada a trayectorias que no vio**: con d = 4, la secuencia de estados se reconstruye bien en el 99 % de las trayectorias de Argentina y sólo en el 34 % de las trayectorias del mundo cuyo proceso sí existe en Argentina.
2. **Según la regla fijada en el protocolo (§6), con d = 4 el autoencoder reproduce lo que vio más que una regla general**: ya en las trayectorias que difieren de una argentina en un solo año (h = 1), la secuencia de estados cae de 0,99 a 0,53. Parte de esa caída se debe a que esas trayectorias suelen tener un estado de un solo año (análisis exploratorio, §3.4), pero aun sin ellas la fidelidad cae de 0,84 (h = 1) a 0,15 (h ≥ 4).
3. **La generalización mejora con d.** Con d = 16 a 31 el autoencoder reconstruye la secuencia de estados del 82–84 % de las trayectorias no vistas de proceso conocido, y casi no depende de la distancia a lo visto (sin tramos de un año: 0,99 con h = 1 y 0,84–0,88 con h ≥ 2).
4. **El autoencoder es el que más reconstruye con pocas dimensiones** en las trayectorias no vistas (por ejemplo, con d = 7: secuencia de estados 0,58 frente a 0,22 del AE lineal y ~0,02 de PCA y MCA). **Con d grande el AE lineal lo alcanza** (d = 31: 0,84 ambos) y, en los procesos que no existen en Argentina, lo supera (0,62 frente a 0,55).
5. **Consecuencia para la elección de d = 4.** d = 4 alcanza para reproducir Argentina, pero no para generalizar. Si la representación tiene que valer más allá de las trayectorias de ajuste, los resultados apuntan a d entre 16 y 31. Es una decisión abierta (§5).

![Secuencia de estados correcta](figuras/p1/fig2_secuencia_de_estados.png)

---

## 2. Qué se midió

Síntesis del protocolo; el detalle está en [`protocolo_evaluacion.md`](protocolo_evaluacion.md).

- **Métodos:** autoencoder (transformer, ~800.000 parámetros), AE lineal (misma pérdida), PCA y MCA del one-hot. Todos ajustados **sólo con las 7.827 trayectorias de Argentina**, con d ∈ {1, 2, 3, 4, 7, 10, …, 31}.
- **Evaluación:** las **62.121 trayectorias dinámicas del mundo que no existen en Argentina**, que ningún método vio. Se separan por:
  - **A:** si su *proceso* (la secuencia de estados sin fechas) existe en Argentina (43.086, "proceso visto") o no (19.035, "proceso nuevo");
  - **B:** **h**, cuántos de los 31 años difieren de la trayectoria argentina más parecida.
- **Reconstrucción.** Principales: **exactitud por año** (*cuándo*) y **secuencia de estados correcta** (*qué*). Secundarias: reconstrucción exacta, número de cambios, error de fechado y F1 macro por clase.

Tamaño de los modelos (número de parámetros):

{{T_par}}

---

## 3. Reconstrucción

### 3.1 Métricas principales

**Exactitud por año (*cuándo*)**

![Exactitud por año](figuras/p1/fig1_exactitud_por_anio.png)

Argentina (trayectorias de ajuste; referencia):

{{T_rec_acc_arg}}

Mundo no visto, proceso visto:

{{T_rec_acc_vis}}

Mundo no visto, proceso nuevo:

{{T_rec_acc_nue}}

**Secuencia de estados correcta (*qué*)**

Argentina (referencia):

{{T_rec_est_arg}}

Mundo no visto, proceso visto:

{{T_rec_est_vis}}

Mundo no visto, proceso nuevo:

{{T_rec_est_nue}}

Las mismas dos métricas ponderadas por superficie (tablas en el anexo A.2):

![Exactitud por año, por superficie](figuras/p1/fig1b_exactitud_por_anio_superficie.png)

![Secuencia de estados, por superficie](figuras/p1/fig2b_secuencia_de_estados_superficie.png)

Lectura:
- **La distancia entre Argentina y el mundo no visto es la medida de cuánto de la "compresión" era memoria.** Con d = 4, el AE pasa de 0,999 a 0,812 en exactitud por año y de 0,992 a 0,335 en secuencia de estados. Con d = 31, de 1,000 a 0,980 y de 1,000 a 0,840.
- **Con d bajo, el autoencoder es el método que más reconstruye en las trayectorias no vistas**, por márgenes muy superiores al rango entre semillas. Por ejemplo, en proceso visto con d = 7: exactitud por año 0,928 frente a 0,807 (AE lineal), 0,605 (PCA) y 0,555 (MCA); secuencia de estados 0,584 frente a 0,221, 0,018 y 0,023.
- **Con d alto la ventaja desaparece.** En proceso visto con d = 31, el AE y el AE lineal quedan iguales en las dos métricas. En proceso nuevo con d = 31, el AE lineal supera al AE en secuencia de estados (0,621 frente a 0,554) y PCA lo iguala.
- **El autoencoder se satura:** en proceso visto, la secuencia de estados deja de crecer a partir de d ≈ 16 (0,82–0,84), lejos del 1,00 de Argentina.

### 3.2 Reconstrucción exacta por dimensión

**Cada tipo pesa 1**

![Reconstrucción exacta por tipo](figuras/p1/fig3_exacta_por_tipo.png)

Argentina (referencia):

{{T_rec_ex_tipo_arg}}

Mundo no visto, proceso visto:

{{T_rec_ex_tipo_vis}}

Mundo no visto, proceso nuevo:

{{T_rec_ex_tipo_nue}}

**Ponderada por superficie** (área en km² en el mundo; píxeles en Argentina)

![Reconstrucción exacta por superficie](figuras/p1/fig4_exacta_por_superficie.png)

Argentina (referencia):

{{T_rec_ex_sup_arg}}

Mundo no visto, proceso visto:

{{T_rec_ex_sup_vis}}

Mundo no visto, proceso nuevo:

{{T_rec_ex_sup_nue}}

Lectura:
- **Por tipo**, la reconstrucción exacta de lo no visto es baja para todos: con d = 4 el AE reconstruye exactas el 14 % de las trayectorias de proceso visto, y con d = 31 el 65 %.
- **Por superficie, el autoencoder se separa mucho más de los métodos lineales**: en proceso visto con d = 31 reconstruye exacta el 76 % de la superficie, frente al 48 % del AE lineal y el 27–31 % de PCA y MCA. Las trayectorias no vistas que más superficie ocupan son las que el AE reconstruye mejor.
- En Argentina, por superficie, el AE ya es exacto con d = 2 (0,996); el AE lineal llega a 0,98 con d = 10.

### 3.3 Cómo cae la reconstrucción con la novedad

![Gradiente de novedad](figuras/p1/fig5_gradiente_novedad.png)

Secuencia de estados correcta, d = 4, por estrato:

{{T_grad_est_4}}

Exactitud por año, d = 4:

{{T_grad_acc_4}}

Secuencia de estados correcta, d = 31:

{{T_grad_est_31}}

Todas las d (media de semillas; tablas con rangos en el anexo A.1):

![Gradiente de novedad, secuencia de estados, todas las d](figuras/p1/fig5b_gradiente_secuencia_todas_d.png)

![Gradiente de novedad, exactitud por año, todas las d](figuras/p1/fig5c_gradiente_exactitud_todas_d.png)

![Gradiente de novedad, reconstrucción exacta, todas las d](figuras/p1/fig5d_gradiente_exacta_todas_d.png)

Lectura según la regla del protocolo (§6):
- **Con d = 4, el AE cae fuerte ya en h = 1** (secuencia de estados 0,529, frente a 0,992 en Argentina) y sigue cayendo con h (0,103 con h ≥ 6). Es el patrón que el protocolo identifica con reproducir lo visto más que una regla general.
- **Con d = 31, la caída en h = 1 es menor** (0,780) y la fidelidad no empeora al alejarse.
- **PCA y MCA con d = 31 mejoran con h**, lo que no es esperable si h midiera sólo novedad. El §3.4 lo explica.
- En proceso nuevo, con d = 4, ningún método reconstruye la secuencia de estados (≤ 0,05).

### 3.4 Análisis exploratorio: estados de un año *(no previsto en el protocolo)*

**Por qué.** El patrón de PCA y MCA del §3.3 sugirió que los estratos de h difieren en el tipo de trayectoria. Una trayectoria que difiere en un solo año de una argentina suele ser una argentina con un estado de **un solo año** intercalado (por ejemplo, bosque → rala un año → bosque). Esos tramos son difíciles de reconstruir para cualquier método y a menudo son ruido del producto.

Trayectorias con algún tramo de un año, por estrato:

{{T_tramos_comp}}

Secuencia de estados correcta, separando las trayectorias con tramos de un año:

{{T_tramos}}

Lectura:
- **h está mezclado con la presencia de tramos de un año** (47 % en proceso visto con h = 1; 0–6 % con h ≥ 2). Parte de la caída del AE en h = 1 se debe a eso: **las trayectorias con un tramo de un año casi no se reconstruyen con ningún método ni d** (AE con d = 31: 0,54; PCA y MCA: 0).
- **Aun sin esos tramos, el AE con d = 4 se degrada con la distancia**: 0,996 en Argentina, 0,843 con h = 1, 0,367 con h = 2–3 y 0,152 con h ≥ 4. La conclusión del §3.3 se mantiene.
- **Con d ≥ 16, el AE generaliza a trayectorias sin tramos de un año** (0,99 con h = 1; 0,84–0,88 con h ≥ 2). El AE lineal con d = 31 da lo mismo.
- PCA y MCA con d = 31 reconstruyen mejor que el AE las trayectorias alejadas sin tramos de un año (0,96 y 0,95 con h ≥ 4, frente a 0,88), y fallan por completo cuando hay un tramo de un año.
- Este análisis es **exploratorio**: se definió después de ver los resultados. Sugiere tratar los tramos de un año como un factor propio en lo que sigue (por ejemplo, con medidas como la turbulencia).

### 3.5 Métricas secundarias

d = 4:

{{T_sec_4}}

d = 31:

{{T_sec_31}}

Todas las d, por tipo y por superficie: anexo A.3.

- Con d = 4 el AE acierta el número de cambios en el 47 % de las trayectorias de proceso visto (AE lineal: 30 %) y, cuando acierta la secuencia de estados, fecha los cambios con un error medio de 0,5 años (AE lineal: 2,3). El error de fechado se calcula sólo donde la secuencia de estados es correcta, así que en cada método se promedia sobre un subconjunto distinto (protocolo §4.2).
- El F1 macro por clase muestra que las clases raras se conservan bastante mejor que las trayectorias completas: con d = 31 está entre 0,94 y 0,98 para todos los métodos en lo no visto.

---

## 4. Respuesta a la Pregunta 1

**¿Qué capacidad de compresión tiene cada método?**

- **Reconstrucción.** El autoencoder tiene la mayor capacidad de compresión **con pocas dimensiones** y es el único que reconstruye bien las trayectorias no vistas que más superficie ocupan. Pero su fidelidad con d bajo está atada a las trayectorias de ajuste: **con d = 4 reproduce Argentina, no una regla general**. Con d ≥ 16 generaliza a procesos conocidos, y ahí el AE lineal lo alcanza en las métricas por tipo.
- **En conjunto**, el autoencoder **comprime mejor con pocas dimensiones dentro de lo que vio**, y esa ventaja no se sostiene como generalización con d bajo. Si su representación sirve para detectar procesos se evalúa en la Pregunta 2.

---

## 5. Decisiones que quedan abiertas

1. **d de trabajo.** d = 4 se eligió con el informe anterior, que medía dentro de Argentina. Los resultados sobre trayectorias no vistas apuntan a d = 16–31 si la representación tiene que generalizar. Hay que decidir si d = 4 se mantiene para la tipología final.
2. **Estados de un año.** Afectan a todos los métodos y están mezclados con la novedad. Para la Pregunta 2 se resolvió con un criterio de persistencia mínima de 3 años (protocolo, Parte II).

## 6. Límites

- **Tres semillas**: los rangos son aproximados.
- **La configuración del autoencoder no se tuneó** (ancho, épocas, peso de ajuste). Un modelo regularizado o entrenado de otra forma podría generalizar mejor con d bajo.
- **El conjunto no visto es casi todo de trayectorias raras** (mediana de 2 a 9 píxeles por tipo) y viene de otras regiones del mundo: la generalización medida es a ese conjunto.
- **El análisis del §3.4 es exploratorio.**

## 7. Desviaciones del protocolo

- **§3.4 (estados de un año)**: análisis agregado después de ver los resultados; se presenta como exploratorio.
- **AE lineal**: se reentrenó con la misma configuración y semillas para guardar sus pesos (requisito de implementación del protocolo). Reproduce el anterior: reconstrucción idéntica en Argentina y códigos con diferencias de 2×10⁻⁵.

## 8. Cómo reproducirlo

```bash
python scripts/modelo/p1_correr_ae.py --workers 3                # autoencoders (GPU)
python scripts/modelo/p1_compresion.py linear                    # PCA y MCA, con su ajuste
python scripts/modelo/p1_compresion.py linae                     # AE lineal, con sus pesos
python scripts/validacion/evaluacion_pregunta1.py no_vistos
python scripts/validacion/evaluacion_pregunta1.py codificar
python scripts/validacion/evaluacion_pregunta1.py reconstruccion
python scripts/viz/pregunta1_figuras.py
python scripts/viz/pregunta1_informe.py                          # este informe (texto en docs/autoencoder_v3/plantillas/)
```

En CPU (8 núcleos), la codificación y la reconstrucción tardan alrededor de media hora.

---

## Anexo: tablas completas

### A.1 Gradiente de novedad, todas las d

Mundo no visto, cada tipo pesa 1, por estrato de h.

{{T_anx_grad}}

### A.2 Métricas principales ponderadas por superficie

{{T_anx_sup}}

### A.3 Métricas secundarias, todas las d

{{T_anx_sec}}
