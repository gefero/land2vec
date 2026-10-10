# Tuneo del autoencoder (autoencoder_v3, 1992-2022)

**Fecha:** 2026-10-09.

**Estado:**
- Protocolo fijado antes de entrenar.
- Piloto hecho (§8).
- Búsqueda pendiente, con S_max = 80.000.

Todo lo que figura acá se fijó antes de ver resultados de las etapas que regula. Los cambios posteriores están en §11 (desviaciones), cada uno con su motivo y con lo que se había visto al decidirlo.

---

## 1. Para qué se tunea y qué se puede afirmar

### 1.1 El uso esperado
El autoencoder (AE) se va a usar para **agrupar trayectorias de uso del suelo de modo que los grupos expresen procesos reales en el territorio**: deforestación, degradación y regeneración de bosque, expansión urbana.

Por eso la configuración no se elige por la reconstrucción, sino por **cuán bien se agrupan los procesos sobre el espacio z** en trayectorias que el modelo no vio.

### 1.2 Dónde entran las etiquetas, y dónde no
Hay que distinguir dos momentos.

- **Entrenamiento: no supervisado.**
  - La pérdida es la entropía cruzada de la reconstrucción: para cada año de cada trayectoria, el AE asigna a partir de z una probabilidad a cada uno de los 9 estados, y la pérdida castiga darle poca probabilidad al estado verdadero.
  - Las etiquetas de proceso no aparecen en esa cuenta. Ninguno de los pasos de ajuste de los pesos "sabe" qué es una deforestación.
  - El criterio S tampoco podría usarse como pérdida: es un conteo de aciertos y no es derivable.
- **Selección: con etiquetas.**
  - Cada 5.000 pasos el modelo se congela, se calcula z para todas las trayectorias y se mide S (§4). Medir no modifica el modelo.
  - S se usa sólo para **elegir entre modelos ya entrenados**: qué configuración pasa a la confirmación y cuál se elige al final.

### 1.3 Qué se puede afirmar
La información que entra por las etiquetas es mínima: elegir una entre 24 configuraciones. Entrenar con etiquetas, en cambio, ajustaría cientos de miles de pesos para separar los procesos. Pero esa información es real.

- **El espacio elegido no es neutral.** Es el que, entre los candidatos, mejor alinea z con estos procesos.
- **Formulación correcta para el paper:** "un AE entrenado sin etiquetas, cuya configuración se eligió por su capacidad de agrupar procesos definidos por reglas". **No** "el AE descubre los procesos sin supervisión".
- **Las etiquetas no traen información externa.** Son reglas aplicadas a la propia secuencia (§2).
- **Las fuentes externas no se usan en el tuneo.** El Monitor de Desmontes, Hansen y GHSL quedan reservados para la evaluación, que es la prueba que vale.

### 1.4 Antecedente
El AE de la v3 anterior nunca se tuneó:
- heredó la arquitectura del barrido de v2, que no discriminaba entre configuraciones (todas dentro de ±0,003 de macro F1, una semilla por configuración, elección final por costo);
- en la evaluación de procesos quedó por debajo de PCA, MCA, OM y one-hot, sin poder distinguir si el problema era el método o la configuración.

Todo eso está archivado en el tag `v3-1992-2022-previo`.

---

## 2. Procesos

Definidos en `src/land2vec/procesos.py`.

- **Tramo:** racha de años consecutivos con el mismo estado.
- **Tramo persistente:** dura al menos 3 años. Las excepciones son por censura: el tramo que empieza en 1992, y el que empieza en uno de los dos últimos años y llega a 2022, cuentan como persistentes.
- **Evento:** el paso entre dos tramos persistentes de estados distintos. Los tramos transitorios intermedios se ignoran.
- **Positiva:** una trayectoria es positiva para un proceso si tiene al menos un evento de ese proceso.

| Familia | Proceso | Eventos |
|---|---|---|
| Deforestación | D1, estricta | F → A |
| | D2, amplia | F, Sh → A |
| | D3, muy amplia | F, Sh, G → A |
| Degradación | | F → G, F → Sh, F → B |
| Regeneración | | A, Sh, G → F |
| Expansión urbana | | cualquier estado → U |

- **Las tres variantes de deforestación** responden a la confusión del producto entre bosque (F), arbustal (Sh) y pastizal (G). Se evalúan en paralelo.
- **La regeneración incluye Sh → F.** En Argentina es la mayor parte de la regeneración.
- **Queda fuera la retracción urbana**, porque el producto no la registra. En todo el mundo hay 2 tipos de trayectoria con U → otro estado, que suman 1 km², frente a 517.000 km² de expansión urbana.

Todos los procesos pasan el umbral de evaluabilidad (≥ 1 % de la superficie dinámica y ≥ 100 tipos) en América Latina (`data/autoencoder_v3/latam/catalogo_procesos.csv`, generado por `scripts/validacion/catalogo_procesos.py`). Porcentajes sobre la superficie dinámica:

| Proceso | Tipos | Superficie |
|---|--:|--:|
| D1 | 3.231 | 33,7 % |
| D2 | 4.814 | 39,4 % |
| D3 | 5.320 | 39,7 % |
| Expansión urbana | 3.182 | 2,7 % |
| Degradación | 5.112 | 18,1 % |
| Regeneración | 8.093 | 25,8 % |

---

## 3. Universo y partición

### 3.1 Universo: América Latina y el Caribe
- **Qué es:** 22.299 tipos de trayectoria, de los cuales 22.290 son dinámicos.
  - Superficie: 20,8 millones de km², el 7,6 % dinámica.
  - Fuente: el censo `data/autoencoder_v3/latam/universo_latam.csv.gz`, hecho con `scripts/datos/censo_mundial.py`.
- **Máscara:** Natural Earth v5.1.2, *map units* con `REGION_WB = "Latin America & Caribbean"`; son 51 unidades (`scripts/datos/mascara_latam.py`).
  - Se usan las *map units* y no los países porque los países incluyen la Guayana Francesa, Guadalupe y Martinica dentro de Francia.
  - La máscara se dilata 5 km hacia el mar, para no perder la costa, pero no hacia los países vecinos.
- **Contiene a Argentina:** las 7.827 trayectorias de Argentina están todas en el universo.

**Por qué América Latina y no Argentina ni el mundo:**
- **Argentina** tiene poca variedad. En la v3 anterior, el AE con d bajo reproducía Argentina pero no generalizaba.
- **El mundo** tiene 69.948 tipos. Entrenar cuesta lo mismo, porque el costo depende de los pasos y no del tamaño del universo. Lo que no escala son las matrices de distancias n × n:
  - América Latina: ~2 GB;
  - el mundo: ~20 GB, más que la memoria disponible.
- **Fuera de América Latina, el 98,7 % de la superficie dinámica está en trayectorias que también aparecen en América Latina.** Los 47.650 tipos del mundo que no están en América Latina suman sólo el 1,1 % de la superficie dinámica mundial.
  - Probar con el mundo no visto mide si el modelo entiende **secuencias raras**, no si generaliza a **territorio nuevo**.
  - Esto importa para el diseño de la evaluación.

### 3.2 Partición
- **Por tipo:** 80 % entrenamiento, 20 % validación, con semilla fija (0). Resultado: 17.839 tipos de entrenamiento y 4.460 de validación.
- **Estratos:** combinación de familias presentes (deforestación según D3, degradación, regeneración, urbana; 16 combinaciones) × decil de superficie (km²) entre los dinámicos.
  - En cada estrato se toma al azar el 20 %, redondeado.
  - Los estratos de un solo tipo van a entrenamiento.
  - Así, cada proceso queda repartido 80/20.
- **Las 9 trayectorias constantes van a entrenamiento.** No tienen procesos.
- **El resto del mundo no participa del tuneo.**
- **Peso de ajuste en el entrenamiento:** min(n_px, tope), con el tope como hiperparámetro (§5). Sin tope, las constantes, que son casi toda la superficie, dominarían la pérdida.

---

## 4. Criterio de selección

### 4.1 Criterio principal S
Se calcula en cada evaluación, con los z de **todas** las trayectorias (entrenamiento y validación), obtenidos del modelo entrenado sólo con entrenamiento.

1. **Agrupar:** k-medias sobre z, con cada tipo pesando 1, 10 arranques y semilla fija, para cada **k ∈ {8, 12, 16, 24, 32, 48, 64}**.
2. **Asignar grupos a procesos:** para cada proceso, un grupo se asigna al proceso si al menos el 50 % de sus tipos de **entrenamiento** son positivos. Un grupo puede quedar asignado a varios procesos, o a ninguno.
3. **Predecir en validación:** un tipo de validación se predice positivo para el proceso si cae en un grupo asignado a ese proceso.
4. **F1 por proceso.** Se calcula sobre los 4.460 tipos de validación, cada uno con peso 1:

   F1 = 2 · aciertos positivos / (2 · aciertos positivos + falsas alarmas + omisiones)

   Vale 1 si el agrupamiento detecta todas las trayectorias del proceso sin falsas alarmas, y 0 si no detecta ninguna.
5. **Promediar:**
   - para cada k, el promedio de las 4 familias: deforestación (promedio de D1, D2 y D3), degradación, regeneración y expansión urbana. Así, la deforestación no pesa la mitad sólo por tener tres variantes;
   - **S = promedio sobre los 7 valores de k.**

La asignación se estima con entrenamiento y se evalúa con validación. Por eso un grupo chico no "acierta" sólo porque su asignación se calculó con sus propias trayectorias.

### 4.2 Por qué agrupamiento y no la sonda de vecinos
El protocolo original usaba una **sonda de 10 vecinos**:
- para cada tipo de validación, se buscan sus 10 vecinos más cercanos en z entre los de entrenamiento;
- se predice positivo si al menos 5 lo son.

Se descartó antes del piloto porque **satura** (desviación 1):

| Espacio | S de la sonda |
|---|--:|
| z al azar | 0,12 |
| AE con sólo 100 pasos de entrenamiento (prueba del código) | 0,85 |
| PCA, d = 8 | 0,90 |
| One-hot, sin comprimir | 0,95 |

- **La causa.** Los vecinos inmediatos de una trayectoria son casi copias suyas: la misma secuencia, con el cambio un año antes o después. Casi cualquier representación que conserve algo de la secuencia los pone cerca, aunque el resto del espacio esté desordenado. La sonda mide el orden **local**, y ese orden lo tienen casi todos los modelos.
- **La consecuencia.** Todas las configuraciones habrían quedado en una franja de ~0,1, con diferencias del orden del ruido entre semillas. Es el problema del barrido de v2.
- **Lo que mide el agrupamiento.** Mide el orden **a mayor escala**: si todas las deforestaciones quedan en una región de z, separadas de las regeneraciones. Eso es lo que importa para agrupar, y tiene escala: va de 0 (azar) a 0,53 (one-hot) (§4.6).

### 4.3 Por qué k-medias y estos k
**El algoritmo: k-medias.** Lo eligió el usuario.
- **Coherencia.** Es el algoritmo con el que se va a agrupar sobre z. Tunear con un algoritmo y agrupar con otro optimizaría para el que no se usa.
- **Encaje con z.** Trabaja directamente sobre vectores, con distancia euclídea.
- **Costo.** Cada evaluación tarda 4–6 s con los 7 valores de k (unas 700 evaluaciones en todo el tuneo, ~1 h de CPU repartida). K-medoides y el jerárquico necesitarían la matriz de distancias completa en cada evaluación.
- **Sesgo.** Favorece los espacios donde los grupos son nubes más o menos redondas y de tamaño parecido. Se acepta, porque es el algoritmo de uso.

**Los k: de 8 a 64.** Los fijó el usuario.
- **Con k muy chico**, hay más tipos de dinámica que grupos. Cada grupo mezcla procesos y casi ninguno llega a tener mayoría de uno: el F1 da 0 para muchos espacios.
- **Con k muy grande** (100 o más), los grupos se vuelven tan chicos como un vecindario y se vuelve a la saturación de la sonda.
- **No hay un k natural.** En el barrido de la v3 2000-2022, ninguna métrica mostró un codo.
- **Por eso se promedia** sobre una grilla que va de una tipología gruesa (8 grupos) a una fina (64), en vez de apostar a un valor.

### 4.4 Ruido del criterio
Medido con PCA d = 8, cambiando sólo la semilla de k-medias (5 semillas):
- con un solo k, el F1 promedio se mueve hasta 0,07;
- promediado sobre los 7 valores de k, **S varía 0,013** y S_sup, 0,024.

Es chico frente a la escala del criterio. El ruido entre semillas del modelo se mide en la confirmación (§6).

### 4.5 Por qué por tipo, y el control por superficie
- **Se pondera por tipo** porque, por superficie, cada proceso depende de pocas decenas de tipos efectivos: los de mayor superficie. En el primer pase de la v3 anterior, eso hacía saltar las métricas hasta 0,3 entre configuraciones vecinas.
- **S_sup** es el mismo S, pero con el F1 ponderado por km². Funciona como control: una configuración no se elige si su S_sup queda por debajo del de la mejor en más que el rango entre semillas de esa mejor.

### 4.6 Secundarios y referencias
**Secundarios.** Se reportan en cada evaluación, pero no deciden:
- la sonda de 10 vecinos (S_vecinos);
- el error de fechado de la sonda: en años, entre el año del primer evento y la mediana de ese año entre los vecinos positivos;
- la reconstrucción en validación y en entrenamiento: exactitud por año y fracción de secuencias exactas;
- el F1 de cada proceso para cada k.

**Referencias.** Se calcularon con la misma partición, antes de entrenar ningún modelo (`data/autoencoder_v3/tuneo/referencias.csv`):

| Referencia | S | S_sup | S_vecinos |
|---|--:|--:|--:|
| Piso: z al azar (gaussiano, d = 8) | 0,000 | 0,000 | 0,124 |
| Lineal: PCA del one-hot, d = 8 (ajustado con entrenamiento) | 0,415 | 0,540 | 0,900 |
| Sin compresión: one-hot de la secuencia | 0,529 | 0,622 | 0,947 |

Si el AE no supera al one-hot, comprimir no aporta para estas tareas. Eso es un resultado en sí.

---

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

- **Calendario de la tasa de aprendizaje**, igual para todas: calentamiento lineal durante el 5 % de S_max, y después coseno hasta el 1 % de la tasa. Se define sobre S_max desde el primer paso, así que una corrida cortada y retomada sigue exactamente igual.
- **El presupuesto se mide en pasos**, no en épocas. Con lote 512, cada paso ve cuatro veces más trayectorias.
- **n_head** queda fijo en 4.

---

## 6. Etapas

1. **Piloto** (hecho, §8). Configuración base, d = 8, una semilla, 100.000 pasos, evaluación cada 5.000.
   - Mide la velocidad: 41,9 pasos/s con un proceso en la GTX 1060.
   - Fija **S_max** como el primer múltiplo de 10.000 pasos en el que la fracción de secuencias exactas en validación alcanza el 99 % de su máximo en el piloto, acotado entre 20.000 y 100.000 (desviación 2). Resultado: **S_max = 80.000**.
2. **Búsqueda.** d = 8, una semilla, 24 configuraciones: la base más 23 sorteadas del espacio de §5 con semilla fija (sin repetir).
   - Las configuraciones se guardan en `data/autoencoder_v3/tuneo/plan_busqueda.json` la primera vez.
   - **Las 24 se entrenan hasta S_max y se comparan por su S en S_max** (sin rondas, desviación 3).
3. **Confirmación.** Las 4 de mayor S en la búsqueda, con 3 semillas (0, 1 y 2) y d ∈ {4, 8, 16, 20}, hasta S_max. Son 48 corridas.
4. **Elección.** Por el S medio entre semillas, promediado sobre los cuatro d (§7).
5. **Modelos finales.** La configuración elegida, reentrenada con todo América Latina (entrenamiento y validación), con d ∈ {2, 3, 4, 6, 8, 12, 16, 20, 24, 31} × 3 semillas. Son 30 corridas, hasta S_max, y son los modelos que entran a la evaluación.

---

## 7. Regla de decisión
- **Diferencias menores que el rango entre las 3 semillas no se interpretan.**
- **Desempates:** entre configuraciones empatadas, se elige la de menos parámetros; si también empatan en eso, la de lote más chico y después la de menos pasos.
- **El control por superficie** (§4.5) puede descartar una configuración, pero no elegirla.
- **d no se elige en el tuneo:** se informa S contra d, y la elección queda para la evaluación.

---

## 8. Resultados del piloto

### 8.1 Curva de entrenamiento
Configuración base, d = 8, semilla 0, 100.000 pasos (`data/autoencoder_v3/tuneo/metricas_piloto.csv`). Las tres primeras columnas son la reconstrucción en validación. Las cuatro últimas son el F1 de cada familia, promediado sobre los 7 k.

| Paso | Años correctos | Secuencias exactas | S | S_sup | Deforestación | Degradación | Regeneración | Urbana |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5.000 | 0,948 | 0,29 | **0,429** | 0,509 | 0,28 | 0,27 | 0,67 | 0,49 |
| 10.000 | 0,965 | 0,42 | 0,392 | 0,492 | 0,22 | 0,18 | 0,66 | 0,51 |
| 20.000 | 0,983 | 0,70 | 0,354 | 0,407 | 0,17 | 0,20 | 0,56 | 0,48 |
| 30.000 | 0,986 | 0,75 | 0,321 | 0,370 | 0,15 | 0,22 | 0,51 | 0,41 |
| 50.000 | 0,991 | 0,84 | 0,295 | 0,292 | 0,13 | 0,25 | 0,48 | 0,32 |
| 80.000 | 0,995 | 0,93 | 0,306 | 0,320 | 0,16 | 0,22 | 0,44 | 0,41 |
| 100.000 | 0,995 | 0,94 | 0,309 | 0,337 | 0,16 | 0,22 | 0,47 | 0,39 |

- **Reconstrucción.** El AE generaliza muy bien: reproduce exactas el 94 % de las trayectorias de validación, que nunca vio. En entrenamiento llega a 1,000.
- **Agrupamiento.** S **baja** a medida que el AE aprende a reconstruir: de 0,43 a ~0,30, donde se estabiliza desde los 40.000 pasos.
  - Ya a los 5.000 pasos queda por debajo del one-hot (0,529).
  - Desde los 10.000, también por debajo de PCA (0,415).
  - La deforestación es la familia peor agrupada.
- **Sonda.** S_vecinos se mantiene entre 0,94 y 0,96 durante todo el entrenamiento: confirma que la sonda no habría distinguido nada.

### 8.2 Diagnóstico: qué organiza z
Para entender la caída se midió con qué se corresponden los grupos de k-medias, con la información mutua normalizada (NMI; 0 = nada, 1 = todo), sobre los tipos dinámicos. Comparación: z del piloto en 100.000 pasos, PCA d = 8 y one-hot. Lo reproduce `scripts/validacion/diagnostico_z.py`.

| k | Espacio | Proceso | Estado inicial | Estado final | Par inicial-final | Año del 1er cambio |
|--:|---|--:|--:|--:|--:|--:|
| 8 | AE (piloto) | **0,06** | 0,17 | **0,04** | 0,16 | **0,10** |
| 8 | PCA, d = 8 | 0,18 | 0,15 | 0,14 | 0,19 | 0,03 |
| 8 | One-hot | 0,25 | 0,09 | 0,22 | 0,20 | 0,01 |
| 24 | AE (piloto) | **0,15** | 0,24 | **0,12** | 0,28 | **0,15** |
| 24 | PCA, d = 8 | 0,24 | 0,29 | 0,14 | 0,32 | 0,07 |
| 24 | One-hot | 0,25 | 0,24 | 0,20 | 0,32 | 0,09 |

**Interpretación:**
- **El AE organiza z por el estado inicial y por la fecha del cambio, y poco por el estado final.** Pone juntas, por ejemplo, las trayectorias que eran bosque y cambiaron alrededor de 2001, cualquiera haya sido su destino.
- **Los procesos se definen por el estado de destino.** F → A es deforestación y A → F, regeneración. Por eso el AE agrupa mal los procesos.
- **Por qué pasa.** Para reconstruir exacto, el AE tiene que guardar en z el año de cada cambio, y lo hace. La pérdida de reconstrucción organiza z **en contra** de esta tarea, y cuanto mejor reconstruye, más pesa la fecha.
- **No es un problema de fragmentación.** Los tres espacios reparten cada proceso en una cantidad parecida de grupos: con k = 24, entre 10 y 13 grupos cubren el 80 % de cada familia. La diferencia está en qué se junta dentro de cada grupo.

**Límites:**
- es un solo modelo (configuración base, una semilla, d = 8);
- el diagnóstico es exploratorio: se hizo después de ver la curva.

### 8.3 Consecuencias para el tuneo
- **S no sirve para decidir cuánto entrenar.** Su máximo está en la primera evaluación, y la regla original diría "entrenar lo mínimo". S_max pasa a fijarse por la reconstrucción (desviación 2): así se comparan sólo AE que efectivamente reconstruyen.
- **La búsqueda por rondas quedó sesgada.** Cortar a S_max/4 y seguir con las de mayor S favorecería a las configuraciones que aprenden más despacio (tasa baja, modelos grandes), no a las que organizan mejor los procesos. Las 24 se entrenan completas (desviación 3).

---

## 9. Qué se espera de la búsqueda y qué sigue

### 9.1 Lo que se espera
La búsqueda responde si **alguna configuración** de un AE de reconstrucción organiza mejor los procesos que la base: más regularización, otro peso de ajuste, otro pooling, otra capacidad.

Con lo que mostró el piloto, lo más probable es que ninguna supere al one-hot (0,529), y que la mejor quede cerca de PCA (0,415) o por debajo. Eso respondería por la negativa la pregunta de si un AE entrenado sólo con reconstrucción aporta para agrupar procesos.

### 9.2 Si ningún AE de reconstrucción supera al one-hot: cambiar lo que aprende el AE
Ya no sería tunear, sino rediseñar el método. La idea es que z guarde **qué pasó** (de qué estado a qué estado) más que **cuándo**. Tres variantes, de la más simple a la más ambiciosa:

1. **AE con ruido temporal (denoising).**
   - Al entrenar, la trayectoria de entrada se corre en el tiempo δ años al azar, con δ entre −m y +m; en los bordes se repite el primer o el último estado. Se le pide reconstruir la original.
   - Como no puede saber la fecha exacta, el AE deja de invertir z en ella.
   - Se pueden agregar ruidos que reflejan problemas conocidos del producto: estados de un año, o la confusión F ↔ Sh.
   - Es casi el mismo entrenamiento y entra en esta infraestructura con el ruido como hiperparámetro. Primer paso propuesto: tres pilotos con m = 1, 2 y 4.
2. **Aprendizaje contrastivo.**
   - Dos versiones de una trayectoria con fechas corridas deben dar z casi iguales, y trayectorias distintas deben quedar lejos. No necesita decodificador.
   - Va directo a la invariancia buscada, pero tiene más hiperparámetros y riesgo de que el espacio colapse.
3. **z dividido en z_proceso y z_fecha.**
   - Las versiones corridas comparten z_proceso; la fecha sólo puede ir a z_fecha.
   - Se agrupa con z_proceso si se quiere "deforestación sin importar el año", o con ambos si se quiere "deforestación de principios de los 2000".
   - No obliga a elegir entre detectar y fechar, pero es lo más complejo.

**Dos condiciones para cualquiera de las tres:**
- **El diseño incorpora conocimiento experto.** Decidir que un corrimiento de fecha no cambia el proceso, pero un cambio de estado final sí, es definir "el mismo proceso". Entra por el diseño, no por las etiquetas, y hay que declararlo. Hace todavía más necesaria la validación externa, para no limitarse a reaprender las reglas.
- **Hace falta una referencia nueva, sin entrenar:** "transiciones sin fecha".
  - Cada trayectoria se representa por qué pasos entre estados persistentes tiene (F → A, F → Sh, Sh → A, …), sin importar cuándo.
  - Si el AE con ruido no la supera, no aporta nada que las reglas no den.
  - Lo que sí podría aportar: robustez al ruido del producto, parecidos graduales entre procesos (F → A se parece más a F → Sh → A que a A → F) y combinaciones no previstas por las reglas.

**Preguntas abiertas antes de diseñarlo:**
- ¿Qué cambios debería tolerar z sin moverse? Corrimientos de fecha de cuántos años, estados de un año, confusión F/Sh.
- ¿La fecha es algo por lo que agrupar, o sólo para caracterizar un grupo ya formado (por ejemplo, "este grupo de deforestación ocurrió sobre todo en 2001–2005")?

---

## 10. Ejecución
- **Interrupción y reanudación.** Todo se puede interrumpir y retomar con el mismo comando:
  - checkpoint atómico por corrida, que incluye el optimizador y los estados de los generadores aleatorios;
  - el orden de los lotes depende sólo de (semilla, época), y la tasa de aprendizaje, sólo del paso;
  - parada ordenada con Ctrl+C, SIGTERM o un archivo `models/autoencoder_v3/tuneo/STOP`;
  - parada por horario con `--max-horas` o `--hasta`;
  - estado guardado en disco, incluida la máquina en la que corre cada corrida.
- **Prueba en CPU.** Una corrida de 200 pasos sin cortes, la misma cortada con SIGTERM, con `STOP`, con `kill -9`, y entrenada en dos tramos (hasta 100 pasos y después hasta 200): los pesos finales dieron idénticos bit a bit.
- **Código:**
  - `scripts/modelo/tuneo_ae.py`, con los subcomandos `particion`, `referencias`, `piloto`, `busqueda`, `confirmar`, `final`, `estado` y `resumen`;
  - `src/land2vec/criterio_procesos.py`;
  - `src/land2vec/universos.py`;
  - `scripts/validacion/diagnostico_z.py`;
  - para Mendieta (CCAD-UNC), `scripts/cluster/`.
- **Salidas:** `models/autoencoder_v3/tuneo/` (modelos y checkpoints) y `data/autoencoder_v3/tuneo/` (partición, referencias, planes y métricas).
- **Costo.** El piloto corrió a 41,9 pasos/s con un proceso en la GTX 1060 (100.000 pasos en ~40 min).
  - La búsqueda (24 × 80.000 = 1,92 millones de pasos) se estima en 6–9 horas en esa GPU con 3–4 corridas en paralelo, y menos en Mendieta.
  - Las configuraciones grandes (n_embd 256, 4 capas, lote 512) son más lentas.
  - En CPU, la búsqueda tardaría días: unos 3 pasos/s por proceso.

```
python scripts/modelo/tuneo_ae.py busqueda --s-max 80000 --workers 3   # --s-max sólo la primera vez
python scripts/modelo/tuneo_ae.py estado
python scripts/modelo/tuneo_ae.py resumen
```

---

## 11. Desviaciones

1. **2026-10-09, antes del piloto: el criterio principal pasó de la sonda de 10 vecinos al agrupamiento con k-medias.**
   - **Motivo:** la sonda satura (§4.2).
     - Con la partición fijada dio 0,124 con z al azar, 0,900 con PCA d = 8 y 0,947 con el one-hot.
     - Una corrida de prueba del AE con sólo 100 pasos (reconstrucción por año 0,61) ya daba 0,852.
     - Todas las configuraciones habrían caído en una franja de ~0,1.
   - **Qué se había visto de un AE al decidir:** sólo esa corrida de prueba de 100 pasos, usada para probar el código.
   - **Cambios:**
     - k-medias pasa de 4 a 10 arranques;
     - k pasa de {12, 24, 48} a {8, 12, 16, 24, 32, 48, 64};
     - S pasa a ser el promedio sobre esos k;
     - la sonda queda como secundaria (S_vecinos).

     El algoritmo y los k los eligió el usuario (§4.3).
2. **2026-10-09, después del piloto: S_max se fija por la reconstrucción, no por S.**
   - **Motivo:** S bajó a medida que el AE aprendía a reconstruir; su máximo está en la primera evaluación (§8.1, §8.3).
   - **Regla nueva:** el primer múltiplo de 10.000 pasos con secuencias exactas en validación ≥ 99 % de su máximo en el piloto. El máximo fue 0,935; el umbral, 0,926; resultado: **S_max = 80.000** (0,930).
3. **2026-10-09, después del piloto: la búsqueda no usa rondas.**
   - **Motivo:** como S es más alto cuanto menos entrenado está el AE, las rondas favorecerían a las configuraciones que aprenden más despacio (§8.3).
   - **Cambio:** las 24 configuraciones se entrenan hasta S_max y se comparan ahí. Costo: 1,92 millones de pasos en vez de 1,44 millones.
