# Plan autoencoder_v3: reformulación del paper

**Fecha:** 2026-10-02
**Estado:** propuesta de trabajo. Parte de la auditoría de v2 ([`reporte_auditoria.md`](../v2/reporte_auditoria.md)) y del censo de trayectorias de Argentina hecho el mismo día.

---

## 1. Por qué se reformula

La auditoría encontró que tres conclusiones del borrador (`docs/v2/paper_metodologia.md`) no se sostienen:
- la generalización a zonas fuera de dominio;
- la superioridad de las tipologías para detectar desmonte;
- la tipología Fina/HDBSCAN como la mejor.

El censo de trayectorias explica buena parte de lo primero: **el universo de dinámicas de ESA CCI en Argentina es chico y enumerable** (§2). Las zonas "nuevas" repiten las mismas trayectorias que el entrenamiento, y eso no cambia alargando la serie. Medido sobre las zonas actuales, el solapamiento queda en ~95 % para ventanas de 12 a 23 años, así que extender a 1992 no lo resuelve (§7).

Se reformula entonces el ejercicio como **descriptivo, no predictivo**. No interesa si el modelo funciona con datos nuevos. Interesa si la información de las secuencias se puede comprimir en un embedding, y qué descripción de las dinámicas produce esa compresión.

## 2. El universo de trayectorias (Argentina, ESA CCI 2000-2022)

Se calculó sobre `data/landcover_timeseries_2000-2022.nc`, recortado con `data/geo/ar_provinces.geojson`. **Corrección del 2026-10-03:** la primera versión del censo enmascaraba con el paso de las coordenadas float32, que se desvía de 1/360° y desplaza el borde hasta 5,6 px en latitud; daba 36.078.445 px. Con la máscara corregida (`preprocess.vector_mask`) son 36.084.989 px y el resto de las cifras se mueve menos de 0,1 %. Las 4.554 trayectorias no cambian.

| Medida | Valor |
|---|---:|
| Píxeles en Argentina | 36.084.989 |
| Trayectorias distintas | **4.554** |
| Constantes (23 años iguales) | 9 tipos, 95,3 % de la superficie |
| Dinámicas (al menos un cambio) | 4.545 tipos, 4,7 % de la superficie (1,68 M px) |
| Máximo de cambios por trayectoria | **3** (1 cambio: 1.092 tipos, 97 % de los px dinámicos; 2 cambios: 3.294; 3 cambios: 159) |
| Trayectorias que cubren el 50 % / 90 % de los px dinámicos | 36 / 281 |
| Trayectorias de un solo píxel | 1.003 |
| Trayectorias dinámicas presentes en las zonas de entrenamiento actuales | 1.357 (30 % de los tipos, 96 % de los px dinámicos) |
| Valores sin dato (Nd o relleno) | ninguno |

**Costura 2015/2016: artefacto confirmado** (verificado el 2026-10-02, §4.0). Los cambios de 2014→2015 son casi nulos (920 px) y los de 2015→2016 dan un salto (100.613 px). Coincide con la costura entre ESA CCI v2.0.7 (hasta 2015) y C3S v2.1.1 (desde 2016). Al concatenar se perdió la versión de cada año: el atributo global del `.nc` es el de 2000 (`v2.0.7cds`). Además, 1.437 tipos (~30,6 mil px) oscilan, es decir, vuelven a un estado anterior.

## 3. Preguntas de investigación

**P1. Compresión.** ¿Puede la información de las trayectorias de cobertura de Argentina 2000-2022 comprimirse en un embedding de baja dimensión? ¿Qué se preserva y qué se pierde?

**P2. Descripción.** ¿Qué descripción de las dinámicas de cobertura se obtiene sobre ese espacio comprimido? ¿En qué se diferencia de describirlas con el análisis de secuencias clásico (distancia OM) o con una codificación explícita (one-hot)?

**Cómo se conectan.** P1 habilita a P2. Si la compresión es fiel, la descripción sobre el embedding es legítima, y tiene sentido preguntar qué tipo de descripción produce y en qué difiere de las alternativas.

## 4. Diseño

### 4.0 El universo como resultado descriptivo
- Censo completo de las 4.554 trayectorias, ponderadas por superficie: concentración, número de cambios, procesos más frecuentes y su distribución espacial.
- Es la referencia contra la que se mide todo lo demás.
- **Costura 2015/2016 (verificada el 2026-10-02).** Tiene dos efectos:
  - **2015 es una copia de 2014.** Sólo cambian 920 px en el país y ninguno de bosque. En Chaco la referencia marca 5.651 px desmontados en 2015. CCI detecta ese desmonte recién en 2016-2018 (~35 %) y nunca en 2015. Para los demás años el retraso típico es de ~1 año.
  - **2016 trae una reclasificación hacia bosque.** Es el único año en que dominan las ganancias de F: Sh→F 36 %, A→F 17 %, Wt→F 5 % de los cambios, frente a 9 %, 5 % y 2 % en el resto de los años. Suma 60.231 px de ganancia de F en el país, 2,3 a 3,6 veces lo de los años vecinos (16.566 en 2016→17, 25.678 en 2013→14). Está repartida en todas las provincias con bosque o arbustal, incluso en San Luis (6.708) y La Pampa (1.460), así que es del producto y no un proceso regional. El 98,6 % persiste hasta 2022: es un cambio de nivel, no ruido.
- **Cuánto toca del universo:**
  - 655 tipos (14 %) y 100.613 px (6,0 % de los px dinámicos) cambian en 2015→16.
  - 184 tipos y 60.231 px (3,6 %) son ganancias de F en ese año.
  - 49 tipos y 87.653 px (5,2 %) tienen ese como único cambio.
- **Descarga nueva de 2015 y 2016 (2026-10-03).** Se bajaron de nuevo del CDS y se compararon byte por byte (`cmp`) con las copias de `raw_unzipped/`: ambas son idénticas (2.311.288.712 y 2.316.204.819 bytes). Los metadatos internos son coherentes (`time` = 2015-01-01 y 2016-01-01, `time_coverage_start` y versión correctos). Se descarta un archivo dañado o mal etiquetado: el 2015 casi congelado es lo que distribuye el CDS. El patrón es global, no sólo de Argentina: sobre el mapa mundial completo, los píxeles que cambian de código LCCS son 3,33 M en 2013→14, 0,49 M en 2014→15, 5,38 M en 2015→16 y 3,03 M en 2016→17.
- **No es un error de construcción del `.nc`** (verificado el 2026-10-02 con los mapas crudos de `data/ESA_data/`). `scripts/datos/build_landcover_nc.py` reconstruye el archivo desde los crudos y da los años 2000-2020 idénticos píxel a píxel, sin años corridos ni duplicados. En los crudos, 2014 y 2015 ya difieren en apenas 1.647 px de la ventana: el congelamiento de 2015 es del producto. Con los 23 globales de `raw_unzipped/`, la reconstrucción 2000-2022 completa también es idéntica, y los recortes `*_clipped_arg.nc` coinciden con los globales.
- **Cómo se arma 2016 en los crudos.** `observation_count`, `processed_flag` y `current_pixel_state` son idénticos en 2014, 2015 y 2016: es la misma base. `change_count` suma 1 en 2016 justo en los px que cambian de clase (220.627 de 220.675). Entonces C3S 2016 no es un mapa nuevo: es el mapa de 2015 más los cambios que detectó PROBA-V en su primer año. Los pares más frecuentes en la ventana son 120→60 (32 mil), 61→120, 50→100 y 30→50. Hipótesis sin verificar: v2.0.7 confirma cada cambio con años posteriores, así que en el último año de la serie (2015) casi no puede registrar cambios.
- **Tratamiento:** pendiente de decisión (§7).

### 4.1 P1: compresión
- **Datos de ajuste:** las 4.554 trayectorias deduplicadas, ponderadas por superficie (o con un tope de repeticiones). Así el entrenamiento corre en CPU.
- **Dimensiones:** d ∈ {1, 2, 3, 4, 6, 8, 12, 16}, con 3 semillas por d. Hay que medir el ruido entre réplicas, porque en el barrido anterior superaba a las diferencias entre configuraciones.
- **Comparación:** compresiones lineales de la misma dimensión, PCA del one-hot o análisis de correspondencias múltiples (MCA).
- **Qué se preserva**, ponderado por superficie y por tipo, y separando constantes de dinámicas:
  - reconstrucción por año y de la secuencia completa;
  - número de cambios, secuencia de estados (ignorando años) y error en el año de cada cambio;
  - clases raras;
  - macro F1 sobre las 9 clases presentes (sin Nd).
- **Preservación de estructura:** correlación entre distancias en el embedding y OM; solapamiento de vecindarios (kNN).
- **Referencia teórica:** la entropía del universo, que mide cuánta información hay que comprimir.
- **Resultado principal:** curva de fidelidad contra dimensión (autoencoder frente a lineal), con el detalle de qué información se pierde primero.

### 4.2 P2: descripción
- **Tipologías en tres espacios**, con la misma granularidad (3 niveles de k):
  - z, con la d elegida en P1;
  - OM, con clustering ponderado sobre las 4.554 trayectorias;
  - one-hot o su PCA.
- **Comparación:**
  - **Qué agrupa cada una.** Por ejemplo, si z agrupa "bosque → agricultura" sin importar el año del cambio mientras OM separa por año, o al revés.
  - Coincidencia entre tipologías (ARI ponderado).
  - Calidad medida en el espacio de secuencias: pseudo-R² y ASW **corregida** (denominador W_k−1).
  - Estabilidad, con submuestras reales del 80 % y `min_cluster_size` reescalado si se usa HDBSCAN.
  - Interpretabilidad: secuencias representativas y legibilidad de los tipos como procesos.
- **Control externo:** correspondencia con la referencia de desmonte, como prueba de validez de la descripción y **no** como competencia de detección. Hay que incluir R0' y las otras reglas triviales, tratar bien las constantes y declarar el techo (~0,64) que impone ESA CCI.
- **Producto:** mapas de la tipología para todo el país y actualización del visor y del atlas.

## 5. Qué pasa con el trabajo actual

**Se reutiliza:**
- el pipeline de extracción;
- la arquitectura del modelo;
- `seqdist`;
- `eval_desmonte`;
- el visor y el atlas.

**Hay que corregir sí o sí** (según la auditoría):
- La asignación por centroide de píxeles constantes a clusters de cambio (bloqueante B; regla en `reporte_auditoria.md` §2.3).
- La ASW de `seqdist.asw` y la cobertura por filas (EMB-05).
- La línea de base R0, mal emparejada; corregir también el docstring de `_r0_year` (bloqueante A).
- La estabilidad: hoy son submuestras de 20.000 filas, no del 80 % (CL-06, A6).
- La versión del producto en §1.1: 2016-2020 viene de C3S v2.1.1 (confirmado en los metadatos de los crudos), no de v2.0.7.
- La tabla de agregación de §1.2 no coincide con el `.nc`. El archivo usa la agregación IPCC estándar: 110 (mosaico herbáceo > árbol/arbusto) → G, no F; 160/170 (bosque inundado) → F, no Sp. Hay que corregir la tabla y el párrafo de las "tres decisiones". La regla real está en `LCCS_TO_STATE` de `scripts/datos/build_landcover_nc.py`.
- La pérdida de entrenamiento calculada en FP16, fuera del autocast (EMB-04). Incluir `clip_grad_norm` y scheduler (A3).

**Sale del texto:**
- "Generaliza a zonas fuera de dominio" y la evaluación OOD por zonas.
- "Dimensionalidad intrínseca baja" como lectura del barrido.
- "Las tipologías superan ampliamente las líneas de base".
- Fina/HDBSCAN como tipología ganadora.
- Periurbano Córdoba como evidencia de detección: la referencia no es visible en ESA CCI (VE-9).

## 6. Resultados posibles

Todos son publicables:

| Resultado | Lectura |
|---|---|
| El autoencoder comprime mejor que lo lineal y su tipología agrupa distinto que OM | El embedding aporta una descripción nueva; hay que caracterizar en qué. |
| El autoencoder ≈ lo lineal | El universo es tan simple que no hace falta no linealidad. Es un hallazgo sobre ESA CCI. |
| La tipología de z ≈ la de OM | El embedding reproduce el análisis clásico. Ojo: con 4.554 trayectorias OM también es barato, así que el argumento de escalabilidad es débil. |

## 7. Decisiones tomadas

- **Encuadre descriptivo, no predictivo** (2026-10-02).
- **v3 trabaja con el `.nc` reconstruido** (2026-10-03): `data/autoencoder_v3/landcover_timeseries_2000-2022_rebuild.nc`, armado desde los mapas crudos con `scripts/datos/build_landcover_nc.py`. Es idéntico píxel a píxel al original (`data/landcover_timeseries_2000-2022.nc`) en los 23 años y la misma grilla, pero trae la procedencia de cada año (archivo y versión), el dict de agrupación y los tokens en los atributos. El original queda para v1/v2, que siguen ejecutables. El código de v3 lo toma por defecto: `paths.NC_V3` es el default de `censo_trayectorias.py` y de la salida de `build_landcover_nc.py`; `paths.NC_FILE` (el original) queda para v1/v2. Con el censo leyendo la serie reconstruida, `universo_argentina.csv` y `universo_rectangulo.csv` salen idénticos byte a byte. Pendiente: regenerarlo si la descarga de 2014 difiere de las copias actuales.
- **No extender la serie a 1992 por ahora.** No reduce el solapamiento: ~95 % en ventanas de 12 a 23 años sobre las zonas actuales. Además requiere descarga desde Copernicus, los años 1992-1999 se apoyan en cambios detectados con AVHRR (~1 km) y la referencia de desmonte empieza en 2001. Queda como posible extensión para otra pregunta.

## 8. Orden de trabajo

1. ~~Verificar la costura 2015/2016.~~ Hecho el 2026-10-02 (§4.0); falta decidir el tratamiento.
2. P1 completo. Es barato y define la d.
3. Corregir los bugs de la auditoría (§5).
4. P2.
5. Escribir la metodología de v3 (en `docs/autoencoder_v3/`) con el nuevo encuadre, partiendo de `docs/v2/paper_metodologia.md`.

## 9. Insumos disponibles

- **Censo del universo:** `python scripts/datos/censo_trayectorias.py [--crecimiento]`. Escribe `data/autoencoder_v3/universo_argentina.csv` y `universo_rectangulo.csv`, con las columnas `traj_id`, `seqs`, `n_px`, `n_cambios`, `constante` y `visto_en_train`. También imprime el resumen de §2 y los cambios por año, que muestran la costura 2014-2016. Con `--crecimiento` agrega la tabla de §7.
- **Preprocesamiento (`land2vec.preprocess`):** funciones por archivo anual (`read_raw`, `group_codes` con un dict, `clip_to_vector` con cualquier vector, `save_year`; encadenadas en `process_year`) y sobre la serie (`stack_years`, `sequences_table`, `write_sequences`). Los anuales pesan ~8 MB (Argentina, polígono: ≤ 5 MB) frente a ~2,3 GB del crudo. Verificado: reproduce el `.nc` actual en los 23 años y la tabla de Chaco (1.424.457 filas) sin diferencias.
- **Reconstrucción del `.nc` desde los crudos:** `python scripts/datos/build_landcover_nc.py [--years 2000-2022]`, luego `--compare data/landcover_timeseries_2000-2022.nc`. Los crudos van en `data/ESA_data/`, que no se versiona. Registra en los atributos el archivo y la versión de cada año. También sirve para 1992-1999.
- **Modelo actual:** `models/v2/autoencoder_v2`. Carga en CPU con `cfg.device = "cpu"`.
- **Rutas:** todo en `src/land2vec/paths.py`. Los productos de v3 van en `data/autoencoder_v3/`, `models/autoencoder_v3/`, `notebooks/v3/` y `docs/autoencoder_v3/`.
