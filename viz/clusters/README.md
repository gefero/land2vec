# Mapa interactivo de clusterizaciones (local)

`index.html` es un visor estático (Leaflet vendorizado, sin build, sin CDN) de
las seis clusterizaciones de trayectoria de la v2 -- la matriz de 3
granularidades × 2 familias de `scripts/tune_clustering.py --select`, ver
`docs/v2_autoencoder_training.md` §7.2 -- **sobre el mapa real** de las 7 zonas
de evaluación out-of-domain.

Complementa a `viz/typology/`: aquel responde *cómo es* cada cluster
(cronograma, secuencia modal, índices); éste, *dónde cae*.

## Qué muestra

- **Corrida**: granularidad (`fina` / `media` / `gruesa`) × familia
  (`HDBSCAN` / `paramétrico`).
- **Set de etiquetas** — los dos archivos que deja `--select` por celda:
  - `dinámico · ajuste` (`clusters_dynamic*.zip`): las ~107k secuencias con
    transición sobre las que se **ajustó** el clustering (in-sample).
  - `pool · aplicado` (`clusters_pooled_subsampled*.zip`): ~126k parcelas
    (incluye las constantes submuestreadas al 15%), **asignadas** por centroide
    más cercano; las que quedan lejos de todo centroide salen como `−1` ("sin
    tipificar"). Ver docs §7.2, Nota metodológica.
- **Vista**: zona (con zoom a su bbox), **modo de color** (`proceso` / `cluster`),
  tamaño y opacidad del punto, el toggle **`tamaño = píxel real (300 m)`** — el
  marcador escala con el zoom para cubrir la huella del píxel ESA CCI
  (`src/land2vec/extract.py`), así en `dinámico` y en las zonas densas los puntos
  se tocan en vez de verse como confeti; con el modo activo el deslizador de
  tamaño pasa a ser un factor (`2.5` = 1× la huella real) — y el toggle
  **`fondo: trayectorias constantes`** con su slider de opacidad.
- **Color por proceso** (por defecto): cada cluster se agrupa en 1 de 10 procesos
  conceptuales (deforestación / pérdida de bosque, degradación forestal, expansión
  agrícola sobre pastizal/estepa, pérdida de vegetación, revegetación, regeneración
  de bosque, dinámica de agua/humedal, urbanización, oscilante, otro) según su
  secuencia modal `inicio»fin`, con hue por proceso y luminosidad/croma por
  antigüedad del cambio (reciente = claro, viejo = oscuro). **Deforestación** y
  **expansión agrícola** se distinguen por el origen: la primera arranca en bosque
  (el impacto es el bosque perdido), la segunda avanza sobre pastizal/arbustal sin
  tocar bosque. Color generado en OKLCh; la clasificación (`classify_process`), las
  glosas y la paleta viven en `scripts/build_cluster_map.py` y se validan con
  `scripts/check_cluster_palette.py`. El modo `cluster` vuelve a la paleta
  cualitativa por id.
- **Click en un píxel**: popup con su trayectoria cruda de 23 años (`F-F-…-A-A`),
  la forma colapsada (`F»A`), la etiqueta del cluster y el proceso. El click
  **suma esa trayectoria a la selección** (multi-select: el 1er click deja solo
  esa, los siguientes se acumulan; los clusters ocultos siguen siendo
  clickeables). Botón `✕ ver todas` arriba a la izquierda para resetear. Las
  secuencias van deduplicadas en `{set}{suffix}.json` (clave `seqs`); cada punto
  guarda `(lat, lon, seqIdx)`.
- **Fondo de trayectorias constantes**: los píxeles cuya cobertura no cambió en
  2000–2022 (el ~97% del territorio, que no se dibuja como puntos) como un raster
  de fondo — un PNG indexado por zona (grilla ESA CCI de 300 m reconstruida,
  encoder stdlib con `zlib`), coloreado por su único estado con una paleta pálida
  **integrada al sistema de procesos** (bosque = verde como "regeneración de
  bosque", agua = azul como "dinámica hídrica"…, pero mucho más claro). Se genera
  de `data/id_seqs_text_*` + `lat_long_df_*`, sin re-correr el modelo.
- **Base**: OpenStreetMap, OSM Humanitarian o Esri World Imagery (satélite; los
  tiles propios de Google no se pueden embeber fuera de su API con clave).
- **Leyenda**: en modo `proceso`, agrupada por proceso (cabecera = color base;
  clic en la cabecera aísla el proceso, clic en una fila agrega/quita el cluster
  de la selección — igual que el click en el mapa). En modo `cluster`, lista
  plana. Etiqueta automática de `viz/typology/typology_browser.json` si está.
- **Exportar vista**: PNG o JPG de lo que se ve, con pie (corrida, zona, leyenda
  compacta de procesos/clusters + la fila de estados del fondo si está activo,
  barra de escala, atribución, logo de factor~data). El fondo de constantes se
  compone en el export entre los tiles y los puntos. `2×` compone con un nivel
  más de detalle.
- **Barra lateral redimensionable**: se arrastra su borde derecho; el ancho se
  guarda en `localStorage`.

## Cómo levantarlo

```bash
python scripts/build_cluster_map.py           # genera viz/clusters/data/*.{json,png}
python -m http.server -d viz/clusters 8001     # -> http://localhost:8001
```

`build_cluster_map.py` solo usa la librería estándar (no necesita pandas/torch);
lee `data/clusters_*.zip` + `data/lat_long_df_*.zip` + `data/id_seqs_text_*.zip`
(trayectorias crudas para el popup + fondo de constantes) y, si está,
`viz/typology/typology_browser.json` (etiquetas y proceso de cada cluster). Con
`--only _medium` procesa una sola granularidad/familia; con `--precision 4`,
archivos más livianos; `--no-constants` salta el raster de fondo,
`--only-constants` regenera solo ese. `scripts/check_cluster_palette.py` reporta
la uniformidad perceptual (ΔE) de las rampas de color por proceso y de la paleta
del fondo.

## Solo local, por ahora

`viz/clusters/data/` está **gitignoreado**: son las coordenadas por parcela
cruzadas con la etiqueta de cluster. Se regeneran en la máquina. El visor abierto
con `file://` no puede hacer `fetch` de los JSON -- hay que servirlo por HTTP
(el comando de arriba).
