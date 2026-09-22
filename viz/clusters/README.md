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
- **Imagen satelital de inicio/fin** — toggle `imagen satelital (inicio/fin)` con
  switch de año (`2000` / `2022`) y su opacidad, para inspeccionar visualmente
  el paisaje real detrás de una trayectoria o un cluster. Se arma con
  `scripts/fetch_zone_imagery_gee.py` (Google Earth Engine, ver más abajo) y es
  opcional: si no se generó ninguna, el control queda oculto. Va debajo del
  fondo de trayectorias constantes y encima de los tiles base.
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
python scripts/build_cluster_map.py                              # genera viz/clusters/data/*.{json,png}
python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP  # opcional: imagen satelital inicio/fin
python -m http.server -d viz/clusters 8001                       # -> http://localhost:8001
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

### Imagen satelital

El manifiesto (`data/imagery/index.json`) y el toggle del visor son los mismos
sin importar cuál de las siguientes vías haya generado cada PNG -- `index.html`
no distingue la fuente.

#### Vía recomendada: `scripts/fetch_zone_imagery_gee.py` (Google Earth Engine)

Landsat 5 (`LANDSAT/LT05/C02/T1_L2`) para 2000, Sentinel-2 SR armonizado
(`COPERNICUS/S2_SR_HARMONIZED`) para 2022, mediana de escenas libres de nubes
sobre una ventana de ±`--pad-months` (default 5) alrededor del año pedido.
Earth Engine resuelve el mosaico y la reproyección puertas adentro -- **no**
es la imagen "Google Earth" propiamente dicha (la mezcla Maxar/DigitalGlobe con
buscador de fechas históricas de la app de escritorio), sigue siendo
Landsat/Sentinel crudo, pero con mejor compositing que armarlo a mano.

Setup, una sola vez (gratis para uso no comercial/investigación):

```bash
pip install earthengine-api requests
python -c "import ee; ee.Authenticate(auth_mode='localhost')"
```

`auth_mode='localhost'` es importante: el modo por defecto (`'notebook'` si se
corre desde un notebook) usa un flujo OAuth que Google empezó a bloquear con el
cartel "Esta aplicación está bloqueada" -- no es un problema de la cuenta, es
el flujo viejo. Si hace falta, pide de paso darse de alta en Earth Engine y
vincular un proyecto de Google Cloud (gratuito) en
https://code.earthengine.google.com/register.

```bash
python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP
```

Sin `--zone`/`--year` corre las 8 zonas (7 OOD + `chaco_santiago_frontier`, la
de training) × (2000, 2022). Es **idempotente**: si `{zona}_{año}.png` ya
existe (y su entrada en el manifiesto), lo saltea -- se puede cortar a mitad de
camino y retomar después con el mismo comando; `--force` reprocesa igual.

Flags relevantes, todos con default razonable:

| Flag | Para qué |
|---|---|
| `--zone` (repetible) | limita a una o más zonas; sin esto, las 8 |
| `--year` (repetible) | limita a 2000 y/o 2022; sin esto, ambos |
| `--cloud-cover N` | nubosidad máxima admitida por escena (default 50). Subir a 80 en zonas con pocas escenas despejadas |
| `--pad-months N` | ancho de la ventana de fechas (default 5) |
| `--max-side N` | lado máximo del PNG en píxeles (default 2000) |
| `--landsat-fallback` | ver más abajo -- solo hace falta en 3 zonas puntuales |
| `--force` | reprocesa aunque ya exista el PNG |

**Límite de memoria del servidor de Earth Engine.** Un compuesto de mediana
sobre demasiadas escenas en una región grande tira `"User memory limit
exceeded"` (o, más raro, `503 Earth Engine memory capacity exceeded`, que
suele ser transitorio -- alcanza con reintentar). El script ya reparte un
presupuesto fijo de escenas **por tile** (`MAX_SCENES`, actualmente 160) en vez
de tomar las N menos nubladas de toda la colección: una zona grande cruza
varios tiles MGRS (Sentinel) o path/row (Landsat), y un tope global le daba
todo el cupo a 1-2 tiles con poca nube dejando el resto de la zona sin ninguna
escena -- eso se veía como agujeros negros rectangulares en el compuesto, no
como ruido. Si una zona puntual sigue fallando por memoria, `--max-side` más
chico reduce la grilla de salida y da margen.

**Costura en el límite de huso UTM (`--landsat-fallback`).** Tres zonas cruzan
un límite de huso -- `patagonia_estepa` (19/20), `chaco_santiago_frontier` y
`pampa_nucleo` (ambas 20/21) -- y la grilla de *tiles* MGRS de Sentinel-2, que
está definida por huso, deja sin cubrir una franja exacta en esa costura, sin
importar cuántas escenas o qué nubosidad se admita (es una falta de dato real,
no un parámetro para ajustar). `--landsat-fallback` arma un segundo compuesto
con Landsat 8/9 (grilla path/row, sin esa discontinuidad) y lo usa *solo* donde
Sentinel-2 quedó sin dato -- el resto de la imagen sigue siendo Sentinel-2 a
10 m. El parche queda visible (30 m, tono distinto) pero acotado a la costura.
Usarlo únicamente en esas 3 zonas:

```bash
python scripts/fetch_zone_imagery_gee.py --project TU_PROYECTO_GCP \
  --zone patagonia_estepa --zone chaco_santiago_frontier --zone pampa_nucleo \
  --cloud-cover 80 --landsat-fallback --force
```

#### Alternativa manual: Google Earth Pro

Si se necesita específicamente la imagen "real" de Google Earth (no
Landsat/Sentinel) o no se quiere dar de alta una cuenta de Earth Engine:

1. `python scripts/make_zone_kml.py` genera `viz/clusters/zonas_imagenes.kml`
   -- un rectángulo por zona (las 7 OOD + `chaco_santiago_frontier`, la de
   training) con vista cenital (`tilt=0`) ya calculada.
2. En Google Earth Pro: `Archivo > Abrir` ese KML, doble click en cada zona del
   panel *Lugares* para volar exacto a su rectángulo (queda top-down, no hace
   falta ajustar inclinación). Ocultar el propio KML antes de exportar para que
   no salga el borde amarillo en la captura.
3. Ícono de reloj (imágenes históricas): elegir la fecha disponible más cercana
   a 2000 y, aparte, a 2022 -- Earth Pro no tiene una escena para cada zona en
   el año exacto, se usa la más próxima.
4. `Archivo > Guardar > Guardar imagen` en cada fecha, nombrando el archivo
   `{zona}_2000.png` / `{zona}_2022.png` (el id de zona es el mismo de
   `data/index.json`, p. ej. `puna_noa`, o `chaco_santiago_frontier` para la de
   training). Todas las capturas juntas en una misma carpeta.
5. `python scripts/import_manual_imagery.py <carpeta>` las copia a
   `data/imagery/` y arma `data/imagery/index.json`. `--date
   puna_noa_2000=1999-08-15` (repetible) es opcional, solo para que el tooltip
   del visor muestre la fecha real de esa captura.

No hay corrección de perspectiva ni calibración de esquinas en esta opción: se
asume que el encuadre del KML (rectángulo + vista cenital) alcanza para que la
imagen se superponga razonablemente con los puntos de cluster.

#### Intento descartado: `scripts/fetch_zone_imagery.py`

Primera versión, vía el catálogo STAC de Microsoft Planetary Computer (sin
cuenta propia), armando la mediana entre escenas a mano en vez de delegarla en
un servidor. Salía con un desalineamiento visible (rayado) en zonas grandes --
cada escena se leía y reproyectaba por separado y no terminaban de calzar entre
sí. Queda en el repo solo como referencia; necesita `pystac-client` +
`planetary-computer` + `rasterio` (`requirements.txt`), no se recomienda usarlo.

## Solo local, por ahora

`viz/clusters/data/` está **gitignoreado**: son las coordenadas por parcela
cruzadas con la etiqueta de cluster. Se regeneran en la máquina. El visor abierto
con `file://` no puede hacer `fetch` de los JSON -- hay que servirlo por HTTP
(el comando de arriba).
