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
- **Base**: OpenStreetMap, OSM Humanitarian o Esri World Imagery (satélite; los
  tiles propios de Google no se pueden embeber fuera de su API con clave).
- **Leyenda**: swatch + etiqueta automática (de `viz/typology/typology_browser.json`
  si está), ordenada por tamaño, clic para aislar un cluster.
- **Exportar vista**: PNG o JPG de lo que se ve, con pie (corrida, zona, leyenda
  compacta, barra de escala, atribución). `2×` compone con un nivel más de
  detalle para una salida nítida.

## Cómo levantarlo

```bash
python scripts/build_cluster_map.py           # genera viz/clusters/data/*.json
python -m http.server -d viz/clusters 8001     # -> http://localhost:8001
```

`build_cluster_map.py` solo usa la librería estándar (no necesita pandas/torch);
lee `data/clusters_*.zip` + `data/lat_long_df_*.zip`. Con `--only _medium` procesa
una sola granularidad/familia; con `--precision 4`, archivos más livianos.

## Solo local, por ahora

`viz/clusters/data/` está **gitignoreado**: son las coordenadas por parcela
cruzadas con la etiqueta de cluster. Se regeneran en la máquina. El visor abierto
con `file://` no puede hacer `fetch` de los JSON -- hay que servirlo por HTTP
(el comando de arriba).
