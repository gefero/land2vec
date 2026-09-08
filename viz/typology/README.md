# Navegador de tipologías de trayectoria (local)

`index.html` es un visor estático (sin build, sin dependencias) de las seis
tipologías de trayectoria de la v2 -- ver `docs/v2_autoencoder_training.md`
§7.5. Elegís una corrida, ves la grilla de clusters (etiqueta automática,
cronograma, tamaño) y hacés clic en uno para el panel completo: cronograma,
secuencia modal vs. prototipo decodificado, índices, las trayectorias más
frecuentes hasta el 80% del cluster, y las secuencias representativas
(`seqrplot`).

## Solo local, por ahora

Ni el navegador ni sus datos se publican. `typology_browser.json` contiene
trayectorias de uso del suelo **textuales** con sus conteos (no solo
agregados), así que está **gitignoreado** -- se genera y se mira en la máquina,
nada más.

```bash
python scripts/describe_clusters.py            # genera viz/typology/typology_browser.json
python -m http.server -d viz/typology 8000     # -> http://localhost:8000
```

O doble clic en `index.html` (arranca con datos de ejemplo) y el botón
**cargar JSON** con `viz/typology/typology_browser.json`.

## Si algún día se quiere publicar

Habría que decidir qué exponer (ver §7.5): el JSON actual trae ~566 de las
1.128 trayectorias dinámicas distintas, verbatim, con frecuencias por cluster.
Una versión publicable tendría que recortar `top_seqs_80` / `representantes` /
`vecinos` a solo prototipos y agregados, o cifrarse.
