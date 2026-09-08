# Navegador de tipologías de trayectoria

`index.html` es un visor estático (sin build, sin dependencias) de las seis
tipologías de trayectoria de la v2 -- ver `docs/v2_autoencoder_training.md`
§7.5. Elegís una corrida, ves la grilla de clusters (etiqueta automática,
cronograma, tamaño) y hacés clic en uno para el panel completo: cronograma,
secuencia modal vs. prototipo decodificado, índices, las trayectorias más
frecuentes hasta el 80% del cluster, y las secuencias representativas
(`seqrplot`).

## Datos

El visor consume `typology_browser.json`, que **no está versionado hasta que lo
generás**:

```bash
python scripts/describe_clusters.py        # escribe docs/typology/typology_browser.json
git add docs/typology/typology_browser.json && git commit && git push
```

## Cómo verlo

- **Local, rápido**: doble clic en `index.html`. El `fetch` del JSON falla bajo
  `file://`, así que arranca con datos de ejemplo; usá el botón **cargar JSON**
  para abrir tu `docs/typology/typology_browser.json`.
- **Local, servido** (toma el JSON solo):
  ```bash
  python -m http.server -d docs/typology 8000
  # -> http://localhost:8000
  ```
- **GitHub Pages**: el workflow `.github/workflows/pages.yml` publica este
  directorio en cada push que lo toque. Una sola vez, en el repo:
  *Settings -> Pages -> Build and deployment -> Source: **GitHub Actions***.
  Queda en `https://<usuario>.github.io/land2vec/`.
