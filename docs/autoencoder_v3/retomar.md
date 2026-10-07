# Retomar: rehacer el ejercicio v3 con 1992-2022

**Fecha:** 2026-10-07. Documento de traspaso entre sesiones: resume qué pasó, dónde quedó todo y qué sigue.

---

## 1. Resumen de la sesión

**Punto de partida.** El ejercicio v3 sobre 2000-2022 (23 años) estaba hecho: censo de trayectorias de Argentina (4.554) y del mundo (35.023), P1 (compresión con autoencoder, PCA, MCA, autoencoder lineal), P2 (tipologías en seis espacios, seis métodos de clustering, control con desmonte), mapas. Todo quedó archivado en el tag `v3-2000-2022` (commit 75ffdc0).

**Qué se discutió y decidió**
1. **Duda de fondo del usuario:** las pruebas hechas no responden la pregunta que importa: *¿qué aporta el autoencoder respecto de OM?* Críticas reconocidas: P1 es casi tautológica (se ajusta y evalúa sobre el mismo universo chico); la calidad en P2 se medía con OM (circular); el control con desmonte está acotado por el techo de ESA CCI (~0,64) y no distingue entre espacios ni métodos; d y k nunca se eligieron con un criterio a priori (k = 12 era sólo el nivel medio de 6/12/24).
2. **Decisión:** rehacer todo con el período **1992-2022 (31 años)**, para Argentina y para el mundo, **reformulando las preguntas antes de correr** P2. Se reutiliza el nombre v3 (mismas rutas). Se borraron los resultados de 2000-2022 (siguen en el tag).
3. **Opción de trabajo "mínima" (elegida):** primero sólo censos (Argentina y mundo) y P1. P2, mapas y control con desmonte **no se tocan** hasta definir la batería de pruebas.
4. **Datos:** los crudos 1992-1999 se obtuvieron del CDS por API (token del usuario en `~/.cdsapirc`) y por subida manual (1996-1999). Los 31 años están en `data/ESA_data/raw_unzipped/`.

**Hallazgos útiles de v3 sobre 2000-2022** (para comparar con la versión de 31 años; detalle en el tag):
- Argentina: 36.084.989 px; 4.554 trayectorias; 9 constantes = 95,3 % de la superficie; máximo 3 cambios; 36/281 trayectorias cubren el 50/90 % de los px dinámicos.
- Mundo (máscara de continentes, colchón de 16 px): 35.023 trayectorias; 35.014 dinámicas = 4,07 % del área; máximo 7 cambios; 92/415/2.154 tipos cubren 50/90/99 % del área dinámica. Las trayectorias que existen en Argentina cubren el 97,9 % del área dinámica mundial.
- P1: con d=3 el autoencoder reconstruye exacto el 99,4 % de los px dinámicos; por tipo hace falta d≈8-16 (88 % → 96 %). El autoencoder lineal entrenado con la misma pérdida es mucho mejor que PCA/MCA; la ventaja de la no linealidad queda en d bajo. El autoencoder no sigue la geometría OM tan bien como lo lineal a d alto.
- P2: las tipologías en z recuperan los mismos procesos que OM/one-hot; sin ventaja medible. El método de clustering importa tanto como el espacio. HDBSCAN es inservible en este universo; k-medoides ≈ k-means ≈ Ward; no hay k natural.
- Costura de producto 2015/16 (v2.0.7cds → C3S v2.1.1): 2015 es casi copia de 2014; 2016 trae una reclasificación hacia bosque. Se marca, no se corrige (opción A).

---

## 2. Estado actual

**Git** (`main`, último commit `f426d2c`; el árbol tiene sin versionar `scripts/datos/descargar_cds.py` y este documento):
- `75ffdc0` censo mundial 2000-2022; tag `v3-2000-2022` (sólo local, falta `git push origin v3-2000-2022`).
- `1be7ca0` generaliza censo y P1 a T años (default 1992-2022).
- `f426d2c` elimina los resultados de 2000-2022.

**Datos:** `data/ESA_data/raw_unzipped/` tiene los 31 mapas globales 1992-2022 (1992-2015: v2.0.7cds; 2016-2022: C3S v2.1.1; ~2,3 GB c/u; verificados: abren, año correcto, grilla 64.800 × 129.600). Se borró `descargas_2026` (duplicados). `data/ESA_data/land_use_clipped/` (Argentina 1992-2020 recortada) sigue ahí, sin uso. `data/geo/World_Continents_*.geojson` (polígono de continentes) es la máscara del censo mundial. `data/autoencoder_v3/` está vacío.

**Código ya generalizado** (default `P.V3_YEARS = (1992, 2022)`): `scripts/datos/censo_trayectorias.py` y `censo_mundial.py` (`--years`, `--out-dir`; clave de 92 bits → 31 años en dos uint64, 16+15), `p1_compresion.py` (largo desde los datos), `build_landcover_nc.py`. **Regresión con 2000-2022 verificada:** censo de Argentina idéntico byte a byte; censo mundial con las mismas 35.023 trayectorias (n_px y n_cambios idénticos, área igual salvo redondeo); PCA/MCA de P1 idénticos.
**Nuevo:** `scripts/datos/descargar_cds.py` (descarga y verifica años del CDS).
**Sin tocar (siguen con 23 años y 2000 fijos):** `p2_*.py`, `eval_desmonte.py`, `p2_desmonte.py`, `p2_mapa_pais.py`, `train_autoencoder.py`.

---

## 3. Plan de trabajo

### Paso 1 — Serie de Argentina 1992-2022
```bash
python scripts/datos/build_landcover_nc.py                       # default 1992-2022, vector = rectángulo de v3 → data/autoencoder_v3/landcover_timeseries_1992-2022_rebuild.nc
```
Verificar que los años 2000-2022 coinciden con `data/landcover_timeseries_2000-2022.nc` (el original de v1/v2, idéntico al v3 viejo): `--compare` está pensado para la misma cantidad de años, así que **revisar si hay que adaptarlo** a 31 vs 23 (comparar por año). Es la prueba de que la nueva serie reproduce la anterior en el tramo común.

### Paso 2 — Censos
```bash
python scripts/datos/censo_trayectorias.py                        # Argentina y rectángulo (31 años); escribe universo_*.csv
python scripts/datos/censo_mundial.py --completo --workers 6      # mundo, ~10 min; escribe data/autoencoder_v3/mundo/
```
Tareas asociadas:
- **Detectar costuras:** el censo imprime los px que cambian por año (1992-2022). Buscar saltos o años casi congelados además de 2015/16 (posibles en ~1999/2000 por el cambio de sensor AVHRR → SPOT, y alrededor de 2012/13). Decidir tratamiento: marcar y reportar con/sin (como en la opción A de 2015/16). `add_costura` hoy marca sólo 2015/16: generalizar a una lista de años.
- Comparar universo de 31 vs 23 años: cuántas trayectorias nuevas aparecen y cuánta área cubren; si el universo mundial sigue siendo enumerable (en 2000-2022: 35.023).
- Chequeo cruzado: Argentina del censo mundial vs censo de Argentina.

### Paso 3 — P1 sobre 31 años
```bash
python scripts/modelo/p1_compresion.py om && python scripts/modelo/p1_compresion.py linear
# AE y AE lineal: train --d <d> --seed <s> (grilla d ∈ {1,2,3,4,6,8,12,16} × 3 semillas), linae, eval
```
Con más años puede hacer falta un d mayor; no asumir la grilla anterior. Declarar antes de mirar la regla para elegir d (o decir que se reportan varias).

### Paso 4 — Reformular las preguntas (antes de P2)
Reescribir `docs/autoencoder_v3/plan.md` con la pregunta **¿qué aporta el autoencoder respecto de OM?** y una batería **fijada a priori**, sin usar OM como juez:
1. **Coherencia espacial:** los píxeles vecinos deberían quedar cerca (ni OM ni el AE usan geografía).
2. **Pares sintéticos con respuesta conocida:** mismo proceso desfasado 1/2/5 años vs procesos distintos; discriminación y tolerancia al desfase de cada distancia.
3. **Análisis de desacuerdos** z vs OM (pares donde discrepan) con inspección de los tipos.
4. **Datos no vistos:** entrenar con una parte de las trayectorias y comprobar si el resto cae cerca de sus vecinas OM.
5. **Escala:** a nivel mundial, costo y fidelidad de OM frente al autoencoder (OM completa sobre ~35 mil tipos ≈ 5 GB y ~5 h; ya no depende de poder o no calcularla).
Fijar también la regla para elegir **k** y mantener el control con desmonte como prueba de validez (no de competencia).

### Paso 5 — Adaptar P2 y el control con desmonte (sólo después del Paso 4)
Cosas ya identificadas:
- La referencia de desmonte empieza en 2001: las reglas R0', F2000∧¬F2022 y "alguna transición" deben evaluarse **sólo sobre 2001-2022** (`toks[9:]` con inicio en 1992). Si no, un F→A de 1995 cuenta como falso positivo.
- Las zonas v2 (`data/zonas/id_seqs_text_2000_2022_<zona>.zip`) son de 23 años y submuestreadas: los pesos IPW (`compute_weights`) deben seguir usando la definición de "dinámica" de 2000-2022; la trayectoria de 31 años de cada píxel hay que sacarla del `.nc` nuevo por coordenadas.
- `typology.YEAR_0`, `T_DEFAULT` y `auto_label` asumen 2000/23; los P2 lo cargan al vuelo.
- Hay que decidir qué hacer con P2 (qué métodos y espacios se rehacen, si se rehacen todos).

### Paso 6 — Documentar, mapas y commit
Resultados, mapas nacionales y commit (con git y el filtro LFS anulado dentro del contenedor, ver §5; el `.nc` y los `.npy` van por LFS desde la máquina del usuario).

---

## 4. Decisiones abiertas
- **Tratamiento de las costuras** que aparezcan (marcar y reportar con/sin vs. otra opción).
- **Elección de d y k:** regla a priori o reportar varias.
- **Alcance de P2 y del control con desmonte** tras reformular las preguntas.
- **Máscara del censo mundial:** colchón de 16 px y qué hacer con Wa constante (océano/lagos/hielo; la Antártida y Groenlandia entran en el polígono).
- **Qué mide el AE frente a OM** (la batería del Paso 4) y qué se considera "aporte".

---

## 5. Trampas y entorno (para no repetir errores)
- **Venv:** el `.venv` del repo no sirve en el contenedor. Hay un venv con uv en el scratchpad de la sesión (`/tmp/claude-1000/.../scratchpad/venv`); si no existe, recrearlo: `uv venv --python 3.13 <dir>` + torch CPU (índice `https://download.pytorch.org/whl/cpu`) + `numpy pandas scikit-learn scipy xarray h5netcdf h5py geopandas shapely matplotlib rasterio tabulate cdsapi`, y `uv pip install -e . --no-deps`. El Python del sistema no tiene el módulo `http`.
- **Git en el contenedor:** falta git-lfs. Usar `git -c filter.lfs.clean=cat -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false <comando>`. Push y migraciones LFS las hace el usuario desde fuera (ver la advertencia de `git lfs migrate`: siempre con `--exclude-ref=refs/remotes/origin/main`).
- **Memoria en el censo mundial:** abrir y cerrar los `.nc` en cada bloque (con handles persistentes cada worker crece ~100 MB por bloque). Lanzar tareas largas con `run_in_background` (los procesos lanzados con `nohup` en un comando se cortaron al cerrar la sesión).
- **No usar `pkill -f` con un patrón que aparezca en el propio comando:** mata la shell.
- **Disco:** `/workspaces/land2vec` está en el disco externo (1,2 TB libres). El disco raíz del contenedor (~9 GB libres) sólo importa para `/tmp` y las cachés.
- **Seguridad:** el token del CDS se pegó en el chat de esa sesión: **revocarlo/regenerarlo** en https://cds.climate.copernicus.eu/profile y, si ya no se usa, borrar `~/.cdsapirc`.
- **Cifras de referencia para regresión** (censo 2000-2022): Argentina 4.554 trayectorias y 36.084.989 px; mundo 35.023 trayectorias y 149,9 M km² dentro de la máscara (2.048 bloques de 2025²; px dentro + fuera = 2.048 × 2025²).
