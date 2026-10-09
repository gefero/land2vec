#!/usr/bin/env bash
# Copia a Mendieta (CCAD-UNC) lo mínimo para correr scripts/modelo/tuneo_ae.py, o trae los resultados.
# Se corre en la PC local, desde la raíz del repo.
#
#   bash scripts/cluster/sincronizar_mendieta.sh subir USUARIO     # código + insumos del tuneo
#   bash scripts/cluster/sincronizar_mendieta.sh bajar USUARIO     # modelos, códigos z, métricas y planes (sin checkpoints)
#   bash scripts/cluster/sincronizar_mendieta.sh subir-estado USUARIO   # tuneo empezado en la PC, CON checkpoints, para
#                                                                      # que el cluster lo retome (parar antes lo local)
#
# Se copia el árbol de trabajo (no un git clone): hay archivos del tuneo sin commitear y en el cluster no hay git-lfs.
# `subir` nunca toca models/autoencoder_v3/tuneo ni los plan_*.json/metricas_*.csv del cluster: mientras el tuneo
# corre, el cluster es la fuente de verdad. `subir-estado` se niega si hay corridas locales "en curso" y no pisa
# archivos más nuevos en el cluster (rsync --update).
set -euo pipefail

ACCION=${1:?subir | subir-estado | bajar}
USUARIO=${2:?usuario del CCAD}
HOST=${HOST:-mendieta.ccad.unc.edu.ar}
DEST=${DEST:-land2vec}          # relativo al home del cluster
R="$USUARIO@$HOST:$DEST"

case "$ACCION" in
subir)
    # logs/ tiene que existir antes del sbatch: Slurm no crea el directorio de --output y el job falla sin aviso
    ssh "$USUARIO@$HOST" "mkdir -p $DEST/logs $DEST/data/autoencoder_v3/latam $DEST/data/autoencoder_v3/tuneo $DEST/scripts/modelo"
    rsync -avz --exclude __pycache__ --exclude '*.egg-info' setup.py src requirements.txt "$R/"
    rsync -avz scripts/cluster "$R/scripts/"
    rsync -avz scripts/modelo/tuneo_ae.py scripts/modelo/p1_compresion.py "$R/scripts/modelo/"
    rsync -avz data/autoencoder_v3/latam/universo_latam.csv.gz "$R/data/autoencoder_v3/latam/"
    rsync -avz data/autoencoder_v3/tuneo/particion.npz data/autoencoder_v3/tuneo/referencias.csv "$R/data/autoencoder_v3/tuneo/"
    ;;
subir-estado)
    corriendo=$(grep -l '"estado": "en curso"' models/autoencoder_v3/tuneo/*/*/estado.json 2>/dev/null || true)
    if [ -n "$corriendo" ] && [ -z "${FORZAR:-}" ]; then
        echo "hay corridas locales en curso (pararlas con: touch models/autoencoder_v3/tuneo/STOP):" >&2
        echo "$corriendo" >&2; exit 1
    fi
    ssh "$USUARIO@$HOST" "mkdir -p $DEST/models/autoencoder_v3/tuneo $DEST/data/autoencoder_v3/tuneo"
    rsync -avzu --exclude STOP --exclude '*.tmp' models/autoencoder_v3/tuneo/ "$R/models/autoencoder_v3/tuneo/"
    rsync -avzu data/autoencoder_v3/tuneo/ "$R/data/autoencoder_v3/tuneo/"
    ;;
bajar)
    mkdir -p models/autoencoder_v3/tuneo data/autoencoder_v3/tuneo
    # ckpt.pt sólo sirve para retomar corridas, y eso pasa en el cluster
    rsync -avz --exclude STOP --exclude ckpt.pt --exclude '*.tmp' "$R/models/autoencoder_v3/tuneo/" models/autoencoder_v3/tuneo/
    rsync -avz "$R/data/autoencoder_v3/tuneo/" data/autoencoder_v3/tuneo/
    rsync -avz "$R/logs/" logs/mendieta/ 2>/dev/null || true
    ;;
*)
    echo "acción desconocida: $ACCION (subir | subir-estado | bajar)" >&2; exit 1 ;;
esac
