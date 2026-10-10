#!/bin/bash
# Una etapa del tuneo (scripts/modelo/tuneo_ae.py) en Mendieta. Desde ~/land2vec:
#
#   sbatch scripts/cluster/tuneo_mendieta.sbatch piloto
#   sbatch scripts/cluster/tuneo_mendieta.sbatch busqueda --s-max 50000      # --s-max sólo la primera vez
#   sbatch scripts/cluster/tuneo_mendieta.sbatch confirmar
#   sbatch scripts/cluster/tuneo_mendieta.sbatch final --config c07
#
# Nodo completo (2 GPUs, 20 cores) en vez de medio nodo: sbatch --gres=gpu:2 --cpus-per-task=20 scripts/...
# Más o menos corridas a la vez por GPU: WORKERS_POR_GPU=6 sbatch ... (default 4; el piloto usa 1).
#
# Reanudación: si la etapa no termina dentro del tiempo del job, se vuelve a encolar EL MISMO comando.
#   - --max-horas se calcula con el tiempo que le queda al job menos un margen, para que las corridas
#     guarden su checkpoint y salgan solas antes de que Slurm las mate.
#   - --dependency=singleton: nunca corren a la vez dos jobs llamados tuneo_ae del mismo usuario. tuneo_ae.py también
#     saltea corridas vivas en otro nodo (por la última actualización de estado.json), pero así ni siquiera compiten.
#   - Parar a mano desde la cabecera (el /home es compartido): touch models/autoencoder_v3/tuneo/STOP
#SBATCH --job-name=tuneo_ae
#SBATCH --dependency=singleton
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --output=logs/tuneo-%x-%j.out

. /etc/profile   # antes de set -u: el /etc/profile del cluster usa variables sin definir (HISTCONTROL)
set -euo pipefail
source scripts/cluster/entorno_scratch.sh

ETAPA=${1:?etapa: piloto | busqueda | confirmar | final}
shift

MARGEN_SEG=${MARGEN_SEG:-900}
FIN=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
MAX_HORAS=$(python3 -c "import datetime as d; print(max(0.1, (d.datetime.fromisoformat('$FIN') - d.datetime.now()).total_seconds() - $MARGEN_SEG) / 3600)")

# Con --gres, Slurm renumera las GPUs asignadas desde 0 en CUDA_VISIBLE_DEVICES.
N_GPU=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:?no hay GPU asignada}")
DEVICES=$(seq -s, -f 'cuda:%g' 0 $((N_GPU - 1)))
if [ "$ETAPA" = piloto ]; then WORKERS=1; else WORKERS=$((N_GPU * ${WORKERS_POR_GPU:-4})); fi
THREADS=$((SLURM_CPUS_PER_TASK / WORKERS)); [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

echo "$(date '+%F %T') job $SLURM_JOB_ID en $(hostname): etapa $ETAPA $*"
echo "  GPUs $DEVICES ($(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd,)), $WORKERS corridas a la vez, $THREADS hilos c/u, para a las ${MAX_HORAS:0:5} h"

python3 scripts/modelo/tuneo_ae.py "$ETAPA" "$@" \
    --device "$DEVICES" --workers "$WORKERS" --threads "$THREADS" --max-horas "$MAX_HORAS"
python3 scripts/modelo/tuneo_ae.py estado
[ "$ETAPA" = piloto ] && python3 scripts/modelo/tuneo_ae.py resumen || true
