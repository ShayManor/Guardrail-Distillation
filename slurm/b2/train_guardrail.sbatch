#!/bin/bash -l
#SBATCH --job-name=b2-4-guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/b2-4-guard_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/b2-4-guard_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODEL=b2
MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Auto-discover latest SKD run (override: SRC_DIR=<path> sbatch ...)
SRC_DIR="${SRC_DIR:-}"
if [ -z "$SRC_DIR" ]; then
  SRC_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SRC_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
  echo "[auto] Found latest SKD run: $SRC_DIR"
fi
[ ! -f "$SRC_DIR/student_skd.ckpt" ] && echo "ERROR: $SRC_DIR/student_skd.ckpt not found." && exit 1

SUPERVISION_TYPE="${SUPERVISION_TYPE:-dense_multi}"
SEED="${SEED:-42}"
OUT="runs/mit-${MODEL}_guard_${SUPERVISION_TYPE}_s${SEED}_j${SLURM_JOB_ID}"

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --checkpoint-dir "$SRC_DIR" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 35 \
  --batch-size 8 \
  --lr 5e-4 \
  --supervision-type "$SUPERVISION_TYPE" \
  --dense-disagree-weight 1.0 \
  --dense-gap-weight 1.0 \
  --corruption-prob 0.5 \
  --use-student-features \
  --seed "$SEED" \
  --log-every 20

echo "[DONE] Guardrail ${SUPERVISION_TYPE} seed=${SEED} complete.  Output: $OUT  Source: $SRC_DIR"
