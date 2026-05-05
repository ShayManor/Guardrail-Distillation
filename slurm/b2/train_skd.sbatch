#!/bin/bash -l
#SBATCH --job-name=b2-3-skd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A your-account
#SBATCH -p your-gpu-partition
#SBATCH --gres=gpu:1
#SBATCH -o $SCRATCH/logs/b2-3-skd_%j.out
#SBATCH -e $SCRATCH/logs/b2-3-skd_%j.err

set -euo pipefail
source $SCRATCH/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=~/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p $SCRATCH/logs
cd "$REPO"

MODEL=b2
MIT_STUDENT=$(find $SCRATCH ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find $SCRATCH ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Auto-discover latest sup run (override: SRC_DIR=<path> sbatch ...)
SRC_DIR="${SRC_DIR:-}"
if [ -z "$SRC_DIR" ]; then
  SRC_DIR=$(ls -td runs/mit-${MODEL}_sup_*/student_sup.ckpt outputs*/student_sup.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SRC_DIR" ] && echo "ERROR: No sup run found for mit-${MODEL}." && exit 1
  echo "[auto] Found latest sup run: $SRC_DIR"
fi
[ ! -f "$SRC_DIR/student_sup.ckpt" ] && echo "ERROR: $SRC_DIR/student_sup.ckpt not found." && exit 1

OUT="runs/mit-${MODEL}_skd_j${SLURM_JOB_ID}"

HF_HOME=$SCRATCH/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --checkpoint-dir "$SRC_DIR" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 1 --epochs-kd 1 --epochs-skd 60 --epochs-guardrail 1 \
  --skip-sup --skip-kd --skip-guardrail \
  --batch-size 8 \
  --lr 6e-5 \
  --alpha-kd 1.0 --alpha-struct 0.5 \
  --kd-temperature 2.0 \
  --log-every 20

echo "[DONE] Stage 3 SKD complete.  Output: $OUT  Source: $SRC_DIR"
