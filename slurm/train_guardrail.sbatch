#!/bin/bash -l
#SBATCH --job-name=gpp-4-guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/gpp-4-guard-v4_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-4-guard-v4_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

# ── Change MODEL to switch backbone (b0, b1, b2, …) ──
MODEL=${MODEL:-b0}

MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Auto-discover latest SKD run for mit-${MODEL} (override: SRC_DIR=<path> sbatch ...)
SRC_DIR="${SRC_DIR:-}"
if [ -z "$SRC_DIR" ]; then
  SRC_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  if [ -z "$SRC_DIR" ]; then
    echo "ERROR: No SKD run found for mit-${MODEL}. Run train_skd.sbatch first or set SRC_DIR."
    exit 1
  fi
  echo "[auto] Found latest SKD run: $SRC_DIR"
fi

OUT="runs/mit-${MODEL}_guard_j${SLURM_JOB_ID}"

# Verify source checkpoint exists
if [ ! -f "$SRC_DIR/student_skd.ckpt" ]; then
  echo "ERROR: $SRC_DIR/student_skd.ckpt not found."
  exit 1
fi

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --checkpoint-dir "$SRC_DIR" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 35 \
  --batch-size 8 \
  --lr 5e-4 \
  --utility-loss-weight 1.0 \
  --margin-loss-weight 1.0 \
  --cf-delta 0.02 \
  --cf-severities 0.25,0.5,0.75,1.0 \
  --corruption-prob 0.5 \
  --log-every 20

echo "[DONE] Guardrail training complete."
echo "  Output dir: $OUT"
echo "  Checkpoint: $OUT/guardrail.ckpt"
echo "  Source dir:  $SRC_DIR"
