#!/bin/bash -l
#SBATCH --job-name=gpp-4-guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/gpp-4-guard_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-4-guard_%j.err

set -euo pipefail

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers

export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p /scratch/gautschi/manors/logs
cd "$REPO"

MIT_B0=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b0/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

OUT=outputs-guardrailpp-v2

# Verify stage 3 finished
if [ ! -f "$OUT/student_skd.ckpt" ]; then
  echo "ERROR: $OUT/student_skd.ckpt not found. Run train_3_skd.sbatch first."
  exit 1
fi

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B0" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --skip-sup \
  --skip-kd \
  --skip-skd \
  --epochs-guardrail 20 \
  --skip-sup \
  --skip-kd \
  --skip-skd \
  --batch-size 8 \
  --lr 5e-4 \
  --utility-w0 0.5 \
  --utility-w1 0.25 \
  --utility-w2 0.25 \
  --utility-loss-weight 3.0 \
  --margin-loss-weight 0.1 \
  --cf-delta 0.05 \
  --cf-severities 0.0,0.5,1.0 \
  --log-every 20

echo "[DONE] Stage 4 guardrail complete. Checkpoint: $OUT/guardrail.ckpt"
