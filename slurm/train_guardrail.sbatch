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

MIT_B0=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b0/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

SRC_DIR=outputs-guardrailpp-v2
OUT=outputs-guardrailpp-v4

mkdir -p "$OUT"
for ckpt in student_sup.ckpt student_kd.ckpt student_skd.ckpt; do
  if [ ! -f "$OUT/$ckpt" ]; then
    cp "$SRC_DIR/$ckpt" "$OUT/$ckpt"
    echo "Copied $SRC_DIR/$ckpt -> $OUT/$ckpt"
  fi
done

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B0" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --utility-loss-weight 1.0 \
  --margin-loss-weight 1.0 \
  --cf-delta 0.02 \
  --cf-severities 0.25,0.5,0.75,1.0 \
  --corruption-prob 0.5 \
  --log-every 20

echo "[DONE] Guardrail training complete. Checkpoint: $OUT/guardrail.ckpt"