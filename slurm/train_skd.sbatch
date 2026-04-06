#!/bin/bash -l
#SBATCH --job-name=gpp-3-skd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/gpp-3-skd_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-3-skd_%j.err

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

# Verify stage 1 finished (SKD loads from student_sup.ckpt)
if [ ! -f "$OUT/student_sup.ckpt" ]; then
  echo "ERROR: $OUT/student_sup.ckpt not found. Run train_1_sup.sbatch first."
  exit 1
fi

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B0" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 1 \
  --epochs-kd 1 \
  --epochs-skd 60 \
  --epochs-guardrail 1 \
  --skip-sup \
  --skip-kd \
  --skip-guardrail \
  --batch-size 8 \
  --lr 6e-5 \
  --alpha-kd 1.0 \
  --alpha-struct 0.5 \
  --kd-temperature 2.0 \
  --log-every 20

echo "[DONE] Stage 3 SKD complete. Checkpoint: $OUT/student_skd.ckpt"
