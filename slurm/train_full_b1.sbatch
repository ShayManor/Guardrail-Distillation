#!/bin/bash -l
#SBATCH --job-name=gpp-b1-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/gpp-b1-full_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-b1-full_%j.err

# Full pipeline for MiT-B1 student: sup -> kd -> skd -> guardrail
# Matches B0 v2 hyperparameters with v4 guardrail fixes

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

# Cache B1 backbone if not already present
python -c "from transformers import AutoModel; AutoModel.from_pretrained('nvidia/mit-b1')" 2>/dev/null || true

MIT_B1=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b1/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

if [ -z "$MIT_B1" ]; then
  echo "ERROR: nvidia/mit-b1 not found in cache. Download it first."
  exit 1
fi

OUT=outputs-mit-b1-v4

# ── Stage 1: Supervised ──
HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B1" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 80 \
  --epochs-kd 1 --epochs-skd 1 --epochs-guardrail 1 \
  --skip-kd --skip-skd --skip-guardrail \
  --batch-size 8 \
  --lr 6e-5 \
  --log-every 20

echo "[DONE] B1 Stage 1 sup complete"

# ── Stage 2: KD ──
HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B1" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 1 --epochs-skd 1 --epochs-guardrail 1 \
  --skip-sup --skip-skd --skip-guardrail \
  --epochs-kd 60 \
  --batch-size 8 \
  --lr 6e-5 \
  --kd-temperature 2.0 \
  --log-every 20

echo "[DONE] B1 Stage 2 KD complete"

# ── Stage 3: SKD ──
HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B1" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 1 --epochs-kd 1 --epochs-guardrail 1 \
  --skip-sup --skip-kd --skip-guardrail \
  --epochs-skd 60 \
  --batch-size 8 \
  --lr 6e-5 \
  --alpha-kd 1.0 --alpha-struct 0.5 \
  --kd-temperature 2.0 \
  --log-every 20

echo "[DONE] B1 Stage 3 SKD complete"

# ── Stage 4: Guardrail ──
HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B1" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --cf-delta 0.02 \
  --cf-severities 0.25,0.5,0.75,1.0 \
  --corruption-prob 0.5 \
  --log-every 20

echo "[DONE] B1 all stages complete. Checkpoints in $OUT/"
