#!/bin/bash -l
#SBATCH --job-name=guardrailpp-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/guardrailpp-full_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/guardrailpp-full_%j.err

# Full pipeline: trains sup -> kd -> skd -> guardrail from scratch.
# If you already have student checkpoints, use train_guardrail.sbatch instead.

set -euo pipefail

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers

export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p /scratch/gautschi/manors/guardrail/logs
cd "$REPO"

# ── Change MODEL to switch backbone (b0, b1, b2, …) ──
MODEL=${MODEL:-b0}

MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

OUT="runs/mit-${MODEL}_guardrailpp_full_j${SLURM_JOB_ID}"

HF_HOME=/scratch/gautschi/manors/hf_cache \
  python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 20 \
  --epochs-kd 20 \
  --epochs-skd 20 \
  --epochs-guardrail 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --utility-w0 0.5 \
  --utility-w1 0.25 \
  --utility-w2 0.25 \
  --cf-delta 0.02 \
  --cf-severities 0.25,0.5,0.75,1.0 \
  --corruption-prob 0.5 \
  --log-every 20
