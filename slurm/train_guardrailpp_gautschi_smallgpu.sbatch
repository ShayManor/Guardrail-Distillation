#!/bin/bash -l
#SBATCH --job-name=guardrailpp-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/guardrailpp-eval_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/guardrailpp-eval_%j.err

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

MIT_B0=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b0/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

HF_HOME=/scratch/gautschi/manors/hf_cache \
  python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir outputs-guardrailpp \
  --student-model MIT_B0 \
  --teacher-model TEACHER \
  --guardrail-mode guardrailpp \
  --epochs-sup 20 \
  --epochs-kd 20 \
  --epochs-skd 20 \
  --epochs-guardrail 20 \
  --batch-size 8 \
  --lr 5e-5 \
  --utility-w0 0.5 \
  --utility-w1 0.25 \
  --utility-w2 0.25 \
  --cf-delta 0.05 \
  --cf-severities 0.0,0.25,0.5,0.75,1.0 \
  --log-every 20