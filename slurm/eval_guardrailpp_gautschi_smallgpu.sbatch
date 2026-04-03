#!/bin/bash -l
#SBATCH --job-name=guardrailpp-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/guardrailpp-train_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/guardrailpp-train_%j.err

set -euo pipefail

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers

export REPO_DIR=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p /scratch/gautschi/manors/guardrail/logs
cd "$REPO_DIR"

python -u src/train/run.py train \
  --dataset-path /home/manors/Guardrail-Distillation/data/acdc \
  --output-dir outputs-guardrailpp \
  --student-model nvidia/mit-b0 \
  --teacher-model nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
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