#!/bin/bash -l
#SBATCH --job-name=b1-1-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A your-account
#SBATCH -p your-gpu-partition
#SBATCH --gres=gpu:1
#SBATCH -o $SCRATCH/logs/b1-1-sup_%j.out
#SBATCH -e $SCRATCH/logs/b1-1-sup_%j.err

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

MODEL=b1
MIT_STUDENT=$(find $SCRATCH ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find $SCRATCH ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

OUT="runs/mit-${MODEL}_sup_j${SLURM_JOB_ID}"

HF_HOME=$SCRATCH/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --epochs-sup 80 \
  --epochs-kd 1 --epochs-skd 1 --epochs-guardrail 1 \
  --skip-kd --skip-skd --skip-guardrail \
  --batch-size 8 \
  --lr 6e-5 \
  --log-every 20

echo "[DONE] Stage 1 sup complete.  Output: $OUT"
