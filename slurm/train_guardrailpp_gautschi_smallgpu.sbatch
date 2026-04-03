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

export REPO_DIR=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p /scratch/gautschi/manors/guardrail/logs
cd "$REPO_DIR"

MIT_B0=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b0/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)


python -u src/eval/full_eval.py eval \
  --run-id guardrailpp_cityscapes_val \
  --dataset-name cityscapes \
  --dataset-path /home/manors/Guardrail-Distillation/data/cityscapes \
  --split val \
  --domain all \
  --student-name student_skd \
  --train-method skd \
  --student-backbone "$MIT_B0" \
  --student-ckpt outputs-guardrailpp/student_skd.ckpt \
  --teacher-backbone "$TEACHER" \
  --guardrail-ckpt outputs-guardrailpp/guardrail.ckpt \
  --guardrail-student-name student_skd \
  --mc-dropout-passes 8 \
  --output-dir paper_eval \
  --batch-size 4 \
  --num-workers 4