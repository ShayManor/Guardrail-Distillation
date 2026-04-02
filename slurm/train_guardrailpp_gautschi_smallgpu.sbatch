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

export REPO_DIR=/scratch/gautschi/manors/Guardrail-Distillation
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
unset DISPLAY

mkdir -p /scratch/gautschi/manors/logs
cd "$REPO_DIR"

python -u src/eval/full_eval.py eval \
  --run-id guardrailpp_cityscapes_val \
  --dataset-name cityscapes \
  --dataset-path /scratch/gautschi/manors/datasets/cityscapes \
  --split val \
  --domain all \
  --student-name student_skd \
  --train-method skd \
  --student-backbone nvidia/mit-b0 \
  --student-ckpt outputs-guardrailpp/student_skd.ckpt \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt outputs-guardrailpp/guardrail.ckpt \
  --guardrail-student-name student_skd \
  --mc-dropout-passes 8 \
  --output-dir paper_eval \
  --batch-size 4 \
  --num-workers 4