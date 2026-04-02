#!/bin/bash
#SBATCH --job-name=guardrailpp-eval
#SBATCH --account=cis240206
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/$USER/Guardrail-Distillation
mkdir -p logs paper_eval
source .venv/bin/activate

python -u src/eval/full_eval.py eval \
  --run-id guardrailpp_cityscapes_val \
  --dataset-name cityscapes \
  --dataset-path /scratch/$USER/datasets/cityscapes \
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
