#!/bin/bash
#SBATCH --job-name=guardrailpp-train
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
mkdir -p logs outputs-guardrailpp

# Optional: load your site modules if needed
# module purge
# module load cuda/12.1

source .venv/bin/activate

python -u src/train/run.py train \
  --dataset-path /scratch/$USER/datasets/cityscapes \
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
