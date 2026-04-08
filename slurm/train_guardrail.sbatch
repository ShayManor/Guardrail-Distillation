#!/bin/bash -l
#SBATCH --job-name=gpp-4-guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/gpp-4-guard-v3_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-4-guard-v3_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MIT_B0=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b0/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

OUT=outputs-guardrailpp-v3

mkdir -p "$OUT"
cp outputs-guardrailpp-v2/student_sup.ckpt "$OUT/"
cp outputs-guardrailpp-v2/student_kd.ckpt "$OUT/"
cp outputs-guardrailpp-v2/student_skd.ckpt "$OUT/"

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --student-model "$MIT_B0" \
  --teacher-model "$TEACHER" \
  --guardrail-mode guardrailpp \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 20 \
  --batch-size 8 \
  --lr 5e-4 \
  --utility-loss-weight 3.0 \
  --margin-loss-weight 0.0 \
  --cf-severities 0.0,0.5,1.0 \
  --log-every 20