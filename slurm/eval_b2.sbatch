#!/bin/bash -l
#SBATCH --job-name=gpp-b2-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0:30:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/gpp-b2-eval_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-b2-eval_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MIT_B2=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--mit-b2/snapshots/*' -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

OUT=outputs-mit-b2-v4

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/eval/full_eval.py eval \
  --run-id guardrailpp_b2_cityscapes_val \
  --dataset-name cityscapes \
  --dataset-path "$REPO/data/cityscapes" \
  --split val \
  --domain all \
  --student-name student_skd \
  --train-method skd \
  --student-backbone "$MIT_B2" \
  --student-ckpt "$OUT/student_skd.ckpt" \
  --teacher-backbone "$TEACHER" \
  --guardrail-ckpt "$OUT/guardrail.ckpt" \
  --guardrail-student-name student_skd \
  --mc-dropout-passes 4 \
  --output-dir "$REPO/paper_eval_b2" \
  --batch-size 2 \
  --num-workers 4
