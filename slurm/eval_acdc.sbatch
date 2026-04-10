#!/bin/bash -l
#SBATCH --job-name=gpp-eval-acdc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/gpp-eval-acdc_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-eval-acdc_%j.err

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

# Use whichever guardrail version is latest
if [ -d outputs-guardrailpp-v5 ] && [ -f outputs-guardrailpp-v5/guardrail.ckpt ]; then
  OUT=outputs-guardrailpp-v5
elif [ -f outputs-guardrailpp-v4/guardrail.ckpt ]; then
  OUT=outputs-guardrailpp-v4
elif [ -f outputs-guardrailpp-v3/guardrail.ckpt ]; then
  OUT=outputs-guardrailpp-v3
elif [ -f outputs-guardrailpp-v2/guardrail.ckpt ]; then
  OUT=outputs-guardrailpp-v2
else
  echo "ERROR: No guardrail checkpoint found"; exit 1
fi

STUDENT_CKPT="$OUT/student_skd.ckpt"
GUARDRAIL_CKPT="$OUT/guardrail.ckpt"
EVAL_DIR="$REPO/paper_eval_acdc"

for f in "$STUDENT_CKPT" "$GUARDRAIL_CKPT"; do
  [ ! -f "$f" ] && echo "ERROR: $f missing" && exit 1
done

# ── Per-condition evals (fog, night, rain, snow) + all ──
for DOMAIN in fog night rain snow all; do
  echo "============================================"
  echo "[ACDC] Evaluating domain=$DOMAIN"
  echo "============================================"

  HF_HOME=/scratch/gautschi/manors/hf_cache \
  python -u src/eval/full_eval.py eval \
    --run-id "acdc_b0_skd_${DOMAIN}" \
    --dataset-name acdc \
    --dataset-path "$REPO/data/acdc" \
    --split val \
    --domain "$DOMAIN" \
    --student-name student_skd \
    --train-method skd \
    --student-backbone "$MIT_B0" \
    --student-ckpt "$STUDENT_CKPT" \
    --teacher-backbone "$TEACHER" \
    --guardrail-ckpt "$GUARDRAIL_CKPT" \
    --guardrail-student-name student_skd \
    --mc-dropout-passes 4 \
    --output-dir "$EVAL_DIR" \
    --batch-size 2 \
    --num-workers 4
done

echo "[DONE] ACDC eval complete. CSVs in $EVAL_DIR/csv/"