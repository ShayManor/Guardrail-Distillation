#!/bin/bash -l
#SBATCH --job-name=b1-eval-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/b1-eval-all_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-eval-all_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODEL=b1
MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Student (newest SKD checkpoint for b1)
SKD_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
[ -z "$SKD_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
[ ! -f "$STUDENT_CKPT" ] && echo "ERROR: $STUDENT_CKPT missing" && exit 1
echo "[auto] SKD dir: $SKD_DIR"

# Resolve newest guardrail dir per supervision mode
declare -A GUARD_DIRS
for MODE in dense_multi dense_disagree dense_gap scalar_benefit gt_disagree gt_risk; do
  D=$(ls -td runs/mit-${MODEL}_guard_${MODE}_j*/guardrail.ckpt runs/mit-${MODEL}_guard_${MODE}_s*_j*/guardrail.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  if [ -z "$D" ] || [ ! -f "$D/guardrail.ckpt" ]; then
    echo "WARNING: no guardrail checkpoint for mode=$MODE — skipping"
    continue
  fi
  GUARD_DIRS[$MODE]="$D"
  echo "[auto] $MODE -> $D"
done

if [ ${#GUARD_DIRS[@]} -eq 0 ]; then
  echo "ERROR: no guardrail checkpoints found at all"; exit 1
fi

run_city () {
  local MODE="$1" GUARD_DIR="$2"
  local EVAL_DIR="$REPO/paper_eval_city_${MODEL}_${GUARD_DIR##*/}"
  echo "============================================"
  echo "[CS  ] mode=$MODE   out=$EVAL_DIR"
  echo "============================================"
  HF_HOME=/scratch/gautschi/manors/hf_cache \
  python -u src/eval/full_eval.py eval \
    --run-id "city_${MODEL}_skd_${MODE}" \
    --dataset-name cityscapes \
    --dataset-path "$REPO/data/cityscapes" \
    --split val \
    --student-name student_skd \
    --train-method skd \
    --student-backbone "$MIT_STUDENT" \
    --student-ckpt "$STUDENT_CKPT" \
    --teacher-backbone "$TEACHER" \
    --guardrail-ckpt "$GUARD_DIR/guardrail.ckpt" \
    --guardrail-student-name student_skd \
    --mc-dropout-passes 4 \
    --output-dir "$EVAL_DIR" \
    --batch-size 4 \
    --num-workers 4
}

run_acdc () {
  local MODE="$1" GUARD_DIR="$2"
  local EVAL_DIR="$REPO/paper_eval_acdc_${MODEL}_${GUARD_DIR##*/}"
  for DOMAIN in fog night rain snow all; do
    echo "============================================"
    echo "[ACDC] mode=$MODE domain=$DOMAIN   out=$EVAL_DIR"
    echo "============================================"
    HF_HOME=/scratch/gautschi/manors/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "acdc_${MODEL}_skd_${MODE}_${DOMAIN}" \
      --dataset-name acdc \
      --dataset-path "$REPO/data/acdc" \
      --split val \
      --domain "$DOMAIN" \
      --student-name student_skd \
      --train-method skd \
      --student-backbone "$MIT_STUDENT" \
      --student-ckpt "$STUDENT_CKPT" \
      --teacher-backbone "$TEACHER" \
      --guardrail-ckpt "$GUARD_DIR/guardrail.ckpt" \
      --guardrail-student-name student_skd \
      --mc-dropout-passes 4 \
      --output-dir "$EVAL_DIR" \
      --batch-size 4 \
      --num-workers 4
  done
}

for MODE in "${!GUARD_DIRS[@]}"; do
  run_city "$MODE" "${GUARD_DIRS[$MODE]}"
  run_acdc "$MODE" "${GUARD_DIRS[$MODE]}"
done

echo "[DONE] All ablation evals complete."
for MODE in "${!GUARD_DIRS[@]}"; do
  echo "  $MODE: paper_eval_{city,acdc}_${MODEL}_${GUARD_DIRS[$MODE]##*/}/csv/"
done
