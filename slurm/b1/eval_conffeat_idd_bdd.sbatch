#!/bin/bash -l
#SBATCH --job-name=b1-eval-conffeat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/b1-eval-conffeat_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-eval-conffeat_%j.err

# IDD + BDD eval for the b1 confidence-feature guardrail heads (seeds 42/137/256).
# eval_all.sbatch's dir-name regex does not match *_conffeat_* dirs, so these
# three checkpoints were only ever evaluated on ACDC and Cityscapes.
set -euo pipefail
source $SCRATCH/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=~/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODEL=b1
MIT_STUDENT=$(find $SCRATCH ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find $SCRATCH ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

SKD_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt 2>/dev/null | grep -v "_skd_de_" | head -1 | xargs dirname)
STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
echo "[cfg] SKD_DIR=$SKD_DIR"

GUARD_DIRS=(
  runs/mit-b1_guard_dense_multi_conffeat_s42_j13171788
  runs/mit-b1_guard_dense_multi_conffeat_s137_j13173397
  runs/mit-b1_guard_dense_multi_conffeat_s256_j13173398
)

for GDIR in "${GUARD_DIRS[@]}"; do
  GUARDRAIL_CKPT="$GDIR/guardrail.ckpt"
  [ ! -f "$GUARDRAIL_CKPT" ] && echo "ERROR: $GUARDRAIL_CKPT missing" && exit 1

  for DS in idd bdd; do
    if [ "$DS" = "idd" ]; then
      DS_NAME=idd; DS_PATH="$REPO/data/idd"
    else
      DS_NAME=bdd; DS_PATH="$REPO/data/bdd100k"
    fi
    EVAL_DIR="$REPO/paper_eval_${DS}_${MODEL}_${GDIR##*/}"
    echo "============================================"
    echo "[${DS}] $GDIR -> $EVAL_DIR"
    echo "============================================"
    HF_HOME=$SCRATCH/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "${DS}_${MODEL}_conffeat_${GDIR##*_}" \
      --dataset-name "$DS_NAME" \
      --dataset-path "$DS_PATH" \
      --split val \
      --student-name student_skd \
      --train-method skd \
      --student-backbone "$MIT_STUDENT" \
      --student-ckpt "$STUDENT_CKPT" \
      --teacher-backbone "$TEACHER" \
      --guardrail-ckpt "$GUARDRAIL_CKPT" \
      --guardrail-student-name student_skd \
      --mc-dropout-passes 4 \
      --output-dir "$EVAL_DIR" \
      --batch-size 4 \
      --num-workers 8
  done
done

echo "[DONE] conffeat IDD+BDD eval complete"
