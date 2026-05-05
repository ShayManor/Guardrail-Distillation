#!/bin/bash -l
#SBATCH --job-name=b1-fig-silent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o $SCRATCH/logs/b1-fig-silent_%j.out
#SBATCH -e $SCRATCH/logs/b1-fig-silent_%j.err

# Qualitative figure: silent confident failure under shift (4 images x 7 panels).
# Loads mit-b1 student + two guardrail heads (teacher-supervised dense_multi,
# GT-supervised gt_disagree) and renders one PNG.
# Runs on highmem (CPU) — workload is 4 forward passes, seconds of compute.

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
export MIT_STUDENT=$(find $SCRATCH ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)

# Student checkpoint (SKD). Override with STUDENT_CKPT=/abs/path.
STUDENT_CKPT="${STUDENT_CKPT:-}"
if [ -z "$STUDENT_CKPT" ]; then
  SKD_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SKD_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
  STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
  echo "[auto] student ckpt: $STUDENT_CKPT"
fi
export STUDENT_CKPT

# Teacher-supervised guardrail head (dense_multi). Override with GUARD_TEACHER_CKPT=...
GUARD_TEACHER_CKPT="${GUARD_TEACHER_CKPT:-}"
if [ -z "$GUARD_TEACHER_CKPT" ]; then
  D=$(ls -td runs/mit-${MODEL}_guard_dense_multi_j*/guardrail.ckpt runs/mit-${MODEL}_guard_dense_multi_s*_j*/guardrail.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$D" ] && echo "ERROR: No dense_multi guardrail found for mit-${MODEL}." && exit 1
  GUARD_TEACHER_CKPT="$D/guardrail.ckpt"
  echo "[auto] teacher-head ckpt: $GUARD_TEACHER_CKPT"
fi
export GUARD_TEACHER_CKPT

# GT-supervised guardrail head (gt_disagree). Override with GUARD_GT_CKPT=...
GUARD_GT_CKPT="${GUARD_GT_CKPT:-}"
if [ -z "$GUARD_GT_CKPT" ]; then
  D=$(ls -td runs/mit-${MODEL}_guard_gt_disagree_j*/guardrail.ckpt runs/mit-${MODEL}_guard_gt_disagree_s*_j*/guardrail.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$D" ] && echo "ERROR: No gt_disagree guardrail found for mit-${MODEL}." && exit 1
  GUARD_GT_CKPT="$D/guardrail.ckpt"
  echo "[auto] gt-head ckpt: $GUARD_GT_CKPT"
fi
export GUARD_GT_CKPT

# Dataset roots (match eval_all.sbatch conventions).
export ACDC_PATH="${ACDC_PATH:-$REPO/data/acdc}"
export IDD_PATH="${IDD_PATH:-$REPO/data/idd}"
export BDD_PATH="${BDD_PATH:-$REPO/data/bdd100k}"
export CITY_PATH="${CITY_PATH:-$REPO/data/cityscapes}"

for f in "$STUDENT_CKPT" "$GUARD_TEACHER_CKPT" "$GUARD_GT_CKPT"; do
  [ ! -f "$f" ] && echo "ERROR: $f missing" && exit 1
done

OUT="$REPO/src/analysis/figures/fig_silent_confident_failure.png"
OUT_STRONG="$REPO/src/analysis/figures/fig_acdc_fog_strong.png"

echo "============================================"
echo "[fig] rendering silent-confident-failure figure"
echo "  out=$OUT"
echo "============================================"

python -u scripts/make_silent_failure_figure.py \
  --output "$OUT" \
  --device cpu

echo "[DONE] figure written to: $OUT"

echo "============================================"
echo "[fig] rendering strong 4x4 figure (4 datasets x 4 panels)"
echo "  out=$OUT_STRONG"
echo "============================================"

# Crops the 4-row source PNG into a tight 4-dataset x 4-panel grid.
# Pure CPU/PIL work, no checkpoints needed.
python -u src/analysis/figure_scripts/fig_acdc_fog_silent_failure.py

echo "[DONE] strong figure written to: $OUT_STRONG"
echo "        (paper includes both via \\graphicspath{{../src/analysis/figures/}})"

# ─── Publish to git remote ──────────────────────────────────────────────────
# User-authorized: commit the rendered PNGs and push so figures are available
# off-cluster. We only stage the figure files themselves, never `git add -A`.
echo "============================================"
echo "[git] staging + pushing figures"
echo "============================================"
cd "$REPO"
REL_OUT="${OUT#$REPO/}"
REL_OUT_STRONG="${OUT_STRONG#$REPO/}"
git add "$REL_OUT" "$REL_OUT_STRONG"
if git diff --cached --quiet; then
  echo "[git] no changes to commit (figures byte-identical to HEAD)"
else
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  git commit -m "figure: refresh silent-confident-failure qualitative grid (b1 job ${SLURM_JOB_ID:-local})"
  git push origin "$BRANCH"
  echo "[git] pushed to origin/$BRANCH"
fi
