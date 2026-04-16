#!/bin/bash -l
#SBATCH --job-name=prep-idd-local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/prep-idd-local_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/prep-idd-local_%j.err

# Lay out IDD (Part I + Part II tarballs already on scratch) in the same
# Cityscapes-style form as $REPO/data/cityscapes and $REPO/data/acdc, so
# eval commands can point at --dataset-path "$REPO/data/idd" with minimal
# loader changes (identity label map since the toolkit remap below emits
# Cityscapes-19 trainIds directly).
#
# Inputs (env overrides):
#   TAR_DIR   — directory containing the .tar.gz files
#               (default: /scratch/gautschi/manors)
#   IDD_ROOT  — where the prepped tree lives on scratch
#               (default: /scratch/gautschi/manors/data/idd)
#   REPO      — repo root (default: /home/manors/Guardrail-Distillation)

set -euxo pipefail
trap 'echo "[FAIL] line $LINENO exited with $?" >&2' ERR
export PYTHONUNBUFFERED=1

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

REPO="${REPO:-/home/manors/Guardrail-Distillation}"
TAR_DIR="${TAR_DIR:-/scratch/gautschi/manors}"
IDD_ROOT="${IDD_ROOT:-/scratch/gautschi/manors/data/idd}"
STAGING="/scratch/gautschi/manors/data/idd_staging"

# ── 1. Locate tarballs ──────────────────────────────────────────────────────
shopt -s nullglob
TARS=( "$TAR_DIR"/idd*.tar.gz "$TAR_DIR"/IDD*.tar.gz )
shopt -u nullglob
[ "${#TARS[@]}" -ge 1 ] || { echo "ERROR: no idd*.tar.gz in $TAR_DIR"; ls -la "$TAR_DIR"; exit 10; }
echo "[tars] found ${#TARS[@]}:"
for t in "${TARS[@]}"; do echo "  $t"; done

# ── 2. Extract both parts into a shared staging dir ─────────────────────────
# IDD Part I and Part II both unpack to the same top-level dir (IDD_Segmentation
# or idd20k_II) and merge naturally when extracted into a common parent.
#
# `rm -rf` on Lustre/NFS scratch intermittently fails with "Directory not empty"
# due to delayed metadata ops / stale handles. Retry with a backoff and fall
# back to find-delete, which tolerates the race better.
robust_rmrf() {
  local target="$1"
  [ -e "$target" ] || return 0
  for i in 1 2 3 4 5; do
    rm -rf "$target" 2>/dev/null && return 0
    find "$target" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$target" 2>/dev/null && return 0
    sleep $((i * 2))
  done
  rm -rf "$target"
}
robust_rmrf "$STAGING"
mkdir -p "$STAGING"
for t in "${TARS[@]}"; do
  echo "[extract] $t -> $STAGING"
  tar -xzf "$t" -C "$STAGING"
done

# Find the directory that actually contains leftImg8bit/ — that's the tree root.
LEFT=$(find "$STAGING" -maxdepth 6 -type d -name leftImg8bit -print -quit)
[ -n "$LEFT" ] || { echo "ERROR: no leftImg8bit/ found under $STAGING"; find "$STAGING" -maxdepth 4 -type d; exit 11; }
TREE_ROOT=$(dirname "$LEFT")
echo "[ok] tree root: $TREE_ROOT"
ls -la "$TREE_ROOT"

# ── 3. Move into canonical location ─────────────────────────────────────────
robust_rmrf "$IDD_ROOT"
mkdir -p "$(dirname "$IDD_ROOT")"
mv "$TREE_ROOT" "$IDD_ROOT"
robust_rmrf "$STAGING"

# ── 4. Apply Cityscapes-19 trainId remap via AutoNUE toolkit ────────────────
# The toolkit writes *_gtFine_labellevel3Ids.png → *_gtFine_csTrainIds.png
# alongside originals. csTrainIds uses the standard 19-class Cityscapes
# ontology with `255` as ignore — identical value semantics to cityscapes
# trainIds, so no further remap is needed at dataloader time.
TOOLKIT="$IDD_ROOT/_autonue_toolkit"
if [ ! -d "$TOOLKIT" ]; then
  git clone --depth 1 https://github.com/AutoNUE/public-code.git "$TOOLKIT"
fi
pip install -q cityscapesscripts numpngw imageio tqdm pandas scipy

cd "$TOOLKIT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python preperation/createLabels.py \
  --datadir "$IDD_ROOT" \
  --id-type csTrainId \
  --num-workers 8

# ── 5. Symlink into $REPO/data alongside cityscapes/acdc ────────────────────
mkdir -p "$REPO/data"
ln -sfn "$IDD_ROOT" "$REPO/data/idd"
echo "[ok] $REPO/data/idd -> $IDD_ROOT"

# ── 6. Validate ─────────────────────────────────────────────────────────────
cd "$IDD_ROOT"
N_IMG_TRAIN=$(find leftImg8bit/train -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
N_IMG_VAL=$(find   leftImg8bit/val   -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
N_LBL_TRAIN=$(find gtFine/train -type f -name '*_csTrainIds.png' | wc -l)
N_LBL_VAL=$(find   gtFine/val   -type f -name '*_csTrainIds.png' | wc -l)

echo "[counts] train: $N_IMG_TRAIN img / $N_LBL_TRAIN cs19 lbl"
echo "[counts] val:   $N_IMG_VAL img / $N_LBL_VAL cs19 lbl"

# IDD Segmentation: ~6993 train / ~981 val drawn from 182 drives. Accept a
# small margin because Part I vs II split isn't guaranteed stable.
[ "$N_IMG_TRAIN" -ge 6500 ] || { echo "ERROR: train images too few"; exit 20; }
[ "$N_IMG_VAL"   -ge  900 ] || { echo "ERROR: val images too few"; exit 21; }
[ "$N_LBL_TRAIN" -ge 6500 ] || { echo "ERROR: train cs19 labels too few — toolkit remap failed?"; exit 22; }
[ "$N_LBL_VAL"   -ge  900 ] || { echo "ERROR: val cs19 labels too few — toolkit remap failed?"; exit 23; }

# Sanity: 20 random val labels must be single-channel in {0..18, 255}.
export IDD_ROOT
python - <<'PY'
import os, random, numpy as np
from pathlib import Path
from PIL import Image
root = Path(os.environ["IDD_ROOT"]) / "gtFine" / "val"
lbls = sorted(root.rglob("*_csTrainIds.png"))
random.seed(0)
allowed = set(range(19)) | {255}
for p in random.sample(lbls, min(20, len(lbls))):
    arr = np.array(Image.open(p))
    assert arr.ndim == 2, (p.name, arr.ndim)
    extra = set(np.unique(arr).tolist()) - allowed
    assert not extra, (p.name, sorted(extra)[:8])
print(f"[ok] 20/20 val cs19 labels are single-channel trainIds")
PY

echo "[DONE] IDD staged at $IDD_ROOT, reachable as $REPO/data/idd"
echo "       tree: leftImg8bit/{train,val}/<drive_id>/*.png"
echo "             gtFine/{train,val}/<drive_id>/*_gtFine_csTrainIds.png"
