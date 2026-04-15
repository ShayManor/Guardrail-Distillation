#!/bin/bash -l
#SBATCH --job-name=prep-bdd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/prep-bdd_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/prep-bdd_%j.err

# Download + unzip + validate BDD100K (solesensei Kaggle mirror) and stage it
# next to the other datasets so existing dataloaders can reach it with a
# minimal code diff.
#
# Existing layout:
#   $REPO/data/cityscapes/{leftImg8bit,gtFine}/{train,val}/...
#   $REPO/data/acdc/...
#
# We stage the big tree on scratch and symlink it into $REPO/data so the
# paper repo doesn't have to hold 10 GB of images:
#   /scratch/gautschi/manors/data/bdd100k/                  (real files)
#   $REPO/data/bdd100k  ->  /scratch/.../data/bdd100k        (symlink)
#
# Inside, we preserve the Kaggle mirror's native seg layout:
#   data/bdd100k/seg/images/{train,val,test}/*.jpg
#   data/bdd100k/seg/labels/{train,val}/*_train_id.png      (19 trainIds)
#   data/bdd100k/seg/color_labels/{train,val}/*_train_color.png
#   data/bdd100k/labels_release/  (attribute JSON, time-of-day/weather)
#
# These labels are already in Cityscapes 19 trainIds — a later BDDDataset
# subclass of CityscapesDataset only needs to change the rglob patterns
# and use an identity label map.
#
# Env overrides:
#   BDD_SCRATCH  — where to stage (default: /scratch/gautschi/manors/data/bdd100k)
#   BDD_ZIP      — path to the raw zip (default: $BDD_SCRATCH/../bdd100k_raw.zip)
#   REPO         — repo root (default: /home/manors/Guardrail-Distillation)
#   KAGGLE_CONFIG_DIR — where kaggle.json lives
#                      (default: /scratch/gautschi/manors/.kaggle)

set -euxo pipefail

# ERR trap so a failed sub-shell actually surfaces in the log
trap 'echo "[FAIL] line $LINENO exited with $?" >&2' ERR

# Line-buffered python so prints land in the log while the job runs
export PYTHONUNBUFFERED=1

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

REPO="${REPO:-/home/manors/Guardrail-Distillation}"
BDD_SCRATCH="${BDD_SCRATCH:-/scratch/gautschi/manors/data/bdd100k}"
BDD_ZIP="${BDD_ZIP:-/scratch/gautschi/manors/data/bdd100k_raw.zip}"
STAGING="/scratch/gautschi/manors/data/bdd100k_staging"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/scratch/gautschi/manors/.kaggle}"

mkdir -p "$(dirname "$BDD_ZIP")" "$STAGING" /scratch/gautschi/manors/logs

# ── 1. Download ─────────────────────────────────────────────────────────────
# Always attempt — kaggle CLI is idempotent (re-downloads the zip if missing).
# If the zip is already there and valid, we still re-verify it below.
if [ ! -f "$BDD_ZIP" ] || ! unzip -tq "$BDD_ZIP" >/dev/null 2>&1; then
  echo "[download] solesensei/solesensei_bdd100k -> $BDD_ZIP"
  [ -f "$KAGGLE_CONFIG_DIR/kaggle.json" ] || {
    echo "ERROR: $KAGGLE_CONFIG_DIR/kaggle.json missing. Copy your Kaggle API token there first."
    exit 10
  }
  chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
  pip install -q -U kaggle
  kaggle datasets download \
    -d solesensei/solesensei_bdd100k \
    -p "$(dirname "$BDD_ZIP")"
  # kaggle writes solesensei_bdd100k.zip — rename to our canonical name
  if [ -f "$(dirname "$BDD_ZIP")/solesensei_bdd100k.zip" ] && [ ! -f "$BDD_ZIP" ]; then
    mv "$(dirname "$BDD_ZIP")/solesensei_bdd100k.zip" "$BDD_ZIP"
  fi
fi

# Sanity: is it actually a zip?
file "$BDD_ZIP" | grep -qi "zip archive" || {
  echo "ERROR: $BDD_ZIP is not a zip archive."
  head -c 500 "$BDD_ZIP"
  exit 11
}
echo "[ok] zip present: $(du -h "$BDD_ZIP" | cut -f1)"

# ── 2. Unzip into staging ───────────────────────────────────────────────────
rm -rf "$STAGING"
mkdir -p "$STAGING"
echo "[unzip] -> $STAGING"
unzip -q "$BDD_ZIP" -d "$STAGING"

# The archive nests as bdd100k/bdd100k/... — find the directory that
# actually contains seg/ and use it as the tree root.
SEG_PARENT=$(find "$STAGING" -maxdepth 6 -type d -name seg -print -quit)
[ -n "$SEG_PARENT" ] || { echo "ERROR: no seg/ directory in archive"; find "$STAGING" -maxdepth 3; exit 12; }
TREE_ROOT=$(dirname "$SEG_PARENT")
echo "[ok] tree root: $TREE_ROOT"

# ── 3. Move into final location (scratch) ───────────────────────────────────
rm -rf "$BDD_SCRATCH"
mkdir -p "$(dirname "$BDD_SCRATCH")"
mv "$TREE_ROOT" "$BDD_SCRATCH"
rm -rf "$STAGING"

# ── 4. Symlink into the repo data dir next to cityscapes/acdc ───────────────
mkdir -p "$REPO/data"
ln -sfn "$BDD_SCRATCH" "$REPO/data/bdd100k"
echo "[ok] $REPO/data/bdd100k -> $BDD_SCRATCH"

# ── 5. Validate ─────────────────────────────────────────────────────────────
IMG_TRAIN="$BDD_SCRATCH/seg/images/train"
IMG_VAL="$BDD_SCRATCH/seg/images/val"
LBL_TRAIN="$BDD_SCRATCH/seg/labels/train"
LBL_VAL="$BDD_SCRATCH/seg/labels/val"

for d in "$IMG_TRAIN" "$IMG_VAL" "$LBL_TRAIN" "$LBL_VAL"; do
  [ -d "$d" ] || { echo "ERROR: missing $d"; ls -la "$BDD_SCRATCH/seg" || true; exit 20; }
done

N_IMG_TRAIN=$(find "$IMG_TRAIN" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
N_IMG_VAL=$(find "$IMG_VAL"   -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
N_LBL_TRAIN=$(find "$LBL_TRAIN" -maxdepth 1 -type f -name '*.png' | wc -l)
N_LBL_VAL=$(find "$LBL_VAL"   -maxdepth 1 -type f -name '*.png' | wc -l)

echo "[counts] train: $N_IMG_TRAIN img / $N_LBL_TRAIN lbl"
echo "[counts] val:   $N_IMG_VAL img / $N_LBL_VAL lbl"

# Expected for the BDD 10K seg subset: 7000 train + 1000 val
[ "$N_IMG_TRAIN" -ge 6900 ] || { echo "ERROR: train images too few ($N_IMG_TRAIN, expected ~7000)"; exit 21; }
[ "$N_IMG_VAL"   -ge  990 ] || { echo "ERROR: val images too few ($N_IMG_VAL, expected ~1000)"; exit 22; }
[ "$N_LBL_TRAIN" -ge 6900 ] || { echo "ERROR: train labels too few ($N_LBL_TRAIN, expected ~7000)"; exit 23; }
[ "$N_LBL_VAL"   -ge  990 ] || { echo "ERROR: val labels too few ($N_LBL_VAL, expected ~1000)"; exit 24; }
[ "$N_IMG_TRAIN" -eq "$N_LBL_TRAIN" ] || { echo "ERROR: train image/label count mismatch"; exit 25; }
[ "$N_IMG_VAL"   -eq "$N_LBL_VAL" ]   || { echo "ERROR: val image/label count mismatch";   exit 26; }

# Verify labels are 19-class trainIds (values in {0..18, 255}), not the
# 34-class Cityscapes labelIds and not RGB color_labels. Sample 20 val pngs.
export BDD_SCRATCH
python - <<'PY'
import os, random, numpy as np
from PIL import Image
root = os.environ["BDD_SCRATCH"] + "/seg/labels/val"
files = sorted(os.listdir(root))
random.seed(0)
sample = random.sample(files, min(20, len(files)))
bad = []
for f in sample:
    arr = np.array(Image.open(os.path.join(root, f)))
    if arr.ndim != 2:
        bad.append((f, f"ndim={arr.ndim} (expected 2, is this color_labels?)"))
        continue
    uniq = set(np.unique(arr).tolist())
    allowed = set(range(19)) | {255}
    extra = uniq - allowed
    if extra:
        bad.append((f, f"unexpected values: {sorted(extra)[:8]}"))
if bad:
    print("[FAIL] label sanity check:")
    for f, why in bad:
        print(f"  {f}: {why}")
    raise SystemExit(30)
print(f"[ok] label sanity: 20/20 val pngs are single-channel with values in {{0..18, 255}}")
PY

echo "[DONE] BDD100K staged at $BDD_SCRATCH and symlinked at $REPO/data/bdd100k"
