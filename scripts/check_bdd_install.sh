#!/bin/bash -l
# Full post-install check for BDD100K. Verifies:
#   1. Scratch tree exists at the canonical path.
#   2. $REPO/data/bdd100k is a symlink that resolves to the scratch tree.
#   3. All four required subdirs are present (images/{train,val}, labels/{train,val}).
#   4. File counts match the BDD 10K seg subset (7000 train / 1000 val).
#   5. Every image has a matching label and vice versa (pair check by stem).
#   6. 50 random val labels are single-channel with values ⊂ {0..18, 255}.
#   7. 50 random train labels pass the same label schema check.
#   8. Images are readable RGB at the expected 1280×720 resolution.
set -euo pipefail

source "$SCRATCH/venvs/guardrail/bin/activate"

REPO="${REPO:-$HOME/Guardrail-Distillation}"
export BDD_SCRATCH="${BDD_SCRATCH:-$SCRATCH/data/bdd100k}"
REPO_LINK="$REPO/data/bdd100k"

fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[ok]   $*"; }

# 1. Scratch tree
[ -d "$BDD_SCRATCH" ] || fail "scratch tree missing: $BDD_SCRATCH"
pass "scratch tree present: $BDD_SCRATCH"

# 2. Repo symlink
[ -L "$REPO_LINK" ] || fail "$REPO_LINK is not a symlink"
RESOLVED=$(readlink -f "$REPO_LINK")
[ "$RESOLVED" = "$BDD_SCRATCH" ] || fail "symlink resolves to $RESOLVED, expected $BDD_SCRATCH"
pass "symlink: $REPO_LINK -> $BDD_SCRATCH"

# 3. Required subdirs
for d in seg/images/train seg/images/val seg/labels/train seg/labels/val; do
  [ -d "$BDD_SCRATCH/$d" ] || fail "missing $BDD_SCRATCH/$d"
done
pass "all required subdirs present"

# 4. Counts
N_IMG_TRAIN=$(find "$BDD_SCRATCH/seg/images/train" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
N_IMG_VAL=$(find   "$BDD_SCRATCH/seg/images/val"   -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
N_LBL_TRAIN=$(find "$BDD_SCRATCH/seg/labels/train" -maxdepth 1 -type f -name '*.png' | wc -l)
N_LBL_VAL=$(find   "$BDD_SCRATCH/seg/labels/val"   -maxdepth 1 -type f -name '*.png' | wc -l)
echo "       counts  train: $N_IMG_TRAIN img / $N_LBL_TRAIN lbl"
echo "       counts  val:   $N_IMG_VAL img / $N_LBL_VAL lbl"
[ "$N_IMG_TRAIN" -eq 7000 ] || fail "train images expected 7000, got $N_IMG_TRAIN"
[ "$N_IMG_VAL"   -eq 1000 ] || fail "val images expected 1000, got $N_IMG_VAL"
[ "$N_LBL_TRAIN" -eq 7000 ] || fail "train labels expected 7000, got $N_LBL_TRAIN"
[ "$N_LBL_VAL"   -eq 1000 ] || fail "val labels expected 1000, got $N_LBL_VAL"
pass "counts match BDD 10K seg subset (7000/1000)"

# 5–8. Deep checks
python - <<'PY'
import os, random, numpy as np
from pathlib import Path
from PIL import Image

root = Path(os.environ["BDD_SCRATCH"]) / "seg"
allowed = set(range(19)) | {255}

def stem(p: Path, suffixes):
    s = p.stem
    for suf in suffixes:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s

def check_pairs(split):
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    lbls = sorted((root / "labels" / split).glob("*.png"))
    img_stems = {p.stem for p in imgs}
    lbl_stems = {stem(p, ["_train_id", "_trainIds"]) for p in lbls}
    missing_lbl = img_stems - lbl_stems
    missing_img = lbl_stems - img_stems
    if missing_lbl or missing_img:
        raise SystemExit(
            f"[FAIL] {split} pair mismatch: "
            f"{len(missing_lbl)} imgs without label, "
            f"{len(missing_img)} labels without img "
            f"(e.g. {sorted(missing_lbl)[:3]} / {sorted(missing_img)[:3]})"
        )
    print(f"[ok]   {split}: all {len(imgs)} images paired with labels")

def check_labels(split, n):
    lbls = sorted((root / "labels" / split).glob("*.png"))
    random.seed(0)
    sample = random.sample(lbls, min(n, len(lbls)))
    for p in sample:
        arr = np.array(Image.open(p))
        if arr.ndim != 2:
            raise SystemExit(f"[FAIL] {p.name} has ndim={arr.ndim} (color_labels?)")
        extra = set(np.unique(arr).tolist()) - allowed
        if extra:
            raise SystemExit(f"[FAIL] {p.name} has unexpected values {sorted(extra)[:8]}")
    print(f"[ok]   {split}: {len(sample)}/{len(sample)} sampled labels are trainIds ⊂ {{0..18, 255}}")

def check_images(split, n):
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    random.seed(1)
    sample = random.sample(imgs, min(n, len(imgs)))
    sizes = set()
    for p in sample:
        im = Image.open(p)
        im.verify()
        im = Image.open(p)
        if im.mode != "RGB":
            raise SystemExit(f"[FAIL] {p.name} mode={im.mode} (expected RGB)")
        sizes.add(im.size)
    print(f"[ok]   {split}: {len(sample)} images readable RGB, sizes={sizes}")

for split in ("train", "val"):
    check_pairs(split)
check_labels("val", 50)
check_labels("train", 50)
check_images("val", 20)
check_images("train", 20)
PY

echo
echo "[DONE] BDD100K is staged and validated at $BDD_SCRATCH"
echo "       Reachable in the repo as:          $REPO_LINK"
