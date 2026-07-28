#!/bin/bash -l
# Sanity-check BDD100K seg val labels: each PNG must be single-channel with
# values in {0..18, 255} (Cityscapes-19 trainIds, not color_labels or 34-class).
set -euo pipefail

source "$SCRATCH/venvs/guardrail/bin/activate"

export BDD_SCRATCH="${BDD_SCRATCH:-$SCRATCH/data/bdd100k}"

python - <<'PY'
import os, random, numpy as np
from PIL import Image
root = os.environ["BDD_SCRATCH"] + "/seg/labels/val"
files = sorted(os.listdir(root))
random.seed(0)
for f in random.sample(files, 20):
    arr = np.array(Image.open(os.path.join(root, f)))
    uniq = set(np.unique(arr).tolist())
    extra = uniq - (set(range(19)) | {255})
    assert arr.ndim == 2 and not extra, (f, arr.ndim, sorted(extra)[:8])
print("[ok] 20/20 val pngs are single-channel trainIds")
PY
