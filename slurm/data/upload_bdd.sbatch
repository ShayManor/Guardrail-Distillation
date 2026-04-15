#!/bin/bash -l
#SBATCH --job-name=upload-bdd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH --mem=16G
#SBATCH -o /scratch/gautschi/manors/logs/upload-bdd_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/upload-bdd_%j.err

# Push the staged BDD100K seg subset to a private HF dataset repo so we have
# a stable, re-pullable snapshot (and somewhere to park downstream models).
#
# Creates the repo private-by-default if it doesn't exist, then uploads only
# what a Cityscapes-style loader needs:
#   - seg/images/{train,val,test}/**   (jpgs — test is unlabeled but cheap)
#   - seg/labels/{train,val}/**        (19 trainIds PNGs)
# Skips seg/color_labels/** (RGB visualizations, 2–3 GB of dead weight) and
# skips bdd100k_labels_release/ (detection JSONs we don't use).
#
# Idempotent: huggingface-cli upload resumes on retry, so re-submitting this
# job after a network blip just continues where it left off.
#
# Env overrides:
#   HF_REPO     — target repo id     (default: ShayManor/bdd100k-cs19)
#   BDD_ROOT    — source tree root   (default: /scratch/gautschi/manors/data/bdd100k)
#   HF_TOKEN    — write-scoped token. If not set, relies on a pre-existing
#                 `huggingface-cli login` cached under $HF_HOME.

set -euxo pipefail
trap 'echo "[FAIL] line $LINENO exited with $?" >&2' ERR
export PYTHONUNBUFFERED=1

source /scratch/gautschi/manors/venvs/guardrail/bin/activate

HF_REPO="${HF_REPO:-ShayManor/bdd100k-cs19}"
BDD_ROOT="${BDD_ROOT:-/scratch/gautschi/manors/data/bdd100k}"
export HF_HOME="${HF_HOME:-/scratch/gautschi/manors/hf_cache}"

SRC="$BDD_ROOT/seg"
[ -d "$SRC/images/val" ]  || { echo "ERROR: $SRC/images/val missing — run prep_bdd.sbatch first"; exit 1; }
[ -d "$SRC/labels/val" ]  || { echo "ERROR: $SRC/labels/val missing — run prep_bdd.sbatch first"; exit 1; }

# Pin hub <1.0: v1.x renamed the CLI to `hf`, removed update_repo_visibility,
# and is incompatible with the transformers pinned in this venv.
pip install -q "huggingface_hub>=0.34,<1.0"

# Auth: prefer HF_TOKEN env, fall back to cached login under $HF_HOME
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Create + force-private + upload all via the Python API so we don't depend
# on whichever CLI shim (huggingface-cli vs hf) happens to be on PATH.
export HF_REPO BDD_ROOT
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo

repo = os.environ["HF_REPO"]
root = os.environ["BDD_ROOT"]
api = HfApi()

create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
print(f"[ok] repo ensured: {repo} (private=True)")

# Defensive: if it pre-existed as public, flip it. (0.34+ API.)
try:
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    print("[ok] repo settings forced private")
except Exception as e:
    print(f"[warn] update_repo_settings failed: {e}")

api.upload_large_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path=root,
    allow_patterns=["seg/images/**", "seg/labels/**"],
)
print(f"[DONE] uploaded seg/images + seg/labels from {root} -> {repo}")
PY

echo "[DONE] pushed $SRC -> https://huggingface.co/datasets/$HF_REPO"
echo "       pull later with:"
echo "         huggingface-cli download $HF_REPO --repo-type dataset \\"
echo "           --local-dir /scratch/gautschi/manors/data/bdd100k"
