#!/bin/bash -l
#SBATCH --job-name=upload-idd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH --mem=16G
#SBATCH -o /scratch/gautschi/manors/logs/upload-idd_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/upload-idd_%j.err

# Push the prepped IDD tree to a private HF dataset repo.
#
# Inputs (env overrides):
#   HF_REPO    — target repo id (default: ShayManor/idd-cs19)
#   IDD_ROOT   — where prep_idd.sbatch wrote the data (default: /scratch/gautschi/manors/data/idd)
#   HF_TOKEN   — write-scoped HF token. If not set, relies on a pre-existing
#                `huggingface-cli login` cached under $HF_HOME.
#
# Uploads only what a Cityscapes-style loader needs:
#   - leftImg8bit/**  (images)
#   - gtFine/**/*_csTrainIds.png  (the remapped CS-19 labels)
# Original 34-class labels and toolkit artifacts are skipped to keep the repo small.

set -euxo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

HF_REPO="${HF_REPO:-ShayManor/idd-cs19}"
IDD_ROOT="${IDD_ROOT:-/scratch/gautschi/manors/data/idd}"
SRC="$IDD_ROOT/idd20k_II"
export HF_HOME="${HF_HOME:-/scratch/gautschi/manors/hf_cache}"

[ -d "$SRC" ] || { echo "ERROR: $SRC missing — run prep_idd.sbatch first"; exit 1; }

pip install -q -U "huggingface_hub[cli]"

# Auth: prefer HF_TOKEN env, fall back to cached login
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Create the private repo if it doesn't exist (no-op if it does)
huggingface-cli repo create "${HF_REPO##*/}" \
  --type dataset \
  --organization "${HF_REPO%%/*}" \
  --private \
  -y || true

cd "$SRC"

# huggingface-cli upload resumes on retry, so re-submitting this job is safe.
huggingface-cli upload "$HF_REPO" . . \
  --repo-type dataset \
  --include "leftImg8bit/**" "gtFine/**/*_csTrainIds.png" \
  --commit-message "IDD Segmentation (Part I+II) with Cityscapes-19 remapped labels"

echo "[DONE] pushed $SRC -> https://huggingface.co/datasets/$HF_REPO"
