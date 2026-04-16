#!/bin/bash -l
#SBATCH --job-name=prep-idd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH --mem=32G
#SBATCH -o /scratch/gautschi/manors/logs/prep-idd_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/prep-idd_%j.err

# Download + unzip + Cityscapes-19 remap for IDD Segmentation (Part I + II).
#
# Inputs (env overrides):
#   IDD_URL    — signed download URL from idd.insaan.iiit.ac.in (required on first run)
#   IDD_ROOT   — where to stage the data (default: /scratch/gautschi/manors/data/idd)
#
# Outputs:
#   $IDD_ROOT/idd20k_II/leftImg8bit/{train,val}/...
#   $IDD_ROOT/idd20k_II/gtFine/{train,val}/.../ *_csTrainIds.png
#
# The *_csTrainIds.png files are the Cityscapes-19 remapped labels written by
# AutoNUE/public-code's preperation/createLabels.py. They drop in as labels for
# a Cityscapes-style dataloader with no further remap.

set -euxo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

IDD_ROOT="${IDD_ROOT:-/scratch/gautschi/manors/data/idd}"
IDD_URL="${IDD_URL:-}"
mkdir -p "$IDD_ROOT"
cd "$IDD_ROOT"

# 1. Download (skip if zip already on disk)
if [ ! -f idd_raw.zip ]; then
  if [ -z "$IDD_URL" ]; then
    echo "ERROR: idd_raw.zip not present and IDD_URL not set. Export IDD_URL and resubmit."
    exit 1
  fi
  echo "[download] $IDD_URL -> $IDD_ROOT/idd_raw.zip"
  wget --tries=3 --continue -O idd_raw.zip "$IDD_URL"
fi

# Sanity check: make sure we got a zip, not an HTML login page
file idd_raw.zip | grep -qi "zip archive" || {
  echo "ERROR: idd_raw.zip is not a zip archive (signed URL expired or IP-bound?)."
  head -c 500 idd_raw.zip
  exit 2
}

# 2. Unzip (idempotent — -n skips existing)
if [ ! -d idd20k_II ] && [ ! -d IDD_Segmentation ]; then
  echo "[unzip] extracting idd_raw.zip"
  unzip -q -n idd_raw.zip
fi

# Normalize directory name — IDD ships as either idd20k_II/ or IDD_Segmentation/
if [ -d IDD_Segmentation ] && [ ! -d idd20k_II ]; then
  mv IDD_Segmentation idd20k_II
fi
[ -d idd20k_II ] || { echo "ERROR: expected idd20k_II/ after unzip"; ls -la; exit 3; }

# 3. Toolkit for the 34-class -> Cityscapes-19 remap
if [ ! -d public-code ]; then
  git clone --depth 1 https://github.com/AutoNUE/public-code.git
fi

# cityscapesScripts is needed by createLabels.py
pip install -q cityscapesscripts numpngw imageio tqdm pandas scipy

# 4. Generate *_csTrainIds.png alongside the original labels
cd public-code
export PYTHONPATH="$PWD:$PWD/helpers:${PYTHONPATH:-}"
python preperation/createLabels.py \
  --datadir "$IDD_ROOT/idd20k_II" \
  --id-type csTrainId \
  --num-workers 8

# 5. Summary
cd "$IDD_ROOT"
N_IMG=$(find idd20k_II/leftImg8bit -name '*.png' -o -name '*.jpg' | wc -l)
N_LBL=$(find idd20k_II/gtFine -name '*_csTrainIds.png' | wc -l)
echo "[DONE] IDD ready at $IDD_ROOT/idd20k_II"
echo "  images:      $N_IMG"
echo "  cs19 labels: $N_LBL"
