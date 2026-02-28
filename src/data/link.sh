#!/usr/bin/env bash
# setup_cityscapes.sh — Link HF-cached Cityscapes into a clean directory structure.
#
# Usage:
#   bash setup_cityscapes.sh [TARGET_DIR]
#
# Defaults:
#   TARGET_DIR = ./data/cityscapes
#   Searches HF_HOME, then ~/.cache/huggingface, then common locations

set -euo pipefail

TARGET="${1:-./data/cityscapes}"
IMG_DATASET="datasets--ShayManor--leftImg8bit_trainvaltest"
LBL_DATASET="datasets--ShayManor--gtFine_trainvaltest"

# ── Locate HF cache ──────────────────────────────────────────────────────────
find_hf_cache() {
    local candidates=(
        "${HF_HOME:-}/hub"
        "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub"
        "$HOME/.cache/huggingface/hub"
    )
    # Also search common workspace locations
    for base in /workspace /root /home/*; do
        candidates+=("$base/.hf_home/hub" "$base/.cache/huggingface/hub")
    done

    for dir in "${candidates[@]}"; do
        if [[ -d "$dir/$IMG_DATASET" && -d "$dir/$LBL_DATASET" ]]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

HF_HUB=$(find_hf_cache) || { echo "ERROR: Could not find HF datasets. Set HF_HOME or pass datasets manually."; exit 1; }
echo "[setup] Found HF cache: $HF_HUB"

# ── Resolve snapshot dirs (handles any hash) ─────────────────────────────────
resolve_snapshot() {
    local dataset_dir="$1"
    local snap_dir="$dataset_dir/snapshots"
    if [[ ! -d "$snap_dir" ]]; then
        echo "ERROR: No snapshots/ in $dataset_dir" >&2; exit 1
    fi
    # Pick the latest snapshot (by modification time)
    local latest
    latest=$(ls -td "$snap_dir"/*/ 2>/dev/null | head -1)
    if [[ -z "$latest" ]]; then
        echo "ERROR: No snapshot found in $snap_dir" >&2; exit 1
    fi
    echo "$latest"
}

IMG_SNAP=$(resolve_snapshot "$HF_HUB/$IMG_DATASET")
LBL_SNAP=$(resolve_snapshot "$HF_HUB/$LBL_DATASET")

# ── Find the actual data root inside each snapshot ───────────────────────────
# Images: look for leftImg8bit/ dir, or fall back to snapshot root
if [[ -d "${IMG_SNAP}leftImg8bit" ]]; then
    IMG_SRC="${IMG_SNAP}leftImg8bit"
else
    # Snapshot IS the images dir (flat structure)
    IMG_SRC="${IMG_SNAP}"
fi

# Labels: look for gtFine/ dir, or fall back to snapshot root
if [[ -d "${LBL_SNAP}gtFine" ]]; then
    LBL_SRC="${LBL_SNAP}gtFine"
else
    LBL_SRC="${LBL_SNAP}"
fi

# ── Create symlinks ──────────────────────────────────────────────────────────
mkdir -p "$TARGET"

for name in leftImg8bit:IMG_SRC gtFine:LBL_SRC; do
    link_name="${name%%:*}"
    var_name="${name##*:}"
    src="${!var_name}"
    dst="$TARGET/$link_name"

    if [[ -L "$dst" ]]; then
        rm "$dst"
    elif [[ -d "$dst" ]]; then
        echo "[WARN] $dst exists as a real directory, skipping"
        continue
    fi

    ln -s "$src" "$dst"
    echo "[setup] $dst -> $src"
done

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
IMG_COUNT=$(find "$TARGET/leftImg8bit" -name "*.png" 2>/dev/null | wc -l)
LBL_COUNT=$(find "$TARGET/gtFine" -name "*labelIds*" 2>/dev/null | wc -l)
echo "[setup] Images: $IMG_COUNT .png files"
echo "[setup] Labels: $LBL_COUNT labelIds files"

if [[ $IMG_COUNT -eq 0 || $LBL_COUNT -eq 0 ]]; then
    echo "[WARN] Missing files — check dataset downloads"
    echo "  leftImg8bit: $IMG_SRC"
    echo "  gtFine: $LBL_SRC"
    exit 1
fi

echo "[setup] Ready: $TARGET"
echo "  leftImg8bit/ -> $IMG_SRC"
echo "  gtFine/      -> $LBL_SRC"