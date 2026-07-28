#!/usr/bin/env bash
# Stage the newest student_skd + guardrail checkpoints into ./checkpoints/.
#
# Run from the repo root on the cluster:
#
#   ./scripts/fetch_checkpoints.sh
#
# Env overrides:
#   DEST    output dir   (default: ./checkpoints)
#   RUNS    runs dir     (default: ./runs)
#
# What it does
#   For each backbone b0/b1/b2, finds the newest
#       $RUNS/mit-b<N>_skd_*/student_skd.ckpt
#   and the newest
#       $RUNS/mit-b<N>_guard_<mode>_*/guardrail.ckpt
#   for every mode in {dense_multi, dense_disagree, dense_gap,
#   gt_disagree, gt_risk, scalar_benefit} that exists. cp -p into
#   $DEST — originals in runs/ are never touched. Modes that don't
#   exist for a backbone (b0 and b2 typically only have dense_multi)
#   are skipped silently.
#
# Resulting layout:
#   $DEST/mit-b<N>/student_skd.ckpt
#   $DEST/mit-b<N>/<mode>/guardrail.ckpt

set -euo pipefail

DEST="${DEST:-./checkpoints}"
RUNS="${RUNS:-./runs}"

BACKBONES=(b0 b1 b2)
MODES=(dense_multi dense_disagree dense_gap gt_disagree gt_risk scalar_benefit)

if [[ ! -d "$RUNS" ]]; then
    echo "ERROR: $RUNS not found. Run from the repo root or set RUNS=..." >&2
    exit 1
fi

# Newest matching file by mtime, or empty if no match. $1 is a glob and
# must be left unquoted so the shell expands it before ls sees it.
newest() {
    # shellcheck disable=SC2012,SC2086
    ls -t $1 2>/dev/null | head -1 || true
}

mkdir -p "$DEST"
echo "[stage] runs=$RUNS  dest=$DEST"

for B in "${BACKBONES[@]}"; do
    OUT_BB="$DEST/mit-$B"
    mkdir -p "$OUT_BB"

    SKD=$(newest "$RUNS/mit-${B}_skd_*/student_skd.ckpt")
    if [[ -n "$SKD" ]]; then
        cp -p "$SKD" "$OUT_BB/student_skd.ckpt"
        echo "  [$B] student_skd  <- $SKD"
    else
        echo "  [$B] student_skd  MISSING"
    fi

    for MODE in "${MODES[@]}"; do
        # Glob matches single-seed (mit-b1_guard_dense_multi_j*) and
        # multi-seed (mit-b1_guard_dense_multi_s137_j*) runs alike.
        CKPT=$(newest "$RUNS/mit-${B}_guard_${MODE}_*/guardrail.ckpt")
        if [[ -n "$CKPT" ]]; then
            mkdir -p "$OUT_BB/$MODE"
            cp -p "$CKPT" "$OUT_BB/$MODE/guardrail.ckpt"
            echo "  [$B] $MODE  <- $CKPT"
        fi
    done
done

echo
echo "[stage] done. Tree:"
find "$DEST" -name '*.ckpt' | sort
echo
du -sh "$DEST"
