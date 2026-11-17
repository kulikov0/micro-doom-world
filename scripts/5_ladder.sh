#!/bin/bash
set -e
cd "$(dirname "$0")/.."
INIT=${1:-dyn_base.pt} TAG=${2:-ladder}
OUT=${MICRO_DOOM_OUT:-out}
export MICRO_DOOM_MPS_AMP=bf16
rm -f "$OUT/dyn_${TAG}_resume.pt"
for stage in "750 2 8 2" "1250 4 4 1" "1750 8 4 1"; do
    set -- $stage
    python adversarial.py --init $INIT --tag $TAG --max-steps $1 --segs $2 --batch $3 --accum $4
    cp "$OUT/dyn_$TAG.pt" "$OUT/dyn_${TAG}_$1.pt"
done
