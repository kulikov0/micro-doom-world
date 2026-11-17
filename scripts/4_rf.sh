#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python train.py --tag rf --init dyn_base.pt --rf 0.75 --repa 0 --batch 8 --lr 2e-4 --max-steps 5000 --schedule cosine \
    --keep-every 2500 --compile default
