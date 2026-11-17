#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python tokenizer.py --tag tokenizer --steps 40500 --lr 3e-4
python tokenizer.py --tag tokenizer_gan --init tokenizer.pt --adv 0.5 --adaptive --lr-d 4e-4 --d-steps 2 \
    --enemy-w 4 --enemy-share 0.5 --steps 9000
python data.py encode --tok tokenizer.pt
