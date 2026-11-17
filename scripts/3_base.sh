#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python train.py --tag base --compile default
