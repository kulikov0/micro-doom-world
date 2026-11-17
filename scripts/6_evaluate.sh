#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python evaluate.py ladder rf dyn_ladder_1750.pt dyn_base.pt
