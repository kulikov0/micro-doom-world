#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python export_web.py dyn_ladder_1750.pt --out web/model --check 3
python export_web.py dyn_rf_ladder_1750.pt --out web/rf --check 3
