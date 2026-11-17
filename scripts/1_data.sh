#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python data.py record
for kind in e1m1 e1m2 e1m3; do python data.py audit --kind $kind; done
python data.py unpack
