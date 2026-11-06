#!/bin/sh
set -eu
BASELINE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$BASELINE_DIR/baseline.py"
