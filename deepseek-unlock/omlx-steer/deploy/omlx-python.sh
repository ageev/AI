#!/bin/sh
# Run oMLX's bundled interpreter with the same environment omlx-cli sets.
set -eu
APP="${OMLX_APP:-/Applications/oMLX.app}"
RESOURCES="$APP/Contents/Resources"
PYROOT="$RESOURCES/Python"
export PYTHONHOME="$PYROOT/cpython-3.11"
export PYTHONDONTWRITEBYTECODE=1
STEER_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$RESOURCES:$PYROOT/framework-mlx-base/lib/python3.11/site-packages:$STEER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHONHOME/bin/python3" "$@"
