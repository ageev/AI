#!/bin/sh
# Restore the oMLX launchd job exactly as before install.sh. Does NOT restart.
#   uninstall.sh [--purge]   --purge also deletes ~/omlx-steer-data and ~/omlx-steer-dev
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DATA="${OMLX_STEER_DATA:-$HOME/omlx-steer-data}"
STATE="$DATA/deploy-state.json"
[ -f "$STATE" ] || { echo "no $STATE: nothing to restore" >&2; exit 1; }
PLIST="$(/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["plist"])' "$STATE")"
# Plain bundled interpreter on purpose: works even if the checkout is broken.
/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 "$ROOT/deploy/plist_edit.py" uninstall "$STATE"
plutil -lint "$PLIST"
if [ "${1:-}" = "--purge" ]; then
    rm -rf "$DATA" "$HOME/omlx-steer-dev"
    echo "purged $DATA and ~/omlx-steer-dev (the code checkout stays)"
fi
echo "To apply (restarts oMLX, the model reloads):"
echo "  launchctl bootout gui/$(id -u) $PLIST; launchctl bootstrap gui/$(id -u) $PLIST"
