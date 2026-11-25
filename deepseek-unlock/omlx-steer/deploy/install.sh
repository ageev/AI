#!/bin/sh
# Point the oMLX launchd job at omlx-steer. Does NOT restart the server.
#   install.sh [--sweep]
# Changes exactly ProgramArguments (launcher in front of the same omlx-cli
# arguments) and two OMLX_STEER_* environment variables. The originals go to
# ~/omlx-steer-data/deploy-state.json; uninstall.sh restores them.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PLIST="${OMLX_PLIST:-$HOME/Library/LaunchAgents/local.omlx.plist}"
DATA="${OMLX_STEER_DATA:-$HOME/omlx-steer-data}"
SWEEP=0; [ "${1:-}" = "--sweep" ] && SWEEP=1
"$ROOT/deploy/omlx-python.sh" "$ROOT/tools/check_pins.py"
mkdir -p "$DATA/profiles"
"$ROOT/deploy/omlx-python.sh" "$ROOT/deploy/plist_edit.py" install "$PLIST" "$DATA/deploy-state.json" "$ROOT" "$DATA" "$SWEEP"
plutil -lint "$PLIST"
echo "To apply (restarts oMLX, the model reloads):"
echo "  launchctl bootout gui/$(id -u) $PLIST; launchctl bootstrap gui/$(id -u) $PLIST"
