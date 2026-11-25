#!/bin/sh
# Isolated test instance of oMLX + omlx-steer on the tiny checkpoint.
#   dev-stand.sh start|stop|log
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DEV="${OMLX_STEER_DEV:-$HOME/omlx-steer-dev}"
PORT="${PORT:-8099}"
case "${1:-start}" in
start)
    mkdir -p "$DEV/base" "$DEV/profiles"
    [ -f "$DEV/base/model_settings.json" ] || cat > "$DEV/base/model_settings.json" <<JSON
{"version": 2, "models": {"DeepSeek-V4.1-tiny": {"model_alias": "tiny", "mtp_enabled": true,
 "is_pinned": true, "is_default": true, "temperature": 1.0, "top_p": 1.0}}}
JSON
    # NOSTEER=1 runs plain oMLX (no hooks) for A/B comparisons.
    if [ -n "${NOSTEER:-}" ]; then ENTRY="-m omlx.cli"; else ENTRY="$ROOT/deploy/launch.py"; fi
    OMLX_BASE_PATH="$DEV/base" OMLX_STEER_PROFILES="$DEV/profiles" OMLX_STEER_SWEEP=1 OMLX_STEER_DEBUG="${OMLX_STEER_DEBUG:-}" OMLX_STEER_PINS="${OMLX_STEER_PINS:-}" \
        nohup "$ROOT/deploy/omlx-python.sh" $ENTRY serve \
        --model-dir "$DEV/models/tiny" --host 127.0.0.1 --port "$PORT" > "$DEV/server.log" 2>&1 &
    echo $! > "$DEV/server.pid"
    echo "started pid $(cat "$DEV/server.pid") on 127.0.0.1:$PORT"
    ;;
stop)
    [ -f "$DEV/server.pid" ] && kill "$(cat "$DEV/server.pid")" 2>/dev/null || true
    rm -f "$DEV/server.pid"
    ;;
log) tail -n "${2:-50}" "$DEV/server.log" ;;
esac
