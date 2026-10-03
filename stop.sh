#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PID_FILE="$PROJECT_ROOT/data/server.pid"
DEV_PID_FILE="$PROJECT_ROOT/data/vite.pid"

stop_one() {
    local label="$1" pid_file="$2" pid
    if [[ ! -f "$pid_file" ]]; then
        echo "$label is not running (no PID file)."
        return 0
    fi
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "Process $pid did not exit gracefully; forcing..." >&2
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "$label stopped (PID $pid)."
    else
        echo "$label is not running (stale PID file removed)."
    fi
    rm -f "$pid_file"
}

stop_one "Backend" "$SERVER_PID_FILE"
stop_one "Dev server" "$DEV_PID_FILE"
