#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="rag-bench"
PORT=6742
DEV_PORT=5173
PID_FILE="$PROJECT_ROOT/data/server.pid"
DEV_PID_FILE="$PROJECT_ROOT/data/vite.pid"
SERVER_LOG="$PROJECT_ROOT/logs/server.log"
DEV_LOG="$PROJECT_ROOT/logs/vite.log"

mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/logs"

alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

# Build the bundle when it is missing — the fresh-clone case, where dist is
# absent because it is gitignored. Keeping it current after an edit belongs to
# the edit, not to the next start: this is a floor, not the mechanism.
if [[ ! -f "$PROJECT_ROOT/frontend/dist/index.html" ]]; then
    echo "Building frontend..."
    cd "$PROJECT_ROOT/frontend"
    [[ -d node_modules ]] || npm install
    npm run build
    cd "$PROJECT_ROOT"
fi

# --- Backend: uvicorn on 6742, serves the API and the built SPA -------------
if alive "$PID_FILE"; then
    echo "Backend already running (PID $(cat "$PID_FILE"))"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    cd "$PROJECT_ROOT/backend"
    nohup python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >>"$SERVER_LOG" 2>&1 &
    echo $! >"$PID_FILE"
    cd "$PROJECT_ROOT"

    echo "Waiting for the backend..."
    for _ in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
        echo "Failed to start the backend. Recent server log:" >&2
        tail -20 "$SERVER_LOG" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
fi

# --- Frontend dev server: Vite on 5173, hot reload --------------------------
# Launched through the vite binary rather than `npm run dev`, so the PID we
# record is the server itself: npm forks a child and would survive our kill.
# --strictPort turns an occupied port into a loud failure instead of a silent
# move to 5174, where the API proxy still works but the URL you bookmarked
# does not.
if alive "$DEV_PID_FILE"; then
    echo "Dev server already running (PID $(cat "$DEV_PID_FILE"))"
else
    cd "$PROJECT_ROOT/frontend"
    [[ -d node_modules ]] || npm install
    nohup ./node_modules/.bin/vite --port "$DEV_PORT" --strictPort >>"$DEV_LOG" 2>&1 &
    echo $! >"$DEV_PID_FILE"
    cd "$PROJECT_ROOT"

    echo "Waiting for the dev server..."
    for _ in $(seq 1 30); do
        if curl -sf "http://localhost:$DEV_PORT/" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -sf "http://localhost:$DEV_PORT/" >/dev/null 2>&1; then
        echo "Failed to start the dev server. Recent log:" >&2
        tail -20 "$DEV_LOG" >&2
        rm -f "$DEV_PID_FILE"
        exit 1
    fi
fi

echo
echo "RAG-BenchLab is running"
echo "  Debug (hot reload): http://localhost:$DEV_PORT"
echo "  Built (acceptance): http://localhost:$PORT"
echo "  App log:            logs/app.log"
echo "  Server log:         logs/server.log"
echo "  Vite log:           logs/vite.log"
