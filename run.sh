#!/usr/bin/env bash
# Launch the Movement Tracker web app.
# Usage: ./run.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PORT=8080

# Kill any existing process using the port
existing=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$existing" ]; then
    echo "Stopping existing server on port $PORT..."
    echo "$existing" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Activate venv if it exists (created by setup.sh)
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "Starting Movement Tracker..."
echo ""
echo "Dashboard will open at http://localhost:$PORT"
echo ""

cd "$PROJECT_DIR"
python -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port $PORT --reload
