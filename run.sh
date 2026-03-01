#!/usr/bin/env bash
# Launch the DLC Labeler web app.
# Usage: ./run.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# Activate venv if it exists (created by setup.sh)
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "Starting DLC Labeler..."
echo ""
echo "Dashboard will open at http://localhost:8080"
echo ""

cd "$PROJECT_DIR"
python -m uvicorn dlc_app.app:app --host 127.0.0.1 --port 8080 --reload
