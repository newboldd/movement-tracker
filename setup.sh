#!/usr/bin/env bash
# One-command setup and launch for DLC Labeler.
#
# Usage:
#   ./setup.sh
#
# What it does:
#   1. Checks for Python 3.9+
#   2. Creates a virtual environment (first run only)
#   3. Installs dependencies (first run only)
#   4. Starts the web app and opens your browser

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
PORT=8080

# ── Find Python ──────────────────────────────────────────────────────────
find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            # Check version >= 3.9
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python) || {
    echo "Error: Python 3.9+ is required but not found."
    echo "Install it from https://www.python.org/downloads/ and try again."
    exit 1
}

echo "Using Python: $PYTHON ($($PYTHON --version))"

# ── Create venv if needed ────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Install dependencies if needed ───────────────────────────────────────
if [ ! -f "$VENV_DIR/.installed" ] || [ "$REQUIREMENTS" -nt "$VENV_DIR/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$REQUIREMENTS" -q
    touch "$VENV_DIR/.installed"
    echo "Dependencies installed."
fi

# ── Kill any existing server on the port ─────────────────────────────────
existing=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$existing" ]; then
    echo "Stopping existing server on port $PORT..."
    echo "$existing" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── Launch ───────────────────────────────────────────────────────────────
echo ""
echo "Starting DLC Labeler at http://localhost:$PORT"
echo "Press Ctrl+C to stop."
echo ""

cd "$PROJECT_DIR"

# Open browser after a short delay (background, non-fatal)
(sleep 2 && open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null) &

python -m uvicorn dlc_app.app:app --host 127.0.0.1 --port "$PORT" --reload
