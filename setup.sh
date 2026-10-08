#!/usr/bin/env bash
# One-command setup and launch for DLC Labeler (macOS / Linux).
#
# Usage:
#   ./setup.sh                 install what labeling needs, then start
#   ./setup.sh --with-dlc      also install DeepLabCut, for training
#   ./setup.sh --reinstall     rebuild the virtual environment from scratch
#
# The install is deliberately two-tier.  The base tier is small and fast
# and is all you need to label frames, run MediaPipe and review
# predictions.  DeepLabCut and torch are several GB and only matter on a
# machine that will actually train, so they are opt-in — via --with-dlc
# here, or the button on the Jobs page later.
#
# Data (videos, DLC projects, the database) lives outside the code, in
# DLC_DATA_DIR.  Set it in a .env file next to this script:
#     DLC_DATA_DIR=~/data/dlc-labeler

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
REQUIREMENTS_DLC="$PROJECT_DIR/requirements-dlc.txt"
PORT="${DLC_PORT:-8080}"

WITH_DLC=0
REINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --with-dlc)  WITH_DLC=1 ;;
        --reinstall) REINSTALL=1 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

# Local overrides (DLC_DATA_DIR, DLC_PORT, …)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# MT_DATA_DIR is honoured too, so a machine already set up for the full
# Movement Tracker app points at the same data without being re-configured.
export DLC_DATA_DIR="${DLC_DATA_DIR:-${MT_DATA_DIR:-$PROJECT_DIR/data}}"
PORT="${DLC_PORT:-$PORT}"
OS="$(uname -s)"   # Darwin | Linux

print_header() { echo ""; echo "── $1 ──────────────────────────────────"; }

# ── Python ────────────────────────────────────────────────────────────────

find_python() {
    # 3.9 is the floor FastAPI and the type hints here need; mediapipe
    # wheels below 0.10.19 stop at 3.12, so prefer 3.11/3.12 and warn
    # rather than fail on anything newer.
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || continue
            major=${version%%.*}
            minor=${version##*.}
            if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

install_python() {
    print_header "Installing Python"
    echo "Python 3.9+ is required but was not found."
    echo ""
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            echo "Installing Python 3.11 via Homebrew..."
            brew install python@3.11
        else
            echo "Homebrew is not installed. Either:"
            echo "  1. Install Homebrew (https://brew.sh) and re-run this script, or"
            echo "  2. Install Python from https://www.python.org/downloads/"
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -q && sudo apt-get install -y python3.11 python3.11-venv python3-pip
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3.11
        else
            echo "Could not detect a package manager."
            echo "Install Python 3.9+ from https://www.python.org/downloads/ and re-run."
            exit 1
        fi
    else
        echo "Automatic install is not supported on this OS."
        echo "Install Python 3.9+ from https://www.python.org/downloads/ and re-run."
        exit 1
    fi
}

PYTHON=$(find_python) || {
    install_python
    PYTHON=$(find_python) || {
        echo "Python 3.9+ still not found. Install it manually and re-run."
        exit 1
    }
}

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

case "$PY_VERSION" in
    3.13|3.14|3.15)
        echo ""
        echo "WARNING: MediaPipe does not publish wheels for Python $PY_VERSION."
        echo "         Hand detection will not install. Everything else works."
        echo "         Install Python 3.11 or 3.12 and re-run to get MediaPipe."
        echo ""
        ;;
esac

# ── ffmpeg ────────────────────────────────────────────────────────────────
# imageio-ffmpeg bundles a binary, so a system ffmpeg is a nicety, not a
# requirement.  Don't make a missing one look like a failure.
if ! command -v ffmpeg &>/dev/null; then
    echo "No system ffmpeg found — the bundled one (imageio-ffmpeg) will be used."
fi

# ── Virtual environment ───────────────────────────────────────────────────

if [ "$REINSTALL" = "1" ] && [ -d "$VENV_DIR" ]; then
    print_header "Removing the old virtual environment"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    print_header "Creating the virtual environment"
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

# Re-install whenever requirements.txt is newer than the last successful run.
if [ ! -f "$VENV_DIR/.installed" ] || [ "$REQUIREMENTS" -nt "$VENV_DIR/.installed" ]; then
    print_header "Installing Python dependencies"
    echo "A few minutes on first run."
    echo ""
    "$VENV_PY" -m pip install --upgrade pip -q
    if ! "$VENV_PY" -m pip install -r "$REQUIREMENTS"; then
        echo ""
        echo "Dependency install failed."
        echo "If it was MediaPipe, check that this Python is 3.9-3.12:"
        echo "  $("$VENV_PY" --version 2>&1)"
        exit 1
    fi
    touch "$VENV_DIR/.installed"
    echo ""
    echo "Dependencies installed."
fi

# ── DeepLabCut (second tier) ──────────────────────────────────────────────

if [ "$WITH_DLC" = "1" ]; then
    if "$VENV_PY" -c "import deeplabcut" 2>/dev/null; then
        echo "DeepLabCut is already installed."
    else
        print_header "Installing DeepLabCut"
        echo "Several GB — this is the slow part. It only has to happen once."
        echo ""
        "$VENV_PY" -m pip install -r "$REQUIREMENTS_DLC"
        echo ""
        echo "DeepLabCut installed."
    fi
fi

# ── Data directory ────────────────────────────────────────────────────────

mkdir -p "$DLC_DATA_DIR/dlc" "$DLC_DATA_DIR/videos" "$DLC_DATA_DIR/calibration"

# Ship the bundled calibration so a stereo rig that matches it gets 3D
# straight away.  Never overwrite one that is already there.
if [ -d "$PROJECT_DIR/calibration" ]; then
    for f in "$PROJECT_DIR"/calibration/*; do
        [ -f "$f" ] || continue
        dest="$DLC_DATA_DIR/calibration/$(basename "$f")"
        [ -e "$dest" ] || cp "$f" "$dest"
    done
fi

# ── Choose a port ─────────────────────────────────────────────────────────
#
# Movement Tracker defaults to this same 8080.  Two rules: never kill
# something that is not this app, and never hand the browser a port we are
# not actually serving on — opening the URL of a *different* app that
# happens to hold 8080 is the most confusing failure there is.
#
# Freeness is decided by trying to bind the port, not by reading lsof or
# netstat.  Binding is the same question the server itself will ask, needs
# no external tool, and cannot be fooled by a listener owned by another
# user or by a machine without lsof installed.

LAUNCHER="$PROJECT_DIR/scripts/launcher.py"

port_free() {
    "$VENV_PY" "$LAUNCHER" free "$1"
}

# Only used to decide whether a busy port is OUR stale server, so a missing
# lsof costs nothing: we simply move to the next free port instead.
port_held_by_us() {
    command -v lsof &>/dev/null || return 1
    local pids pid
    pids=$(lsof -ti ":$1" 2>/dev/null) || return 1
    [ -n "$pids" ] || return 1
    for pid in $pids; do
        # Match the uvicorn target exactly.  A loose "dlc_labeler" would
        # also match an editor or a tail -f with the path open.
        ps -p "$pid" -o args= 2>/dev/null | grep -q "dlc_labeler.app:app" || return 1
    done
    echo "$pids"
}

if ! port_free "$PORT"; then
    if ours=$(port_held_by_us "$PORT"); then
        echo "Stopping a previous DLC Labeler on port $PORT..."
        echo "$ours" | xargs kill 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            port_free "$PORT" && break
            sleep 0.5
        done
        if ! port_free "$PORT"; then
            echo "$ours" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    fi
fi

if ! port_free "$PORT"; then
    busy=$PORT
    for try in $(seq $((PORT + 1)) $((PORT + 20))); do
        if port_free "$try"; then
            PORT=$try
            break
        fi
    done
    if [ "$PORT" = "$busy" ]; then
        echo ""
        echo "Port $busy is in use and no free port was found in "
        echo "$((busy + 1))-$((busy + 20)). Set DLC_PORT in .env to a free one."
        echo ""
        exit 1
    fi
    echo ""
    echo "Port $busy is in use by another program (Movement Tracker, perhaps)."
    echo "Using port $PORT instead. Set DLC_PORT in .env to pin this."
    echo ""
fi

# ── Launch ────────────────────────────────────────────────────────────────

echo ""
echo "Starting DLC Labeler at http://localhost:$PORT"
echo "Data directory: $DLC_DATA_DIR"
if ! "$VENV_PY" -c "import deeplabcut" 2>/dev/null; then
    echo ""
    echo "DeepLabCut is not installed, so training and analysis are off."
    echo "Add it any time with:  ./setup.sh --with-dlc"
fi
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$PROJECT_DIR"

# Open the browser only once this server is answering, and only on the
# port it is answering on.  A fixed sleep would send the browser to
# whatever else holds the port when our start fails.
("$VENV_PY" "$LAUNCHER" open "$PORT" 60 >/dev/null 2>&1 || true) &

# Exit code 42 means "restart" — the Settings page uses it when the data
# directory changes, because DATA_DIR is resolved at import time.
while true; do
    set +e
    "$VENV_PY" -m uvicorn dlc_labeler.app:app \
        --host 127.0.0.1 --port "$PORT" --timeout-graceful-shutdown 3
    exit_code=$?
    set -e
    [ "$exit_code" -eq 42 ] || break
    echo ""
    echo "Restarting..."
    echo ""
done
