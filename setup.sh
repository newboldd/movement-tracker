#!/usr/bin/env bash
# One-command setup and launch for Movement Tracker.
#
# Usage:
#   ./setup.sh
#
# What it does:
#   1. Checks for Python 3.9+ — installs it if missing (where possible)
#   2. Checks for ffmpeg — installs it if missing (where possible)
#   3. Creates a virtual environment (first run only)
#   4. Installs Python dependencies (first run only)
#   5. Downloads sample video if not already present
#   6. Starts the web app and opens your browser

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
PORT=8080
OS="$(uname -s)"   # Darwin | Linux

# ── Helpers ───────────────────────────────────────────────────────────────

print_header() { echo ""; echo "── $1 ──────────────────────────────────"; }

# Attempt a brew / apt / dnf install; return 1 if nothing worked
try_install() {
    local pkg_brew="$1" pkg_apt="$2" pkg_dnf="$3"
    if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
        brew install "$pkg_brew"
        return 0
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y "$pkg_apt"
        return 0
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y "$pkg_dnf"
        return 0
    elif command -v yum &>/dev/null; then
        sudo yum install -y "$pkg_dnf"
        return 0
    fi
    return 1
}

# ── Python ────────────────────────────────────────────────────────────────

find_python() {
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &>/dev/null; then
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

install_python() {
    print_header "Installing Python"
    echo "Python 3.9+ is required but was not found."
    echo ""

    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            echo "Installing Python 3.11 via Homebrew..."
            brew install python@3.11
        else
            echo "Homebrew is not installed."
            echo ""
            echo "The easiest options:"
            echo "  1. Install Homebrew (https://brew.sh), then re-run this script."
            echo "  2. Download Python directly from https://www.python.org/downloads/"
            echo "     Install it, then re-run this script."
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            echo "Installing Python 3.11 via apt..."
            sudo apt-get update -q
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
        elif command -v dnf &>/dev/null; then
            echo "Installing Python 3.11 via dnf..."
            sudo dnf install -y python3.11
        elif command -v yum &>/dev/null; then
            echo "Installing Python 3 via yum..."
            sudo yum install -y python3
        else
            echo "Could not detect a package manager."
            echo ""
            echo "Please install Python 3.9+ manually:"
            echo "  → https://www.python.org/downloads/"
            exit 1
        fi
    else
        echo "Automatic install is not supported on this OS."
        echo ""
        echo "Please install Python 3.9+ from:"
        echo "  → https://www.python.org/downloads/"
        exit 1
    fi
}

# Find Python, install if missing, then find again
PYTHON=$(find_python) || {
    install_python
    PYTHON=$(find_python) || {
        echo ""
        echo "Python 3.9+ still not found after install attempt."
        echo "Please install it manually from https://www.python.org/downloads/ and re-run."
        exit 1
    }
}

echo "Using Python: $PYTHON ($($PYTHON --version))"

# ── ffmpeg ────────────────────────────────────────────────────────────────

if ! command -v ffmpeg &>/dev/null; then
    print_header "Installing ffmpeg"
    echo "ffmpeg is required for video trimming and face de-identification."
    echo ""
    if ! try_install ffmpeg ffmpeg ffmpeg; then
        echo "Could not install ffmpeg automatically."
        echo ""
        if [ "$OS" = "Darwin" ]; then
            echo "Install it with:"
            echo "  brew install ffmpeg"
            echo ""
            echo "Homebrew: https://brew.sh"
        else
            echo "Install it with your package manager, e.g.:"
            echo "  sudo apt install ffmpeg     (Debian/Ubuntu)"
            echo "  sudo dnf install ffmpeg     (Fedora)"
        fi
        echo ""
        echo "Or download from: https://ffmpeg.org/download.html"
        echo ""
        echo "Re-run this script after installing ffmpeg."
        exit 1
    fi
    echo "ffmpeg installed."
fi

# ── Virtual environment ───────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    print_header "Creating virtual environment"
    "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Python dependencies ───────────────────────────────────────────────────

if [ ! -f "$VENV_DIR/.installed" ] || [ "$REQUIREMENTS" -nt "$VENV_DIR/.installed" ]; then
    print_header "Installing Python dependencies"
    pip install --upgrade pip -q
    pip install -r "$REQUIREMENTS" -q
    touch "$VENV_DIR/.installed"
    echo "Dependencies installed."
fi

# ── Sample data ───────────────────────────────────────────────────────────

if [ ! -f "$PROJECT_DIR/sample_data/Con01_R1.mp4" ]; then
    print_header "Downloading sample data"
    "$PYTHON" "$PROJECT_DIR/scripts/download_sample.py"
fi

# ── Kill any existing server on the port ─────────────────────────────────

existing=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$existing" ]; then
    echo "Stopping existing server on port $PORT..."
    echo "$existing" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── Launch ────────────────────────────────────────────────────────────────

echo ""
echo "Starting Movement Tracker at http://localhost:$PORT"
echo "Press Ctrl+C to stop."
echo ""

cd "$PROJECT_DIR"

# Open browser after a short delay (background, non-fatal)
(sleep 2 && open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null) &

python -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port "$PORT" --reload
