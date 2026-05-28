#!/usr/bin/env bash
# Update the icon for Movement Tracker.app from a single source image.
#
# Usage:
#   scripts/update_app_icon.sh path/to/new_icon.png
#
# The source image should be square and at least 1024x1024 for a clean
# result.  Any format `sips` can read (png, jpg, tiff…) is fine.
#
# What it does:
#   1. Render the source into the 10 standard macOS icon sizes.
#   2. Pack them into icon.icns via iconutil.
#   3. Drop the new icns into Movement Tracker.app/Contents/Resources/.
#   4. Touch the bundle so Finder / Dock refresh the cached icon.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <source-image>" >&2
  exit 2
fi

SRC="$1"
if [ ! -f "$SRC" ]; then
  echo "Source image not found: $SRC" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$REPO_ROOT/Movement Tracker.app"
DEST="$APP/Contents/Resources/icon.icns"

if [ ! -d "$APP" ]; then
  echo "App bundle not found: $APP" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ICONSET="$TMP/icon.iconset"
mkdir -p "$ICONSET"

# Standard macOS icon ladder: (size, filename).
sizes=(
  "16    icon_16x16.png"
  "32    icon_16x16@2x.png"
  "32    icon_32x32.png"
  "64    icon_32x32@2x.png"
  "128   icon_128x128.png"
  "256   icon_128x128@2x.png"
  "256   icon_256x256.png"
  "512   icon_256x256@2x.png"
  "512   icon_512x512.png"
  "1024  icon_512x512@2x.png"
)

for entry in "${sizes[@]}"; do
  px="${entry%% *}"
  name="${entry##* }"
  sips -z "$px" "$px" "$SRC" --out "$ICONSET/$name" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$TMP/icon.icns"
cp "$TMP/icon.icns" "$DEST"
touch "$APP"
touch "$APP/Contents/Info.plist"

# Also refresh icon.ico at the repo root — run.bat reads it on
# first launch to set the IconLocation on the generated
# "Movement Tracker.lnk" shortcut for Windows users.
ICO="$REPO_ROOT/icon.ico"
python3 - "$SRC" "$ICO" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGBA")
# Standard Windows shortcut icon sizes — Pillow writes all of them
# into a single multi-resolution .ico.
sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]
img.save(dst, format="ICO", sizes=sizes)
PY

# ── Force macOS to actually reload the icon ───────────────────
# macOS aggressively caches app icons in the IconServices store.
# Just touching the bundle + restarting Dock/Finder usually isn't
# enough — re-register the bundle with LaunchServices, blow away
# the per-user IconServices cache, and restart its daemons.
LSREG="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
if [ -x "$LSREG" ]; then
  "$LSREG" -v -f "$APP" >/dev/null 2>&1 || true
fi
rm -rf "$HOME/Library/Caches/com.apple.iconservices.store" 2>/dev/null || true
if command -v killall >/dev/null 2>&1; then
  killall iconservicesd          >/dev/null 2>&1 || true
  killall iconservicesagent      >/dev/null 2>&1 || true
  killall com.apple.iconservices >/dev/null 2>&1 || true
  killall Dock                   >/dev/null 2>&1 || true
  killall Finder                 >/dev/null 2>&1 || true
fi

echo "Updated $DEST and $ICO from $SRC"
echo
echo "If the Dock/Finder icon still looks stale, drag the .app to the"
echo "Trash (don't empty), drag it back, and relaunch Finder.  Some"
echo "icon caches are only refreshed on application identity change."
