#!/bin/bash
# ─────────────────────────────────────────────────────────────
# DLC Labeler — double-click to launch.
#
# Drag this file to the Dock or Desktop for quick access.
# First launch on a downloaded copy: macOS Gatekeeper will refuse it
# until you allow it once under System Settings → Privacy & Security.
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
exec ./setup.sh
