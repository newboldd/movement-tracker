#!/usr/bin/env python3
"""Port checks and browser opening for the launch scripts.

Both launchers used to decide these two things with shell tools — lsof or
netstat for "is this port free", and a fixed sleep before opening the
browser.  Both are unreliable in the way that matters most here: a
machine that also runs Movement Tracker has something on port 8080, and
if the check misses it, the server fails to bind and the browser still
opens that port — so Movement Tracker's UI appears and looks like this
app started the wrong thing.

Asking the kernel directly works the same on every machine, needs no
external tool, and cannot be fooled by a listener owned by another user.

Usage:
    launcher.py free <port>            exit 0 if the port can be bound
    launcher.py open <port> [timeout]  wait for the port to answer, then
                                       open it in a browser
"""
from __future__ import annotations

import socket
import sys
import time


def is_free(port: int) -> bool:
    """True when a server could bind this port on the loopback address."""
    sock = socket.socket()
    # Match what uvicorn does, so this answers the same question it will.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def is_answering(port: int, timeout: float = 0.5) -> bool:
    """True when something accepts a connection on this port."""
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def open_when_ready(port: int, timeout: float) -> int:
    """Open the app once it answers. Never opens a port we aren't serving."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_answering(port):
            import webbrowser
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception:
                # A headless or locked-down machine has no browser to open.
                # The server is up and the URL is on screen; that is enough.
                pass
            return 0
        time.sleep(0.3)
    # The server never came up.  It reports its own failure, so say nothing
    # here — and above all, do not open some other app's port.
    return 1


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    action, raw_port = argv[1], argv[2]
    try:
        port = int(raw_port)
    except ValueError:
        print(f"Not a port number: {raw_port}", file=sys.stderr)
        return 2

    if action == "free":
        return 0 if is_free(port) else 1
    if action == "open":
        timeout = float(argv[3]) if len(argv) > 3 else 60.0
        return open_when_ready(port, timeout)

    print(f"Unknown action: {action}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(1)
