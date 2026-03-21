@echo off
echo Starting Movement Tracker...
echo.
echo Dashboard will open at http://localhost:8080
echo.

cd /d "%~dp0"
python -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port 8080

pause
