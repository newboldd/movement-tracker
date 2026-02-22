@echo off
echo Starting DLC Labeler...
echo.
echo Dashboard will open at http://localhost:8080
echo.

cd /d "%~dp0\.."
python -m uvicorn dlc_app.app:app --host 127.0.0.1 --port 8080

pause
