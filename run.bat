@echo off
setlocal

cd /d "%~dp0"

:: ── Find Python ────────────────────────────────────────────────
:: Priority: .venv in repo > mano conda env > any conda env with uvicorn > current python
set "PYTHON="
set "ACTIVATE="

:: 1. Check for local .venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    set "ACTIVATE=.venv\Scripts\activate.bat"
    goto :found
)

:: 2. Check for 'mano' conda env
if defined CONDA_EXE (
    for /f "delims=" %%P in ('where conda 2^>nul') do set "CONDA_CMD=%%P"
)
if defined CONDA_PREFIX (
    :: Already in a conda env — check if it has uvicorn
    "%CONDA_PREFIX%\python.exe" -c "import uvicorn" 2>nul && (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        goto :found
    )
)
:: Try to find and activate 'mano' env
where conda >nul 2>nul && (
    for /f "delims=" %%D in ('conda info --envs 2^>nul ^| findstr /R "^mano "') do (
        for /f "tokens=2*" %%A in ("%%D") do (
            if exist "%%A\python.exe" (
                call conda activate mano 2>nul
                set "PYTHON=%%A\python.exe"
                goto :found
            )
        )
    )
)

:: 3. Fall back to system python
where python >nul 2>nul && (
    set "PYTHON=python"
    goto :found
)

echo ERROR: Python not found.
echo.
echo Install Python 3.9-3.11 from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found

:: ── Check dependencies ─────────────────────────────────────────
echo Checking dependencies...
%PYTHON% -c "import uvicorn, fastapi, cv2, numpy, pandas, mediapipe" 2>nul
if errorlevel 1 (
    echo Installing missing dependencies...
    %PYTHON% -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies. Try manually:
        echo   pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: ── Launch ──────────────────────────────────────────────────────
echo.
echo Starting Movement Tracker...
echo Dashboard will open at http://localhost:8080
echo.
%PYTHON% -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port 8080

pause
