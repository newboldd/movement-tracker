@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: ── Find Python ────────────────────────────────────────────────
:: Priority: .venv > active conda env > mano conda env > Anaconda base > system python
set "PYTHON="

:: 1. Check for local .venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :found
)

:: 2. Already in a conda env with uvicorn?
if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" -c "import uvicorn" 2>nul && (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        goto :found
    )
)

:: 3. Try to find conda and activate 'mano' env
:: Check common conda locations when launched from File Explorer (no conda on PATH)
set "CONDA_BAT="
where conda >nul 2>nul && (
    for /f "delims=" %%C in ('where conda') do set "CONDA_BAT=%%~dpCactivate.bat"
)
if not defined CONDA_BAT (
    for %%D in (
        "%USERPROFILE%\anaconda3\Scripts\activate.bat"
        "%USERPROFILE%\miniconda3\Scripts\activate.bat"
        "C:\ProgramData\anaconda3\Scripts\activate.bat"
        "C:\ProgramData\miniconda3\Scripts\activate.bat"
        "%USERPROFILE%\Anaconda3\Scripts\activate.bat"
        "%USERPROFILE%\Miniconda3\Scripts\activate.bat"
    ) do (
        if exist %%D set "CONDA_BAT=%%~D"
    )
)

if defined CONDA_BAT (
    :: Try activating 'mano' env
    call "%CONDA_BAT%" mano 2>nul
    if defined CONDA_PREFIX (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        if exist "!PYTHON!" goto :found
    )
    :: Fall back to conda base
    call "%CONDA_BAT%" 2>nul
    if defined CONDA_PREFIX (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        if exist "!PYTHON!" goto :found
    )
)

:: 4. Fall back to system python (verify it's real, not the Windows Store alias)
where python >nul 2>nul && (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        echo %%P | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            set "PYTHON=python"
            goto :found
        )
    )
)

:: 5. No Python found — try to auto-install via winget
echo.
echo Python not found. Attempting automatic install...
echo.
where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found and winget is not available.
    echo.
    echo Please install Python manually:
    echo   1. Download from https://www.python.org/downloads/
    echo   2. Check "Add Python to PATH" during installation
    echo   3. Re-run this script
    echo.
    pause
    exit /b 1
)

echo Installing Python 3.11 via winget (this may take a minute^)...
winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo Automatic install failed. Please install Python manually:
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Refresh PATH so we can find the newly installed python
echo Refreshing PATH...
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

where python >nul 2>nul && (
    set "PYTHON=python"
    goto :found
)

echo.
echo Python was installed but could not be found on PATH.
echo Please close this window, open a new terminal, and run this script again.
echo.
pause
exit /b 1

:found
echo Using Python: %PYTHON%

:: ── Check dependencies ─────────────────────────────────────────
echo Checking dependencies...
%PYTHON% -c "import uvicorn, fastapi, cv2, numpy, pandas, mediapipe" 2>nul
if errorlevel 1 (
    echo Installing missing dependencies (this may take a few minutes^)...
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies. Try manually:
        echo   %PYTHON% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: ── Sample data ──────────────────────────────────────────────────
if not exist "sample_data\Con01_R1.mp4" (
    echo Downloading sample video...
    %PYTHON% scripts\download_sample.py
)

:: ── Ensure default directories exist ────────────────────────────
if not exist "dlc" mkdir dlc

:: ── Launch ──────────────────────────────────────────────────────
echo.
echo Starting Movement Tracker...
echo Dashboard will open at http://localhost:8080
echo.
%PYTHON% -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port 8080

pause
