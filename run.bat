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

:: 5. No Python found — try to auto-install
echo.
echo Python not found. Attempting automatic install...
echo.

set "INSTALL_OK=0"

:: 5a. Portable Python — download zip, extract locally (no install, no admin)
echo Setting up portable Python...
set "PORTABLE_DIR=%~dp0.python"
set "PY_ZIP=%TEMP%\python-3.11-embed.zip"
set "GET_PIP=%TEMP%\get-pip.py"

if not exist "!PORTABLE_DIR!\python.exe" (
    echo Downloading portable Python 3.11...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '!PY_ZIP!' }" 2>nul
    if not exist "!PY_ZIP!" (
        echo.
        echo Could not download Python. Check your internet connection.
        echo.
        pause
        exit /b 1
    )

    echo Extracting...
    mkdir "!PORTABLE_DIR!" 2>nul
    powershell -Command "Expand-Archive -Path '!PY_ZIP!' -DestinationPath '!PORTABLE_DIR!' -Force" 2>nul
    del "!PY_ZIP!" 2>nul

    :: Enable pip in embeddable Python (uncomment 'import site' in python311._pth)
    powershell -Command "(Get-Content '!PORTABLE_DIR!\python311._pth') -replace '^#import site','import site' | Set-Content '!PORTABLE_DIR!\python311._pth'" 2>nul

    :: Install pip
    echo Installing pip...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '!GET_PIP!' }" 2>nul
    "!PORTABLE_DIR!\python.exe" "!GET_PIP!" --no-warn-script-location 2>nul
    del "!GET_PIP!" 2>nul
)

if exist "!PORTABLE_DIR!\python.exe" (
    set "PYTHON=!PORTABLE_DIR!\python.exe"
    goto :found
)

echo.
echo All automatic install methods failed.
echo Please ask IT to install Python 3.11 or Anaconda, then re-run this script.
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
