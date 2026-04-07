@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: ── Upgrade: migrate data from a previous installation ─────────
if "%~1"=="upgrade" (
    if "%~2"=="" (
        echo Usage: run.bat upgrade "C:\path\to\old\movement-tracker-master"
        echo.
        echo This copies your database, videos, DLC data, and settings from
        echo a previous installation into this one.
        pause
        exit /b 1
    )
    set "OLD=%~2"

    echo.
    echo Migrating data from: !OLD!
    echo                  to: %~dp0
    echo.

    :: Database
    if exist "!OLD!\movement_tracker\dlc_app.db" (
        echo Copying database...
        copy /y "!OLD!\movement_tracker\dlc_app.db" "movement_tracker\dlc_app.db" >nul
        echo   OK: dlc_app.db
    )

    :: Settings
    if exist "!OLD!\movement_tracker\settings.json" (
        echo Copying settings...
        copy /y "!OLD!\movement_tracker\settings.json" "movement_tracker\settings.json" >nul
        echo   OK: settings.json
    )

    :: Videos
    if exist "!OLD!\videos" (
        echo Copying videos...
        xcopy /E /I /Y /Q "!OLD!\videos" "videos" >nul
        echo   OK: videos\
    )

    :: DLC data
    if exist "!OLD!\dlc" (
        echo Copying DLC data...
        xcopy /E /I /Y /Q "!OLD!\dlc" "dlc" >nul
        echo   OK: dlc\
    )

    :: Portable Python (reuse if present, saves re-download)
    if exist "!OLD!\.python" (
        echo Copying portable Python...
        xcopy /E /I /Y /Q "!OLD!\.python" ".python" >nul
        echo   OK: .python\
    )
    :: Also check AppData location from previous installs
    if exist "%LOCALAPPDATA%\MovementTracker\python\python.exe" (
        echo Found existing portable Python in AppData.
    )

    echo.
    echo Migration complete! Starting Movement Tracker...
    echo.
)

:: ── Find Python ────────────────────────────────────────────────
:: Priority: .venv > active conda env > mano conda env > Anaconda base > system python
set "PYTHON="

:: 1. Check for local .venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :found
)

:: 1b. Check for portable Python in AppData (from previous run on locked-down machine)
if exist "%LOCALAPPDATA%\MovementTracker\python\python.exe" (
    "%LOCALAPPDATA%\MovementTracker\python\python.exe" -c "print('ok')" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=%LOCALAPPDATA%\MovementTracker\python\python.exe"
        goto :found
    )
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

set "PY_ZIP=%TEMP%\python-3.11-embed.zip"

:: Try two locations for portable Python:
::   1. %LOCALAPPDATA%\MovementTracker  (AppLocker usually allows AppData\Local)
::   2. .python\ next to this script     (fallback, may be blocked in Downloads)
set "PORTABLE_DIR_APPDATA=%LOCALAPPDATA%\MovementTracker\python"
set "PORTABLE_DIR_LOCAL=%~dp0.python"

:: Check if we already have a working portable Python in either location
if exist "!PORTABLE_DIR_APPDATA!\python.exe" (
    "!PORTABLE_DIR_APPDATA!\python.exe" -c "print('ok')" >nul 2>nul
    if not errorlevel 1 (
        set "PORTABLE_DIR=!PORTABLE_DIR_APPDATA!"
        goto :portable_ready
    )
    echo Note: Python in AppData is blocked by Group Policy, trying next...
)
if exist "!PORTABLE_DIR_LOCAL!\python.exe" (
    "!PORTABLE_DIR_LOCAL!\python.exe" -c "print('ok')" >nul 2>nul
    if not errorlevel 1 (
        set "PORTABLE_DIR=!PORTABLE_DIR_LOCAL!"
        goto :portable_ready
    )
    echo Note: Python in local folder is blocked by Group Policy...
)

:: Download and extract portable Python
echo Setting up portable Python...
echo Downloading portable Python 3.11...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '!PY_ZIP!' }" 2>nul
if not exist "!PY_ZIP!" (
    echo.
    echo Could not download Python. Check your internet connection.
    echo.
    pause
    exit /b 1
)

:: Try extracting to AppData\Local first (less likely to be blocked by Group Policy)
set "PORTABLE_DIR=!PORTABLE_DIR_APPDATA!"
echo Extracting to %LOCALAPPDATA%\MovementTracker\...
mkdir "!PORTABLE_DIR!" 2>nul
powershell -Command "Expand-Archive -Path '!PY_ZIP!' -DestinationPath '!PORTABLE_DIR!' -Force" 2>nul

:: Enable pip in embeddable Python (uncomment 'import site' in python311._pth)
powershell -Command "(Get-Content '!PORTABLE_DIR!\python311._pth') -replace '^#import site','import site' | Set-Content '!PORTABLE_DIR!\python311._pth'" 2>nul

:: Verify it actually runs (Group Policy check)
"!PORTABLE_DIR!\python.exe" -c "print('ok')" >nul 2>nul
if errorlevel 1 (
    echo Python in AppData blocked by Group Policy, trying local folder...
    rmdir /s /q "!PORTABLE_DIR!" 2>nul

    :: Fall back to .python\ next to the script
    set "PORTABLE_DIR=!PORTABLE_DIR_LOCAL!"
    echo Extracting to !PORTABLE_DIR!...
    mkdir "!PORTABLE_DIR!" 2>nul
    powershell -Command "Expand-Archive -Path '!PY_ZIP!' -DestinationPath '!PORTABLE_DIR!' -Force" 2>nul
    powershell -Command "(Get-Content '!PORTABLE_DIR!\python311._pth') -replace '^#import site','import site' | Set-Content '!PORTABLE_DIR!\python311._pth'" 2>nul

    :: Verify again
    "!PORTABLE_DIR!\python.exe" -c "print('ok')" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ============================================================
        echo ERROR: Python is blocked by Group Policy in all locations.
        echo ============================================================
        echo.
        echo Your IT department blocks .exe files from running in:
        echo   - %LOCALAPPDATA%\MovementTracker\
        echo   - %~dp0
        echo.
        echo Please ask IT to do ONE of the following:
        echo   1. Install Python 3.11 system-wide (recommended^)
        echo   2. Whitelist this folder: %LOCALAPPDATA%\MovementTracker\
        echo   3. Install Anaconda for your user account
        echo.
        del "!PY_ZIP!" 2>nul
        pause
        exit /b 1
    )
)
del "!PY_ZIP!" 2>nul

:portable_ready
echo Using portable Python at: !PORTABLE_DIR!

:: Download pip as a standalone zip app (no .exe files — avoids Group Policy blocks)
set "PIP_PYZ=!PORTABLE_DIR!\pip.pyz"
if not exist "!PIP_PYZ!" (
    echo Downloading pip...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/pip/pip.pyz' -OutFile '!PIP_PYZ!' }" 2>nul
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

:: ── Determine pip command ───────────────────────────────────────
:: Prefer pip.pyz (no .exe, avoids Group Policy blocks on locked-down machines)
set "PIP_CMD="
:: Check both possible pip.pyz locations
set "PIP_PYZ="
if exist "%LOCALAPPDATA%\MovementTracker\python\pip.pyz" set "PIP_PYZ=%LOCALAPPDATA%\MovementTracker\python\pip.pyz"
if exist "%~dp0.python\pip.pyz" set "PIP_PYZ=%~dp0.python\pip.pyz"
if defined PIP_PYZ (
    set "PIP_CMD=%PYTHON% "%PIP_PYZ%""
) else (
    set "PIP_CMD=%PYTHON% -m pip"
)

:: ── Check dependencies ─────────────────────────────────────────
echo Checking dependencies...
%PYTHON% -c "import uvicorn, fastapi, cv2, numpy, pandas, mediapipe" 2>nul
if errorlevel 1 (
    echo Installing missing dependencies (this may take a few minutes^)...

    :: Strategy 1: offline wheels directory (for air-gapped / locked-down machines)
    if exist "%~dp0wheels" (
        echo Found local wheels directory, installing offline...
        %PIP_CMD% install --no-index --find-links "%~dp0wheels" -r requirements.txt --no-build-isolation
        if not errorlevel 1 goto :deps_ok
        echo Offline install failed, trying online...
    )

    :: Strategy 2: pip.pyz with --only-binary (no .exe created, no compiling)
    if defined PIP_PYZ (
        %PYTHON% "%PIP_PYZ%" install --only-binary :all: --no-cache-dir -r requirements.txt
        if not errorlevel 1 goto :deps_ok
        echo pip.pyz binary-only install failed, trying with source builds...
        %PYTHON% "%PIP_PYZ%" install --no-cache-dir -r requirements.txt
        if not errorlevel 1 goto :deps_ok
    )

    :: Strategy 3: standard pip module (works when pip.exe isn't blocked)
    %PYTHON% -m pip install --only-binary :all: -r requirements.txt
    if not errorlevel 1 goto :deps_ok
    %PYTHON% -m pip install -r requirements.txt
    if not errorlevel 1 goto :deps_ok

    echo.
    echo ============================================================
    echo ERROR: Failed to install dependencies.
    echo ============================================================
    echo.
    echo This is often caused by hospital/enterprise Group Policy
    echo blocking programs in the Downloads folder.
    echo.
    echo Try these fixes (easiest first^):
    echo.
    echo  1. MOVE this folder to C:\MovementTracker and re-run
    echo     (paths outside Downloads are less likely to be blocked^)
    echo.
    echo  2. Ask IT to whitelist this folder:
    echo     %~dp0
    echo.
    echo  3. Ask IT to install Python 3.11 system-wide, then re-run
    echo.
    echo  4. On another (unrestricted^) PC, run:
    echo       pip download -r requirements.txt -d wheels\
    echo     Copy the "wheels" folder into this directory, then re-run.
    echo     (This enables fully offline installation.^)
    echo.
    pause
    exit /b 1
)
:deps_ok

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

:: Open browser after a short delay (background, non-blocking)
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8080"

:: ── Launch with restart loop (exit code 42 = restart after update) ──
:launch
%PYTHON% -m uvicorn movement_tracker.app:app --host 127.0.0.1 --port 8080
if %errorlevel%==42 (
    echo.
    echo Restarting after update...
    echo.
    goto :launch
)

pause
