@echo off
setlocal enabledelayedexpansion

:: DLC Labeler — one-click setup and launch for Windows.
::
::   run.bat               install what labeling needs, then start
::   run.bat --with-dlc    also install DeepLabCut, for training
::
:: Written for locked-down machines.  On a managed laptop the usual
:: failure is not a missing Python but Group Policy refusing to run one:
:: so this tries, in order, an existing venv, a portable Python under
:: AppData\Local, then a portable Python next to this script, and uses
:: pip.pyz (a zipapp, no .exe) rather than pip.exe wherever it can.

cd /d "%~dp0"

set "WITH_DLC="
if /i "%~1"=="--with-dlc" set "WITH_DLC=1"

:: ── First run: make a shortcut with the app icon ───────────────
if exist "icon.ico" if not exist "DLC Labeler.lnk" (
    powershell -NoProfile -Command ^
        "$ws = New-Object -ComObject WScript.Shell;" ^
        "$sc = $ws.CreateShortcut((Join-Path (Get-Location) 'DLC Labeler.lnk'));" ^
        "$sc.TargetPath = (Join-Path (Get-Location) 'run.bat');" ^
        "$sc.WorkingDirectory = (Get-Location).Path;" ^
        "$sc.IconLocation = (Join-Path (Get-Location) 'icon.ico');" ^
        "$sc.Description = 'DLC Labeler';" ^
        "$sc.Save()" >nul 2>&1
)

:: ── Data directory ─────────────────────────────────────────────
:: .env next to this script may set DLC_DATA_DIR (MT_DATA_DIR is read
:: too, so a machine already set up for Movement Tracker keeps its data).
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "_k=%%A"
        set "_k=!_k: =!"
        if /i "!_k!"=="DLC_DATA_DIR" set "DLC_DATA_DIR=%%B"
        if /i "!_k!"=="MT_DATA_DIR" if not defined DLC_DATA_DIR set "DLC_DATA_DIR=%%B"
        if /i "!_k!"=="DLC_PORT" set "DLC_PORT=%%B"
    )
)
if not defined DLC_DATA_DIR if defined MT_DATA_DIR set "DLC_DATA_DIR=%MT_DATA_DIR%"
if not defined DLC_DATA_DIR set "DLC_DATA_DIR=%~dp0data"
if not defined DLC_PORT set "DLC_PORT=8080"

:: ── Find Python ────────────────────────────────────────────────
:: Priority: local .venv > portable Python from a previous run >
:: active conda env > conda base > system python > fresh portable install.
set "PYTHON="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :found
)

if exist "%LOCALAPPDATA%\DLCLabeler\python\python.exe" (
    "%LOCALAPPDATA%\DLCLabeler\python\python.exe" -c "print('ok')" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=%LOCALAPPDATA%\DLCLabeler\python\python.exe"
        goto :found
    )
)

if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" -c "import uvicorn" 2>nul && (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        goto :found
    )
)

:: Conda is often installed but not on PATH when launched from Explorer.
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
    call "%CONDA_BAT%" dlc 2>nul
    if defined CONDA_PREFIX (
        set "PYTHON=!CONDA_PREFIX!\python.exe"
        if exist "!PYTHON!" goto :found
    )
    call "%CONDA_BAT%" 2>nul
    if defined CONDA_PREFIX (
        set "PYTHON=!CONDA_PREFIX!\python.exe"
        if exist "!PYTHON!" goto :found
    )
)

:: System python, but not the Windows Store stub, which exits silently.
where python >nul 2>nul && (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        echo %%P | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            set "PYTHON=python"
            goto :found
        )
    )
)

echo.
echo Python not found. Installing a private copy...
echo.

set "PY_ZIP=%TEMP%\python-3.11-embed.zip"
set "PORTABLE_DIR_APPDATA=%LOCALAPPDATA%\DLCLabeler\python"
set "PORTABLE_DIR_LOCAL=%~dp0.python"

if exist "!PORTABLE_DIR_LOCAL!\python.exe" (
    "!PORTABLE_DIR_LOCAL!\python.exe" -c "print('ok')" >nul 2>nul
    if not errorlevel 1 (
        set "PORTABLE_DIR=!PORTABLE_DIR_LOCAL!"
        goto :portable_ready
    )
)

echo Downloading portable Python 3.11...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '!PY_ZIP!' }" 2>nul
if not exist "!PY_ZIP!" (
    echo.
    echo Could not download Python. Check the internet connection.
    echo.
    pause
    exit /b 1
)

:: AppData\Local first — it is the location most policies still allow.
set "PORTABLE_DIR=!PORTABLE_DIR_APPDATA!"
echo Extracting to %LOCALAPPDATA%\DLCLabeler\...
mkdir "!PORTABLE_DIR!" 2>nul
powershell -Command "Expand-Archive -Path '!PY_ZIP!' -DestinationPath '!PORTABLE_DIR!' -Force" 2>nul
:: The embeddable build ships with site imports disabled, which also
:: disables pip; re-enable it.
powershell -Command "(Get-Content '!PORTABLE_DIR!\python311._pth') -replace '^#import site','import site' | Set-Content '!PORTABLE_DIR!\python311._pth'" 2>nul

"!PORTABLE_DIR!\python.exe" -c "print('ok')" >nul 2>nul
if errorlevel 1 (
    echo AppData is blocked by policy here. Trying this folder instead...
    rmdir /s /q "!PORTABLE_DIR!" 2>nul
    set "PORTABLE_DIR=!PORTABLE_DIR_LOCAL!"
    mkdir "!PORTABLE_DIR!" 2>nul
    powershell -Command "Expand-Archive -Path '!PY_ZIP!' -DestinationPath '!PORTABLE_DIR!' -Force" 2>nul
    powershell -Command "(Get-Content '!PORTABLE_DIR!\python311._pth') -replace '^#import site','import site' | Set-Content '!PORTABLE_DIR!\python311._pth'" 2>nul
    "!PORTABLE_DIR!\python.exe" -c "print('ok')" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ============================================================
        echo Python is blocked by Group Policy in every location tried.
        echo ============================================================
        echo.
        echo Programs cannot run from:
        echo   - %LOCALAPPDATA%\DLCLabeler\
        echo   - %~dp0
        echo.
        echo Ask IT for ONE of these:
        echo   1. Python 3.11 installed system-wide ^(simplest^)
        echo   2. %LOCALAPPDATA%\DLCLabeler\ added to the allow list
        echo   3. Anaconda installed for your user account
        echo.
        echo Or move this folder out of Downloads ^(e.g. to C:\DLCLabeler^)
        echo and run it again — that alone often fixes it.
        echo.
        del "!PY_ZIP!" 2>nul
        pause
        exit /b 1
    )
)
del "!PY_ZIP!" 2>nul

:portable_ready
echo Using portable Python at: !PORTABLE_DIR!

:: pip as a zipapp: no pip.exe is created, so nothing new has to pass policy.
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
echo Every automatic install path failed. Ask IT to install Python 3.11
echo or Anaconda, then run this again.
echo.
pause
exit /b 1

:found
echo Using Python: %PYTHON%

:: ── Pick a pip ─────────────────────────────────────────────────
set "PIP_PYZ="
if exist "%LOCALAPPDATA%\DLCLabeler\python\pip.pyz" set "PIP_PYZ=%LOCALAPPDATA%\DLCLabeler\python\pip.pyz"
if exist "%~dp0.python\pip.pyz" set "PIP_PYZ=%~dp0.python\pip.pyz"

:: ── Base dependencies ──────────────────────────────────────────
echo Checking dependencies...
%PYTHON% -c "import uvicorn, fastapi, cv2, numpy, yaml, mediapipe" 2>nul
if errorlevel 1 (
    echo Installing dependencies ^(a few minutes^)...

    :: A wheels\ folder lets a locked-down or offline machine install
    :: from media copied off an unrestricted one.
    if exist "%~dp0wheels" (
        echo Found a local wheels folder — installing offline...
        if defined PIP_PYZ (
            %PYTHON% "%PIP_PYZ%" install --no-index --find-links "%~dp0wheels" -r requirements.txt
        ) else (
            %PYTHON% -m pip install --no-index --find-links "%~dp0wheels" -r requirements.txt
        )
        if not errorlevel 1 goto :deps_ok
        echo Offline install failed; trying online...
    )

    if defined PIP_PYZ (
        %PYTHON% "%PIP_PYZ%" install --only-binary :all: --no-cache-dir -r requirements.txt
        if not errorlevel 1 goto :deps_ok
        %PYTHON% "%PIP_PYZ%" install --no-cache-dir -r requirements.txt
        if not errorlevel 1 goto :deps_ok
    )

    %PYTHON% -m pip install --only-binary :all: -r requirements.txt
    if not errorlevel 1 goto :deps_ok
    %PYTHON% -m pip install -r requirements.txt
    if not errorlevel 1 goto :deps_ok

    echo.
    echo ============================================================
    echo Could not install the dependencies.
    echo ============================================================
    echo.
    echo On managed laptops this is usually Group Policy blocking the
    echo Downloads folder. In order of how often it works:
    echo.
    echo  1. MOVE this folder to C:\DLCLabeler and run it again
    echo  2. Ask IT to allow: %~dp0
    echo  3. Ask IT to install Python 3.11 system-wide, then re-run
    echo  4. On an unrestricted PC run:
    echo       pip download -r requirements.txt -d wheels\
    echo     copy the wheels folder in here, and run again ^(fully offline^)
    echo.
    pause
    exit /b 1
)
:deps_ok

:: ── DeepLabCut (second tier) ───────────────────────────────────
if defined WITH_DLC (
    %PYTHON% -c "import deeplabcut" 2>nul
    if errorlevel 1 (
        echo.
        echo Installing DeepLabCut ^(several GB — the slow part, once^)...
        if defined PIP_PYZ (
            %PYTHON% "%PIP_PYZ%" install -r requirements-dlc.txt
        ) else (
            %PYTHON% -m pip install -r requirements-dlc.txt
        )
        if errorlevel 1 (
            echo.
            echo DeepLabCut did not install. Labeling still works; you can
            echo retry from the Jobs page.
            echo.
        )
    ) else (
        echo DeepLabCut is already installed.
    )
)

:: ── Data directory ─────────────────────────────────────────────
if not exist "%DLC_DATA_DIR%" mkdir "%DLC_DATA_DIR%" 2>nul
if not exist "%DLC_DATA_DIR%\dlc" mkdir "%DLC_DATA_DIR%\dlc" 2>nul
if not exist "%DLC_DATA_DIR%\videos" mkdir "%DLC_DATA_DIR%\videos" 2>nul
if not exist "%DLC_DATA_DIR%\calibration" mkdir "%DLC_DATA_DIR%\calibration" 2>nul
if not exist "%DLC_DATA_DIR%\packages" mkdir "%DLC_DATA_DIR%\packages" 2>nul
if exist "calibration" (
    xcopy /I /Y /Q /D "calibration\*" "%DLC_DATA_DIR%\calibration\" >nul 2>nul
)

:: ── Port ───────────────────────────────────────────────────────
:: Movement Tracker defaults to this same port.  Ask Python whether the
:: port can be bound rather than parsing netstat: netstat's output format
:: varies by Windows version and locale, and getting it wrong means
:: uvicorn fails to bind while the browser still opens that port — which
:: on a machine running Movement Tracker shows ITS interface.
set /a PORT_TRIES=0
:portcheck
%PYTHON% "%~dp0scripts\launcher.py" free %DLC_PORT% >nul 2>&1
if errorlevel 1 (
    set /a PORT_TRIES+=1
    if !PORT_TRIES! GEQ 20 (
        echo.
        echo Could not find a free port near %DLC_PORT%.
        echo Set DLC_PORT in .env to one you know is free.
        echo.
        pause
        exit /b 1
    )
    set /a DLC_PORT+=1
    goto :portcheck
)
if !PORT_TRIES! GTR 0 (
    echo.
    echo The default port was busy ^(Movement Tracker, perhaps^).
    echo Using port !DLC_PORT! instead. Set DLC_PORT in .env to pin this.
    echo.
)

:: ── Launch ─────────────────────────────────────────────────────
echo.
echo Starting DLC Labeler at http://localhost:%DLC_PORT%
echo Data directory: %DLC_DATA_DIR%
%PYTHON% -c "import deeplabcut" 2>nul
if errorlevel 1 (
    echo.
    echo DeepLabCut is not installed, so training and analysis are off.
    echo Add it with:  run.bat --with-dlc
)
echo.

:: Open the browser only once the server answers, and only on the port it
:: answers on — never after a fixed delay, which would open whatever else
:: holds the port if our start fails.
start "" /b %PYTHON% "%~dp0scripts\launcher.py" open %DLC_PORT% 60

:: Exit code 42 means "restart" — the Settings page uses it when the
:: data directory changes.
:launch
%PYTHON% -m uvicorn dlc_labeler.app:app --host 127.0.0.1 --port %DLC_PORT% --timeout-graceful-shutdown 3
if %errorlevel%==42 (
    echo.
    echo Restarting...
    echo.
    goto :launch
)

pause
