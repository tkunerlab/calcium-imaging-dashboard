@echo off
setlocal EnableExtensions
title Calcium Imaging Dashboard Installer
cd /d "%~dp0"

set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

for %%I in ("%~dp0.") do set "SOURCE_DIR=%%~fI"
set "APP_DIR=%LOCALAPPDATA%\CalciumImagingDashboard"
set "VENV_DIR=%APP_DIR%\venv"
set "PYTHON_EXE="

echo ============================================================
echo  Calcium Imaging Dashboard Installer
echo ============================================================
echo.
echo The application will be installed for your Windows account.
echo Source: %SOURCE_DIR%
echo Install location: %VENV_DIR%
echo.

if not defined LOCALAPPDATA (
    echo ERROR: Windows did not provide a Local AppData directory.
    echo Please contact your system administrator.
    call :finish_failure
    exit /b 1
)

echo [1/4] Detecting Python 3.11 or 3.12...
for %%V in (3.12 3.11) do (
    if not defined PYTHON_EXE (
        for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    echo ERROR: Python 3.11 or 3.12 was not found.
    echo Install Python from https://www.python.org/downloads/ and run this installer again.
    call :finish_failure
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Found Python at:
    echo   %PYTHON_EXE%
    echo but this software requires Python 3.11 or 3.12.
    echo Install a supported version from https://www.python.org/downloads/.
    call :finish_failure
    exit /b 1
)
echo       Using %PYTHON_EXE%
echo.

echo [2/4] Preparing the local installation directory...
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create:
    echo   %APP_DIR%
    echo Check that your Windows account can write to Local AppData.
    call :finish_failure
    exit /b 1
)
echo       Ready: %APP_DIR%
echo.

echo [3/4] Creating the isolated Python environment...
echo       This may take a minute. Existing application files will be refreshed.
"%PYTHON_EXE%" -m venv --clear "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create the Python environment.
    echo Close any running dashboard or database-builder windows, then try again.
    call :finish_failure
    exit /b 1
)
echo       Environment created.
echo.

echo [4/4] Installing Calcium Imaging Dashboard and dependencies...
echo       Dependency installation can take several minutes on the first run.
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check "%SOURCE_DIR%"
if errorlevel 1 (
    echo ERROR: Package installation failed.
    echo Review the messages above. Check your internet connection and try again.
    call :finish_failure
    exit /b 1
)

echo.
echo ============================================================
echo  Installation complete
echo ============================================================
echo.
echo Double-click either launcher in this repository:
echo   launch-dashboard.bat
echo   launch-db-builder.bat
echo.
call :finish_success
exit /b 0

:finish_failure
echo.
echo Installation did not complete.
echo No databases or analysis files were changed.
if "%NO_PAUSE%"=="0" (
    echo.
    echo Press any key to close this window...
    pause >nul
)
exit /b 0

:finish_success
if "%NO_PAUSE%"=="0" (
    echo Press any key to close this window...
    pause >nul
)
exit /b 0
