@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3.12 -c "import sys" >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py -3.12"
    ) else (
        py -3.11 -c "import sys" >nul 2>nul
        if %errorlevel% equ 0 set "PYTHON_CMD=py -3.11"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python 3.11 or 3.12 was not found.
    echo Install Python from https://www.python.org/downloads/ and run this file again.
    exit /b 1
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo This software requires Python 3.11 or 3.12.
    exit /b 1
)

%PYTHON_CMD% -m venv --clear .venv
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install .
if errorlevel 1 exit /b 1

echo.
echo Installation complete.
echo Run ".venv\Scripts\cell-registration-dashboard.exe" to launch the dashboard.
echo Run ".venv\Scripts\db-builder.exe" to launch the database builder.
endlocal
