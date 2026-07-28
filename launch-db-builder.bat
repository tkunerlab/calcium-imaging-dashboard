@echo off
setlocal EnableExtensions
title Calcium Imaging Database Builder

set "BUILDER_EXE=%LOCALAPPDATA%\CalciumImagingDashboard\venv\Scripts\db-builder.exe"

if not exist "%BUILDER_EXE%" (
    echo Calcium Imaging Database Builder is not installed for this Windows account.
    echo.
    echo Double-click install-windows.bat first, then try this launcher again.
    echo Expected application:
    echo   %BUILDER_EXE%
    echo.
    echo Press any key to close this window...
    pause >nul
    endlocal
    exit /b 1
)

echo Launching Calcium Imaging Database Builder...
"%BUILDER_EXE%" %*
set "LAUNCH_EXIT=%ERRORLEVEL%"

if not "%LAUNCH_EXIT%"=="0" (
    echo.
    echo The database builder stopped with exit code %LAUNCH_EXIT%.
    echo Review the message above, then press any key to close this window.
    pause >nul
)

endlocal
exit /b %LAUNCH_EXIT%
