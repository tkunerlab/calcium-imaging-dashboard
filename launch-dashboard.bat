@echo off
setlocal EnableExtensions
title Calcium Imaging Dashboard

set "DASHBOARD_EXE=%LOCALAPPDATA%\CalciumImagingDashboard\venv\Scripts\cell-registration-dashboard.exe"

if not exist "%DASHBOARD_EXE%" (
    echo Calcium Imaging Dashboard is not installed for this Windows account.
    echo.
    echo Double-click install-windows.bat first, then try this launcher again.
    echo Expected application:
    echo   %DASHBOARD_EXE%
    echo.
    echo Press any key to close this window...
    pause >nul
    endlocal
    exit /b 1
)

echo Launching Calcium Imaging Dashboard...
"%DASHBOARD_EXE%" %*
set "LAUNCH_EXIT=%ERRORLEVEL%"

if not "%LAUNCH_EXIT%"=="0" (
    echo.
    echo The dashboard stopped with exit code %LAUNCH_EXIT%.
    echo Review the message above, then press any key to close this window.
    pause >nul
)

endlocal
exit /b %LAUNCH_EXIT%
