@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Mode smoke
set ACES_EXIT=%ERRORLEVEL%
echo.
echo Exit code: %ACES_EXIT%
pause
exit /b %ACES_EXIT%
