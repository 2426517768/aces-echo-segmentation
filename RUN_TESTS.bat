@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Mode tests
set ACES_EXIT=%ERRORLEVEL%
pause
exit /b %ACES_EXIT%
