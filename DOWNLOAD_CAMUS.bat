@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Mode smoke
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -X utf8 -u download_public.py
set ACES_EXIT=%ERRORLEVEL%
pause
exit /b %ACES_EXIT%
:failed
echo Setup failed.
pause
exit /b 1
