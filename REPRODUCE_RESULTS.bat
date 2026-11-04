@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv-analysis\Scripts\python.exe"
if not exist "%PY%" (
  where py >nul 2>&1
  if errorlevel 1 (
    python -m venv .venv-analysis
  ) else (
    py -3 -m venv .venv-analysis
  )
  if errorlevel 1 goto failed
)
"%PY%" -c "import numpy"
if errorlevel 1 (
  "%PY%" -m pip install -r requirements-analysis.txt
  if errorlevel 1 goto failed
)
"%PY%" scripts\verify_release.py
if errorlevel 1 goto failed
"%PY%" scripts\reproduce_results.py
if errorlevel 1 goto failed
echo Results saved to outputs\reproduced
pause
exit /b 0
:failed
echo Verification did not finish. Keep the error shown above.
pause
exit /b 1
