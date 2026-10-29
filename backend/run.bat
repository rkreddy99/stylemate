@echo off
REM ============================================================
REM  StyleMate backend - Windows one-click launcher
REM  First run creates a virtual environment and installs deps.
REM  Later runs reuse it, so you are not reinstalling every time.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [StyleMate] Creating virtual environment ...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo [StyleMate] Could not create a virtual environment.
    echo             Install Python 3.11 or 3.12 from python.org and retry.
    pause & exit /b 1
  )
  echo [StyleMate] Installing dependencies ^(one time^) ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 ( echo [StyleMate] Install failed. & pause & exit /b 1 )
  echo. > .venv\.stylemate-ready
) else (
  REM Only reinstall when requirements.txt is newer than the marker.
  for %%F in (requirements.txt) do set REQ=%%~tF
  if not exist ".venv\.stylemate-ready" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    echo. > .venv\.stylemate-ready
  )
)

echo.
echo [StyleMate] Starting on http://localhost:8000
echo [StyleMate] Open that address in your browser. Ctrl+C stops the server.
echo.
start "" http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
