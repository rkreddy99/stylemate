@echo off
title StyleMate
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo   ========================================================
echo      StyleMate  -  AI search, Shop a Look, Fitting Room
echo   ========================================================
echo.

REM ---- 1. Locate the backend folder, wherever this file was dropped -------
set "APP="
if exist "app\main.py"           set "APP=."
if exist "backend\app\main.py"   set "APP=backend"
if exist "..\backend\app\main.py" set "APP=..\backend"
if not defined APP (
  echo   [X] Could not find the StyleMate backend next to this file.
  echo.
  echo       Put Start_StyleMate.bat in the "stylemate" folder - the one
  echo       that contains the "backend" folder - and run it again.
  echo.
  pause & exit /b 1
)
cd /d "%APP%"
echo   [1/5] Found the backend.

REM ---- 2. Find a usable Python -------------------------------------------
set "PY="
for %%V in (3.12 3.11 3.13 3) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo   [X] Python is not installed, or not on your PATH.
  echo.
  echo       Install Python 3.12 from  https://www.python.org/downloads/
  echo       On the first screen, TICK "Add python.exe to PATH".
  echo       Then run this file again.
  echo.
  pause & exit /b 1
)
echo   [2/5] Using Python: !PY!

REM ---- 3. Create the environment once, reuse it after ---------------------
if not exist ".venv\Scripts\python.exe" (
  echo   [3/5] First run - setting up. This takes a few minutes, once only.
  echo.
  !PY! -m venv .venv
  if errorlevel 1 (
    echo   [X] Could not create the virtual environment.
    pause & exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
  echo       Installing packages...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   [X] Installing the packages failed. The message above says why.
    echo       A common cause is no internet connection.
    pause & exit /b 1
  )
  echo ready > ".venv\.stylemate-ready"
  echo.
  echo       Setup complete. Future runs start in seconds.
) else (
  if not exist ".venv\.stylemate-ready" (
    echo   [3/5] Finishing an interrupted setup...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 ( echo   [X] Install failed. & pause & exit /b 1 )
    echo ready > ".venv\.stylemate-ready"
  ) else (
    echo   [3/5] Environment ready.
  )
)

REM ---- 4. Serve the newest page --------------------------------------------
REM The highest-numbered frontend\StyleMate_frontend_*_gemini_vN.html (and the
REM 3D room files) are copied into app\static, as start_stylemate.command does
REM on a Mac - so a new page version is picked up without editing this file.
echo   [4/5] Picking the newest page...
".venv\Scripts\python.exe" sync_static.py
if errorlevel 1 echo   [!] Some 3D room files are missing - the rest of the app still works.

REM ---- 5. Start, and open the browser once the port answers ---------------
echo   [5/5] Starting StyleMate on http://localhost:8000
echo.
echo   ---------------------------------------------------------
echo     Sign in with      test  /  test
echo     To stop StyleMate, close this window.
echo   ---------------------------------------------------------
echo.

REM Wait for the port to answer, then open the browser. Single level of
REM quoting only - nesting double quotes inside a quoted cmd /c string is a
REM classic way to make a .bat fail on some machines.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:8000/api/health'; foreach($i in 1..45){ Start-Sleep -Milliseconds 700; try{ if((Invoke-WebRequest $u -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200){ Start-Process 'http://localhost:8000'; break } }catch{} }"

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo   StyleMate has stopped.
pause
