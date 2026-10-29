#!/usr/bin/env bash
# StyleMate backend - macOS / Linux launcher.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[StyleMate] Creating virtual environment ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt
fi
echo "[StyleMate] http://localhost:8000"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
