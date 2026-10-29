#!/bin/bash
# start_stylemate.command — the one file to run. Double-click it, or:
#
#     bash start_stylemate.command
#
# Starts, in order:
#   1. configuration      loads backend/.env, reports which provider will answer
#   2. catalogue          checks the 1,000-SKU Myntra catalogue is built; builds
#                         it from the CSV only if it is missing
#   3. Gemini             checks the key actually works (nothing to "start":
#                         Gemini is a hosted API, there is no local server)
#   4. audit log sink     writes logs/*.json, port 8765
#   5. the backend        FastAPI on :8000, serving the newest Gemini page AND the 3D room
#                         then opens your browser
#
# Ollama is no longer started. The project has moved to Gemini; the local-model
# path is retained behind --with-ollama as a fallback for working offline.
#
# Ctrl-C (or closing the window) stops the sink and the backend. Ollama is left
# running on purpose — reloading a 7B is slow and you usually want it up.
#
# Flags:
#   --no-restart   leave a running Ollama, and a running StyleMate backend, alone
#   --no-open      do not open the browser
#   --pull         download any missing models (one-off, several GB)
#   --with-ollama  also start a local Ollama (legacy fallback; off by default)
#   --set-key      paste your Gemini API key in, written to backend/.env

set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run from the stylemate folder, or from the project folder above it. The
# launcher finds the app rather than assuming it sits beside it, so the same
# file works in either place and there is never a second copy to drift.
if   [ -d "$SELF/backend" ];           then HERE="$SELF"
elif [ -d "$SELF/stylemate/backend" ]; then HERE="$SELF/stylemate"
else
  printf '\033[31m✗\033[0m Could not find the app: no backend/ here or in stylemate/.\n'
  printf '  Put this file in the project folder or in stylemate/.\n'
  exit 1
fi
cd "$HERE"

BACKEND="$HERE/backend"
ENV_FILE="$BACKEND/.env"
FRONTEND="$HERE/frontend"
STATIC="$BACKEND/app/static"
PORT_API=8000
PORT_SINK=8765

RESTART=1; OPEN=1; PULL=0; USE_OLLAMA=0; SET_KEY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-restart) RESTART=0 ;;
    --no-open)    OPEN=0 ;;
    --pull)       PULL=1 ;;
    --set-key)    SET_KEY=1 ;;
    --with-ollama) USE_OLLAMA=1 ;;
    --no-ollama)   USE_OLLAMA=0 ;;   # kept so old notes still work
    -h|--help)    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

STEPS=5
say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }

PIDS=()
cleanup() {
  printf '\n'
  say "Stopping"
  for pid in ${PIDS+"${PIDS[@]}"}; do
    kill "$pid" 2>/dev/null && ok "stopped pid $pid"
  done
  ok "Ollama left running (pkill -f 'ollama serve' to stop it)"
  exit 0
}
trap cleanup INT TERM

up() { curl -fsS --max-time 2 "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------ 1. .env
say "1/${STEPS}  Configuration"
if [ ! -f "$ENV_FILE" ]; then
  # A .env holds secrets, so it is created here on your own machine rather than
  # shipped. env.template is the version-controlled copy with an empty key.
  if [ -f "$BACKEND/env.template" ]; then
    cp "$BACKEND/env.template" "$ENV_FILE"
    ok "created backend/.env from env.template — put your Gemini key in it when you have one"
  elif [ -f "$BACKEND/.env.example" ]; then
    cp "$BACKEND/.env.example" "$ENV_FILE"
    warn "no env.template — copied .env.example to backend/.env"
  else
    bad "backend/.env is missing and there is no template to copy."
  fi
fi
if [ -f "$ENV_FILE" ]; then
  # Export for Ollama and uvicorn. config.py also reads the file itself, so the
  # backend is correct even when started by hand; this makes the same values
  # available to the shell steps below.
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  ok "backend/.env loaded"
if [ "$SET_KEY" = "1" ]; then
  # read -s keeps the key out of the terminal and out of shell history.
  printf '  Paste your Gemini API key (input hidden), then Enter: '
  read -r -s NEWKEY; printf '\n'
  if [ -n "$NEWKEY" ]; then
    if grep -q '^GEMINI_API_KEY=' "$ENV_FILE" 2>/dev/null; then
      tmp="$ENV_FILE.tmp.$$"
      sed "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEWKEY|" "$ENV_FILE" > "$tmp" && mv "$tmp" "$ENV_FILE"
    else
      printf '\nGEMINI_API_KEY=%s\n' "$NEWKEY" >> "$ENV_FILE"
    fi
    chmod 600 "$ENV_FILE" 2>/dev/null
    GEMINI_API_KEY="$NEWKEY"
    ok "key written to backend/.env (file set to owner-only)"
    unset NEWKEY
  else
    warn "nothing entered; the key was left unchanged"
  fi
fi
  if [ -n "${GEMINI_API_KEY:-}" ]; then
    ok "GEMINI_API_KEY is set — the page calls the backend, the key never reaches the browser"
    ok "provider: ${SM_PROVIDER:-auto} -> Gemini (${SM_GEMINI_TEXT_MODEL:-gemini-3.1-flash-lite} / ${SM_GEMINI_VISION_MODEL:-gemini-3.5-flash-lite})"
  else
    warn "GEMINI_API_KEY is empty."
    warn "to add it:  bash $(basename "${BASH_SOURCE[0]}") --set-key"
    warn "Until then search uses the rule parser, and the colour studio uses its"
    warn "pixel estimate. Add --with-ollama to use a local model instead."
  fi
fi
TEXT_MODEL="${SM_TEXT_MODEL:-qwen2.5:7b}"
VISION_MODEL="${SM_VISION_MODEL:-qwen3-vl:8b}"
CTX="${SM_NUM_CTX:-32768}"
# Only mention the local model names when a local model is actually in play;
# otherwise they are noise from the previous architecture.
[ "$USE_OLLAMA" = "1" ] && ok "local models: $TEXT_MODEL · $VISION_MODEL · num_ctx $CTX"

# ------------------------------------------------------------- 2. catalogue
# The shelf is static data: 1,000 Myntra SKUs that build_myntra_catalogue.py
# picks out of the 1.4 GB CSV. The backend loads backend/app/data/
# myntra_catalogue.json when it starts, so the check has to happen BEFORE the
# backend. Built here only when it is missing or unreadable - never on every
# run - and once built, the CSV is not needed again.
say "2/${STEPS}  Catalogue"
CAT_JSON="$BACKEND/app/data/myntra_catalogue.json"
CAT_JS="$FRONTEND/catalogue.js"
CAT_CSV="${SM_MYNTRA_CSV:-$HERE/../myntra dataset/myntra202305041052.csv}"
PYC=""
for c in "$BACKEND/.venv/bin/python" python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  # Run it, do not just find it: macOS's python3 can be a stub that only
  # opens the Xcode installer.
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'print(1)' >/dev/null 2>&1; then PYC="$c"; break; fi
done
cat_info() {   # "<rows> <built date> <built time>", or nothing if unreadable
  "$PYC" -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); print(len(d.get("rows") or []), d.get("built",""))' "$1" 2>/dev/null
}
CAT_ROWS=0
[ -n "$PYC" ] && [ -s "$CAT_JSON" ] && CAT_ROWS="$(cat_info "$CAT_JSON" | cut -d' ' -f1)"
if [ -z "$PYC" ]; then
  bad "no working Python to check or build the catalogue. Fix with:  xcode-select --install"
elif [ "${CAT_ROWS:-0}" -gt 0 ] 2>/dev/null; then
  ok "already built: $(cat_info "$CAT_JSON" | awk '{print $1" Myntra SKUs, built "$2" "$3}') - not rebuilding"
elif [ ! -f "$CAT_CSV" ]; then
  bad "the catalogue is not built, and the Myntra CSV it is built from is not here:"
  warn "  $CAT_CSV"
  warn "Put the CSV back (it is also inside 'archive (1).zip'), or point at it with"
  warn "  SM_MYNTRA_CSV=/path/to/myntra202305041052.csv bash $(basename "${BASH_SOURCE[0]}")"
  warn "Until then the storefront shows its offline demo catalogue."
else
  warn "catalogue not built yet - building it now from the Myntra CSV (one-off, ~15s)"
  if "$PYC" "$HERE/build_myntra_catalogue.py" --force --csv "$CAT_CSV" 2>&1 | sed 's/^/      /'; then
    if [ "$(cat_info "$CAT_JSON" | cut -d' ' -f1)" -gt 0 ] 2>/dev/null; then
      ok "built: $(cat_info "$CAT_JSON" | cut -d' ' -f1) SKUs"
      ok "the backend embeds the new catalogue once at startup (~30s, in the background)"
    else
      bad "the build ran but produced no catalogue - see the lines above"
    fi
  else
    bad "the build failed - see the lines above"
  fi
fi
# The page beside the backend gets the catalogue from the backend. This copy is
# only for opening the HTML file directly, and is cheap to recreate.
if [ -n "$PYC" ] && [ -s "$CAT_JSON" ] && [ ! -s "$CAT_JS" ]; then
  "$PYC" -c 'import sys
open(sys.argv[2],"w",encoding="utf-8").write("/* Generated by build_myntra_catalogue.py - do not edit. */\nwindow.STYLEMATE_CATALOGUE="+open(sys.argv[1],encoding="utf-8").read()+";\n")' \
    "$CAT_JSON" "$CAT_JS" && ok "recreated frontend/catalogue.js from the built catalogue"
fi

# ---------------------------------------------------------------- 3. Gemini
say "3/${STEPS}  Gemini"
# Nothing to launch: Gemini is a hosted API. What is worth doing is proving the
# key works NOW, rather than letting the first search fail with a 403.
if [ -z "${GEMINI_API_KEY:-}" ]; then
  warn "no GEMINI_API_KEY in backend/.env — add it with:  bash $(basename "${BASH_SOURCE[0]}") --set-key"
  warn "Without it the backend falls back to Ollama, or to rule-based search."
else
  GRESP="$(curl -fsS --max-time 20 \
    "https://generativelanguage.googleapis.com/v1beta/models?key=${GEMINI_API_KEY}" 2>/dev/null)"
  if [ -z "$GRESP" ]; then
    bad "could not reach the Gemini API, or the key was rejected."
    warn "Check the key at https://aistudio.google.com (Get API key), and your network."
  elif printf '%s' "$GRESP" | grep -q '"error"'; then
    bad "Gemini rejected the key:"
    printf '%s' "$GRESP" | tr ',' '\n' | grep -oE '"(message|status)":[^,}]*' | head -3 | sed 's/^/        /'
  else
    NMODELS="$(printf '%s' "$GRESP" | grep -o '"name"' | wc -l | tr -d ' ')"
    ok "key accepted — $NMODELS models visible"
    for m in "${SM_GEMINI_TEXT_MODEL:-}" "${SM_GEMINI_VISION_MODEL:-}"; do
      [ -z "$m" ] && continue
      if printf '%s' "$GRESP" | grep -q "models/$m\""; then
        ok "$m available"
      else
        warn "$m is NOT in the list for this key — check the id, or pick another"
      fi
    done
  fi
fi

# ---------------------------------------------------------- LEGACY: Ollama
# Off unless --with-ollama. Kept, not deleted: it is the only thing that works
# with no internet and no key, and the backend still routes to it when
# SM_PROVIDER=ollama or no key is set.
say "   Local Ollama (legacy)"
if [ "$USE_OLLAMA" = "0" ]; then
  ok "not started — the project runs on Gemini (--with-ollama to start it anyway)"
elif ! command -v ollama >/dev/null 2>&1; then
  bad "ollama is not on PATH — install from https://ollama.com/download"
  warn "The app still runs: search falls back to its rule parser."
else
  if up "http://127.0.0.1:11434/api/tags" && [ "$RESTART" = "1" ]; then
    warn "already running — restarting, because OLLAMA_CONTEXT_LENGTH is only"
    warn "read at startup and the page needs $CTX."
    pkill -f 'Ollama.app' 2>/dev/null
    pkill -f 'ollama serve' 2>/dev/null
    osascript -e 'quit app "Ollama"' >/dev/null 2>&1
    for _ in $(seq 1 20); do up "http://127.0.0.1:11434/api/tags" || break; sleep 0.5; done
  fi
  if ! up "http://127.0.0.1:11434/api/tags"; then
    OLLAMA_ORIGINS="*" OLLAMA_CONTEXT_LENGTH="$CTX" OLLAMA_MAX_LOADED_MODELS=2 \
      nohup ollama serve > "$HERE/ollama.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 40); do up "http://127.0.0.1:11434/api/tags" && break; sleep 0.5; done
  fi
  if up "http://127.0.0.1:11434/api/tags"; then
    ok "listening on 11434 (log: ollama.log)"
    ok "OLLAMA_ORIGINS=*  so the browser may call it"
    ok "OLLAMA_MAX_LOADED_MODELS=2  so text and vision models stay loaded together"
  else
    bad "did not come up within 20s — see ollama.log"
  fi
fi

# ---------------------------------------------------------------- 3. models
say "   Local Ollama models (legacy)"
if [ "$USE_OLLAMA" = "1" ] && up "http://127.0.0.1:11434/api/tags"; then
  HAVE="$(curl -fsS http://127.0.0.1:11434/api/tags | tr ',' '\n' \
          | grep -o '"name":"[^"]*"' | sed 's/"name":"//;s/"//')"
  for m in "$TEXT_MODEL" "$VISION_MODEL"; do
    if printf '%s\n' "$HAVE" | grep -qx "$m"; then
      ok "$m"
    elif [ "$PULL" = "1" ]; then
      warn "$m missing — pulling (one-off, several GB)"
      ollama pull "$m" && ok "$m pulled" || bad "pull failed for $m"
    else
      warn "$m is NOT pulled.  ollama pull $m    (or re-run with --pull)"
    fi
  done
  VRAM="$(grep -o 'total_vram="[^"]*"' "$HERE/ollama.log" 2>/dev/null | tail -1 | sed 's/.*="//;s/"//')"
  [ -n "$VRAM" ] && warn "memory: $VRAM. A 7B text model and an 8B vision model together is"
  [ -n "$VRAM" ] && warn "tight; Ollama may unload one and reload it on demand."
elif [ "$USE_OLLAMA" = "1" ]; then
  warn "skipped — Ollama is not answering"
fi

# ------------------------------------------------------------- 4. log sink
say "4/${STEPS}  Audit log sink"
SINK=""
for c in "$HERE/serve_with_logs.js" "$HERE/../serve_with_logs.js"; do
  [ -f "$c" ] && command -v node >/dev/null 2>&1 && { SINK="node \"$c\""; break; }
done
if [ -z "$SINK" ]; then
  for c in "$HERE/serve_with_logs.py" "$HERE/../serve_with_logs.py"; do
    if [ -f "$c" ]; then
      for py in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -c 'print(1)' >/dev/null 2>&1; then
          SINK="$py \"$c\""; break 2
        fi
      done
    fi
  done
fi
if up "http://127.0.0.1:${PORT_SINK}/ping"; then
  ok "already running on $PORT_SINK"
elif [ -n "$SINK" ]; then
  ( cd "$HERE" && eval "PORT=$PORT_SINK nohup $SINK > \"$HERE/sink.log\" 2>&1 &" )
  for _ in $(seq 1 25); do up "http://127.0.0.1:${PORT_SINK}/ping" && break; sleep 0.2; done
  if up "http://127.0.0.1:${PORT_SINK}/ping"; then
    ok "logs -> $HERE/logs/   (written silently; nothing downloads)"
  else
    warn "sink did not start — see sink.log. The app is unaffected."
  fi
else
  warn "no sink script and no runtime for one; logs stay in the browser tab."
fi

# --------------------------------------------------------------- 5. backend
say "5/${STEPS}  StyleMate backend and page"
if [ ! -d "$BACKEND" ]; then
  bad "no backend/ folder next to this script."; exit 1
fi

# The versioned page is the source of truth; the backend serves app/static.
# Gemini pages win. The ollama_v* files are the previous line and are only
# used if no Gemini page is present.
# Highest version by the numbers in the name (v15_gemini_v2 beats v15_gemini_v1,
# v16_gemini_v1 beats both) - not by file date, so opening or copying an old
# page can never make it the one that is served.
pick_highest() {
  local best="" bestkey=-1 f k
  for f in "$FRONTEND"/*_"$1"_v*.html; do
    [ -f "$f" ] || continue
    k=0
    [[ "$(basename "$f")" =~ _v([0-9]+)_$1_v([0-9]+)\.html$ ]] && \
      k=$(( BASH_REMATCH[1] * 1000 + BASH_REMATCH[2] ))
    if [ "$k" -gt "$bestkey" ]; then bestkey=$k; best="$f"; fi
  done
  printf '%s' "$best"
}
NEWEST="$(pick_highest gemini)"
[ -z "$NEWEST" ] && NEWEST="$(pick_highest ollama)"
[ -n "$NEWEST" ] && case "$(basename "$NEWEST")" in
  *ollama_v*) warn "no Gemini page found — falling back to $(basename "$NEWEST"), which calls Ollama directly" ;;
esac
# "StyleMate_frontend_v15_gemini_v7.html" -> "v15 · gemini v7"
page_version() {
  if [[ "$1" =~ _v([0-9]+)_(gemini|ollama)_v([0-9]+)\.html$ ]]; then
    printf 'v%s · %s v%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  else
    printf '%s' "$1"
  fi
}
PAGE_NAME="$(basename "${NEWEST:-}")"
PAGE_VER="$(page_version "$PAGE_NAME")"
if [ -n "$NEWEST" ] && [ -d "$STATIC" ]; then
  if ! cmp -s "$NEWEST" "$STATIC/index.html"; then
    cp "$NEWEST" "$STATIC/index.html"
    ok "serving $PAGE_NAME  (copied into app/static/index.html)"
  else
    ok "serving $PAGE_NAME"
  fi
  printf '  \033[1m→ PAGE VERSION: %s\033[0m   (newest of %s pages in frontend/)\n' \
    "$PAGE_VER" "$(ls "$FRONTEND"/*_v*_gemini_v*.html 2>/dev/null | wc -l | tr -d ' ')"
  for a in three.module.min.js OrbitControls.js stylemate-room.js; do
    # Copy when different, not only when missing, so edits to these files reach the server.
    [ -f "$FRONTEND/$a" ] && ! cmp -s "$FRONTEND/$a" "$STATIC/$a" && cp "$FRONTEND/$a" "$STATIC/$a"
  done
  MISSING=""
  for a in three.module.min.js OrbitControls.js stylemate-room.js; do
    [ -f "$STATIC/$a" ] || MISSING="$MISSING $a"
  done
  [ -z "$MISSING" ] && ok "3D room assets present (three.js, OrbitControls, stylemate-room)" \
                    || bad "3D assets missing:$MISSING — the fitting room will not load"
fi

# The app uses "float | None" annotations, which Python 3.9 cannot evaluate at
# runtime — pydantic raises TypeError and uvicorn never starts. macOS ships 3.9
# with the Xcode command line tools, so the first venv is easily built with the
# wrong interpreter. Prefer 3.10+, and say so plainly when only 3.9 exists.
PY=""; PYV=""
for c in python3.13 python3.12 python3.11 python3.10 \
         /opt/homebrew/bin/python3 /usr/local/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
         python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  v="$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" || continue
  [ -z "$v" ] && continue
  maj="${v%%.*}"; min="${v##*.}"
  if [ "$maj" = "3" ] && [ "$min" -ge 10 ] 2>/dev/null; then PY="$c"; PYV="$v"; break; fi
  [ -z "$PY" ] && { PY="$c"; PYV="$v"; }          # remember a fallback
done
if [ -z "$PY" ]; then
  bad "no working Python 3 found."
  warn "macOS ships a stub that only opens the Xcode installer. Fix with:"
  warn "    xcode-select --install"
  exit 1
fi
PYMIN="${PYV##*.}"
if [ "${PYMIN:-0}" -lt 10 ] 2>/dev/null; then
  warn "the newest Python here is $PYV. The app uses 3.10+ syntax, so it needs a"
  warn "compatibility package to run on $PYV. It will be installed automatically."
  warn "A cleaner fix: install Python 3.12 from https://www.python.org/downloads/"
  warn "then delete backend/.venv and run this again."
else
  ok "python $PYV"
fi

# A venv built earlier with 3.9 will keep failing until it is rebuilt.
if [ -x "$BACKEND/.venv/bin/python" ]; then
  VV="$("$BACKEND/.venv/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  VMIN="${VV##*.}"
  if [ -z "$VV" ]; then
    warn "backend/.venv is broken — rebuilding it"
    rm -rf "$BACKEND/.venv"
  elif [ "${VMIN:-0}" -lt 10 ] 2>/dev/null && [ "${PYMIN:-0}" -ge 10 ] 2>/dev/null; then
    warn "backend/.venv was built with Python $VV, which cannot run this app."
    warn "Rebuilding it with $PYV."
    rm -rf "$BACKEND/.venv"
  fi
fi

if [ ! -x "$BACKEND/.venv/bin/python" ]; then
  warn "first run — creating the virtual environment and installing packages."
  warn "This takes a few minutes once, then starts in seconds."
  "$PY" -m venv "$BACKEND/.venv" || { bad "could not create .venv"; exit 1; }
  "$BACKEND/.venv/bin/python" -m pip install --upgrade pip -q
  "$BACKEND/.venv/bin/python" -m pip install -r "$BACKEND/requirements.txt" \
    || { bad "pip install failed"; exit 1; }
  ok "environment ready"
fi

# Needed only on 3.9: lets pydantic evaluate "float | None". Harmless elsewhere.
VV="$("$BACKEND/.venv/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
if [ "${VV##*.}" -lt 10 ] 2>/dev/null; then
  if ! "$BACKEND/.venv/bin/python" -c 'import eval_type_backport' >/dev/null 2>&1; then
    warn "installing eval_type_backport so the app can run on Python $VV"
    "$BACKEND/.venv/bin/python" -m pip install -q eval_type_backport \
      && ok "compatibility package installed" || bad "could not install it"
  fi
fi
ok "virtual environment ready (python ${VV:-unknown})"

# A backend left over from an earlier run (or from another copy of the app) used
# to be reused as-is, so it kept running whatever code and page it started with.
# A StyleMate backend is restarted instead; anything else on the port is reported.
if up "http://127.0.0.1:${PORT_API}/api/health"; then
  OLD_PID="$(lsof -tiTCP:"$PORT_API" -sTCP:LISTEN 2>/dev/null | head -1)"
  OLD_CMD="$(ps -o args= -p "${OLD_PID:-0}" 2>/dev/null)"
  OLD_DIR="$(lsof -a -p "${OLD_PID:-0}" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
  if [ "$RESTART" = "1" ] && [[ "$OLD_CMD" == *"uvicorn app.main:app"* ]]; then
    if [ "$OLD_DIR" = "$BACKEND" ]; then
      warn "a StyleMate backend from an earlier run is still up (pid $OLD_PID) — restarting it"
    else
      warn "a StyleMate backend from ANOTHER folder is on :$PORT_API (pid $OLD_PID, ${OLD_DIR:-unknown}) — stopping it"
    fi
    kill "$OLD_PID" 2>/dev/null
    for _ in $(seq 1 20); do up "http://127.0.0.1:${PORT_API}/api/health" || break; sleep 0.5; done
  else
    warn "something already answers on $PORT_API (pid ${OLD_PID:-?}: ${OLD_CMD:-unknown}); reusing it"
  fi
fi
if ! up "http://127.0.0.1:${PORT_API}/api/health"; then
  # exec: the background job IS the server, so its pid is $! and Ctrl-C stops it.
  ( cd "$BACKEND" && exec nohup .venv/bin/python -m uvicorn app.main:app \
      --host 127.0.0.1 --port "$PORT_API" ) > "$HERE/backend.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 60); do up "http://127.0.0.1:${PORT_API}/api/health" && break; sleep 0.5; done
fi

if up "http://127.0.0.1:${PORT_API}/api/health"; then
  ok "http://localhost:${PORT_API}  (log: backend.log)"
  CONF="$(curl -fsS "http://127.0.0.1:${PORT_API}/api/config" 2>/dev/null)"
  if [ -n "$CONF" ]; then
    echo "$CONF" | tr ',' '\n' | grep -oE '"(textModel|visionModel|provider|geminiKeyPresent)":[^,}]*' \
      | sed 's/^/        /'
    echo "$CONF" | grep -q '"loaded":0' && warn ".env parsed but set nothing new (values already in the shell)"
  fi
  # Ask the running server for the page and compare it with the files, so the
  # version printed is the one actually being served, not the one we meant.
  LIVE_VER=""
  LIVE="$(mktemp -t stylemate-page)"
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT_API}/" -o "$LIVE" 2>/dev/null; then
    for f in "$FRONTEND"/*_v*.html; do
      [ -f "$f" ] && cmp -s "$LIVE" "$f" && LIVE_VER="$(page_version "$(basename "$f")")"
    done
    if [ -n "$NEWEST" ] && cmp -s "$LIVE" "$NEWEST"; then
      ok "checked: the server is serving $PAGE_VER"
    else
      bad "the server is serving ${LIVE_VER:-a page that matches no file in frontend/}, not $PAGE_VER"
      warn "stop whatever is on :$PORT_API (lsof -iTCP:$PORT_API) and run this again"
    fi
  fi
  rm -f "$LIVE"
else
  bad "backend did not start — see backend.log"
  warn "The standalone page still works: open frontend/$(basename "${NEWEST:-the html file}") directly."
fi

URL="http://localhost:${PORT_API}/"
# The version in the address makes it a new URL to the browser, so a copy of an
# older page it cached is never shown in place of this one.
OPEN_URL="${URL}?page=$(printf '%s' "$PAGE_VER" | tr -d ' ·' )"
[ "$OPEN" = "1" ] && up "http://127.0.0.1:${PORT_API}/api/health" && { open "$OPEN_URL" 2>/dev/null || true; }

say "Running"
echo "  Page        ${PAGE_NAME:-none found}   ($PAGE_VER)"
echo "  App         $URL        (sign in: test / test)"
echo "  API docs    http://localhost:${PORT_API}/docs"
echo "  Model       Gemini via the backend — the key never reaches the browser"
[ "$USE_OLLAMA" = "1" ] && echo "  Ollama      http://localhost:11434          ollama.log  (legacy, --with-ollama)"
echo "  Audit logs  $HERE/logs/                     sink.log"
echo
echo "  Ctrl-C stops the backend and the sink. Ollama stays up."
echo
while true; do sleep 3600; done
