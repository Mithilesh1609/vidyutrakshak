#!/usr/bin/env bash
# Driver for the VidyutRakshak API + console. Works in Git Bash on Windows
# and in bash on Linux/macOS. Run from the repo root:
#
#   .claude/skills/run-vidyutrakshak/driver.sh setup     # .venv + requirements (+ pytest)
#   .claude/skills/run-vidyutrakshak/driver.sh build     # pipeline --fresh + offline console
#   .claude/skills/run-vidyutrakshak/driver.sh start     # uvicorn in background, waits for /api/health
#   .claude/skills/run-vidyutrakshak/driver.sh smoke     # GET every read route, check the payloads
#   .claude/skills/run-vidyutrakshak/driver.sh feedback  # POST one outcome, check the queue reflects it
#   .claude/skills/run-vidyutrakshak/driver.sh screenshot [out.png]  # headless Edge/Chrome capture of /
#   .claude/skills/run-vidyutrakshak/driver.sh stop      # kill whatever listens on $PORT
#
# Env: PORT (default 8000), BASE (default http://localhost:$PORT).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PORT=${PORT:-8000}
BASE=${BASE:-http://localhost:$PORT}
LOG=${LOG:-${TMPDIR:-/tmp}/vidyutrakshak-server.log}
export PYTHONUTF8=1          # Windows cp1252 console can't print the pipeline's "→" / "•"
unset VR_DATA_DIR VR_MODELS_DIR

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=""; fi

base_python() {
  for c in "${LOCALAPPDATA:-}/Programs/Python/Python312/python.exe" python3.12 python3 python; do
    if "$c" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then echo "$c"; return; fi
  done
  echo "No Python >= 3.10 found (on Windows: winget install Python.Python.3.12 --scope user)" >&2; exit 1
}

listening_pid() {   # prints the PID listening on $PORT, or nothing
  if command -v netstat >/dev/null && netstat -ano >/dev/null 2>&1 && [ "$(uname -o 2>/dev/null)" = Msys ]; then
    netstat -ano | awk -v p=":$PORT" '$2 ~ p"$" && $4 == "LISTENING" {print $5; exit}'
  else
    (lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true) | head -1
  fi
}

j() { "$PY" -c "import json,sys; d=json.load(sys.stdin); $1"; }   # tiny JSON assert helper
ok() { echo "  ok  $*"; }

case "${1:-}" in
setup)
  [ -n "$PY" ] || { "$(base_python)" -m venv .venv; exec "$0" setup; }
  "$PY" -m pip install -q --disable-pip-version-check -r requirements.txt pytest
  "$PY" -c "import fastapi, lightgbm, sklearn, pandas; print('deps ok, pandas', pandas.__version__)"
  ;;
build)
  "$PY" -m vidyutrakshak.pipeline --fresh | grep -E "v4:|scored" || true
  "$PY" scripts/build_static_dashboard.py
  ;;
start)
  if [ -n "$(listening_pid)" ]; then echo "port $PORT already in use (pid $(listening_pid)); run: $0 stop" >&2; exit 1; fi
  nohup "$PY" -m uvicorn vidyutrakshak.api:app --port "$PORT" >"$LOG" 2>&1 &
  for _ in $(seq 1 60); do
    if h=$(curl -sf "$BASE/api/health"); then echo "up: $h"; echo "log: $LOG"; exit 0; fi
    sleep 1
  done
  echo "server did not answer /api/health in 60 s; log:" >&2; tail -30 "$LOG" >&2; exit 1
  ;;
smoke)
  curl -sf "$BASE/api/health" | j "assert d['status']=='ok', d; print('  ok  health', d)"
  [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/")" = 200 ] && ok "console / -> 200"
  curl -sf "$BASE/api/whoami" | j "print('  ok  whoami', d)"
  curl -sf "$BASE/api/dashboard" | j "m=d['meta']; assert m['consumers']>0; print('  ok  dashboard', m['consumers'], 'consumers, model', m['active_model'])"
  TOP=$(curl -sf "$BASE/api/queue" | j "assert d['count']>0 and d['items'][0]['reasons'] and d['items'][0]['reasons_gu']; print(d['items'][0]['consumer_id'])")
  ok "queue (top: $TOP)"
  curl -sf "$BASE/api/consumers/$TOP" | j "assert d['consumer_id']=='$TOP'; print('  ok  consumer', d['consumer_id'], d['band'], d['risk_score'])"
  curl -sf "$BASE/api/dts" | j "assert len(d)>0; print('  ok  dts', len(d))"
  curl -sf "$BASE/api/feeders" | j "assert len(d)>0; print('  ok  feeders', len(d))"
  curl -sf "$BASE/api/models" | j "print('  ok  models active', d['active'], [v['version'] for v in d['versions']])"
  curl -sf "$BASE/api/queue.csv" | head -1 | grep -q consumer_id && ok "queue.csv header"
  echo "smoke passed"
  ;;
feedback)   # mutates data/: undo with `$0 build`
  TOP=$(curl -sf "$BASE/api/queue" | j "print(d['items'][0]['consumer_id'])")
  curl -sf -X POST "$BASE/api/feedback" -H 'content-type: application/json' \
    -d "{\"consumer_id\":\"$TOP\",\"outcome\":\"theft_confirmed\",\"officer\":\"driver\",\"notes\":\"smoke\"}" \
    | j "assert d['saved']; print('  ok  feedback saved', d)"
  curl -sf "$BASE/api/consumers/$TOP" | "$PY" -c "import sys; t=sys.stdin.read(); assert 'theft_confirmed' in t, t[:400]; print('  ok  consumer $TOP now shows theft_confirmed')"
  echo "feedback flow passed (reset demo state with: $0 build, then restart)"
  ;;
screenshot)   # headless Edge/Chrome; arg 2 = output png
  OUT=${2:-${TMPDIR:-/tmp}/vidyutrakshak-console.png}
  for b in "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" "/c/Program Files/Google/Chrome/Application/chrome.exe" \
           "$(command -v chromium 2>/dev/null || true)" "$(command -v google-chrome 2>/dev/null || true)"; do
    [ -n "$b" ] && [ -x "$b" ] && break
  done
  PROFILE=$(mktemp -d)   # throwaway profile, kept apart from the user's running browser
  "$b" --headless=new --disable-gpu --hide-scrollbars --window-size=1440,1000 --virtual-time-budget=8000 \
       --user-data-dir="$PROFILE" --screenshot="$OUT" "$BASE/" 2>/dev/null
  rm -rf "$PROFILE"; [ -s "$OUT" ] && echo "screenshot: $OUT"
  ;;
stop)
  pid=$(listening_pid)
  if [ -z "$pid" ]; then echo "nothing on port $PORT"; exit 0; fi
  if [ "$(uname -o 2>/dev/null)" = Msys ]; then taskkill //F //PID "$pid" >/dev/null; else kill "$pid"; fi
  echo "stopped pid $pid"
  ;;
*) sed -n '2,14p' "$0"; exit 2 ;;
esac
