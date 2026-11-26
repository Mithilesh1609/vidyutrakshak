#!/usr/bin/env bash
# Reset the demo to a clean state (model v4, no live outcomes) and start the
# console at http://localhost:8000. Run before every rehearsal and before the
# jury slot. Ctrl+C stops the server.
#
#   scripts/demo_reset.sh            # open demo mode (no keys)
#   scripts/demo_reset.sh --keys     # with viewer / vigilance / admin keys
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

if ss -ltn 2>/dev/null | grep -q ':8000 '; then
  echo "Port 8000 is in use. Stop the old server (Ctrl+C in its terminal) and run this again." >&2
  exit 1
fi

echo "• rebuilding simulated data and models v1..v4 (about 10 s)"
unset VR_DATA_DIR VR_MODELS_DIR
"$PY" -m vidyutrakshak.pipeline --fresh | grep -E "v4:|scored" || true
"$PY" scripts/build_static_dashboard.py >/dev/null   # offline fallback: dist/vidyutrakshak_console.html

if [ "${1:-}" = "--keys" ]; then
  export VR_API_KEYS="viewer:view-demo,vigilance:vig-demo,admin:admin-demo"
  echo "• keys: viewer=view-demo  vigilance=vig-demo  admin=admin-demo"
else
  unset VR_API_KEYS
fi
echo "• console: http://localhost:8000   (offline copy: dist/vidyutrakshak_console.html)"
exec "$PY" -m uvicorn vidyutrakshak.api:app --port 8000
