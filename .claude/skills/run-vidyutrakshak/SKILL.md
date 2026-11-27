---
name: run-vidyutrakshak
description: Set up, build, start, smoke-test, screenshot and stop the VidyutRakshak FastAPI server and risk console (localhost:8000). Use when asked to run or start the server, rebuild the demo models, check the API routes, record a field outcome, screenshot the console, or run the tests, including on Windows.
---

VidyutRakshak is a FastAPI app (`vidyutrakshak.api:app`) that serves a theft-risk console at `/` and JSON at `/api/*`,
from data and models that `vidyutrakshak.pipeline` builds. Drive it with
`.claude/skills/run-vidyutrakshak/driver.sh`: curl checks for the API, headless Edge/Chrome for the console.
All paths are relative to the repo root. Verified on Windows 11 under Git Bash (the Bash tool); the driver also has
Linux branches (`lsof`, `kill`, `.venv/bin/python`), but those haven't been run.

## Prerequisites (Windows)

There is no Python on a clean Windows machine: `python`/`python3` are only Microsoft Store stubs. LightGBM's DLL also
needs the VC++ runtime (`vcomp140.dll`). Install both once, from PowerShell:

```powershell
winget install Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
winget install Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements   # UAC prompt
```

## Setup and build

```bash
D=.claude/skills/run-vidyutrakshak/driver.sh
$D setup    # creates .venv from Python 3.12 and installs requirements.txt + pytest (~minutes the first time)
$D build    # pipeline --fresh: simulated data, models v1..v4, plus dist/vidyutrakshak_console.html (~30 s)
```

`build` is also the **demo reset**. It overwrites the data in `data/` and `models/`, and it modifies the tracked
files `static/dashboard_data.json` and `dist/vidyutrakshak_console.html`. Afterwards, `git restore` those two
files if you don't mean to commit them.

## Run (agent path)

```bash
$D start        # uvicorn in background on $PORT (8000); waits for /api/health; log in $TMPDIR/vidyutrakshak-server.log
$D smoke        # health, /, whoami, dashboard, queue (EN + Gujarati reasons), consumer detail, dts, feeders, models, queue.csv
$D feedback     # POSTs theft_confirmed for the top queue item and checks the consumer now shows it (mutates data; `build` resets)
$D screenshot /path/out.png   # headless Edge or Chrome, 1440x1000, of the console
$D stop         # kills whatever listens on $PORT (netstat + taskkill on Windows)
```

Use `$D start` rather than a `run_in_background` uvicorn: `stop` finds the server by port, so it works for either.
The server is in open demo mode (role `admin`, no keys). For role checks, set
`VR_API_KEYS="viewer:view-demo,vigilance:vig-demo,admin:admin-demo"` before `start` and send `X-API-Key`.

## Run (human path)

`scripts/demo_reset.sh` does build + foreground server, but it **does not work on Windows**: it looks for
`.venv/bin/python` (Windows has `.venv/Scripts/python.exe`), and its port check uses `ss`. Use the driver instead.

## Test

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest -q     # ~70 s
```

On Windows the result is 12 passed, 1 failed. `test_ingest_validates_and_pseudonymises` asserts that `lookup.csv` has
mode `0o600`, and Windows has no POSIX permission bits (`os.stat` reports `0o666`). That failure is expected, not a
regression. The tests use temp dirs and don't change the demo state.

## Gotchas

- **pandas 3 breaks the pipeline.** Under copy-on-write, `DataFrame.to_numpy()` returns a read-only array, and
  `features.py` writes into one (`D[~metered] = np.nan`). `requirements.txt` now pins `pandas>=2.2,<3`. If someone
  removes that pin, the crash comes back.
- **`PYTHONUTF8=1` is required on Windows** for the pipeline and the server. The default cp1252 console can't encode
  the `→`/`•` in log lines, so the pipeline dies mid-run *after* training. The driver exports it.
- **The API never trains on startup.** If `models/` is empty it tells you to run the pipeline, so run `build`
  before `start` on a fresh clone. (`VR_AUTO_BUILD=1` makes it build on startup instead.)
- **Field feedback persists** in `data/` across restarts. Re-run `build` to get back to the clean v4 demo.

## Troubleshooting

- **`Python was not found; run without arguments to install from the Microsoft Store`**: no real Python. Run the
  winget line above.
- **`FileNotFoundError: Could not find module '...lightgbm\bin\lib_lightgbm.dll' (or one of its dependencies)`**:
  the DLL is there, but `vcomp140.dll` is missing. Install `Microsoft.VCRedist.2015+.x64`.
- **`ValueError: assignment destination is read-only`** in `features.py build_features`: pandas 3 got installed. Run
  `.venv/Scripts/python -m pip install "pandas>=2.2,<3"`.
- **`UnicodeEncodeError: 'charmap' codec can't encode character '→'`**: `PYTHONUTF8=1` isn't set.
- **`port 8000 already in use`** from `start`: an old server is still running. `$D stop` (or `PORT=8001 $D start`).
