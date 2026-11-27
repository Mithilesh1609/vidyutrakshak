# VidyutRakshak

Explainable electricity theft and abnormal-consumption risk scoring for DISCOMs.
Prototype for **SFIC 2026, PS000205** (Gujarat Urja Vikas Nigam Limited).

VidyutRakshak fuses smart-meter, billing, feeder and DT data to rank consumers
and areas by theft risk. It explains every alert in plain English and Gujarati,
sizes a daily inspection queue to field capacity, and retrains on inspection
outcomes behind a promotion gate.

> **Data note.** Real DISCOM data is not public, so `vidyutrakshak/simulate.py`
> generates a realistic network (10 feeders, 81 DTs, 5,035 consumers, one year
> of daily reads). The meter-data path is also validated on the public, labelled
> SGCC theft dataset (42,372 real consumers); see [Evidence](#evidence).
> `vidyutrakshak/ingest.py` loads real MDMS, billing/GIS, DT and vigilance exports.

## Quick start

```bash
pip install -r requirements.txt
scripts/demo_reset.sh                       # simulate, train v1..v4, serve http://localhost:8000
```

Or step by step: `python -m vidyutrakshak.pipeline --fresh`, then
`uvicorn vidyutrakshak.api:app --port 8000`. With Docker: `docker compose up --build`.

| Command | Effect on models and recorded outcomes |
|---|---|
| `python -m vidyutrakshak.pipeline` | Builds if nothing is built; otherwise rescores with the active model. Deletes nothing. |
| `... --rebuild` | Retrains v1.. from scratch; keeps live field outcomes. |
| `... --fresh` | Demo reset: new simulated data, v1..v4, no live outcomes. |

`--rebuild` and `--fresh` build in a staging folder and swap in only a registry that loads; the previous
registry and outcome log are kept in `models.prev/`. The API never trains on startup: it serves the built
artifacts and asks you to run the pipeline if they are missing (`VR_AUTO_BUILD=1` builds them instead).
A recorded outcome updates the queue, KPIs and files immediately. State is file-based: run one worker.

Tests: `pytest -q` (17 tests: quality bars, calibration, promotion gate, every API
route, access roles, ingest and pseudonymisation, Gujarati reasons, assistant, PDF reports). Tests run in
their own temporary folders and never change the demo state.

Demo video: `.venv/bin/python scripts/demo_video/make_video.py` (with the demo server running) records the
console, narrates it with a Piper voice and cuts in the deck slides; output in `dist/`. Run
`scripts/demo_reset.sh` afterwards. Set `VR_VIDEO_URL` when building the deck to put the video link on slide 3.

Live demo: [docs/demo_run_sheet.md](docs/demo_run_sheet.md). Jury questions:
[docs/jury_qa.md](docs/jury_qa.md). Offline console (no server):
`dist/vidyutrakshak_console.html`.

## Results on the simulated network

The holdout area is never inspected by the feedback loop, so every approach is
scored against the same consumers. Capacity = top 71 alerts (4%). ML rows are
averaged over 5 training runs. These are the numbers the console and deck quote
(`headline` in `static/dashboard_data.json`).

| Approach | Hit rate | Thefts caught | Visits to vacant / solar homes | Use fell >25%: vacant / solar in top 69 |
|---|---|---|---|---|
| Random inspection (status quo) | 5% | 4% | 5 of 71 | 32 |
| Rule: flag >50% consumption drop | 25% | 22% | 53 of 71 | 51 |
| Anomaly detection only (no labels) | 54% | 46% | 28 of 71 | 30 |
| ML on meter data only | 85% | 74% | 6 of 71 | 12.4 |
| **VidyutRakshak: ML + DT energy balance** | **85%** | **73%** | **6 of 71** | **10.0** |

- **Feedback loop:** the hit rate rises from 68% (v1, initial vigilance records) to 86% (v4,
  after three rounds of 60 field inspections), one training run per version.
- **Where the DT balance pays off:** on the hard cases (use fell >25%: 69 thieves among 112 vacant or
  solar homes) it cuts wasted visits by 19% and lifts PR-AUC from 0.86 to 0.89. For blatant
  theft, meter signals alone already rank well, so the overall hit rate is the same.
- **Calibrated scores:** on the holdout, the 80–100 band is 96% theft, 50–80 is 64%, 20–50 is 24%.
  Brier score 0.012, against 0.044 for a flat guess.

## Evidence

| Check | Result | File |
|---|---|---|
| Real data: SGCC, 42,372 consumers, 8.5% thieves, 5-fold CV, meter-only | Top 4%: **40% thieves (4.7× random)** with 1 year; 43% with 2 years; black-box on raw series 32% / 44%; drop rule 3% | `docs/sgcc_validation.json` |
| Long-running theft (half of thieves stealing before the data starts) | 85% hit rate; 85% of those thieves still in the top list | `docs/stress_tests.json` |
| Subtle theft (10–25% bypass) | 73% (69% without DT meters) | 〃 |
| Half / no DT meters | 83% / 86% | 〃 |
| 10% GIS mis-tagging; 20% meters with MDMS gaps | 81%; 83% | 〃 |
| Scale (one laptop) | 1 lakh consumers (95,880) in 15 s (2.1 GB); 1.9 lakh in 33 s (3.9 GB) | `docs/benchmark.json` |

Regenerate: `python scripts/validate_sgcc.py` (needs `external/sgcc/data.csv`, see the
script's docstring), `python scripts/stress_test.py`, `python scripts/benchmark.py --feeders 200`.

## How it works

| Stage | What it does | Where |
|---|---|---|
| Ingest | Validates MDMS, billing/GIS, DT, tamper and vigilance exports; replaces consumer numbers with keyed HMAC pseudonyms | `ingest.py` |
| Data quality | Imputes MDMS gaps from peers so missing reads never look like theft; meters with >20% gaps, or reported faulty in the field, go to the metering team | `features.py`, `pipeline.py` |
| Peer comparison | Compares each consumer daily to same-DT, same-tariff peers, which cancels summer AC peaks, Rabi pumping and Sunday closures | `features.py` |
| Signatures | Sustained drop (change-point), low-read days, flat-lining, capping, tamper alarms, load vs sanctioned kW | `features.py` |
| DT energy balance | DT input minus billed energy; checks whether the DT's losses stepped up in the same weeks the consumer's reads fell. Unmetered DTs are treated as unknown | `features.py` |
| Unsupervised layer | Isolation Forest, works with zero labels | `model.py` |
| Supervised layer | LightGBM on confirmed outcomes; Platt calibration on cross-validated scores (tip-offs excluded) so a score reads as a probability | `model.py` |
| Explanations | TreeSHAP contributions → reason codes → English and Gujarati text | `model.py`, `i18n.py` |
| Area scores | DT and feeder risk from excess loss and top consumer risk | `model.py` |
| Feedback loop | Outcomes → versioned retrain → **promotion gate** (live only if at least as good on outcomes the live model never saw; otherwise kept as a candidate) → roll back any time | `api.py` |
| Drift | PSI per feature, month over month | `pipeline.py` |

## API

| Method | Route | Role |
|---|---|---|
| GET | `/api/whoami` | viewer |
| GET | `/api/queue?capacity=50&feeder=FDR-01` · `/api/queue.csv` (route sheet) | viewer |
| GET | `/api/consumers/{id}` (reasons in English and Gujarati) | viewer |
| GET | `/api/dts`, `/api/dts/{id}`, `/api/feeders`, `/api/models` | viewer |
| POST | `/api/feedback` `{consumer_id, outcome, officer, notes}` | vigilance |
| POST | `/api/retrain` (returns `promoted` and the gate's reason) | admin |
| POST | `/api/models/{v}/activate` | admin |
| POST | `/api/demo/field-round` (demo only: open mode on simulated data) | admin |
| POST | `/api/chat` `{messages: [{role, content}], lang}` (read-only assistant) | viewer |
| GET | `/api/reports/consumer/{id}.pdf?lang=en\|gu` · `/api/reports/queue.pdf?capacity=50&feeder=&band=` · `/api/reports/model.pdf` | viewer |

Set `VR_API_KEYS="viewer:k1,vigilance:k2,admin:k3"` to enforce roles; clients send
`X-API-Key`. The console asks for a key and shows only the actions that role
allows. With no keys set it runs in open demo mode. Interactive docs at `/docs`.

## Assistant and explainable PDFs

**Assistant.** The console's "Ask VidyutRakshak" panel answers questions in English or Gujarati:
why a consumer was flagged, the riskiest consumers on a feeder or DT, transformer losses, network totals,
model quality, and what a metric means. IDs in answers link to the consumer, DT or feeder. It is read-only:
it cannot record outcomes, retrain or switch models.

- With `ANTHROPIC_API_KEY` set on the server, Claude (`claude-opus-5-5`) answers, calling read-only tools
  over the live scores (`vidyutrakshak/assistant.py`). It is told to quote only tool results and never to call
  a consumer a thief. What leaves the server: the question and the tool results (pseudonymised consumer IDs,
  scores, reason text, area totals). No meter series or real consumer numbers.
- Without a key, or with `VR_CHAT_LLM=0`, a keyword matcher answers the same questions from the same tools,
  with no network. Any API error also falls back to it. The panel shows which engine answered.

**PDFs.** Built on the server with fpdf2, with Gujarati shaped by HarfBuzz (Noto fonts in
`vidyutrakshak/fonts/`, SIL OFL):

| Report | Contents | Console button |
|---|---|---|
| Case report | Score and band; reasons in English and Gujarati; TreeSHAP contributions (what pushed the score up or down); 12 months of metered vs peer-expected use; DT loss vs baseline; estimates; a field record box | Queue detail: "Case report (PDF)" |
| Route sheet | The day's queue by feeder and DT, top reasons, outcome tick-boxes | Queue toolbar: "Route sheet (PDF)" |
| Model summary | Model vs random and rule baselines, global feature importance, calibration, reason-code glossary, version history and gate verdict | Model tab |

The offline console (`dist/vidyutrakshak_console.html`) has no server, so the assistant and PDFs are hidden
there; "Print case" prints the selected consumer instead.

## Plugging in real data

```bash
export VR_PSEUDO_SALT="<long secret held by the DISCOM>"
python -m vidyutrakshak.ingest --reads mdms_daily.csv --consumers consumer_master.csv \
    --dt-reads dt_meter_daily.csv --tamper mdms_events.csv --inspections vigilance_log.csv \
    --dts dt_gis.csv --out data_pilot --lookup secure/lookup.csv
VR_DATA_DIR=data_pilot VR_MODELS_DIR=models_pilot python -m vidyutrakshak.pipeline
VR_DATA_DIR=data_pilot VR_MODELS_DIR=models_pilot uvicorn vidyutrakshak.api:app --port 8000
```

| Export | Columns | Source system |
|---|---|---|
| `--reads` | consumer_no, date, kwh | MDMS |
| `--consumers` | consumer_no, dt_code, feeder_code, category, sanctioned_kw | Billing + GIS indexing |
| `--dt-reads` (optional) | dt_code, date, kwh | DT metering / energy audit |
| `--tamper` (optional) | consumer_no, date | MDMS event logs |
| `--inspections` (optional) | consumer_no, outcome[, source] | Vigilance records |
| `--dts` (optional) | dt_code, lat, lng, capacity_kva | GIS |

Dates may be ISO or day-first. Tariff names like "Residential" or "Commercial" are
mapped automatically; pass `--category-map` for local codes. Without simulator
ground truth, the pipeline trains on the vigilance records, reports cross-validated
quality, and writes the console data into the pilot folder (never over the demo).
`--fresh` refuses to overwrite ingested data.

## Project layout

```
vidyutrakshak/
  simulate.py   synthetic DISCOM network, ground truth, stress-test knobs
  ingest.py     real exports -> validated, pseudonymised tables
  features.py   peer comparison, signatures, DT balance, PSI
  model.py      anomaly + LightGBM, calibration, reasons, area scores, registry
  i18n.py       reason text in English and Gujarati
  pipeline.py   end-to-end run (simulated or real data), feedback loop, ablation, export
  api.py        FastAPI service: roles, promotion gate, route sheet
  views.py      read-only views shared by the API, assistant and reports
  assistant.py  chat: Claude tool use, offline rules fallback
  reports.py    explainable PDFs (case report, route sheet, model summary)
  fonts/        Noto Sans + Noto Sans Gujarati (SIL OFL)
static/index.html     inspector console (served at /)
scripts/              demo_reset.sh, build_static_dashboard.py, validate_sgcc.py,
                      stress_test.py, benchmark.py, demo_video/make_video.py
deck/build_deck.js    synopsis deck + appendix, numbers read from the outputs above
docs/                 evidence JSON, jury Q&A, demo run sheet
tests/                end-to-end, gate, roles, ingest tests
```
