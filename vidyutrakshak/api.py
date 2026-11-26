"""VidyutRakshak REST API.

    uvicorn vidyutrakshak.api:app --port 8000

Endpoints
  GET  /                         inspector dashboard
  GET  /api/health
  GET  /api/dashboard            full dashboard payload
  GET  /api/queue?capacity=50&feeder=FDR-01&dt=DT-004
  GET  /api/queue.csv            same, as a route sheet for field teams
  GET  /api/consumers/{id}       score, reasons, features, daily series
  GET  /api/dts  | /api/dts/{id}
  GET  /api/feeders
  POST /api/feedback             record a field inspection outcome
  POST /api/retrain              retrain on all outcomes, new model version
  GET  /api/models
  POST /api/models/{v}/activate  roll back / forward
  GET  /api/whoami               role of the caller's key

Access control: set VR_API_KEYS="viewer:key1,vigilance:key2,admin:key3".
Viewers read, vigilance officers can also submit outcomes, admins can also
retrain and switch models. With no keys set the API runs in open demo mode.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score

from . import features as fx
from .model import FEATURES, Registry, add_anomaly_score, cv_oof, eval_ranking, predict, train
from .pipeline import DASH, DATA, MODELS, STATIC, publish_state, run

_lock = threading.Lock()
STATE: dict = {}

ROLES = {"viewer": 0, "vigilance": 1, "admin": 2}


def _keys() -> dict[str, str]:
    raw = os.getenv("VR_API_KEYS", "")
    return {k: r for r, k in (p.split(":", 1) for p in raw.split(",") if ":" in p)}


def require(role: str):
    def dep(x_api_key: str | None = Header(default=None)):
        keys = _keys()
        if not keys:
            return "admin"  # open demo mode
        have = keys.get(x_api_key or "")
        if have is None:
            raise HTTPException(401, "Missing or unknown X-API-Key header.")
        if ROLES[have] < ROLES[role]:
            raise HTTPException(403, f"This action needs the '{role}' role; your key has '{have}'.")
        return have
    return dep


def _load_state():
    if not DASH.exists() or not (MODELS / "registry.json").exists():
        run(verbose=False)
    ds = fx.load(DATA)
    F, ctx = fx.build_features(ds)
    F = add_anomaly_score(F)
    live = DATA / "inspections_live.csv"
    insp = pd.read_csv(live) if live.exists() else ds.inspections.copy()
    STATE.update(ds=ds, F=F, ctx=ctx, insp=insp, reg=Registry(MODELS),
                 dash=json.loads(DASH.read_text()))


@asynccontextmanager
async def lifespan(_app):
    _load_state()
    yield


app = FastAPI(title="VidyutRakshak", version="0.1.0", lifespan=lifespan,
              description="Explainable electricity-theft risk scoring for DISCOMs")


def _extras() -> dict:
    d = STATE["dash"]
    return {"ablation": d.get("ablation", []), "drift": d.get("drift", {}),
            "capacity_k": d["meta"].get("capacity_k"), "calibration": d.get("calibration")}


def _scores() -> pd.DataFrame:
    return pd.read_csv(DATA / "scores.csv")


def _split(s) -> list[str]:
    return [x for x in str(s).split(" | ") if x and x != "nan"]


# ------------------------------------------------------------------ reads
@app.get("/", include_in_schema=False)
def index():
    # The page is authored as a body fragment; give it a standards-mode shell.
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse('<!doctype html><html lang="en"><meta charset="utf-8"><link rel="icon" href="data:,">'
                        + html + "</html>")


@app.get("/api/health")
def health():
    return {"status": "ok", "active_model": STATE["dash"]["meta"]["active_model"],
            "consumers": STATE["dash"]["kpis"]["consumers"]}


@app.get("/api/whoami")
def whoami(role=Depends(require("viewer"))):
    """Lets the console show only the actions the caller's key allows."""
    return {"role": role, "open_mode": not _keys(), "demo_tools": _demo_tools()}


@app.get("/api/dashboard")
def dashboard(_=Depends(require("viewer"))):
    return JSONResponse(STATE["dash"])


@app.get("/api/queue")
def queue(capacity: int = 50, feeder: str | None = None, dt: str | None = None,
          _=Depends(require("viewer"))):
    S = _scores()
    S = S[S.inspection_status == "not_inspected"]
    if feeder:
        S = S[S.feeder_id == feeder]
    if dt:
        S = S[S.dt_id == dt]
    S = S.nlargest(max(1, min(capacity, 1000)), "risk_prob")
    cols = ["consumer_id", "dt_id", "feeder_id", "category", "risk_score", "band", "reasons", "reasons_gu",
            "est_leak_kwh_month", "est_revenue_inr_month", "data_issue"]
    out = S[cols].to_dict(orient="records")
    for r in out:
        r["reasons"] = _split(r["reasons"])
        r["reasons_gu"] = _split(r["reasons_gu"])
    return {"capacity": capacity, "count": len(out), "items": out}


@app.get("/api/queue.csv")
def queue_csv(capacity: int = 50, feeder: str | None = None, dt: str | None = None,
              _=Depends(require("viewer"))):
    """Route sheet for field teams (opens in Excel): the day's queue sorted by
    feeder and DT, with empty columns for the officer to fill in."""
    items = queue(capacity, feeder, dt, _)["items"]
    rows = [{"feeder": r["feeder_id"], "dt": r["dt_id"], "consumer_id": r["consumer_id"], "category": r["category"],
             "risk_score": r["risk_score"], "band": r["band"],
             **{f"reason_{i + 1}": (r["reasons"][i] if i < len(r["reasons"]) else "") for i in range(3)},
             "outcome": "", "officer": "", "notes": ""} for r in items]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["feeder", "dt", "risk_score"], ascending=[True, True, False])
    return Response("﻿" + df.to_csv(index=False), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="inspection_list.csv"'})


@app.get("/api/consumers/{cid}")
def consumer(cid: str, _=Depends(require("viewer"))):
    S = _scores().set_index("consumer_id")
    if cid not in S.index:
        raise HTTPException(404, f"No consumer with ID {cid}.")
    r = S.loc[cid]
    i = int(np.where(STATE["F"].consumer_id.to_numpy() == cid)[0][0])
    ctx = STATE["ctx"]
    last = ctx["X"].shape[1]
    window = slice(last - 180, last)
    reads = ctx["X"][i, window]
    return {
        "consumer_id": cid, "dt_id": r.dt_id, "feeder_id": r.feeder_id, "category": r.category,
        "risk_score": float(r.risk_score), "band": r.band,
        "reasons": _split(r.reasons), "reasons_gu": _split(r.reasons_gu),
        "inspection_status": r.inspection_status, "data_issue": bool(r.data_issue),
        "features": {f: float(r[f]) for f in FEATURES},
        "estimates": {"leak_kwh_month": float(r.est_leak_kwh_month),
                      "revenue_inr_month": float(r.est_revenue_inr_month)},
        "daily": {"dates": ctx["dates"][window],
                  "recorded_kwh": [None if np.isnan(v) else round(float(v), 2) for v in reads],
                  "expected_kwh": [round(float(v), 2) for v in (ctx["pr_base"][i] * ctx["peer"][i, window])]},
    }


@app.get("/api/dts")
def dts(_=Depends(require("viewer"))):
    return [{k: v for k, v in d.items() if k != "loss_series"} for d in STATE["dash"]["dts"]]


@app.get("/api/dts/{dt_id}")
def dt_detail(dt_id: str, _=Depends(require("viewer"))):
    d = next((x for x in STATE["dash"]["dts"] if x["id"] == dt_id), None)
    if not d:
        raise HTTPException(404, f"No transformer with ID {dt_id}.")
    S = _scores()
    top = S[S.dt_id == dt_id].nlargest(10, "risk_prob")[["consumer_id", "risk_score", "band", "category"]]
    return {**d, "top_consumers": top.to_dict(orient="records")}


@app.get("/api/feeders")
def feeders(_=Depends(require("viewer"))):
    return STATE["dash"]["feeders"]


@app.get("/api/models")
def models(_=Depends(require("viewer"))):
    return STATE["reg"].list()


# ------------------------------------------------------------------ writes
class Feedback(BaseModel):
    consumer_id: str
    outcome: Literal["theft_confirmed", "no_irregularity", "meter_fault"]
    officer: str = Field(default="field-team", max_length=60)
    notes: str = Field(default="", max_length=500)


@app.post("/api/feedback")
def feedback(fb: Feedback, role=Depends(require("vigilance"))):
    if fb.consumer_id not in set(STATE["F"].consumer_id):
        raise HTTPException(404, f"No consumer with ID {fb.consumer_id}.")
    with _lock:
        insp = STATE["insp"]
        rnd = int(insp["round"].max()) + 1 if "round" in insp else 1
        row = {"consumer_id": fb.consumer_id,
               # A meter fault is not theft: it trains as a negative and is
               # routed to the metering team instead.
               "outcome": "theft_confirmed" if fb.outcome == "theft_confirmed" else "no_irregularity",
               "source": "field_feedback", "round": rnd,
               "recorded_outcome": fb.outcome, "officer": fb.officer, "notes": fb.notes,
               "recorded_at": datetime.now().isoformat(timespec="seconds")}
        STATE["insp"] = pd.concat([insp, pd.DataFrame([row])], ignore_index=True)
        STATE["insp"].to_csv(DATA / "inspections_live.csv", index=False)
        pending = int((STATE["insp"].source == "field_feedback").sum())
    return {"saved": True, "consumer_id": fb.consumer_id, "outcome": fb.outcome,
            "feedback_since_start": pending}


GATE_MIN_NEW = 20      # new outcomes needed before a model change can be judged
GATE_MIN_EACH = 3      # ... with at least this many thefts and non-thefts
GATE_TOLERANCE = 0.02  # candidate may not be worse than the active model by more than this (PR-AUC)


def _gate(F: pd.DataFrame, insp: pd.DataFrame, L: pd.DataFrame, oof: np.ndarray, active: dict) -> dict:
    """Compare candidate vs active model on outcomes the active model never saw.

    The active model is scored on field outcomes recorded after it was trained
    (genuinely unseen data). The candidate is scored on the same consumers with
    out-of-fold predictions, so neither model is graded on its own training rows.
    """
    start = int(active.get("trained_on_rows") or 0)
    new_ids = set(insp.iloc[start:].consumer_id)
    mask = L.consumer_id.isin(new_ids).to_numpy()
    y = L.y.to_numpy()[mask]
    n_new, n_pos = int(mask.sum()), int(y.sum())
    base = {"against": active["version"], "new_outcomes": n_new, "new_thefts": n_pos}
    if n_new < GATE_MIN_NEW or n_pos < GATE_MIN_EACH or n_new - n_pos < GATE_MIN_EACH:
        return {**base, "promoted": False,
                "reason": (f"Only {n_new} new field outcome(s) ({n_pos} theft) since {active['version']} was trained. "
                           f"At least {GATE_MIN_NEW}, with {GATE_MIN_EACH}+ thefts and {GATE_MIN_EACH}+ clean visits, "
                           f"are needed to judge a new model, so {active['version']} stays in service.")}
    ap_active = float(average_precision_score(y, predict(active, L[mask])))
    ap_cand = float(average_precision_score(y, oof[mask]))
    promoted = ap_cand >= ap_active - GATE_TOLERANCE
    if not promoted:
        verdict = f"worse by more than {GATE_TOLERANCE:.2f}, so it is rejected and {active['version']} stays in service"
    elif ap_cand >= ap_active:
        verdict = "at least as good, so it is promoted"
    else:
        verdict = f"within the {GATE_TOLERANCE:.2f} tolerance and trained on more outcomes, so it is promoted"
    return {**base, "promoted": promoted, "active_pr_auc": round(ap_active, 3), "candidate_pr_auc": round(ap_cand, 3),
            "reason": (f"On {n_new} field outcomes {active['version']} never saw, the new model scores PR-AUC "
                       f"{ap_cand:.3f} vs {ap_active:.3f} for {active['version']}: {verdict}.")}


@app.post("/api/retrain")
def retrain(_=Depends(require("admin"))):
    with _lock:
        F, insp, reg = STATE["F"], STATE["insp"], STATE["reg"]
        # Out-of-fold predictions on every labelled outcome: quality without
        # ground truth, and the candidate's side of the promotion gate.
        L, oof = cv_oof(F, insp, FEATURES)
        k = max(5, int(0.2 * len(L)))
        met = {**eval_ranking(L.y.to_numpy(), oof, k), "eval": "cv"}
        active = reg.load()
        gate = _gate(F, insp, L, oof, active)
        m = train(F, insp, FEATURES)
        v = reg.save(m, met, f"retrained on {len(L)} field outcomes", activate=gate["promoted"],
                     trained_on_rows=len(insp), gate=gate)
        STATE["dash"] = publish_state(F, STATE["ctx"], STATE["ds"], reg, insp, _extras())
    return {"version": v, "promoted": gate["promoted"], "reason": gate["reason"], "gate": gate,
            "labels": int(len(L)), "positives": int(L.y.sum()), "cv_pr_auc": round(met["pr_auc"], 3),
            "active": STATE["dash"]["meta"]["active_model"]}


@app.post("/api/models/{version}/activate")
def activate(version: str, _=Depends(require("admin"))):
    reg = STATE["reg"]
    with _lock:
        try:
            reg.set_active(version)
        except KeyError:
            raise HTTPException(404, f"No model version {version}.")
        STATE["dash"] = publish_state(STATE["F"], STATE["ctx"], STATE["ds"], reg, STATE["insp"], _extras())
    return {"active": version}


# ------------------------------------------------------------------ demo helper
def _demo_tools() -> bool:
    """Demo-only actions exist only in open demo mode on simulated data."""
    return not _keys() and (DATA / "_ground_truth.csv").exists()


@app.post("/api/demo/field-round")
def demo_field_round(n: int = 60, _=Depends(require("admin"))):
    """Simulate one day of field work: inspect the top-n of the current queue and
    record the (simulated) true outcome for each. Lets a live demo show a model
    being promoted by the gate without typing 60 outcomes by hand."""
    if not _demo_tools():
        raise HTTPException(404, "Not found.")
    gt = pd.read_csv(DATA / "_ground_truth.csv").set_index("consumer_id")
    with _lock:
        S = _scores()
        picked = S[S.inspection_status == "not_inspected"].nlargest(max(1, min(n, 200)), "risk_prob").consumer_id
        insp = STATE["insp"]
        rnd = int(insp["round"].max()) + 1 if "round" in insp else 1
        rows = pd.DataFrame({
            "consumer_id": picked.to_numpy(),
            "outcome": np.where(gt.loc[picked, "is_theft"].to_numpy() == 1, "theft_confirmed", "no_irregularity"),
            "source": "field_feedback", "round": rnd, "recorded_outcome": None,
            "officer": "demo-field-round", "notes": "", "recorded_at": datetime.now().isoformat(timespec="seconds"),
        })
        rows["recorded_outcome"] = rows.outcome
        STATE["insp"] = pd.concat([insp, rows], ignore_index=True)
        STATE["insp"].to_csv(DATA / "inspections_live.csv", index=False)
        STATE["dash"] = publish_state(STATE["F"], STATE["ctx"], STATE["ds"], STATE["reg"], STATE["insp"], _extras())
    return {"recorded": int(len(rows)), "thefts": int((rows.outcome == "theft_confirmed").sum())}
