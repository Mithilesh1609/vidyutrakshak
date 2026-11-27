"""Read-only views over the live scored state, shared by the API routes, the
assistant (chatbot) and the PDF reports, so all three quote the same numbers.

`state` is the api.STATE dict: S (scored frame), F (features), ctx, dash, reg.
"""
from __future__ import annotations

import numpy as np

from .model import FEATURES, contributions

# Plain-language names for model features (reports, chatbot, SHAP bars).
FEATURE_LABELS = {
    "peer_ratio_change": "Use vs same-DT peers (change)",
    "cp_drop_ratio": "Size of the consumption step-down",
    "cp_weeks_ago": "Weeks since the step-down",
    "low_day_frac": "Share of days far below expected",
    "flatness_change": "Change in day-to-day variation",
    "cap_frac": "Days pinned at a usage ceiling",
    "missing_frac": "Missing meter reads",
    "tamper_events": "Meter tamper alarms",
    "load_utilisation": "Use of sanctioned load",
    "dt_loss_rise": "DT unaccounted energy rise",
    "dt_step_at_drop": "DT loss step at the same time",
    "dt_step_match": "DT loss step matches shortfall",
    "anomaly_score": "Statistical outlier score",
}

DISCLAIMER = ("A risk score ranks consumers for inspection. It is not evidence of theft: "
              "only a field inspection can confirm an irregularity.")


class NotFound(LookupError):
    pass


def queue_items(state: dict, capacity: int = 50, feeder: str | None = None, dt: str | None = None,
                band: str | None = None, include_inspected: bool = False) -> list[dict]:
    S = state["S"]
    if not include_inspected:
        S = S[S.inspection_status == "not_inspected"]
    if feeder:
        S = S[S.feeder_id == feeder.upper()]
    if dt:
        S = S[S.dt_id == dt.upper()]
    if band:
        S = S[S.band == band.capitalize()]
    S = S.nlargest(max(1, min(int(capacity), 1000)), "risk_prob")
    cols = ["consumer_id", "dt_id", "feeder_id", "category", "risk_score", "band", "reasons", "reasons_gu",
            "est_leak_kwh_month", "est_revenue_inr_month", "data_issue"]
    out = S[cols].to_dict(orient="records")
    for r in out:
        r["reasons"], r["reasons_gu"] = list(r["reasons"]), list(r["reasons_gu"])
        r["risk_score"] = float(r["risk_score"])
        r["data_issue"] = bool(r["data_issue"])
    return out


def shap_top(state: dict, cid: str, top: int = 6) -> list[dict]:
    """The features that moved this consumer's score most (TreeSHAP, log-odds)."""
    F = state["F"]
    i = np.where(F.consumer_id.to_numpy() == cid)[0]
    if not len(i):
        raise NotFound(cid)
    m = state.get("model") or state["reg"].load()
    c = contributions(m, F.iloc[i])[0, :-1]
    order = np.argsort(-np.abs(c))[:top]
    return [{"feature": m["features"][j], "label": FEATURE_LABELS.get(m["features"][j], m["features"][j]),
             "contribution": round(float(c[j]), 3)} for j in order]


def consumer_detail(state: dict, cid: str, days: int = 180, with_shap: bool = False) -> dict:
    cid = cid.strip().upper()
    S = state["S"].set_index("consumer_id")
    if cid not in S.index:
        raise NotFound(cid)
    r = S.loc[cid]
    i = int(np.where(state["F"].consumer_id.to_numpy() == cid)[0][0])
    ctx = state["ctx"]
    last = ctx["X"].shape[1]
    window = slice(last - days, last)
    reads = ctx["X"][i, window]
    out = {
        "consumer_id": cid, "dt_id": r.dt_id, "feeder_id": r.feeder_id, "category": r.category,
        "sanctioned_kw": float(r.sanctioned_kw) if "sanctioned_kw" in r else None,
        "risk_score": float(r.risk_score), "band": r.band,
        "reasons": list(r.reasons), "reasons_gu": list(r.reasons_gu),
        "inspection_status": r.inspection_status, "data_issue": bool(r.data_issue),
        "tamper_events": int(r.tamper_events),
        "features": {f: float(r[f]) for f in FEATURES},
        "estimates": {"leak_kwh_month": float(r.est_leak_kwh_month),
                      "revenue_inr_month": float(r.est_revenue_inr_month)},
        "daily": {"dates": list(ctx["dates"][window]),
                  "recorded_kwh": [None if np.isnan(v) else round(float(v), 2) for v in reads],
                  "expected_kwh": [round(float(v), 2) for v in (ctx["pr_base"][i] * ctx["peer"][i, window])]},
    }
    if with_shap:
        out["drivers"] = shap_top(state, cid)
    return out


def dt_summary(state: dict, dt_id: str) -> dict:
    d = next((x for x in state["dash"]["dts"] if x["id"] == dt_id.strip().upper()), None)
    if not d:
        raise NotFound(dt_id)
    S = state["S"]
    top = S[S.dt_id == d["id"]].nlargest(5, "risk_prob")[["consumer_id", "risk_score", "band", "category"]]
    return {**{k: v for k, v in d.items() if k != "loss_series"}, "top_consumers": top.to_dict(orient="records")}


def rank_dts(state: dict, by: str = "excess_kwh_day", n: int = 5, feeder: str | None = None) -> list[dict]:
    by = by if by in ("excess_kwh_day", "loss_recent", "risk", "high") else "excess_kwh_day"
    rows = [{k: v for k, v in d.items() if k not in ("loss_series", "lat", "lng")} for d in state["dash"]["dts"]
            if not feeder or d["feeder"] == feeder.upper()]
    return sorted(rows, key=lambda d: -(d.get(by) or 0))[:max(1, min(n, 50))]


def model_summary(state: dict) -> dict:
    dash = state["dash"]
    reg = state["reg"].list()
    active = next((v for v in reg["versions"] if v["version"] == reg["active"]), {})
    return {"active_model": reg["active"], "trained_on_outcomes": active.get("n_labels"),
            "confirmed_thefts_in_training": active.get("n_pos"), "gate": active.get("gate"),
            "headline": dash.get("headline"), "calibration_brier": (dash.get("calibration") or {}).get("brier"),
            "drift_status": (dash.get("drift") or {}).get("status"),
            "versions": [{"version": v["version"], "status": v["status"], "created": v["created"], "note": v["note"]}
                         for v in reg["versions"]]}
