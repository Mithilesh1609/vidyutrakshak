"""End-to-end checks: pipeline quality bars and every API route.

    pytest -q
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vidyutrakshak.pipeline import DATA, STATIC, run


@pytest.fixture(scope="module")
def dash():
    return run(fresh=True, verbose=False)


def test_model_beats_random_and_rules(dash):
    abl = {a["approach"]: a for a in dash["ablation"]}
    ours = abl["VidyutRakshak: ML + DT energy balance"]
    assert ours["precision_at_k"] > 5 * abl["Random inspection (status quo)"]["precision_at_k"]
    assert ours["precision_at_k"] > abl["Rule: flag >50% consumption drop"]["precision_at_k"]
    # Legitimate drops (vacant homes, rooftop solar) must not flood the queue.
    assert ours["legit_drop_false_alarms_at_k"] < abl["Rule: flag >50% consumption drop"]["legit_drop_false_alarms_at_k"]


def test_feedback_loop_improves(dash):
    hv = [v for v in dash["models"] if v["metrics"].get("eval") == "holdout"]
    assert hv[-1]["metrics"]["pr_auc"] > hv[0]["metrics"]["pr_auc"]


def test_every_alert_is_explained(dash):
    high = [q for q in dash["queue"] if q["band"] == "High"]
    assert high and all(q["reasons"] for q in high)
    # Every reason has a Gujarati version for field officers.
    assert all(len(q["reasons_gu"]) == len(q["reasons"]) for q in high)
    assert any("ગ્રાહક" in r or "મીટર" in r for q in high for r in q["reasons_gu"])


def test_scores_are_calibrated(dash):
    """A risk score should mean what it says: in each band the observed theft
    rate on the holdout should be close to the predicted one."""
    cal = dash["calibration"]
    assert cal["brier"] < cal["brier_base_rate"]
    for b in cal["bins"]:
        if b["n"] >= 20:
            assert abs(b["predicted"] - b["observed"]) < 0.2, b


def test_api_routes(dash):
    from vidyutrakshak.api import app
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        assert "<title>VidyutRakshak" in c.get("/").text
        q = c.get("/api/queue?capacity=5").json()
        assert q["count"] == 5
        cid = q["items"][0]["consumer_id"]
        d = c.get(f"/api/consumers/{cid}").json()
        assert len(d["daily"]["dates"]) == 180
        assert c.get("/api/consumers/NOPE").status_code == 404
        assert c.get("/api/dts/DT-001").status_code == 200
        r = c.post("/api/feedback", json={"consumer_id": cid, "outcome": "meter_fault"})
        assert r.json()["saved"]
        assert c.post("/api/feedback", json={"consumer_id": cid, "outcome": "maybe"}).status_code == 422
        csv = c.get("/api/queue.csv?capacity=10")
        assert csv.status_code == 200 and csv.text.lstrip("﻿").startswith("feeder,dt,consumer_id")
        assert len(csv.text.strip().splitlines()) == 11

        # One new outcome is too few to judge a model: saved as a candidate, v4 stays live.
        j = c.post("/api/retrain").json()
        assert j["promoted"] is False and j["active"] == "v4"
        reg = c.get("/api/models").json()
        assert reg["active"] == "v4"
        assert next(v for v in reg["versions"] if v["version"] == j["version"])["status"] == "candidate"
        # The field-reported meter fault lands on the metering team's list.
        dq = c.get("/api/dashboard").json()["data_quality"]
        assert any(q["id"] == cid and q["field_fault"] for q in dq)

        # A full field round gives the gate enough evidence to decide.
        fr = c.post("/api/demo/field-round?n=60").json()
        assert fr["recorded"] == 60
        j2 = c.post("/api/retrain").json()
        assert j2["gate"]["new_outcomes"] >= 60 and "candidate_pr_auc" in j2["gate"]
        assert (c.get("/api/models").json()["active"] == j2["version"]) == j2["promoted"]

        assert c.post("/api/models/v4/activate").json()["active"] == "v4"
        assert c.get("/api/whoami").json() == {"role": "admin", "open_mode": True, "demo_tools": True,
                                               "chat_mode": "rules"}


def test_gate_rejects_a_worse_model(dash):
    """The promotion gate must refuse a candidate that ranks worse than the
    active model on outcomes the active model never saw."""
    import numpy as np
    from vidyutrakshak import api
    from vidyutrakshak.model import label_frame, predict
    with TestClient(api.app):
        F, reg = api.STATE["F"], api.STATE["reg"]
        active = reg.load()
        insp = pd.read_csv(DATA / "inspections_live.csv")
        # Pretend the active model was trained before the last 120 outcomes.
        active = {**active, "trained_on_rows": len(insp) - 120}
        L = label_frame(F, insp)
        good = predict(active, L)
        assert api._gate(F, insp, L, good, active)["promoted"] is True
        noise = np.random.default_rng(0).random(len(L))
        g = api._gate(F, insp, L, noise, active)
        assert g["promoted"] is False and g["candidate_pr_auc"] < g["active_pr_auc"]


def test_access_control(dash, monkeypatch):
    from vidyutrakshak.api import app
    monkeypatch.setenv("VR_API_KEYS", "viewer:v1ew,vigilance:v1g,admin:adm")
    with TestClient(app) as c:
        assert c.get("/api/dashboard").status_code == 401
        assert c.get("/api/dashboard", headers={"X-API-Key": "v1ew"}).status_code == 200
        for key, role in (("v1ew", "viewer"), ("v1g", "vigilance"), ("adm", "admin")):
            w = c.get("/api/whoami", headers={"X-API-Key": key}).json()
            assert w["role"] == role and w["open_mode"] is False and w["demo_tools"] is False
        body = {"consumer_id": "C00001", "outcome": "no_irregularity"}
        assert c.post("/api/feedback", json=body, headers={"X-API-Key": "v1ew"}).status_code == 403
        assert c.post("/api/feedback", json=body, headers={"X-API-Key": "v1g"}).status_code == 200
        assert c.post("/api/retrain", headers={"X-API-Key": "v1g"}).status_code == 403
        # Demo shortcuts don't exist once real keys are configured.
        assert c.post("/api/demo/field-round", headers={"X-API-Key": "adm"}).status_code == 404


def test_feedback_is_visible_immediately_and_after_restart(dash):
    """A recorded outcome must leave the queue and move the KPIs at once,
    and stay that way after a page reload or an API restart."""
    from vidyutrakshak.api import app
    from vidyutrakshak.pipeline import DASH
    with TestClient(app) as c:
        before = c.get("/api/dashboard").json()
        top = before["queue"][0]
        cid = top["id"]
        assert c.post("/api/feedback", json={"consumer_id": cid, "outcome": "theft_confirmed"}).json()["saved"]
        assert cid not in {i["consumer_id"] for i in c.get("/api/queue?capacity=1000").json()["items"]}
        assert c.get(f"/api/consumers/{cid}").json()["inspection_status"] == "theft_confirmed"
        after = c.get("/api/dashboard").json()
        assert cid not in {q["id"] for q in after["queue"]}
        assert after["meta"]["labels"] == before["meta"]["labels"] + 1
        assert after["meta"]["confirmed_theft"] == before["meta"]["confirmed_theft"] + 1
        if top["band"] == "High":
            assert after["kpis"]["high_uninspected"] == before["kpis"]["high_uninspected"] - 1
    assert cid not in {q["id"] for q in json.loads(DASH.read_text())["queue"]}
    with TestClient(app) as c:  # restart: state is reloaded from disk
        assert cid not in {i["consumer_id"] for i in c.get("/api/queue?capacity=1000").json()["items"]}


def test_assistant_answers_from_live_data(dash):
    """Offline rules engine: answers quote the same IDs and scores as the API."""
    from vidyutrakshak.api import app
    with TestClient(app) as c:
        top = c.get("/api/queue?capacity=1").json()["items"][0]
        cid, feeder = top["consumer_id"], top["feeder_id"]
        detail = c.get(f"/api/consumers/{cid}").json()

        def ask(q, lang="en"):
            r = c.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "lang": lang})
            assert r.status_code == 200
            return r.json()

        a = ask(f"Why was {cid} flagged?")
        assert a["mode"] == "rules" and cid in a["ids"]
        assert f"{detail['risk_score']:.0f}/100" in a["answer"] and detail["reasons"][0] in a["answer"]
        a = ask(f"{cid} શા માટે ફ્લેગ થયો?")
        assert detail["reasons_gu"][0] in a["answer"]
        a = ask(f"top 3 risky consumers on {feeder}")
        assert cid in a["answer"] and a["answer"].count(feeder) >= 1
        assert "unaccounted" in ask("Which DT has the highest loss?")["answer"]
        assert "share of inspections" in ask("What does hit rate mean?")["answer"]
        assert "revenue at risk" in ask("How much revenue is at risk?")["answer"].lower()
        assert "couldn't find" in ask("Why was C99999 flagged?")["answer"]
        # History is accepted and trimmed; an empty or oversized body is rejected.
        hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(15)]
        hist.append({"role": "user", "content": f"Tell me about {top['dt_id']}"})
        assert top["dt_id"] in c.post("/api/chat", json={"messages": hist}).json()["answer"]
        assert c.post("/api/chat", json={"messages": []}).status_code == 422


def test_assistant_claude_path_uses_tools_and_falls_back(dash, monkeypatch):
    """Claude mode with a stubbed client (no network): the tools return live data,
    and any API failure falls back to the offline answer instead of erroring."""
    from types import SimpleNamespace
    from vidyutrakshak import assistant, api
    calls = {}

    class Runner:
        def __init__(self, **kw):
            calls.update(kw)

        def until_done(self):
            tools = {t.name: t for t in calls["tools"]}
            out = json.loads(tools["top_consumers"].call({"n": 2}))
            missing = json.loads(tools["get_consumer"].call({"consumer_id": "C99999"}))
            assert "error" in missing
            return SimpleNamespace(stop_reason="end_turn", content=[
                SimpleNamespace(type="text", text=f"Top: {out[0]['consumer_id']}")])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=lambda **kw: Runner(**kw))))
    monkeypatch.setattr(assistant, "mode", lambda: "claude")
    monkeypatch.setattr(assistant, "_client", lambda: fake)
    with TestClient(api.app) as c:
        top = c.get("/api/queue?capacity=1").json()["items"][0]["consumer_id"]
        r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "who first?"}]}).json()
        assert r == {"answer": f"Top: {top}", "mode": "claude", "ids": [top]}
        assert calls["model"] == assistant.MODEL and len(calls["tools"]) == 8
        assert calls["system"][0]["cache_control"] == {"type": "ephemeral"}

        def boom(**kw):
            raise ConnectionError("offline")
        monkeypatch.setattr(assistant, "_client", lambda: SimpleNamespace(
            beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=boom))))
        r = c.post("/api/chat", json={"messages": [{"role": "user", "content": f"Why was {top} flagged?"}]}).json()
        assert r["mode"] == "rules" and top in r["answer"] and "AI assistant unavailable" in r["answer"]


def test_explainable_pdf_reports(dash):
    from vidyutrakshak.api import app
    with TestClient(app) as c:
        cid = c.get("/api/queue?capacity=1").json()["items"][0]["consumer_id"]
        for path in (f"/api/reports/consumer/{cid}.pdf", f"/api/reports/consumer/{cid}.pdf?lang=gu",
                     "/api/reports/queue.pdf?capacity=30", "/api/reports/queue.pdf?capacity=10&feeder=FDR-01&band=High",
                     "/api/reports/model.pdf"):
            r = c.get(path)
            assert r.status_code == 200, path
            assert r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF"), path
            assert len(r.content) > 5000, path
        assert c.get("/api/reports/consumer/NOPE.pdf").status_code == 404
        assert c.get(f"/api/reports/consumer/{cid}.pdf?lang=fr").status_code == 422


def test_assistant_and_reports_need_a_key(dash, monkeypatch):
    from vidyutrakshak.api import app
    monkeypatch.setenv("VR_API_KEYS", "viewer:v1ew")
    with TestClient(app) as c:
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert c.post("/api/chat", json=body).status_code == 401
        assert c.get("/api/reports/model.pdf").status_code == 401
        assert c.post("/api/chat", json=body, headers={"X-API-Key": "v1ew"}).status_code == 200
        assert c.get("/api/reports/model.pdf", headers={"X-API-Key": "v1ew"}).status_code == 200


def test_api_refuses_to_start_without_artifacts(dash, tmp_path, monkeypatch):
    """Startup must never train or wipe anything implicitly."""
    from vidyutrakshak import api
    monkeypatch.setattr(api, "MODELS", tmp_path / "models")
    monkeypatch.delenv("VR_AUTO_BUILD", raising=False)
    with pytest.raises(RuntimeError, match="python -m vidyutrakshak.pipeline"):
        api._load_state()
    assert not (tmp_path / "models").exists()


def test_rerunning_the_pipeline_is_non_destructive(dash):
    """A plain rerun keeps the registry and live outcomes; --rebuild keeps the
    live field outcomes and parks the old registry in models.prev/."""
    from vidyutrakshak import pipeline as P
    reg_path, live_path = P.MODELS / "registry.json", P.DATA / P.LIVE
    reg_before = json.loads(reg_path.read_text())
    live_before = pd.read_csv(live_path)
    n_field = int((live_before.source == "field_feedback").sum())
    assert n_field > 0 and len(reg_before["versions"]) > 4   # earlier tests recorded outcomes and retrained

    out = P.run(verbose=False)
    assert json.loads(reg_path.read_text()) == reg_before
    assert len(pd.read_csv(live_path)) == len(live_before)
    assert out["meta"]["active_model"] == reg_before["active"]

    P.run(rebuild=True, verbose=False)
    assert int((pd.read_csv(live_path).source == "field_feedback").sum()) == n_field
    assert [v["version"] for v in json.loads(reg_path.read_text())["versions"]] == ["v1", "v2", "v3", "v4"]
    prev = P.MODELS.with_name(P.MODELS.name + ".prev")
    assert json.loads((prev / "registry.json").read_text()) == reg_before
    assert len(pd.read_csv(prev / P.LIVE)) == len(live_before)
    assert not P.MODELS.with_name(P.MODELS.name + ".staging").exists()
