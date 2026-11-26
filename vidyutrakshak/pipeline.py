"""End-to-end pipeline: data -> features -> models -> feedback rounds ->
scores -> dashboard export.

    python -m vidyutrakshak.pipeline            # build if nothing is built yet, otherwise
                                                # rescore with the active model (never deletes)
    python -m vidyutrakshak.pipeline --rebuild  # retrain v1.. from scratch, keep live field outcomes
    python -m vidyutrakshak.pipeline --fresh    # demo reset: regenerate synthetic data, models, feedback

--rebuild and --fresh build the new registry in a staging folder and swap it
in only once it loads; the previous registry and live outcomes are kept in
`<models>.prev/`.

Real data: write it with `python -m vidyutrakshak.ingest ... --out DIR`, then
    VR_DATA_DIR=DIR python -m vidyutrakshak.pipeline
With no simulator ground truth in the folder, the pipeline trains on the
vigilance records, reports cross-validated quality, and publishes the console.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import features as fx
from .model import (FEATURES, NO_DT_FEATURES, Registry, add_anomaly_score, area_scores,
                    cv_oof, eval_ranking, predict, reliability, score, train)
from .simulate import SimConfig, simulate

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("VR_DATA_DIR", ROOT / "data")).resolve()
MODELS = Path(os.getenv("VR_MODELS_DIR", ROOT / "models")).resolve()
STATIC = ROOT / "static"
# Console data: next to the demo console by default; inside the data folder
# when VR_DATA_DIR points at a pilot, so a pilot run never overwrites the demo.
DASH = DATA / "dashboard_data.json" if os.getenv("VR_DATA_DIR") else STATIC / "dashboard_data.json"

FEEDBACK_ROUNDS = 3
BATCH = 60            # field inspections per feedback round
CAPACITY_SHARE = 0.04 # inspection capacity as share of consumers scored
DROP_SLICE = 0.25     # 'use fell by more than this' defines the hard-case slice


def _weekly(v: np.ndarray) -> list[float]:
    T = len(v) // 7 * 7
    w = v[:T].reshape(-1, 7)
    n = (~np.isnan(w)).sum(axis=1)
    m = np.where(n > 0, np.nansum(w, axis=1) / np.maximum(n, 1), np.nan)
    return [None if np.isnan(x) else round(float(x), 2) for x in m]


LIVE = "inspections_live.csv"


# ---------------------------------------------------------------- safe file state
def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def load_inspections(ds: fx.Dataset) -> pd.DataFrame:
    """All outcomes so far: the live log (initial records + field outcomes)
    when there is one, otherwise the initial vigilance records."""
    live = DATA / LIVE
    return pd.read_csv(live) if live.exists() else ds.inspections.copy()


def _field_rows(insp: pd.DataFrame) -> pd.DataFrame:
    return insp[insp.source == "field_feedback"] if "source" in insp else insp.iloc[:0]


def _staging() -> Path:
    st = MODELS.with_name(MODELS.name + ".staging")
    if st.exists():
        shutil.rmtree(st)           # left over from an interrupted build: never live
    return st


def _swap_in(staging: Path) -> Registry:
    """Make a fully built registry live. The old registry and the live
    outcome log are moved to <models>.prev/ rather than deleted."""
    Registry(staging).load()        # refuse to publish a registry that cannot load
    prev = MODELS.with_name(MODELS.name + ".prev")
    if MODELS.exists() or (DATA / LIVE).exists():
        if prev.exists():
            shutil.rmtree(prev)
        if MODELS.exists():
            os.replace(MODELS, prev)
        prev.mkdir(parents=True, exist_ok=True)
        if (DATA / LIVE).exists():
            shutil.copy2(DATA / LIVE, prev / LIVE)
    os.replace(staging, MODELS)
    return Registry(MODELS)


def _is_built() -> bool:
    return (MODELS / "registry.json").exists()


def run(fresh: bool = False, verbose: bool = True, rebuild: bool = False) -> dict:
    """Default: build only when nothing is built; otherwise rescore with the
    active model and the live outcomes (non-destructive). rebuild=True
    retrains from scratch keeping live field outcomes; fresh=True resets the
    simulated demo (new data, v1..v4, no live outcomes)."""
    log = print if verbose else (lambda *a, **k: None)
    meta_path = DATA / "meta.json"
    is_real = meta_path.exists() and json.loads(meta_path.read_text()).get("simulated") is False
    if fresh and is_real:
        raise SystemExit(f"{DATA} holds real (ingested) data; --fresh would overwrite it with simulated data.")
    if not fresh and not rebuild and meta_path.exists() and _is_built():
        return refresh(log)
    if fresh or not meta_path.exists():
        log("• simulating DISCOM data …")
        simulate(SimConfig(), DATA)
    if not (DATA / "_ground_truth.csv").exists():
        return run_real(log)
    staging = _staging()
    reg = Registry(staging)

    ds = fx.load(DATA)
    F, ctx = fx.build_features(ds)
    F = add_anomaly_score(F)
    gt = pd.read_csv(DATA / "_ground_truth.csv").set_index("consumer_id")
    y_all = gt.loc[F.consumer_id, "is_theft"].to_numpy()

    insp = ds.inspections.copy()
    rng = np.random.default_rng(5)
    # A fixed holdout area that is never inspected by the loop: every model
    # version is scored on it against ground truth, so gains are comparable.
    uninspected = F.consumer_id[~F.consumer_id.isin(insp.consumer_id)].to_numpy()
    holdout = set(rng.choice(uninspected, int(0.4 * len(uninspected)), replace=False))
    is_hold = F.consumer_id.isin(holdout).to_numpy()
    k_hold = max(10, int(CAPACITY_SHARE * is_hold.sum()))
    legit = gt.loc[F.consumer_id, "confounder"].isin(["vacancy", "rooftop_solar"]).to_numpy()
    # The hard cases: consumers whose use fell >25%. Thieves, vacant homes and
    # rooftop-solar homes all look alike here; the DT balance is what separates them.
    drop_h = (F.cp_drop_ratio.to_numpy() < 1 - DROP_SLICE)[is_hold]

    def evaluate(scores: np.ndarray) -> dict:
        met = eval_ranking(y_all[is_hold], scores, k_hold)
        met["eval"] = "holdout"
        top = np.argsort(-scores)[:k_hold]
        met["legit_drop_false_alarms_at_k"] = int(legit[is_hold][top].sum())
        yd, sd, ld = y_all[is_hold][drop_h], scores[drop_h], legit[is_hold][drop_h]
        kd = max(1, int(yd.sum()))
        topd = np.argsort(-sd)[:kd]
        met["drop_slice"] = {"n": int(drop_h.sum()), "thefts": int(yd.sum()), "legit": int(ld.sum()), "k": kd,
                             "pr_auc": float(average_precision_score(yd, sd)),
                             "hit_rate": float(yd[topd].mean()), "legit_in_top": int(ld[topd].sum())}
        return met

    # ---------- inspection-feedback loop ----------
    gtl = gt["is_theft"]
    for r in range(FEEDBACK_ROUNDS + 1):
        m = train(F, insp, FEATURES)
        met = evaluate(predict(m, F[is_hold]))
        note = "initial vigilance records" if r == 0 else f"after feedback round {r} (+{BATCH} field outcomes)"
        v = reg.save(m, met, note, trained_on_rows=len(insp))
        log(f"• {v}: {m['n_labels']} labels ({m['n_pos']} theft) → holdout P@{k_hold}={met['precision_at_k']:.2f} "
            f"PR-AUC={met['pr_auc']:.2f}")
        if r == FEEDBACK_ROUNDS:
            break
        # Field teams inspect the model's top-ranked, not-yet-inspected
        # consumers in the operational area; outcomes come back as labels.
        cand = F[~is_hold & ~F.consumer_id.isin(insp.consumer_id)].copy()
        cand["p"] = predict(m, cand)
        picked = cand.nlargest(BATCH, "p").consumer_id
        insp = pd.concat([insp, pd.DataFrame({
            "consumer_id": picked,
            "outcome": np.where(gtl.loc[picked].to_numpy() == 1, "theft_confirmed", "no_irregularity"),
            "source": "model_queue", "round": r + 1,
        })], ignore_index=True)

    # ---------- ablation on the holdout area, final labels, 5 seeds ----------
    def avg(runs: list[dict]) -> dict:
        return {k: (avg([r[k] for r in runs]) if isinstance(runs[0][k], dict) else float(np.mean([r[k] for r in runs])))
                for k in runs[0] if k != "eval"}

    def ml_runs(feats):
        runs = []
        for seed in range(5):
            mm = train(F, insp, feats, seed=seed, calibrate=False)  # ranking only
            runs.append(evaluate(predict(mm, F[is_hold])))
        return avg(runs)

    rand_runs = [evaluate(np.random.default_rng(s).random(is_hold.sum())) for s in range(50)]
    ablation = [
        {"approach": "Random inspection (status quo)", **avg(rand_runs)},
        {"approach": "Rule: flag >50% consumption drop", **evaluate(1 - F.cp_drop_ratio.to_numpy()[is_hold])},
        {"approach": "Anomaly detection only (no labels)", **evaluate(F.anomaly_score.to_numpy()[is_hold])},
        {"approach": "ML on meter data only", **ml_runs(NO_DT_FEATURES)},
        {"approach": "VidyutRakshak: ML + DT energy balance", **ml_runs(FEATURES)},
    ]
    for a in ablation:
        log(f"  {a['approach']:<40} P@{k_hold}={a['precision_at_k']:.2f} R@k={a['recall_at_k']:.2f} "
            f"PR-AUC={a['pr_auc']:.2f} lift={a['lift_vs_random']:.1f}x "
            f"legit-drop FPs={a['legit_drop_false_alarms_at_k']:.1f} | use fell >25%: "
            f"PR-AUC={a['drop_slice']['pr_auc']:.2f} vacant/solar in top-{a['drop_slice']['k']:.0f}="
            f"{a['drop_slice']['legit_in_top']:.1f}")

    m = reg.load()
    drift, drift_summary = _drift(ds, F, m)

    # Does "risk 80" mean ~80% likely theft? Check on the holdout area.
    calibration = reliability(y_all[is_hold], predict(m, F[is_hold]))
    log("• calibration (holdout): " + ", ".join(f"{b['predicted']:.2f}->{b['observed']:.2f} (n={b['n']})"
                                                 for b in calibration["bins"]) + f", Brier {calibration['brier']:.4f}")

    # Field outcomes recorded in the console are real work: a rebuild keeps
    # them (after the ablation, so the evaluation stays clean). They count as
    # new outcomes for the promotion gate at the next retrain.
    if rebuild and not fresh and (DATA / LIVE).exists():
        field = _field_rows(pd.read_csv(DATA / LIVE))
        if len(field):
            insp = pd.concat([insp, field], ignore_index=True)
            log(f"• kept {len(field)} field outcome(s) from the live log")
    reg = _swap_in(staging)
    atomic_csv(insp, DATA / LIVE)
    out = publish_state(F, ctx, ds, reg, insp, extras={
        "ablation": ablation, "drift": {"features": drift, **drift_summary}, "capacity_k": k_hold,
        "calibration": calibration})
    log(f"• scored {out['kpis']['consumers']} consumers with {out['meta']['active_model']}: "
        f"{out['kpis']['high']} high, {out['kpis']['medium']} medium, {out['kpis']['data_issues']} data issues")
    log(f"• drift: max feature PSI {drift[0]['psi']} ({drift[0]['feature']}), status {drift_summary['status']}")
    return out


def _drift(ds: fx.Dataset, F: pd.DataFrame, m: dict) -> tuple[list[dict], dict]:
    """Population Stability Index per feature, this month vs last month."""
    T = ds.reads.shape[1]
    ds_prev = fx.Dataset(ds.consumers, ds.dts, ds.feeders, ds.reads.iloc[:, :T - 30],
                         ds.events.iloc[:, :T - 30], ds.dt_reads.iloc[:, :T - 30], ds.inspections)
    F_prev, _ = fx.build_features(ds_prev)
    F_prev = add_anomaly_score(F_prev)
    drift = []
    for f in FEATURES:
        a, b = F_prev[f].to_numpy(), F[f].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        drift.append({"feature": f, "psi": round(fx.psi(a[ok], b[ok]), 3) if ok.sum() > 50 else 0.0})
    drift.sort(key=lambda d: -d["psi"])
    summary = {"score_psi": round(fx.psi(predict(m, F_prev), predict(m, F)), 3),
               "status": "stable" if max(d["psi"] for d in drift) < 0.25 else "review"}
    return drift, summary


def run_real(log=print) -> dict:
    """Real (ingested) data: no ground truth, so no holdout or simulated
    feedback rounds. Train on the vigilance records, measure quality by
    cross-validation on them, score everyone and publish the console.
    Live outcomes recorded in the console are always kept."""
    staging = _staging()
    reg = Registry(staging)
    ds = fx.load(DATA)
    F, ctx = fx.build_features(ds)
    F = add_anomaly_score(F)
    insp = load_inspections(ds)
    L = F.merge(insp.drop_duplicates("consumer_id", keep="last"), on="consumer_id")
    n_pos, n_neg = int((L.outcome == "theft_confirmed").sum()), int((L.outcome != "theft_confirmed").sum())
    if n_pos < 10 or n_neg < 10:
        raise SystemExit(f"Need at least 10 confirmed thefts and 10 clean inspections to train; found {n_pos} and {n_neg}.")
    L, oof = cv_oof(F, insp, FEATURES)
    k = max(5, int(0.2 * len(L)))
    met = {**eval_ranking(L.y.to_numpy(), oof, k), "eval": "cv"}
    m = train(F, insp, FEATURES)
    reg.save(m, met, f"initial vigilance records ({len(L)} outcomes)", trained_on_rows=len(insp))
    log(f"• trained on {len(L)} inspections ({n_pos} thefts): cross-validated PR-AUC {met['pr_auc']:.2f}, "
        f"top-{k} hit rate {met['precision_at_k']:.2f}")
    drift, drift_summary = _drift(ds, F, reg.load())
    reg = _swap_in(staging)
    atomic_csv(insp, DATA / LIVE)
    out = publish_state(F, ctx, ds, reg, insp, extras={
        "ablation": [], "drift": {"features": drift, **drift_summary},
        "capacity_k": max(10, int(CAPACITY_SHARE * len(F))), "calibration": None})
    log(f"• scored {out['kpis']['consumers']:,} consumers: {out['kpis']['high']} high, {out['kpis']['medium']} medium, "
        f"{out['kpis']['data_issues']} data issues")
    return out


def saved_extras() -> dict:
    """Evaluation results from the last build (ablation, calibration, ...),
    which a rescore keeps rather than recomputes."""
    d = json.loads(DASH.read_text()) if DASH.exists() else {}
    return {"ablation": d.get("ablation", []), "drift": d.get("drift", {}),
            "capacity_k": d.get("meta", {}).get("capacity_k"), "calibration": d.get("calibration")}


def refresh(log=print) -> dict:
    """Non-destructive: rescore everyone with the active model and the live
    outcomes, republish the console. Models and outcomes are untouched."""
    reg = Registry(MODELS)
    ds = fx.load(DATA)
    F, ctx = fx.build_features(ds)
    F = add_anomaly_score(F)
    insp = load_inspections(ds)
    extras = saved_extras()
    if extras["capacity_k"] is None:
        extras["capacity_k"] = max(10, int(CAPACITY_SHARE * len(F)))
    drift, drift_summary = _drift(ds, F, reg.load())
    extras["drift"] = {"features": drift, **drift_summary}
    out = publish_state(F, ctx, ds, reg, insp, extras)
    log(f"• models already built: rescored {out['kpis']['consumers']:,} consumers with active "
        f"{out['meta']['active_model']} and {out['meta']['labels']} outcomes (nothing deleted). "
        f"Use --rebuild to retrain from scratch, or --fresh to reset the demo.")
    return out


def _headline(ablation: list[dict], versions: list[dict]) -> dict:
    """The numbers quoted in the console, deck and README, from one place.

    Hit rates come from the ablation (holdout area, ML rows averaged over 5
    training runs), not from a single model run, so they don't move when a
    model is retrained live. The v1 -> vN climb is a single run per version.
    """
    by = lambda p: next((a for a in ablation if a["approach"].startswith(p)), None)
    ours, rule, anom = by("VidyutRakshak"), by("Rule"), by("Anomaly")
    hv = [v for v in versions if v["metrics"].get("eval") == "holdout"]
    if not ours:
        return {}
    return {
        "k": int(ours["k"]), "holdout_n": int(ours["n"]),
        "base_rate": ours["positives"] / ours["n"],                     # what random checks find
        "hit_rate": ours["precision_at_k"], "recall": ours["recall_at_k"],
        "legit_visits": int(round(ours["legit_drop_false_alarms_at_k"])),
        "rule_hit_rate": rule["precision_at_k"] if rule else None,
        "rule_legit_visits": int(round(rule["legit_drop_false_alarms_at_k"])) if rule else None,
        "anomaly_hit_rate": anom["precision_at_k"] if anom else None,
        "drop_slice": ours.get("drop_slice"),
        "meter_only_drop_slice": (by("ML on meter") or {}).get("drop_slice"),
        "first_version": hv[0]["version"] if hv else None, "first_version_hit_rate": hv[0]["metrics"]["precision_at_k"] if hv else None,
        "last_version": hv[-1]["version"] if hv else None, "last_version_hit_rate": hv[-1]["metrics"]["precision_at_k"] if hv else None,
        "basis": "Holdout area never inspected by the feedback loop; ML rows averaged over 5 training runs.",
    }


def _with_layout(dts: pd.DataFrame) -> pd.DataFrame:
    """The network map needs coordinates. When GIS has none for a DT, place
    it on a schematic: feeders round a circle, their DTs clustered nearby."""
    d = dts.copy()
    for c in ("lat", "lng", "capacity_kva"):
        if c not in d:
            d[c] = np.nan
    miss = d.lat.isna() | d.lng.isna()
    if miss.any():
        feeders = sorted(d.feeder_id.unique())
        for fi, f in enumerate(feeders):
            idx = d.index[miss & (d.feeder_id == f)]
            ang = 2 * np.pi * fi / max(len(feeders), 1)
            for j, i in enumerate(idx):
                a2 = 2 * np.pi * j / max(len(idx), 1)
                d.loc[i, "lat"] = np.cos(ang) + 0.25 * np.cos(a2)
                d.loc[i, "lng"] = np.sin(ang) + 0.25 * np.sin(a2)
    return d


def publish_state(F, ctx, ds, reg: Registry, insp: pd.DataFrame, extras: dict) -> dict:
    """Score everyone with the active model and write scores + dashboard JSON."""
    out, S, dt = build_state(F, ctx, ds, reg, insp, extras)
    write_state(out, S, dt)
    return out


def write_state(out: dict, S: pd.DataFrame, dt: pd.DataFrame) -> None:
    """Persist scores, area scores and the console payload, each atomically."""
    atomic_csv(S.assign(reasons=S.reasons.map(" | ".join), reasons_gu=S.reasons_gu.map(" | ".join)), DATA / "scores.csv")
    atomic_csv(dt, DATA / "dt_scores.csv")
    atomic_write_text(DASH, json.dumps(out, default=float))


def build_state(F, ctx, ds, reg: Registry, insp: pd.DataFrame, extras: dict,
                raw: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Console payload + scored frame + DT scores, in memory.

    `raw` is the active model's scoring (score(m, F)). Pass it to skip
    rescoring when only the outcomes changed (e.g. after a field outcome);
    inspection status, queue and KPIs are always recomputed from `insp`.
    """
    if raw is None:
        raw = score(reg.load(), F)
    S = raw.copy()
    last = insp.drop_duplicates("consumer_id", keep="last").set_index("consumer_id")
    S["inspection_status"] = S.consumer_id.map(last.outcome).fillna("not_inspected")
    dt, fd = area_scores(S, ctx, _with_layout(ds.dts))

    dates = ctx["dates"]
    week_starts = [dates[i] for i in range(0, len(dates) // 7 * 7, 7)]
    pos = {c: i for i, c in enumerate(F.consumer_id)}
    queue = S[(S.inspection_status == "not_inspected")].nlargest(150, "risk_prob")
    # Metering team list: long MDMS gaps, plus meters a field officer reported faulty.
    faulty = set(insp.consumer_id[insp["recorded_outcome"].eq("meter_fault")]) if "recorded_outcome" in insp else set()
    S["meter_fault_reported"] = S.consumer_id.isin(faulty)
    dq = pd.concat([S[S.meter_fault_reported], S[S.data_issue & ~S.meter_fault_reported].nlargest(25, "missing_frac")])
    listing = reg.list()
    versions = listing["versions"]
    eval_metrics = next((v["metrics"] for v in reversed(versions) if v["metrics"].get("eval") == "holdout"),
                        versions[-1]["metrics"])
    headline = _headline(extras.get("ablation", []), versions)

    def c_row(r):
        i = pos[r.consumer_id]
        return {
            "id": r.consumer_id, "dt": r.dt_id, "feeder": r.feeder_id, "category": r.category,
            "sanctioned_kw": r.sanctioned_kw, "score": float(r.risk_score), "band": r.band,
            "reasons": r.reasons, "reasons_gu": r.reasons_gu, "tamper": int(r.tamper_events),
            "leak_kwh_month": float(r.est_leak_kwh_month), "revenue_inr_month": float(r.est_revenue_inr_month),
            "missing_pct": round(100 * float(r.missing_frac), 1), "status": r.inspection_status,
            "series": _weekly(ctx["X"][i]), "expected": _weekly(ctx["pr_base"][i] * ctx["peer"][i]),
        }

    dt_pos = {d: i for i, d in enumerate(ctx["dt_ids"])}
    high = S[S.band == "High"]
    outcomes = insp.drop_duplicates("consumer_id", keep="last")
    out = {
        "meta": {**json.loads((DATA / "meta.json").read_text()), "as_of": dates[-1],
                 "active_model": listing["active"], "week_starts": week_starts,
                 "capacity_k": extras.get("capacity_k"),
                 "labels": int(len(outcomes)), "confirmed_theft": int((outcomes.outcome == "theft_confirmed").sum())},
        "kpis": {
            "consumers": int(len(S)), "dts": int(len(dt)), "feeders": int(len(fd)),
            "high": int((S.band == "High").sum()), "medium": int((S.band == "Medium").sum()),
            "data_issues": int(S.data_issue.sum()),
            "high_uninspected": int(((S.band == "High") & (S.inspection_status == "not_inspected")).sum()),
            "revenue_at_risk_inr_month": float(high[high.inspection_status == "not_inspected"].est_revenue_inr_month.sum()),
            "leak_kwh_month_high": float(high[high.inspection_status == "not_inspected"].est_leak_kwh_month.sum()),
            "holdout": eval_metrics,
        },
        "queue": [c_row(r) for r in queue.itertuples()],
        "headline": headline,
        "data_quality": [{"id": r.consumer_id, "dt": r.dt_id, "missing_pct": round(100 * r.missing_frac, 1),
                          "score": float(r.risk_score), "field_fault": bool(r.meter_fault_reported)}
                         for r in dq.itertuples()],
        "dts": [{
            "id": r.dt_id, "feeder": r.feeder_id, "lat": r.lat, "lng": r.lng,
            "kva": None if pd.isna(r.capacity_kva) else int(r.capacity_kva),
            "consumers": int(r.consumers), "high": int(r.high_risk), "medium": int(r.medium_risk),
            "loss_base": round(r.loss_base_pct, 2), "loss_recent": round(r.loss_recent_pct, 2),
            "excess_kwh_day": round(r.excess_loss_kwh_day, 1), "risk": float(r.dt_risk_score),
            "loss_series": [round(100 * x, 2) for x in _weekly(ctx["dt_loss"][dt_pos[r.dt_id]])],
        } for r in dt.sort_values("dt_risk_score", ascending=False).itertuples()],
        "feeders": [{"id": r.feeder_id, "dts": int(r.dts), "consumers": int(r.consumers),
                     "high": int(r.high_risk), "excess_loss_pct": round(r.excess_loss_pct, 2),
                     "risk": round(r.mean_dt_risk, 0)} for r in fd.itertuples()],
        "models": versions,
        "ablation": extras.get("ablation", []),
        "drift": extras.get("drift", {}),
        "calibration": extras.get("calibration"),
    }
    return out, S, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fresh", action="store_true",
                   help="demo reset: regenerate synthetic data, retrain v1.., clear live outcomes")
    g.add_argument("--rebuild", action="store_true",
                   help="retrain from scratch on the current data, keeping live field outcomes")
    a = ap.parse_args()
    run(fresh=a.fresh, rebuild=a.rebuild)
