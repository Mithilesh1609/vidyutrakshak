"""Stress tests: how the model holds up when field reality is messier than the
baseline simulation.

    python scripts/stress_test.py            # writes docs/stress_tests.json

Each scenario regenerates the network (5 seeds), runs the same protocol as the
pipeline (initial vigilance records + 3 feedback rounds of 60 inspections from
the model's queue), then scores a holdout area the loop never touches.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidyutrakshak import features as fx  # noqa: E402
from vidyutrakshak.model import FEATURES, NO_DT_FEATURES, add_anomaly_score, predict, train  # noqa: E402
from vidyutrakshak.pipeline import BATCH, CAPACITY_SHARE, DROP_SLICE, FEEDBACK_ROUNDS  # noqa: E402
from vidyutrakshak.simulate import SimConfig, simulate  # noqa: E402

OUT = ROOT / "docs" / "stress_tests.json"
SEEDS = (7, 8, 9, 10, 11)

SCENARIOS = [
    ("Baseline", "Simulated network as in the demo", {}),
    ("Long-running theft", "Half of all thefts began before the data starts (no change point)", {"chronic_theft_share": 0.5}),
    ("Subtle theft", "Half of all thefts bypass only 10-25% of load", {"subtle_theft_share": 0.5}),
    ("Half the DTs unmetered", "Only 50% of transformers have a working boundary meter", {"dt_meter_coverage": 0.5}),
    ("No DT meters", "No transformer boundary meters at all", {"dt_meter_coverage": 0.0}),
    ("GIS errors", "10% of consumers indexed to the wrong DT", {"gis_mistag_rate": 0.10}),
    ("Poor MDMS data", "20% of meters with communication gaps (vs 5%)", {"comm_fault_rate": 0.20}),
]
VARIANTS = {
    "meter_only": NO_DT_FEATURES,
    "current": FEATURES,
    "current_plus_level": FEATURES + ["peer_level_recent"],
}


def run_once(cfg: SimConfig) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp)
        ds = fx.load(tmp)
        gt = pd.read_csv(Path(tmp) / "_ground_truth.csv").set_index("consumer_id")
    F, _ = fx.build_features(ds)
    F = add_anomaly_score(F)
    y = gt.loc[F.consumer_id, "is_theft"].to_numpy()
    chronic = (gt.loc[F.consumer_id, "theft_start_day"].to_numpy() == 0) & (y == 1)
    legit = gt.loc[F.consumer_id, "confounder"].isin(["vacancy", "rooftop_solar"]).to_numpy()

    insp = ds.inspections.copy()
    rng = np.random.default_rng(5)
    unins = F.consumer_id[~F.consumer_id.isin(insp.consumer_id)].to_numpy()
    hold = F.consumer_id.isin(set(rng.choice(unins, int(0.4 * len(unins)), replace=False))).to_numpy()
    k = max(10, int(CAPACITY_SHARE * hold.sum()))
    drop = (F.cp_drop_ratio.to_numpy() < 1 - DROP_SLICE)[hold]

    for r in range(FEEDBACK_ROUNDS):  # same loop as the pipeline, with the current model
        m = train(F, insp, FEATURES, calibrate=False)
        cand = F[~hold & ~F.consumer_id.isin(insp.consumer_id)].copy()
        cand["p"] = predict(m, cand)
        picked = cand.nlargest(BATCH, "p").consumer_id
        insp = pd.concat([insp, pd.DataFrame({
            "consumer_id": picked, "source": "model_queue", "round": r + 1,
            "outcome": np.where(gt.loc[picked, "is_theft"].to_numpy() == 1, "theft_confirmed", "no_irregularity")})],
            ignore_index=True)

    out = {}
    yh, lh, ch = y[hold], legit[hold], chronic[hold]
    for name, feats in VARIANTS.items():
        runs = []
        for seed in range(3):
            s = predict(train(F, insp, feats, seed=seed, calibrate=False), F[hold])
            top = np.argsort(-s)[:k]
            yd, sd, ld = yh[drop], s[drop], lh[drop]
            topd = np.argsort(-sd)[:max(1, int(yd.sum()))]
            runs.append({
                "hit_rate": yh[top].mean(), "recall": yh[top].sum() / max(yh.sum(), 1),
                "pr_auc": average_precision_score(yh, s), "legit_visits": lh[top].sum(),
                "drop_slice_legit_in_top": ld[topd].sum(),
                "chronic_recall": ch[top].sum() / ch.sum() if ch.sum() else np.nan,
            })
        out[name] = {m: float(np.nanmean([r[m] for r in runs])) for m in runs[0]}
    out["k"], out["base_rate"] = k, float(yh.mean())
    return out


def main():
    t0 = time.time()
    rows = []
    for name, desc, knobs in SCENARIOS:
        per_seed = [run_once(replace(SimConfig(seed=s), **knobs)) for s in SEEDS]
        avg = {v: {m: float(np.nanmean([p[v][m] for p in per_seed])) for m in per_seed[0][v]} for v in VARIANTS}
        row = {"scenario": name, "description": desc, "knobs": knobs, "k": per_seed[0]["k"],
               "base_rate": float(np.mean([p["base_rate"] for p in per_seed])), **avg}
        rows.append(row)
        c = avg["current"]
        print(f"  {name:<24} hit={c['hit_rate']:.2f} PR-AUC={c['pr_auc']:.2f} legit={c['legit_visits']:.1f} "
              f"hard-case legit={c['drop_slice_legit_in_top']:.1f} (meter-only {avg['meter_only']['drop_slice_legit_in_top']:.1f}) "
              f"| meter-only hit={avg['meter_only']['hit_rate']:.2f} | +level hit={avg['current_plus_level']['hit_rate']:.2f} "
              f"PR-AUC={avg['current_plus_level']['pr_auc']:.2f}"
              + (f" | chronic recall {c['chronic_recall']:.2f} -> {avg['current_plus_level']['chronic_recall']:.2f}"
                 if not np.isnan(c["chronic_recall"]) else ""))
    OUT.parent.mkdir(exist_ok=True)
    clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else
                       [clean(v) for v in o] if isinstance(o, list) else
                       None if isinstance(o, float) and np.isnan(o) else o)   # strict JSON: NaN -> null
    OUT.write_text(json.dumps(clean({"protocol": __doc__.strip().splitlines()[-3:], "seeds": list(SEEDS),
                                     "scenarios": rows, "runtime_s": round(time.time() - t0)}), indent=2))
    print(f"• wrote {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
