"""Validate the meter-data path of VidyutRakshak on real, labelled theft data.

Dataset: SGCC (State Grid Corporation of China) electricity theft dataset,
42,372 consumers, daily kWh 2014-01-01 .. 2016-10-31, FLAG=1 for confirmed
thieves. Public copy: github.com/henryRDlab/ElectricityTheftDetection
(split zip data.zip/.z01/.z02 -> data.csv). Put data.csv in external/sgcc/.

    python scripts/validate_sgcc.py                 # last 365 and 730 days
    python scripts/validate_sgcc.py --days 365

What this does and does not show
- It runs the *same* feature code the prototype uses (peer comparison,
  change-point, low-read days, flatness, capping, missing reads, Isolation
  Forest), then 5-fold cross-validated LightGBM on the real labels.
- SGCC has no transformer meters, tariff categories, sanctioned load or tamper
  logs, so the DT energy balance and those signals are absent. Peers are the
  whole population rather than same-DT neighbours. This is a lower bound for
  the meter-only path, not a test of the full system.
- SGCC labels have no theft start date, so long-running theft (no change point
  in the window) is included -- a harder case than the simulator's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidyutrakshak import features as fx  # noqa: E402
from vidyutrakshak.model import LGB_PARAMS, add_anomaly_score  # noqa: E402

SRC = ROOT / "external" / "sgcc" / "data.csv"
OUT = ROOT / "docs" / "sgcc_validation.json"
CAPACITY = 0.04  # inspect the top 4%, as in the prototype

# Signals that exist in SGCC (no DT, tamper, tariff or sanctioned load).
SGCC_FEATURES = ["peer_ratio_change", "peer_level_recent", "cp_drop_ratio", "cp_weeks_ago", "low_day_frac",
                 "flatness_change", "cap_frac", "missing_frac", "zero_day_frac", "anomaly_score"]


def load_sgcc(days: int) -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(SRC)
    raw = raw.drop_duplicates("CONS_NO").set_index("CONS_NO")
    y = raw.pop("FLAG").to_numpy().astype(int)
    raw.columns = pd.to_datetime(raw.columns, format="%Y/%m/%d")
    raw = raw.sort_index(axis=1).astype("float32")
    raw = raw.iloc[:, -days:]
    raw[raw < 0] = np.nan
    raw.index = raw.index.astype(str)
    return raw, y


def build(reads: pd.DataFrame) -> pd.DataFrame:
    n, cols = len(reads), reads.columns.strftime("%Y-%m-%d")
    reads = reads.set_axis(cols, axis=1)
    cons = pd.DataFrame({"consumer_id": reads.index, "dt_id": "DT-NA", "feeder_id": "F-NA",
                         "category": "RES", "sanctioned_kw": np.nan})
    ds = fx.Dataset(
        consumers=cons, dts=pd.DataFrame({"dt_id": ["DT-NA"]}), feeders=pd.DataFrame({"feeder_id": ["F-NA"]}),
        reads=reads, events=pd.DataFrame(np.zeros((n, len(cols)), dtype=np.int8), index=reads.index, columns=cols),
        dt_reads=pd.DataFrame(columns=cols, dtype=float), inspections=pd.DataFrame(),
    )
    F, ctx = fx.build_features(ds)
    X = ctx["X"]
    F["zero_day_frac"] = np.nan_to_num(X[:, -fx.RECENT_DAYS:] == 0).mean(axis=1)
    return add_anomaly_score(F)


def weekly_raw(reads: pd.DataFrame) -> pd.DataFrame:
    """Black-box reference: the raw series as weekly means, scaled per consumer."""
    X = reads.to_numpy(dtype=float)
    T = X.shape[1] // 7 * 7
    with np.errstate(all="ignore"):
        W = np.nanmean(X[:, :T].reshape(len(X), -1, 7), axis=2)
        W = W / np.nanmean(W, axis=1, keepdims=True)
    return pd.DataFrame(W, columns=[f"w{i:02d}" for i in range(W.shape[1])])


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    k = int(CAPACITY * len(y))
    top = np.argsort(-s)[:k]
    p = float(y[top].mean())
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "precision_at_k": p, "recall_at_k": float(y[top].sum() / y.sum()), "k": k,
            "lift_vs_random": p / float(y.mean())}


def cv_lgb(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        b = lgb.train({**LGB_PARAMS, "num_leaves": 31, "min_child_samples": 40, "seed": seed},
                      lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=400)
        oof[te] = b.predict(X.iloc[te])
    return oof


def run_window(days: int) -> dict:
    t0 = time.time()
    reads, y = load_sgcc(days)
    print(f"• SGCC: {len(y):,} consumers, {reads.shape[1]} days, {y.sum():,} thieves ({100 * y.mean():.1f}%)")
    F = build(reads)
    raw = weekly_raw(reads)
    rows = []

    def add(name, s, note=""):
        m = metrics(y, s)
        rows.append({"approach": name, "note": note, **m})
        print(f"  {name:<46} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
              f"P@top{100 * CAPACITY:.0f}%={m['precision_at_k']:.2f} lift={m['lift_vs_random']:.1f}x")

    add("Random inspection", np.random.default_rng(0).random(len(y)), "base rate")
    add("Rule: flag >50% consumption drop", 1 - F.cp_drop_ratio.to_numpy())
    add("Anomaly detection only (no labels)", F.anomaly_score.to_numpy())
    add("Black-box: LightGBM on raw weekly series", cv_lgb(raw, y), "reference, not explainable")
    add("VidyutRakshak meter-only signals (explainable)", cv_lgb(F[SGCC_FEATURES].reset_index(drop=True), y),
        "same feature code as the prototype")
    both = pd.concat([F[SGCC_FEATURES].reset_index(drop=True), raw], axis=1)
    add("VidyutRakshak signals + raw weekly series", cv_lgb(both, y), "upper reference")
    return {"days": int(reads.shape[1]), "consumers": int(len(y)), "thieves": int(y.sum()),
            "window": [str(reads.columns[0].date()), str(reads.columns[-1].date())],
            "results": rows, "runtime_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[365, 730])
    args = ap.parse_args()
    if not SRC.exists():
        sys.exit(f"Missing {SRC}. Download the SGCC data first (see the docstring).")
    runs = [run_window(d) for d in args.days]
    out = {"dataset": "SGCC electricity theft (State Grid Corporation of China)",
           "source": "github.com/henryRDlab/ElectricityTheftDetection",
           "evaluation": "5-fold stratified cross-validation; top 4% = inspection capacity",
           "limits": "No DT meters, tariff category, sanctioned load or tamper logs in SGCC; peers are the whole "
                     "population. Lower bound for the meter-only path, not the full system.",
           "runs": runs}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"• wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
