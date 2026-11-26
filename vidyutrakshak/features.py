"""Feature engineering: consumer behaviour, peer comparison, data quality,
MDMS events, and DT energy-balance coupling.

Key idea: compare every consumer to same-DT, same-tariff peers on the same
day. That cancels seasonality (summer AC, Rabi pumping, Sunday closures), so
a drop that peers don't share stands out. Then ask whether the DT's
unaccounted energy stepped up at the same time: theft makes energy vanish
from the DT balance, a vacant house or rooftop solar does not.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DAYS = 90      # reference window at the start of the history
RECENT_DAYS = 90    # window being scored
EVENT_DAYS = 120

TARIFF_INR = {"RES": 5.5, "COM": 8.0, "AGR": 2.5, "IND": 7.5}  # indicative ₹/kWh

FEATURES = [
    "peer_ratio_change", "cp_drop_ratio", "cp_weeks_ago", "low_day_frac",
    "flatness_change", "cap_frac", "missing_frac", "tamper_events",
    "load_utilisation", "dt_loss_rise", "dt_step_at_drop", "dt_step_match",
    "anomaly_score",
]


@dataclass
class Dataset:
    consumers: pd.DataFrame
    dts: pd.DataFrame
    feeders: pd.DataFrame
    reads: pd.DataFrame
    events: pd.DataFrame
    dt_reads: pd.DataFrame
    inspections: pd.DataFrame


def load(data_dir: str | Path = "data") -> Dataset:
    d = Path(data_dir)
    return Dataset(
        consumers=pd.read_csv(d / "consumers.csv"),
        dts=pd.read_csv(d / "dts.csv"),
        feeders=pd.read_csv(d / "feeders.csv"),
        reads=pd.read_parquet(d / "meter_reads.parquet"),
        events=pd.read_parquet(d / "tamper_events.parquet"),
        dt_reads=pd.read_parquet(d / "dt_reads.parquet"),
        inspections=pd.read_csv(d / "inspections.csv"),
    )


def _peer_median(X: np.ndarray, cons: pd.DataFrame) -> np.ndarray:
    """Daily median of same-DT/same-category peers, falling back to feeder
    then category level when a peer group is too small."""
    peer = np.full_like(X, np.nan)
    for level in (["category"], ["feeder_id", "category"], ["dt_id", "category"]):
        for _, g in cons.groupby(level):
            if len(g) < 6:
                continue
            idx = g.index.values
            with np.errstate(all="ignore"):
                peer[idx] = np.nanmedian(X[idx], axis=0)
    return np.maximum(peer, 1e-3)


def _weekly(a: np.ndarray) -> np.ndarray:
    T = a.shape[-1] // 7 * 7
    with np.errstate(all="ignore"):
        return np.nanmean(a[..., :T].reshape(*a.shape[:-1], -1, 7), axis=-1)


def build_features(ds: Dataset) -> tuple[pd.DataFrame, dict]:
    cons = ds.consumers.reset_index(drop=True)
    # float32 halves memory for the consumer x day matrices (1 lakh x 365 = 146 MB each).
    X = ds.reads.loc[cons.consumer_id].to_numpy(dtype=np.float32)
    N, T = X.shape
    missing = np.isnan(X)

    peer = _peer_median(X, cons)
    with np.errstate(all="ignore"):
        ratio = X / peer
    # Impute gaps from the consumer's own typical peer ratio (not zero!) so a
    # communication failure is never mistaken for a consumption drop.
    med_ratio = np.nanmedian(ratio, axis=1, keepdims=True)
    X_imp = np.where(missing, peer * med_ratio, X)
    ratio = np.where(missing, med_ratio, ratio)

    base, recent = slice(0, BASE_DAYS), slice(T - RECENT_DAYS, T)
    pr_base = np.median(ratio[:, base], axis=1)
    pr_recent = np.median(ratio[:, recent], axis=1)

    # Single change-point on weekly peer ratio: largest sustained drop.
    W = _weekly(ratio)
    nW = W.shape[1]
    best, best_k = np.ones(N), np.zeros(N, dtype=int)
    for k in range(8, nW - 3):
        before = W[:, max(0, k - 12):k].mean(axis=1)
        after = W[:, k:].mean(axis=1)
        r = after / np.maximum(before, 1e-6)
        upd = r < best
        best[upd], best_k[upd] = r[upd], k

    # Behaviour signatures in the recent window.
    exp_recent = pr_base[:, None] * peer[:, recent]
    xr = X_imp[:, recent]
    low_day_frac = (xr < 0.3 * exp_recent).mean(axis=1)
    rr = ratio[:, recent]
    rb = ratio[:, base]
    cv = lambda a: a.std(axis=1) / np.maximum(a.mean(axis=1), 1e-6)
    flatness_change = cv(rr) / np.maximum(cv(rb), 1e-6)
    hi = np.quantile(xr, 0.98, axis=1, keepdims=True)
    cap_frac = (xr >= 0.97 * hi).mean(axis=1)
    load_util = xr.mean(axis=1) / (cons.sanctioned_kw.values * 24)

    ev = ds.events.loc[cons.consumer_id].to_numpy()
    tamper = ev[:, T - EVENT_DAYS:].sum(axis=1)

    # ---------- DT energy balance ----------
    # DTs without a working boundary meter (common in the field) get NaN DT
    # features; the model treats them as unknown, not as "no loss".
    dt_ids = ds.dts.dt_id.values
    dt_pos = {d: k for k, d in enumerate(dt_ids)}
    cdt = cons.dt_id.map(dt_pos).values
    billed = np.zeros((len(dt_ids), T))
    np.add.at(billed, cdt, X_imp)
    D = ds.dt_reads.reindex(dt_ids).to_numpy(dtype=float)
    metered = np.isfinite(D).mean(axis=1) >= 0.5
    D[~metered] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows = unmetered DTs
        D = np.where(np.isnan(D), billed * np.nanmedian(D / billed, axis=1, keepdims=True), D)
        loss = (D - billed) / D                                   # daily loss fraction
        loss_base = np.median(loss[:, base], axis=1)
        loss_recent = np.median(loss[:, recent], axis=1)
    dt_loss_rise = loss_recent - loss_base

    # Did the consumer's DT lose energy when the consumer's reads dropped?
    # Vectorised over consumers: windows of up to 8 weeks either side of each
    # consumer's change-point, built with cumulative sums.
    Lw, Dw = _weekly(loss), _weekly(D)
    gap_w = _weekly(pr_base[:, None] * peer) - _weekly(X_imp)        # consumer shortfall, kWh/day
    k = best_k
    lo, hi_ = np.maximum(0, k - 8), np.minimum(nW, k + 8)
    rows_c, rows_d = np.arange(N), cdt

    def win_mean(A, rows, a, b):
        c = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)
        return (c[rows, b] - c[rows, a]) / np.maximum(b - a, 1)

    with np.errstate(all="ignore"):
        dl = win_mean(Lw, rows_d, k, hi_) - win_mean(Lw, rows_d, lo, k)   # loss step, fraction
        lost_kwh = dl * win_mean(Dw, rows_d, k, hi_)                      # DT energy vanished / day
        drop_kwh = win_mean(gap_w, rows_c, k, hi_) - np.maximum(0.0, win_mean(gap_w, rows_c, lo, k))
        step_pp = np.where(k > 0, dl * 100, 0.0)
        step_match = np.where(k > 0, np.clip(lost_kwh / np.maximum(drop_kwh, 0.5), -2, 3), 0.0)
    unmetered = ~metered[cdt]
    step_pp[unmetered] = np.nan
    step_match[unmetered] = np.nan

    deficit_recent = np.clip(exp_recent - xr, 0, None).sum(axis=1)

    F = pd.DataFrame({
        "consumer_id": cons.consumer_id,
        "dt_id": cons.dt_id,
        "feeder_id": cons.feeder_id,
        "category": cons.category,
        "sanctioned_kw": cons.sanctioned_kw,
        "peer_ratio_change": pr_recent / np.maximum(pr_base, 1e-6),
        # Level, not change: catches long-running theft that has no change point.
        "peer_level_recent": pr_recent,
        "cp_drop_ratio": best,
        "cp_weeks_ago": np.where(best_k > 0, nW - best_k, 0),
        "low_day_frac": low_day_frac,
        "flatness_change": flatness_change,
        "cap_frac": cap_frac,
        "missing_frac": missing[:, recent].mean(axis=1),
        "tamper_events": tamper,
        "load_utilisation": load_util,
        "dt_loss_rise": dt_loss_rise[cdt],
        "dt_step_at_drop": step_pp,
        "dt_step_match": step_match,
        "recent_kwh_day": xr.mean(axis=1),
        "expected_kwh_day": exp_recent.mean(axis=1),
        "deficit_kwh_90d": deficit_recent,
    })

    ctx = {
        "dates": ds.reads.columns.tolist(),
        "X": X, "X_imp": X_imp, "peer": peer, "pr_base": pr_base,
        "dt_ids": dt_ids, "dt_loss": loss, "dt_reads": D, "dt_billed": billed,
        "loss_base": loss_base, "loss_recent": loss_recent,
    }
    return F, ctx


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index, used for drift monitoring."""
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, qs[0], qs[-1]), qs)[0] / len(expected)
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
