"""Risk engine: anomaly detection + gradient boosting on scarce labels,
TreeSHAP reason codes, prior-shift calibration, area roll-ups, and a
file-based model registry for the inspection-feedback retraining loop."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import FEATURES, TARIFF_INR
from .i18n import LANGS, render

UNSUP_FEATURES = [f for f in FEATURES if f != "anomaly_score"]
NO_DT_FEATURES = [f for f in FEATURES if not f.startswith("dt_")]
REASONS_MIN = 500   # always explain at least this many top-ranked consumers

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=7, min_child_samples=8,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    lambda_l2=2.0, verbose=-1, seed=11,
)


# ---------------------------------------------------------------- models
def add_anomaly_score(F: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """Unsupervised layer: works with zero labels, catches novel patterns."""
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    # Isolation Forest needs complete rows: drop features that are entirely
    # unknown (e.g. no DT meters at all) and fill the rest with the median.
    Zdf = F[UNSUP_FEATURES].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    Z = Zdf.fillna(Zdf.median()).to_numpy()
    iso.fit(Z)
    s = -iso.score_samples(Z)
    F = F.copy()
    F["anomaly_score"] = (s - s.min()) / (s.max() - s.min())
    return F


def label_frame(F: pd.DataFrame, insp: pd.DataFrame) -> pd.DataFrame:
    last = insp.drop_duplicates("consumer_id", keep="last")
    L = F.merge(last[["consumer_id", "outcome", "source"]], on="consumer_id")
    L["y"] = (L.outcome == "theft_confirmed").astype(int)
    return L


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_platt(F: pd.DataFrame, L: pd.DataFrame, features, rounds: int, seed: int) -> list[float] | None:
    """Calibrate raw scores into probabilities on out-of-fold predictions.

    Tip-offs are left out: they are chosen *because* theft looked blatant, so
    they would inflate the probabilities. Random checks and model-queue visits
    are chosen from the data alone, so P(theft | score) is estimated fairly
    on them. The map is monotone: rankings (and every hit-rate) are unchanged.
    """
    C = L[L.source != "tip_off"].reset_index(drop=True)
    if len(C) < 60 or C.y.sum() < 10 or (C.y == 0).sum() < 10:
        return None
    oof = np.zeros(len(C))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(C, C.y):
        # Train each fold on everything except the held-out calibration rows.
        keep = ~L.consumer_id.isin(C.consumer_id.iloc[te])
        b = lgb.train({**LGB_PARAMS, "seed": seed}, lgb.Dataset(L.loc[keep, features], L.y[keep]),
                      num_boost_round=rounds)
        oof[te] = b.predict(C.iloc[te][features])
    lr = LogisticRegression(C=1e4).fit(_logit(oof)[:, None], C.y)
    return [float(lr.coef_[0, 0]), float(lr.intercept_[0])]


def train(F: pd.DataFrame, insp: pd.DataFrame, features=FEATURES, rounds: int = 220, seed: int = 11,
          calibrate: bool = True) -> dict:
    """calibrate=False skips the probability calibration (5 extra fits). Use it
    where only the ranking matters (evaluation runs, CV folds)."""
    L = label_frame(F, insp)
    params = {**LGB_PARAMS, "seed": seed}
    booster = lgb.train(params, lgb.Dataset(L[features], L.y), num_boost_round=rounds)
    # Fallback when there are too few labels to calibrate: re-anchor to the
    # hit-rate seen on random checks (prior-shift correction).
    rand = L[L.source == "random_check"]
    prior_true = float(np.clip(rand.y.mean() if len(rand) >= 50 else 0.05, 0.01, 0.5))
    prior_train = float(L.y.mean())
    platt = _fit_platt(F, L, list(features), rounds, seed) if calibrate else None
    return {"booster": booster, "features": list(features),
            "prior_true": prior_true, "prior_train": prior_train, "platt": platt,
            "n_labels": int(len(L)), "n_pos": int(L.y.sum())}


def cv_oof(F: pd.DataFrame, insp: pd.DataFrame, features=FEATURES, folds: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Out-of-fold scores for every labelled consumer: model quality measured
    without ground truth, on outcomes each fold's model never trained on."""
    L = label_frame(F, insp)
    oof = np.zeros(len(L))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(L, L.y):
        sub = insp[insp.consumer_id.isin(L.consumer_id.iloc[tr])]
        oof[te] = predict(train(F, sub, features, calibrate=False), L.iloc[te])
    return L, oof


def predict(m: dict, F: pd.DataFrame) -> np.ndarray:
    p = m["booster"].predict(F[m["features"]])
    if m.get("platt"):
        a, b = m["platt"]
        return 1 / (1 + np.exp(-(a * _logit(p) + b)))
    a = m["prior_true"] / m["prior_train"]
    b = (1 - m["prior_true"]) / (1 - m["prior_train"])
    return a * p / (a * p + b * (1 - p))


def reliability(y: np.ndarray, p: np.ndarray, edges=(0, 0.05, 0.2, 0.5, 0.8, 1.0)) -> dict:
    """Predicted vs observed theft rate per risk band, plus the Brier score."""
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= 1))
        if m.sum():
            rows.append({"from": a, "to": b, "n": int(m.sum()), "predicted": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    return {"bins": rows, "brier": float(np.mean((p - y) ** 2)),
            "brier_base_rate": float(np.mean((y.mean() - y) ** 2))}


def contributions(m: dict, F: pd.DataFrame) -> np.ndarray:
    """TreeSHAP contributions (log-odds), last column is the bias."""
    return m["booster"].predict(F[m["features"]], pred_contrib=True)


# ---------------------------------------------------------------- reasons
def _fmt_reason(feat: str, r: pd.Series) -> tuple[str, dict] | None:
    """Map a feature that pushed the score up to a reason code + parameters.
    Text is rendered per language in i18n.py."""
    if feat in ("peer_ratio_change", "cp_drop_ratio") and r.cp_drop_ratio < 0.85:
        return "drop", {"pct": f"{100 * (1 - r.cp_drop_ratio):.0f}", "cat": r.category, "weeks": int(r.cp_weeks_ago)}
    if feat == "low_day_frac" and r.low_day_frac > 0.05:
        return "low_days", {"pct": f"{100 * r.low_day_frac:.0f}"}
    if feat == "flatness_change":
        if r.flatness_change < 0.7:
            return "flat", {"pct": f"{100 * (1 - r.flatness_change):.0f}"}
        if r.flatness_change > 1.6:
            return "erratic", {"x": f"{r.flatness_change:.1f}"}
    if feat == "cap_frac" and r.cap_frac > 0.12:
        return "cap", {"pct": f"{100 * r.cap_frac:.0f}"}
    if feat == "tamper_events" and r.tamper_events > 0:
        return "tamper", {"n": int(r.tamper_events)}
    if feat == "load_utilisation" and r.load_utilisation < 0.05:
        return "low_load", {"pct": f"{100 * r.load_utilisation:.1f}", "kw": f"{r.sanctioned_kw:g}"}
    if feat == "dt_loss_rise" and r.dt_loss_rise > 0.01:
        return "dt_rise", {"dt": r.dt_id, "pp": f"{100 * r.dt_loss_rise:.1f}"}
    if feat in ("dt_step_at_drop", "dt_step_match") and r.dt_step_at_drop > 0.15:
        return "dt_step", {"dt": r.dt_id, "pp": f"{r.dt_step_at_drop:.1f}",
                           "match": f"{100 * min(max(r.dt_step_match, 0), 1):.0f}"}
    if feat == "anomaly_score" and r.anomaly_score > 0.5:
        return "outlier", {}
    return None


def reasons_for(F: pd.DataFrame, contrib: np.ndarray, features: list, top: int = 3) -> list[list[tuple[str, dict]]]:
    out = []
    for i, (_, r) in enumerate(F.iterrows()):
        order = [j for j in np.argsort(-contrib[i, :-1]) if contrib[i, j] > 0]
        # The generic "outlier" reason is least actionable: list it last.
        order.sort(key=lambda j: features[j] == "anomaly_score")
        rs, seen = [], set()
        for j in order:
            if len(rs) >= top:
                break
            f = features[j]
            key = "dt_step" if f.startswith("dt_step") else ("drop" if f in ("peer_ratio_change", "cp_drop_ratio") else f)
            if key in seen:
                continue
            txt = _fmt_reason(f, r)
            if txt:
                rs.append(txt)
                seen.add(key)
        out.append(rs)
    return out


# ---------------------------------------------------------------- scoring
def score(m: dict, F: pd.DataFrame) -> pd.DataFrame:
    S = F.copy()
    S["risk_prob"] = predict(m, F)
    S["risk_score"] = np.round(100 * S.risk_prob, 1)
    S["band"] = pd.cut(S.risk_prob, [-1, 0.2, 0.5, 2], labels=["Low", "Medium", "High"]).astype(str)
    S["data_issue"] = S.missing_frac > 0.2
    shortfall_day = np.clip(S.expected_kwh_day - S.recent_kwh_day, 0, None)
    S["est_leak_kwh_month"] = np.round(shortfall_day * 30, 1)
    S["est_revenue_inr_month"] = np.round(S.est_leak_kwh_month * S.category.map(TARIFF_INR), 0)
    # Reasons only where someone will read them: every High/Medium consumer
    # and at least the top REASONS_MIN. Keeps scoring fast at lakh scale.
    need = S.index[(S.risk_prob >= 0.2) | S.risk_prob.rank(ascending=False, method="first").le(REASONS_MIN)]
    codes = pd.Series([[] for _ in range(len(S))], index=S.index, dtype=object)
    if len(need):
        Fn = F.loc[need]
        codes.loc[need] = pd.Series(reasons_for(Fn, contributions(m, Fn), m["features"]), index=need)
    for lang in LANGS:
        col = "reasons" if lang == "en" else f"reasons_{lang}"
        S[col] = codes.map(lambda cs, lang=lang: [render(c, p, lang) for c, p in cs])
    return S


def area_scores(S: pd.DataFrame, ctx: dict, dts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    D = ctx["dt_reads"]
    recent = slice(D.shape[1] - 90, D.shape[1])
    dt = pd.DataFrame({
        "dt_id": ctx["dt_ids"],
        "loss_base_pct": 100 * ctx["loss_base"],
        "loss_recent_pct": 100 * ctx["loss_recent"],
        "dt_kwh_day": D[:, recent].mean(axis=1),
    })
    dt["excess_loss_kwh_day"] = np.clip(dt.loss_recent_pct - dt.loss_base_pct, 0, None) / 100 * dt.dt_kwh_day
    g = S.groupby("dt_id").agg(
        consumers=("consumer_id", "size"),
        high_risk=("band", lambda b: int((b == "High").sum())),
        medium_risk=("band", lambda b: int((b == "Medium").sum())),
        top3_risk=("risk_prob", lambda p: float(np.sort(p.values)[-3:].sum())),
        exp_leak_kwh_month=("est_leak_kwh_month", lambda x: float((x * S.loc[x.index, "risk_prob"]).sum())),
    ).reset_index()
    dt = dt.merge(g, on="dt_id").merge(dts, on="dt_id")
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    comb = 0.5 * z(dt.loss_recent_pct - dt.loss_base_pct) + 0.5 * z(dt.top3_risk)
    dt["dt_risk_score"] = np.round(100 * comb.rank(pct=True), 0)
    fd = dt.groupby("feeder_id").agg(
        dts=("dt_id", "size"), consumers=("consumers", "sum"), high_risk=("high_risk", "sum"),
        excess_loss_kwh_day=("excess_loss_kwh_day", "sum"), dt_kwh_day=("dt_kwh_day", "sum"),
        mean_dt_risk=("dt_risk_score", "mean"),
    ).reset_index()
    fd["excess_loss_pct"] = 100 * fd.excess_loss_kwh_day / fd.dt_kwh_day
    return dt, fd


# ---------------------------------------------------------------- evaluation
def eval_ranking(y: np.ndarray, s: np.ndarray, k: int) -> dict:
    order = np.argsort(-s)
    top = order[:k]
    tp = int(y[top].sum())
    neg = int((y == 0).sum())
    return {
        "precision_at_k": tp / k,
        "recall_at_k": tp / max(int(y.sum()), 1),
        "false_positive_rate_at_k": (k - tp) / max(neg, 1),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "lift_vs_random": (tp / k) / max(y.mean(), 1e-9),
        "k": k, "positives": int(y.sum()), "n": int(len(y)),
    }


# ---------------------------------------------------------------- registry
class Registry:
    def __init__(self, root: str | Path = "models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"

    def _read(self) -> dict:
        return json.loads(self.index.read_text()) if self.index.exists() else {"active": None, "versions": []}

    def save(self, m: dict, metrics: dict, note: str, activate: bool = True,
             trained_on_rows: int | None = None, gate: dict | None = None) -> str:
        """Store a version. With activate=False it is kept as a candidate and
        the current active model stays in service."""
        reg = self._read()
        v = f"v{len(reg['versions']) + 1}"
        d = self.root / v
        d.mkdir(exist_ok=True)
        m["booster"].save_model(str(d / "model.txt"))
        card = {"version": v, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                "features": m["features"], "prior_true": m["prior_true"], "prior_train": m["prior_train"],
                "platt": m.get("platt"), "n_labels": m["n_labels"], "n_pos": m["n_pos"], "metrics": metrics,
                "trained_on_rows": trained_on_rows, "gate": gate}
        (d / "model_card.json").write_text(json.dumps(card, indent=2))
        reg["versions"].append({k: card[k] for k in ("version", "created", "note", "n_labels", "n_pos",
                                                     "metrics", "trained_on_rows", "gate")})
        if activate or reg["active"] is None:
            reg["active"] = v
        self.index.write_text(json.dumps(reg, indent=2))
        return v

    def set_active(self, version: str) -> None:
        reg = self._read()
        if version not in [x["version"] for x in reg["versions"]]:
            raise KeyError(version)
        reg["active"] = version
        self.index.write_text(json.dumps(reg, indent=2))

    def card(self, version: str | None = None) -> dict:
        v = version or self._read()["active"]
        return json.loads((self.root / v / "model_card.json").read_text())

    def load(self, version: str | None = None) -> dict:
        reg = self._read()
        v = version or reg["active"]
        card = json.loads((self.root / v / "model_card.json").read_text())
        return {"booster": lgb.Booster(model_file=str(self.root / v / "model.txt")), "version": v,
                "trained_on_rows": card.get("trained_on_rows"), "platt": card.get("platt"), **{
                    k: card[k] for k in ("features", "prior_true", "prior_train", "n_labels", "n_pos")}}

    def list(self) -> dict:
        """Versions with a status: active, candidate (trained, not promoted) or retired."""
        reg = self._read()
        for x in reg["versions"]:
            gate = x.get("gate") or {}
            x["status"] = ("active" if x["version"] == reg["active"]
                           else "candidate" if gate.get("promoted") is False else "retired")
        return reg
