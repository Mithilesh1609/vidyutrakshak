"""Synthetic DISCOM data simulator.

Generates a realistic feeder -> DT -> consumer network with one year of daily
smart-meter reads, DT boundary-meter reads, sanctioned load, MDMS tamper
events, data-quality faults, legitimate behaviour changes (hard negatives),
injected theft patterns, and a small set of historical vigilance inspections.

Real DISCOM data (MDMS, billing, GIS, vigilance) replaces this module in a
pilot; everything downstream reads the same table shapes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    #        share  base kWh/day  sanctioned kW
    "RES": (0.74, 6.0, 3.0),
    "COM": (0.15, 24.0, 8.0),
    "AGR": (0.08, 28.0, 7.5),
    "IND": (0.03, 120.0, 30.0),
}

THEFT_TYPES = [
    "partial_bypass",   # constant fraction of load bypassed
    "intermittent",     # bypass switched on some days
    "capping",          # reads clipped at a ceiling
    "flatline",         # tampered register reports a low, flat value
    "full_bypass",      # near-zero recorded consumption
    "gradual",          # bypass fraction ramps up over weeks
]


@dataclass
class SimConfig:
    seed: int = 7
    start: str = "2025-11-01"
    days: int = 365
    n_feeders: int = 10
    dts_per_feeder: tuple = (6, 10)
    consumers_per_dt: tuple = (40, 80)
    theft_rate: float = 0.06
    legit_drop_rate: float = 0.04      # vacancy / relocation
    solar_rate: float = 0.03           # rooftop solar adoption (net reads fall)
    comm_fault_rate: float = 0.05      # MDMS communication gaps
    random_inspections: int = 500
    tipoff_inspections: int = 90
    # Stress-test knobs (all off by default, so the baseline data is unchanged).
    chronic_theft_share: float = 0.0   # thefts already running before the history starts
    subtle_theft_share: float = 0.0    # thefts that bypass only 10-25% of load
    dt_meter_coverage: float = 1.0     # share of DTs with a working boundary meter
    gis_mistag_rate: float = 0.0       # consumers indexed to the wrong DT in GIS


def _seasonal_profile(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    doy = dates.dayofyear.values
    month = dates.month.values
    dow = dates.dayofweek.values
    # Residential: summer AC peak (Apr-Jun), mild winter dip.
    res = 1.0 + 0.40 * np.exp(-((doy - 140) ** 2) / (2 * 30 ** 2)) - 0.12 * np.isin(month, [12, 1])
    # Commercial: summer peak + Sunday closure.
    com = (1.0 + 0.25 * np.exp(-((doy - 140) ** 2) / (2 * 35 ** 2))) * np.where(dow == 6, 0.55, 1.0)
    # Agricultural pumping: heavy in Rabi (Nov-Feb) and summer, low in monsoon.
    agr = np.select(
        [np.isin(month, [11, 12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [7, 8, 9])],
        [1.35, 1.0, 0.30],
        default=0.7,
    )
    ind = np.where(dow == 6, 0.35, 1.0) * (1.0 + 0.05 * np.sin(2 * np.pi * doy / 365))
    return {"RES": res, "COM": com, "AGR": agr, "IND": ind}


def simulate(cfg: SimConfig = SimConfig(), out_dir: str | Path = "data") -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(cfg.start, periods=cfg.days, freq="D")
    T = cfg.days
    season = _seasonal_profile(dates)

    # ---------------- network topology ----------------
    feeders, dts, consumers = [], [], []
    center = (23.2156, 72.6369)  # Gandhinagar, for a plausible GIS layer
    cid = 0
    for f in range(cfg.n_feeders):
        fid = f"FDR-{f + 1:02d}"
        ang = 2 * np.pi * f / cfg.n_feeders
        f_lat = center[0] + 0.045 * np.cos(ang)
        f_lng = center[1] + 0.055 * np.sin(ang)
        feeders.append({"feeder_id": fid, "lat": f_lat, "lng": f_lng})
        n_dt = rng.integers(*cfg.dts_per_feeder, endpoint=True)
        for d in range(n_dt):
            did = f"DT-{len(dts) + 1:03d}"
            tech_loss = rng.uniform(0.04, 0.085)
            dts.append({
                "dt_id": did, "feeder_id": fid,
                "lat": f_lat + rng.normal(0, 0.012), "lng": f_lng + rng.normal(0, 0.014),
                "tech_loss": tech_loss,
                "capacity_kva": int(rng.choice([63, 100, 160, 250])),
            })
            n_c = rng.integers(*cfg.consumers_per_dt, endpoint=True)
            cats = rng.choice(list(CATEGORIES), size=n_c, p=[v[0] for v in CATEGORIES.values()])
            for cat in cats:
                share, base_mean, sl_mean = CATEGORIES[cat]
                base = rng.lognormal(np.log(base_mean), 0.45)
                sanctioned = max(1.0, round(sl_mean * (base / base_mean) ** 0.8 * rng.lognormal(0, 0.25), 1))
                consumers.append({
                    "consumer_id": f"C{cid:05d}", "dt_id": did, "feeder_id": fid,
                    "category": str(cat), "base_kwh": base, "sanctioned_kw": sanctioned,
                    "season_strength": rng.uniform(0.6, 1.3),
                })
                cid += 1

    cons = pd.DataFrame(consumers)
    dt_df = pd.DataFrame(dts)
    fdr_df = pd.DataFrame(feeders)
    N = len(cons)

    # ---------------- true consumption ----------------
    true = np.empty((N, T))
    for i, r in enumerate(cons.itertuples()):
        prof = 1 + (season[r.category] - 1) * r.season_strength
        noise = rng.lognormal(0, 0.22 if r.category != "IND" else 0.12, T)
        trend = 1 + rng.normal(0, 0.06) * np.linspace(0, 1, T)
        true[i] = r.base_kwh * prof * noise * trend

    recorded = true.copy()
    label = np.zeros(N, dtype=int)
    theft_type = np.array([""] * N, dtype=object)
    theft_start = np.full(N, -1)
    confounder = np.array([""] * N, dtype=object)

    idx = rng.permutation(N)
    n_theft = int(cfg.theft_rate * N)
    n_legit = int(cfg.legit_drop_rate * N)
    n_solar = int(cfg.solar_rate * N)
    theft_idx = idx[:n_theft]
    legit_idx = idx[n_theft:n_theft + n_legit]
    solar_idx = idx[n_theft + n_legit:n_theft + n_legit + n_solar]

    # Theft: meter records less than is actually drawn.
    for i in theft_idx:
        t0 = int(rng.integers(110, 300))
        kind = THEFT_TYPES[int(rng.integers(len(THEFT_TYPES)))]
        if cfg.chronic_theft_share > 0 and rng.random() < cfg.chronic_theft_share:
            t0 = 0                                   # stealing since before the data starts
        if cfg.subtle_theft_share > 0 and rng.random() < cfg.subtle_theft_share:
            kind = "subtle_bypass"
        ref = true[i, :max(t0, 90)]                  # what normal use looked like
        x = recorded[i, t0:]
        n = len(x)
        if kind == "subtle_bypass":
            x *= 1 - rng.uniform(0.10, 0.25)
        elif kind == "partial_bypass":
            x *= rng.uniform(0.25, 0.6)
        elif kind == "intermittent":
            on = rng.random(n) < rng.uniform(0.35, 0.7)
            x[on] *= rng.uniform(0.05, 0.2)
        elif kind == "capping":
            cap = np.quantile(ref, rng.uniform(0.15, 0.4))
            np.minimum(x, cap, out=x)
        elif kind == "flatline":
            x[:] = np.mean(ref) * rng.uniform(0.35, 0.6) * rng.lognormal(0, 0.02, n)
        elif kind == "full_bypass":
            x *= rng.uniform(0.0, 0.08)
        elif kind == "gradual":
            ramp = np.clip(np.arange(n) / rng.uniform(40, 90), 0, 1)
            x *= 1 - ramp * rng.uniform(0.5, 0.8)
        recorded[i, t0:] = x
        label[i], theft_type[i], theft_start[i] = 1, kind, t0

    # Legitimate drops: house vacated / business shut. True AND recorded fall
    # together, so the DT balance stays clean -> hard negatives for naive rules.
    for i in legit_idx:
        t0 = int(rng.integers(110, 320))
        f = rng.uniform(0.08, 0.35)
        true[i, t0:] *= f
        recorded[i, t0:] *= f
        confounder[i] = "vacancy"

    # Rooftop solar: net import falls; grid draw genuinely falls too.
    for i in solar_idx:
        t0 = int(rng.integers(90, 300))
        f = rng.uniform(0.45, 0.7)
        true[i, t0:] *= f
        recorded[i, t0:] *= f
        confounder[i] = "rooftop_solar"

    # ---------------- MDMS data quality ----------------
    meter_reads = recorded.copy()
    comm_idx = rng.choice(N, int(cfg.comm_fault_rate * N), replace=False)
    for i in comm_idx:
        for _ in range(int(rng.integers(1, 4))):
            s = int(rng.integers(0, T - 20))
            meter_reads[i, s:s + int(rng.integers(4, 21))] = np.nan
        if confounder[i] == "":
            confounder[i] = "comm_gaps"
    # Sporadic single-day drops everywhere.
    meter_reads[rng.random((N, T)) < 0.004] = np.nan

    # Tamper events (cover-open / magnetic influence) logged by smart meters.
    events = np.zeros((N, T), dtype=np.int16)
    for i in range(N):
        if label[i] == 1 and rng.random() < 0.45:
            t0 = theft_start[i]
            k = int(rng.integers(1, 6))
            events[i, rng.integers(max(0, t0 - 5), T, k)] += 1
        elif rng.random() < 0.03:  # false/benign alarms
            events[i, rng.integers(0, T, int(rng.integers(1, 3)))] += 1

    # ---------------- DT boundary meters ----------------
    dt_index = {d: k for k, d in enumerate(dt_df.dt_id)}
    dt_true = np.zeros((len(dt_df), T))
    np.add.at(dt_true, cons.dt_id.map(dt_index).values, true)
    dt_reads = dt_true * (1 + dt_df.tech_loss.values[:, None]) * rng.lognormal(0, 0.012, dt_true.shape)
    dt_reads[rng.random(dt_reads.shape) < 0.01] = np.nan
    if cfg.dt_meter_coverage < 1:
        # Unmetered or dead DT meters: no boundary reads at all.
        dead = rng.random(len(dt_df)) >= cfg.dt_meter_coverage
        dt_reads[dead] = np.nan

    # GIS indexing errors: the billing system thinks some consumers hang off a
    # different DT on the same feeder (DT reads above follow the true wiring).
    if cfg.gis_mistag_rate > 0:
        wrong = np.where(rng.random(N) < cfg.gis_mistag_rate)[0]
        for i in wrong:
            same_feeder = dt_df.dt_id[(dt_df.feeder_id == cons.at[i, "feeder_id"]) & (dt_df.dt_id != cons.at[i, "dt_id"])]
            if len(same_feeder):
                cons.at[i, "dt_id"] = same_feeder.iloc[int(rng.integers(len(same_feeder)))]

    # ---------------- historical vigilance inspections ----------------
    # Random checks + tip-offs (tip-offs are biased towards blatant theft).
    pool = np.setdiff1d(np.arange(N), theft_idx)
    rand_insp = rng.choice(N, cfg.random_inspections, replace=False)
    blatant = [i for i in theft_idx if theft_type[i] in ("full_bypass", "partial_bypass", "flatline")]
    tip = rng.choice(blatant, min(cfg.tipoff_inspections, len(blatant)), replace=False)
    inspected = np.unique(np.concatenate([rand_insp, tip]))
    insp = pd.DataFrame({
        "consumer_id": cons.consumer_id.values[inspected],
        "outcome": np.where(label[inspected] == 1, "theft_confirmed", "no_irregularity"),
        "source": np.where(np.isin(inspected, tip), "tip_off", "random_check"),
        "round": 0,
    })

    # ---------------- persist ----------------
    cons_out = cons.drop(columns=["base_kwh", "season_strength"])
    cons_out.to_csv(out / "consumers.csv", index=False)
    dt_df.drop(columns=["tech_loss"]).to_csv(out / "dts.csv", index=False)
    fdr_df.to_csv(out / "feeders.csv", index=False)
    pd.DataFrame(meter_reads, index=cons.consumer_id, columns=dates.strftime("%Y-%m-%d")) \
        .to_parquet(out / "meter_reads.parquet")
    pd.DataFrame(events, index=cons.consumer_id, columns=dates.strftime("%Y-%m-%d")) \
        .to_parquet(out / "tamper_events.parquet")
    pd.DataFrame(dt_reads, index=dt_df.dt_id, columns=dates.strftime("%Y-%m-%d")) \
        .to_parquet(out / "dt_reads.parquet")
    insp.to_csv(out / "inspections.csv", index=False)
    # Ground truth is kept separately and only used to simulate field outcomes
    # and to score the prototype -- the model never reads it.
    pd.DataFrame({
        "consumer_id": cons.consumer_id, "is_theft": label, "theft_type": theft_type,
        "theft_start_day": theft_start, "confounder": confounder,
    }).to_csv(out / "_ground_truth.csv", index=False)

    meta = {
        "consumers": N, "dts": len(dt_df), "feeders": len(fdr_df), "days": T,
        "start": cfg.start, "simulated": True, "theft_cases": int(label.sum()),
        "inspected": int(len(insp)), "confirmed_theft_labels": int((insp.outcome == "theft_confirmed").sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(simulate(), indent=2))
