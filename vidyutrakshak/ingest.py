"""Turn DISCOM system exports into the tables VidyutRakshak reads.

    export VR_PSEUDO_SALT="<long random secret held by the DISCOM>"
    python -m vidyutrakshak.ingest \\
        --reads mdms_daily.csv --consumers consumer_master.csv \\
        --dt-reads dt_meter_daily.csv --tamper mdms_events.csv \\
        --inspections vigilance_log.csv --out data_pilot --lookup secure/lookup.csv

Expected columns (names are case-insensitive; extra columns are ignored):

  --reads        consumer_no, date, kwh                 MDMS daily energy (one row per meter-day)
  --consumers    consumer_no, dt_code, feeder_code,     billing + GIS consumer index
                 category, sanctioned_kw
  --dt-reads     dt_code, date, kwh                     DT boundary meters (optional: DTs
                                                        without a meter are fine)
  --tamper       consumer_no, date                      MDMS tamper/event log (optional)
  --dts          dt_code[, lat, lng, capacity_kva]      GIS transformer list (optional; the map
                                                        uses a schematic layout without it)
  --inspections  consumer_no, outcome[, date, source]   vigilance records (optional)

Consumer numbers are replaced by keyed pseudonyms (HMAC-SHA256 with a secret
salt) before anything is written, so scores, logs and the console never hold
the real number. The pseudonym -> consumer number lookup is written only if
--lookup is given, to a separate file readable by its owner alone; field
teams resolve it at dispatch time.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_ALIASES = {
    "RES": "RES", "RESIDENTIAL": "RES", "DOMESTIC": "RES", "RGP": "RES", "LTMD-RES": "RES",
    "COM": "COM", "COMMERCIAL": "COM", "NON-RESIDENTIAL": "COM", "NRGP": "COM", "GLP": "COM",
    "AGR": "AGR", "AGRICULTURAL": "AGR", "AGRICULTURE": "AGR", "AG": "AGR",
    "IND": "IND", "INDUSTRIAL": "IND", "LTP": "IND", "HT": "IND",
}
OUTCOMES = {
    "theft_confirmed": "theft_confirmed", "theft": "theft_confirmed", "confirmed": "theft_confirmed",
    "pilferage": "theft_confirmed", "malpractice": "theft_confirmed", "yes": "theft_confirmed", "1": "theft_confirmed",
    "no_irregularity": "no_irregularity", "no irregularity": "no_irregularity", "ok": "no_irregularity",
    "normal": "no_irregularity", "no": "no_irregularity", "0": "no_irregularity",
    "meter_fault": "no_irregularity", "meter fault": "no_irregularity", "defective meter": "no_irregularity",
}


class IngestError(ValueError):
    """Input files don't match the expected export format."""


def pseudonymise(values: pd.Series, salt: bytes) -> pd.Series:
    return values.astype(str).str.strip().map(
        lambda v: "C" + hmac.new(salt, v.encode(), hashlib.sha256).hexdigest()[:12])


def _read(path: str | Path, required: list[str], label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"{label}: file not found: {p}")
    df = pd.read_csv(p, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise IngestError(f"{label} ({p.name}) is missing column(s) {missing}. Found: {list(df.columns)}")
    return df


DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y", "%d-%b-%Y", "%d-%b-%y"]


def _dates(s: pd.Series, label: str) -> pd.Series:
    """Exports use ISO or Indian day-first dates; pick the format that reads
    the most rows (never guess month-first)."""
    s = s.astype(str).str.strip().str.slice(0, 11)
    best = None
    for fmt in DATE_FORMATS:
        d = pd.to_datetime(s, errors="coerce", format=fmt)
        if best is None or d.isna().sum() < best.isna().sum():
            best = d
        if not best.isna().any():
            break
    d = best
    bad = int(d.isna().sum())
    if bad > 0.01 * len(s):
        raise IngestError(f"{label}: {bad} of {len(s)} dates could not be read (examples: {list(s[d.isna()][:3])})")
    return d.dt.normalize()


def _kwh(s: pd.Series, label: str, report: dict) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    neg = int((v < 0).sum())
    if neg:
        report.setdefault("warnings", []).append(f"{label}: {neg} negative readings treated as missing")
    return v.where(v >= 0)


def _wide(df: pd.DataFrame, key: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    w = df.pivot_table(index=key, columns="date", values="kwh", aggfunc="sum")
    w = w.reindex(columns=dates)
    w.columns = dates.strftime("%Y-%m-%d")
    return w.astype("float32")


def ingest(reads: str, consumers: str, out: str, dt_reads: str | None = None, tamper: str | None = None,
           inspections: str | None = None, lookup: str | None = None, salt: bytes | None = None,
           category_map: dict | None = None, dts_gis: str | None = None) -> dict:
    report: dict = {"warnings": []}
    salt = salt if salt is not None else os.environb.get(b"VR_PSEUDO_SALT", b"")
    if len(salt) < 16:
        raise IngestError("Set VR_PSEUDO_SALT to a secret of at least 16 characters; it keys the consumer pseudonyms.")
    aliases = {**CATEGORY_ALIASES, **{k.upper(): v for k, v in (category_map or {}).items()}}

    # -------- consumer master (billing + GIS) --------
    cm = _read(consumers, ["consumer_no", "dt_code", "feeder_code", "category", "sanctioned_kw"], "Consumer master")
    dup = int(cm.consumer_no.duplicated().sum())
    if dup:
        report["warnings"].append(f"Consumer master: {dup} duplicate consumer numbers, kept the last row")
        cm = cm.drop_duplicates("consumer_no", keep="last")
    cm["category"] = cm.category.str.strip().str.upper().map(aliases)
    unknown = cm.category.isna()
    if unknown.any():
        raise IngestError(f"Consumer master: {int(unknown.sum())} rows have a tariff category not mapped to "
                          f"RES/COM/AGR/IND. Pass --category-map with e.g. {{\"<code>\": \"COM\"}}.")
    cm["sanctioned_kw"] = pd.to_numeric(cm.sanctioned_kw, errors="coerce")
    cm["consumer_id"] = pseudonymise(cm.consumer_no, salt)

    # -------- MDMS daily reads --------
    r = _read(reads, ["consumer_no", "date", "kwh"], "Meter reads")
    r["date"] = _dates(r.date, "Meter reads")
    r["kwh"] = _kwh(r.kwh, "Meter reads", report)
    orphan = ~r.consumer_no.isin(cm.consumer_no)
    if orphan.any():
        report["warnings"].append(f"Meter reads: {orphan.sum()} rows for {r.consumer_no[orphan].nunique()} consumers "
                                  f"not in the consumer master were dropped (fix GIS/billing indexing)")
        r = r[~orphan]
    dates = pd.date_range(r.date.min(), r.date.max(), freq="D")
    if len(dates) < 180:
        raise IngestError(f"Meter reads cover only {len(dates)} days; at least 180 are needed (365 recommended).")
    r["consumer_id"] = r.consumer_no.map(dict(zip(cm.consumer_no, cm.consumer_id)))
    X = _wide(r, "consumer_id", dates)
    no_reads = ~cm.consumer_id.isin(X.index)
    if no_reads.any():
        report["warnings"].append(f"{int(no_reads.sum())} consumers in the master have no meter reads; skipped")
    cm = cm[~no_reads]
    X = X.reindex(cm.consumer_id)

    # -------- DT boundary meters --------
    dts = cm.groupby("dt_code").feeder_code.first().reset_index().rename(columns={"dt_code": "dt_id", "feeder_code": "feeder_id"})
    if dts_gis:
        g = _read(dts_gis, ["dt_code"], "DT list").rename(columns={"dt_code": "dt_id"})
        keep = [c for c in ("lat", "lng", "capacity_kva") if c in g]
        g[keep] = g[keep].apply(pd.to_numeric, errors="coerce")
        dts = dts.merge(g[["dt_id", *keep]].drop_duplicates("dt_id"), on="dt_id", how="left")
    if dt_reads:
        d = _read(dt_reads, ["dt_code", "date", "kwh"], "DT reads")
        d["date"] = _dates(d.date, "DT reads")
        d["kwh"] = _kwh(d.kwh, "DT reads", report)
        D = _wide(d.rename(columns={"dt_code": "dt_id"}), "dt_id", dates)
    else:
        D = pd.DataFrame(columns=dates.strftime("%Y-%m-%d"), dtype="float32")
    D = D.reindex(dts.dt_id)
    report["dt_meter_coverage"] = float(np.isfinite(D.to_numpy()).mean(axis=1).__ge__(0.5).mean()) if len(D) else 0.0

    # -------- tamper events --------
    E = pd.DataFrame(0, index=cm.consumer_id, columns=X.columns, dtype="int16")
    if tamper:
        t = _read(tamper, ["consumer_no", "date"], "Tamper events")
        t["date"] = _dates(t.date, "Tamper events").dt.strftime("%Y-%m-%d")
        t = t[t.consumer_no.isin(cm.consumer_no) & t.date.isin(E.columns)]
        t["consumer_id"] = t.consumer_no.map(dict(zip(cm.consumer_no, cm.consumer_id)))
        counts = t.groupby(["consumer_id", "date"]).size()
        for (cid, day), n in counts.items():
            E.at[cid, day] = n

    # -------- vigilance records --------
    insp = pd.DataFrame(columns=["consumer_id", "outcome", "source", "round"])
    if inspections:
        v = _read(inspections, ["consumer_no", "outcome"], "Inspections")
        v["outcome_norm"] = v.outcome.str.strip().str.lower().map(OUTCOMES)
        bad = v.outcome_norm.isna()
        if bad.any():
            raise IngestError(f"Inspections: unknown outcome value(s) {sorted(v.outcome[bad].unique())[:5]}. "
                              f"Use theft_confirmed / no_irregularity / meter_fault.")
        v = v[v.consumer_no.isin(cm.consumer_no)]
        src = v["source"].str.strip().str.lower() if "source" in v else pd.Series("random_check", index=v.index)
        insp = pd.DataFrame({"consumer_id": v.consumer_no.map(dict(zip(cm.consumer_no, cm.consumer_id))),
                             "outcome": v.outcome_norm,
                             "source": src.where(src.isin(["random_check", "tip_off", "model_queue"]), "random_check"),
                             "round": 0})
        if (insp.outcome == "theft_confirmed").sum() < 20:
            report["warnings"].append("Fewer than 20 confirmed thefts: the model will lean on the unsupervised layer")

    # -------- write --------
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    cm[["consumer_id", "dt_code", "feeder_code", "category", "sanctioned_kw"]].rename(
        columns={"dt_code": "dt_id", "feeder_code": "feeder_id"}).to_csv(o / "consumers.csv", index=False)
    dts.to_csv(o / "dts.csv", index=False)
    pd.DataFrame({"feeder_id": sorted(dts.feeder_id.unique())}).to_csv(o / "feeders.csv", index=False)
    X.to_parquet(o / "meter_reads.parquet")
    E.to_parquet(o / "tamper_events.parquet")
    D.to_parquet(o / "dt_reads.parquet")
    insp.to_csv(o / "inspections.csv", index=False)
    meta = {"consumers": int(len(cm)), "dts": int(len(dts)), "feeders": int(dts.feeder_id.nunique()),
            "days": int(len(dates)), "start": str(dates[0].date()), "simulated": False,
            "inspected": int(len(insp)), "confirmed_theft_labels": int((insp.outcome == "theft_confirmed").sum()),
            "dt_meter_coverage": round(report["dt_meter_coverage"], 3)}
    (o / "meta.json").write_text(json.dumps(meta, indent=2))
    if lookup:
        lp = Path(lookup)
        lp.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            cm[["consumer_id", "consumer_no"]].to_csv(fh, index=False)
    return {**meta, "warnings": report["warnings"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--consumers", required=True)
    ap.add_argument("--dt-reads")
    ap.add_argument("--tamper")
    ap.add_argument("--inspections")
    ap.add_argument("--dts", help="GIS transformer list with lat, lng, capacity_kva")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookup", help="where to write pseudonym -> consumer number (owner-only file)")
    ap.add_argument("--category-map", help="JSON file mapping tariff codes to RES/COM/AGR/IND")
    a = ap.parse_args(argv)
    cmap = json.loads(Path(a.category_map).read_text()) if a.category_map else None
    try:
        res = ingest(a.reads, a.consumers, a.out, a.dt_reads, a.tamper, a.inspections, a.lookup,
                     category_map=cmap, dts_gis=a.dts)
    except IngestError as e:
        sys.exit(f"Ingest failed: {e}")
    for w in res.pop("warnings"):
        print(f"  ! {w}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
