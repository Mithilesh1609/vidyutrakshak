"""Real-data path: system exports -> ingest (validated, pseudonymised) ->
pipeline in real-data mode -> console payload. Uses a slice of the simulated
network rewritten in the long CSV formats an MDMS/billing export would have."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from vidyutrakshak.ingest import IngestError, ingest
from vidyutrakshak.pipeline import DATA, ROOT, run

SALT = b"test-salt-not-for-production-0123"
CAT_NAMES = {"RES": "Residential", "COM": "Commercial", "AGR": "Agriculture", "IND": "Industrial"}


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    if not (DATA / "_ground_truth.csv").exists():
        run(fresh=True, verbose=False)
    d = tmp_path_factory.mktemp("exports")
    cons = pd.read_csv(DATA / "consumers.csv")
    cons = cons[cons.feeder_id.isin(["FDR-01", "FDR-02", "FDR-03", "FDR-04"])]
    real_no = {c: f"GJ{7000000 + i:08d}" for i, c in enumerate(cons.consumer_id)}
    pd.DataFrame({"Consumer_No": cons.consumer_id.map(real_no), "DT_Code": cons.dt_id, "Feeder_Code": cons.feeder_id,
                  "Category": cons.category.map(CAT_NAMES), "Sanctioned_kW": cons.sanctioned_kw}
                 ).to_csv(d / "master.csv", index=False)
    reads = pd.read_parquet(DATA / "meter_reads.parquet").loc[cons.consumer_id]
    long = reads.stack(future_stack=True).rename("kwh").reset_index()
    long.columns = ["consumer_no", "date", "kwh"]
    long["consumer_no"] = long.consumer_no.map(real_no)
    long["date"] = pd.to_datetime(long.date).dt.strftime("%d-%m-%Y")      # Indian day-first format
    long.to_csv(d / "reads.csv", index=False)
    dt = pd.read_parquet(DATA / "dt_reads.parquet").loc[cons.dt_id.unique()]
    dl = dt.stack(future_stack=True).rename("kwh").reset_index()
    dl.columns = ["dt_code", "date", "kwh"]
    dl.to_csv(d / "dt.csv", index=False)
    ev = pd.read_parquet(DATA / "tamper_events.parquet").loc[cons.consumer_id]
    evl = ev.stack().rename("n").reset_index()
    evl = evl[evl.n > 0]
    pd.DataFrame({"consumer_no": evl.iloc[:, 0].map(real_no), "date": evl.iloc[:, 1]}).to_csv(d / "tamper.csv", index=False)
    insp = pd.read_csv(DATA / "inspections.csv")
    insp = insp[insp.consumer_id.isin(cons.consumer_id)]
    pd.DataFrame({"consumer_no": insp.consumer_id.map(real_no),
                  "outcome": insp.outcome.map({"theft_confirmed": "Theft", "no_irregularity": "OK"}),
                  "source": insp.source}).to_csv(d / "vigilance.csv", index=False)
    return d, set(real_no.values())


def test_ingest_validates_and_pseudonymises(exports, tmp_path):
    d, real = exports
    out, lookup = tmp_path / "pilot", tmp_path / "secure" / "lookup.csv"
    res = ingest(d / "reads.csv", d / "master.csv", out, d / "dt.csv", d / "tamper.csv", d / "vigilance.csv",
                 lookup=lookup, salt=SALT)
    assert res["simulated"] is False and res["consumers"] > 1000 and res["days"] == 365
    assert res["confirmed_theft_labels"] >= 10 and res["dt_meter_coverage"] > 0.9
    cons = pd.read_csv(out / "consumers.csv")
    assert cons.consumer_id.str.match(r"^C[0-9a-f]{12}$").all()
    for f in out.iterdir():                     # no real consumer number in any output
        text = f.read_bytes().decode("latin-1")
        assert not any(r in text for r in list(real)[:200]), f.name
    assert stat.S_IMODE(os.stat(lookup).st_mode) == 0o600
    # Same salt -> same pseudonyms (stable across monthly runs).
    res2 = ingest(d / "reads.csv", d / "master.csv", tmp_path / "again", salt=SALT)
    assert set(pd.read_csv(tmp_path / "again" / "consumers.csv").consumer_id) == set(cons.consumer_id)


def test_ingest_rejects_bad_exports(exports, tmp_path):
    d, _ = exports
    pd.read_csv(d / "master.csv").drop(columns=["DT_Code"]).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(IngestError, match="missing column"):
        ingest(d / "reads.csv", tmp_path / "m.csv", tmp_path / "o", salt=SALT)
    pd.read_csv(d / "vigilance.csv").assign(outcome="maybe").to_csv(tmp_path / "v.csv", index=False)
    with pytest.raises(IngestError, match="unknown outcome"):
        ingest(d / "reads.csv", d / "master.csv", tmp_path / "o2", inspections=tmp_path / "v.csv", salt=SALT)
    with pytest.raises(IngestError, match="VR_PSEUDO_SALT"):
        ingest(d / "reads.csv", d / "master.csv", tmp_path / "o3", salt=b"short")


def test_pipeline_runs_on_ingested_data(exports, tmp_path):
    d, real = exports
    out = tmp_path / "pilot"
    ingest(d / "reads.csv", d / "master.csv", out, d / "dt.csv", d / "tamper.csv", d / "vigilance.csv", salt=SALT)
    env = {**os.environ, "VR_DATA_DIR": str(out), "VR_MODELS_DIR": str(tmp_path / "models")}
    r = subprocess.run([sys.executable, "-m", "vidyutrakshak.pipeline"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    dash = json.loads((out / "dashboard_data.json").read_text())
    assert dash["meta"]["simulated"] is False and dash["queue"] and dash["ablation"] == []
    assert dash["kpis"]["holdout"]["eval"] == "cv"
    assert not any(q["id"] in real for q in dash["queue"])
    # --fresh must refuse to overwrite real data with simulated data.
    r2 = subprocess.run([sys.executable, "-m", "vidyutrakshak.pipeline", "--fresh"], cwd=ROOT, env=env,
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode != 0 and "real (ingested) data" in (r2.stderr + r2.stdout)
