"""Scale benchmark: time and memory for one daily scoring run.

    python scripts/benchmark.py                  # ~1 lakh consumers (200 feeders)
    python scripts/benchmark.py --feeders 400    # ~2 lakh

Simulates a network of the requested size into a temp folder, then times the
production steps on it: load, features, anomaly layer, training on the
vigilance records, scoring with reasons, and area roll-ups. Peak memory is
the process's maximum resident set size.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidyutrakshak import features as fx  # noqa: E402
from vidyutrakshak.model import FEATURES, add_anomaly_score, area_scores, score, train  # noqa: E402
from vidyutrakshak.simulate import SimConfig, simulate  # noqa: E402

OUT = ROOT / "docs" / "benchmark.json"


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeders", type=int, default=200)
    args = ap.parse_args()
    timings = {}

    def step(name, fn):
        t = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t, 1)
        print(f"  {name:<34} {timings[name]:>7.1f}s   peak {peak_mb():,.0f} MB", flush=True)
        return out

    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(SimConfig(), n_feeders=args.feeders,
                      random_inspections=int(500 * args.feeders / 10), tipoff_inspections=int(90 * args.feeders / 10))
        meta = step("simulate network (test data only)", lambda: simulate(cfg, tmp))
        print(f"• {meta['consumers']:,} consumers, {meta['dts']:,} DTs, {meta['feeders']} feeders, {meta['days']} days")
        sim_peak = peak_mb()
        ds = step("load MDMS / DT / billing tables", lambda: fx.load(tmp))
    F, ctx = step("build features (peers, DT balance)", lambda: fx.build_features(ds))
    F = step("anomaly layer (Isolation Forest)", lambda: add_anomaly_score(F))
    m = step("train model (+ calibration)", lambda: train(F, ds.inspections, FEATURES))
    S = step("score all + reasons (EN/GU)", lambda: score(m, F))
    step("DT and feeder roll-ups", lambda: area_scores(S, ctx, ds.dts))
    total = round(sum(v for k, v in timings.items() if not k.startswith("simulate")), 1)
    out = {"consumers": meta["consumers"], "dts": meta["dts"], "feeders": meta["feeders"], "days": meta["days"],
           "seconds": timings, "total_seconds_excluding_simulation": total,
           "peak_memory_mb": round(peak_mb()), "peak_memory_mb_simulation": round(sim_peak),
           "machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()}}
    OUT.parent.mkdir(exist_ok=True)
    runs = json.loads(OUT.read_text()).get("runs", []) if OUT.exists() else []
    runs = [r for r in runs if r["feeders"] != meta["feeders"]] + [out]
    OUT.write_text(json.dumps({"runs": sorted(runs, key=lambda r: r["consumers"])}, indent=2))
    print(f"• daily run for {meta['consumers']:,} consumers: {total:.0f}s, peak {peak_mb():,.0f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
