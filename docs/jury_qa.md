# VidyutRakshak: jury Q&A preparation

Short answers first, then the evidence. Every number comes from a file in this
repo, so you can open it in the room. Regenerate them with the commands at the end.

| Evidence | File | Command |
|---|---|---|
| Headline results (simulated network) | `static/dashboard_data.json` → `headline`, `ablation`, `calibration` | `python -m vidyutrakshak.pipeline --fresh` |
| Real-data check (SGCC) | `docs/sgcc_validation.json` | `python scripts/validate_sgcc.py` |
| Stress tests | `docs/stress_tests.json` | `python scripts/stress_test.py` |
| Scale | `docs/benchmark.json` | `python scripts/benchmark.py --feeders 200` |

---

### 1. "Your results are on data you simulated yourselves. Why should we believe them?"
**Answer:** Fair challenge, so we also tested on real, labelled theft data. On the
public SGCC dataset (42,372 real consumers, 3,615 confirmed thieves), the same feature
code puts **40% thieves in the top 4% of alerts, against an 8.5% base rate (4.7× random)**.
This uses meter data only: SGCC has no transformer meters, tamper logs or tariffs. With 2 years of
history it is 43%. The simulator is there to measure things real data can't show, such
as the holdout area and vacant/solar confounders. The pilot re-measures everything on GUVNL outcomes.
*Appendix A.*

### 2. "What about consumers who have been stealing for years? There's no drop to detect."
**Answer:** We tested exactly that. When half of all thieves were already stealing
before the data starts, the hit rate stays at **85% (baseline 86%)**, and **85% of
those long-running thieves are still in the top list**. Capping, flat readings, tamper alarms
and comparison with same-DT peers don't need a change point. SGCC is mostly
long-running theft, and that's why a simple drop rule scores 3% there (worse than random). *Appendix B.*

### 3. "Many of our DTs have no working meter. Does the system still work?"
**Answer:** Yes. With **no DT meters at all, the hit rate is 86%**, and with half the DTs metered it is 83%.
The model treats a missing DT meter as unknown, not as zero loss. DT metering adds
value on the hard cases (see Q4) and gives the DT- and feeder-level loss view. *Appendix B.*

### 4. "Then what does the DT energy balance actually add?"
**Answer:** It matters on the consumers who are hardest to tell apart: those whose use fell by more than 25%, where
thieves, vacant houses and rooftop-solar homes all look alike. There, it **cuts visits to
innocent homes by 19%** (12.4 → 10.0 in the top 69) and raises PR-AUC from 0.86 to 0.89.
For subtle 10–25% bypass it lifts the hit rate from 69% to 73%. For blatant theft, meter
signals alone are already strong. *Appendix C.*

### 5. "How many innocent households will you send inspectors to?"
**Answer:** Of the top 71 alerts, **6 are vacant or rooftop-solar homes, against 53 for a
"flag a >50% drop" rule**. 85% of alerts are real theft, against 4.6% for random checks.
Every alert shows its reasons, so the officer can skip one that doesn't hold up.

### 6. "What does a risk score of 80 mean? Can an officer rely on it?"
**Answer:** It means about 80% likely theft. We calibrate the scores and check them on a holdout
area: in the 80–100 band, 96% were really stealing; in 50–80, 64%; in 20–50, 24%.
Tip-offs are left out of calibration because they are biased towards blatant theft.
*Model tab, "Do risk scores mean what they say?"; appendix C.*

### 7. "Will this hold up in a case file? Can you explain a decision?"
**Answer:** Every alert lists up to three reasons computed from the model (TreeSHAP), in
plain English and **Gujarati**. For example: "Consumption fell 23% vs same-DT commercial
peers, starting ~28 weeks ago"; "Usage pinned at a ceiling on 84% of recent days";
"2 meter tamper alarms in the last 120 days". Each one is a number the officer can
verify from MDMS. The score is a reason to inspect, never proof of theft.

### 8. "How many past cases do you need?"
**Answer:** The first model trained on about 586 past inspections (112 thefts) and already hit
68%; three rounds of 60 field outcomes took it to 86%. The unsupervised layer works
with no labels at all. For a pilot: one sub-division's existing vigilance records.

### 9. "What if a retrained model is worse? Who controls that?"
**Answer:** A retrained model **goes live only if it scores at least as well as the live
model on field outcomes the live model never saw**. Otherwise it is kept as a
"candidate" and the current model stays in service. With too few new outcomes (fewer than 20), it
is never promoted. An admin can roll back to any version. *Demo: Model tab → Retrain.*

### 10. "Consumer data is sensitive. How do you handle privacy?"
**Answer:** At ingest, consumer numbers are replaced by keyed pseudonyms (HMAC-SHA256
with a secret salt held by GUVNL). Scores, logs and the console never hold the real
number. The pseudonym→consumer lookup is a separate owner-only file, used at dispatch.
Role-based keys: viewer, vigilance (records outcomes), admin (retrains). Runs on-premise.

### 11. "How does it connect to our MDMS, billing and GIS?"
**Answer:** `python -m vidyutrakshak.ingest` reads standard CSV exports: daily kWh per meter, the
consumer master with DT, feeder, tariff and sanctioned load, DT meter reads, tamper events and
the vigilance log. It accepts ISO or Indian day-first dates and tariff names like
"Residential". It rejects bad files with a clear message (missing columns, unknown
tariff codes or outcomes). An API adapter can replace the CSVs later; the tables stay the same.

### 12. "Will it scale to a whole DISCOM?"
**Answer:** Measured on one laptop: **1 lakh consumers in 15 s (2.1 GB RAM), 1.9 lakh
in 33 s**, roughly linear. Peers are compared within each DT, so a DISCOM runs
division by division overnight on one server. *Appendix D.*

### 13. "What does it cost?"
**Answer:** ₹0 in licences: Python, LightGBM, scikit-learn, FastAPI, all open source.
It needs one VM in the DISCOM data centre. The main cost is staff time for data
onboarding and officer review of the Gujarati wording.

### 14. "Seasons, monsoon, Rabi pumping, festivals: won't those trigger false alarms?"
**Answer:** Every consumer is compared each day with same-DT, same-tariff peers, so a
summer AC peak, Rabi pumping or Sunday closure moves the peers too and cancels out. A
monthly drift check (PSI per signal) flags when patterns shift enough to review the model.

### 15. "Why not deep learning?"
**Answer:** On SGCC, a black-box model on the raw series reached 32% (1 year) and 44%
(2 years); our explainable signals reached 40% and 43%. That's the same accuracy, but every alert comes with
a reason an officer can check and defend. Adding the raw weekly shape on top (56%)
is a pilot option once officers trust the system.

---

## Honest limits (say these before the jury finds them)
- The headline numbers are on a simulated network. The real-data check (SGCC) covers only the meter-data path.
- Subtle bypass of 10–25% is the hardest case (73% hit rate).
- 10% GIS mis-tagging costs about 5 points of hit rate. Data onboarding includes a consumer–DT index check.
- The Gujarati reason text is a first draft and needs review by a native speaker and a vigilance officer.
