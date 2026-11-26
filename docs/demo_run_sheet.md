# Live demo run sheet (about 6 minutes)

## Before the slot
1. `scripts/demo_reset.sh`: clean state (model v4), console at http://localhost:8000.
   Run it again after every rehearsal: recording outcomes and retraining change the state.
   (`pytest` runs in its own folders and does not touch the demo.)
2. Open the console in a fresh browser window at 110–125% zoom and close other tabs.
3. Keep the offline fallback ready: `dist/vidyutrakshak_console.html` opens with no server
   (retraining is disabled there).
4. Have the deck open on the appendix slides (A–D) for questions.

## Flow

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Deck slide 2 | — | Theft hides among lakhs of consumers; random checks find theft in about 5 of 100 visits; drop rules flag vacant and solar homes. |
| 0:40 | Deck slide 4 | — | Compare with same-DT peers, check the DT energy balance, explain every alert, learn from every inspection. |
| 1:20 | Console, top strip | Point at the four KPIs | 76 high-risk consumers not yet inspected; about ₹60k a month at risk; 85% hit rate vs 4.6% for random checks. |
| 1:50 | Queue, consumer C01881 | Click it | Read the three reasons. Show metered vs expected-from-peers, then the DT loss stepping up in the same weeks. |
| 2:40 | Header | Click **ગુજરાતી** | The same reasons in Gujarati for field officers. Click **English** to switch back. |
| 3:00 | Queue toolbar | Click **Download list** | The route sheet for today's crew, grouped by feeder and DT. |
| 3:15 | Transformers tab | Click the reddest DT (DT-002) | Loss went from 4.3% to 14.9%; here are the consumers behind it. |
| 3:45 | Model tab, comparison table | — | Drop rule: 53 of 71 visits to vacant or solar homes; ours: 6. The last column shows where the DT balance helps. |
| 4:15 | Model tab, calibration panel | — | A score of 80 means about 80% likely theft; checked on a holdout area. |
| 4:35 | Queue → consumer C03217 | Type an officer name, click **Theft confirmed** | The outcome goes into the vigilance log. |
| 4:50 | Model tab | Click **Retrain** | With one new outcome, the gate keeps v4 live and saves a candidate. Read the message. |
| 5:10 | Model tab | Click **Simulate a field round**, then **Retrain** | 60 inspections later there is enough evidence: the new model is compared on outcomes v4 never saw, and promoted. |
| 5:40 | Deck appendix A | — | On real SGCC data: 40% vs 8.5% at random, using only explainable meter signals. |
| 6:00 | — | — | Explainable, Gujarati, on-premise, pseudonymised, ₹0 licences; ready for a one-sub-division pilot. |

## If something goes wrong
- Console won't load: run `scripts/demo_reset.sh` again, or open `dist/vidyutrakshak_console.html`.
- A sign-in box appears: the server was started with keys. Use `admin-demo`, or restart without `--keys`.
- Retrain takes a few seconds: keep talking about the gate while the button reads "Retraining…".
