"""Query assistant for the console: answers inspectors' questions about the
live risk scores in English or Gujarati.

Two engines share the same read-only tools (views.py):
  claude  Claude (Anthropic API) picks and calls the tools, then answers in
          free form. Used when ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is
          set and VR_CHAT_LLM is not "0".
  rules   A keyword/intent matcher with fixed answer templates. No network,
          always available, and the fallback whenever the API call fails.

Neither engine can record outcomes, retrain or switch models: chat is read-only.
What leaves the server in claude mode: the question, and tool results
(pseudonymised consumer IDs, scores, reason text, area aggregates).
"""
from __future__ import annotations

import json
import os
import re

from .views import NotFound, consumer_detail, dt_summary, model_summary, queue_items, rank_dts

MODEL = "claude-opus-5-5"
MAX_TURNS, MAX_CHARS = 10, 1000

GLOSSARY = {
    "hit rate": ("Hit rate is the share of inspections that find theft. VidyutRakshak's hit rate is measured on a "
                 "holdout area the feedback loop never inspected, against random checks and a simple drop rule."),
    "risk score": ("The risk score (0–100) is the model's estimated probability, in percent, that an inspection of "
                   "this consumer finds theft. It ranks who to visit first; it is not proof of theft."),
    "band": ("Bands group risk scores: High is 50 or more, Medium 20–50, Low below 20. The daily queue lists "
             "uninspected consumers from the highest score down."),
    "pr-auc": ("PR-AUC (area under the precision-recall curve) summarises how well the model ranks thefts above "
               "legitimate consumers across all queue sizes. 1.0 is perfect; random ranking scores the theft base rate."),
    "calibration": ("Calibration checks that scores mean what they say: of consumers scored around 60%, about 60% "
                    "should turn out to be theft. The Brier score measures this; lower is better."),
    "promotion gate": ("A retrained model goes live only if it ranks at least as well as the active model on field "
                       "outcomes the active model never saw (needs 20+ new outcomes with 3+ thefts and 3+ clean visits)."),
    "unaccounted energy": ("Unaccounted energy is the share of a transformer's (DT) input that no consumer meter "
                           "records. A rise against its own baseline points to hooking or bypass on that DT."),
    "data issue": ("A meter data issue means more than 20% of reads are missing. These consumers go to the metering "
                   "team first, because the risk signals are unreliable without reads."),
    "meter fault": ("A field outcome of 'meter fault' is not theft: it trains the model as a non-theft case and sends "
                    "the consumer to the metering team instead of vigilance."),
    "drift": ("Drift compares the current feature and score distributions with those at training (PSI). Large drift "
              "means the model should be retrained or reviewed."),
}
_GLOSSARY_KEYS = {
    "hit rate": ["hit rate", "hit-rate", "precision", "હિટ રેટ"], "risk score": ["risk score", "score mean", "જોખમ સ્કોર", "સ્કોર"],
    "band": ["band", "high risk mean", "medium", "બેન્ડ"], "pr-auc": ["pr-auc", "pr auc", "auc"],
    "calibration": ["calibrat", "brier"], "promotion gate": ["gate", "promot", "retrain"],
    "unaccounted energy": ["unaccounted", "energy balance", "dt loss", "બિનહિસાબી"],
    "data issue": ["data issue", "missing read", "data quality"], "meter fault": ["meter fault", "મીટર ખામી"],
    "drift": ["drift", "psi"],
}

SYSTEM = """You are the VidyutRakshak assistant inside a DISCOM electricity-theft risk console. Users are vigilance officers, field inspectors and analysts.

Rules:
- Answer only from tool results. Never invent consumer IDs, scores, kWh, rupee figures or reasons. If a tool returns an error or nothing, say so plainly.
- Quote IDs exactly as tools return them (consumer IDs like C01881, DTs like DT-032, feeders like FDR-05).
- A risk score ranks consumers for inspection. It is not proof of theft; never call a consumer a thief.
- Reply in the user's language: Gujarati if they write in Gujarati, otherwise English. Use the tool's reasons_gu text for Gujarati reasons.
- Be brief: a short answer first, then at most a few bullet points. No tables wider than 4 columns.
- You can only read data. If asked to record an outcome, retrain or change a model, tell the user to use the console buttons.
- For what a metric or term means, call explain_term."""


# ------------------------------------------------------------------ tools (shared)
def _tools(state: dict) -> dict:
    """name -> (callable(**kwargs) -> JSON-able)."""
    def get_consumer(consumer_id: str):
        d = consumer_detail(state, consumer_id, days=30, with_shap=True)
        d.pop("daily")
        d["features"] = {k: round(v, 3) for k, v in d["features"].items()}
        return d

    def top_consumers(feeder: str | None = None, dt: str | None = None, band: str | None = None, n: int = 10,
                      include_inspected: bool = False):
        items = queue_items(state, min(int(n or 10), 50), feeder, dt, band, include_inspected)
        return [{k: r[k] for k in ("consumer_id", "dt_id", "feeder_id", "category", "risk_score", "band")}
                | {"top_reason": (r["reasons"] or ["Statistical outlier among peers"])[0],
                   "top_reason_gu": (r["reasons_gu"] or [""])[0]} for r in items]

    def get_transformer(dt_id: str):
        return dt_summary(state, dt_id)

    def rank_transformers(by: str = "excess_kwh_day", n: int = 5, feeder: str | None = None):
        return rank_dts(state, by, int(n or 5), feeder)

    def feeder_summary():
        return state["dash"]["feeders"]

    def network_summary():
        return {"kpis": {k: v for k, v in state["dash"]["kpis"].items() if k != "holdout"},
                "as_of": state["dash"]["meta"]["as_of"], "simulated": state["dash"]["meta"].get("simulated", True)}

    def get_model_summary():
        return model_summary(state)

    def explain_term(term: str):
        k = _match_term(term) or term.lower()
        return {"term": k, "meaning": GLOSSARY.get(k, "No glossary entry. Known terms: " + ", ".join(GLOSSARY))}

    return {"get_consumer": get_consumer, "top_consumers": top_consumers, "get_transformer": get_transformer,
            "rank_transformers": rank_transformers, "feeder_summary": feeder_summary,
            "network_summary": network_summary, "model_summary": get_model_summary, "explain_term": explain_term}


def _match_term(text: str) -> str | None:
    t = text.lower()
    for k, keys in _GLOSSARY_KEYS.items():
        if any(x in t for x in keys):
            return k
    return None


# ------------------------------------------------------------------ engine choice
def mode() -> str:
    if os.getenv("VR_CHAT_LLM", "1") == "0":
        return "rules"
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return "rules"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "rules"
    return "claude"


def _client():
    import anthropic
    return anthropic.Anthropic(timeout=60.0, max_retries=1)


def _clean(messages: list[dict]) -> list[dict]:
    """Last MAX_TURNS text turns, starting with a user turn, roles alternating."""
    out: list[dict] = []
    for m in messages[-MAX_TURNS:]:
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = str(m.get("content", ""))[:MAX_CHARS].strip()
        if not text:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


ID_RE = re.compile(r"\b(C\d{5}|DT-\d{3}|FDR-\d{2})\b")


def answer(state: dict, messages: list[dict], lang: str = "en") -> dict:
    msgs = _clean(messages)
    if not msgs or msgs[-1]["role"] != "user":
        return {"answer": _help(lang), "mode": "rules", "ids": []}
    note = None
    if mode() == "claude":
        try:
            text = answer_llm(state, msgs, lang)
            if text:
                return {"answer": text, "mode": "claude", "ids": sorted(set(ID_RE.findall(text)))}
        except Exception as e:  # network, auth, rate limit, refusal: fall back, never fail the chat
            note = f"AI assistant unavailable ({type(e).__name__}); answered by the offline rules."
    text = answer_rules(state, msgs[-1]["content"], lang)
    if note:
        text += "\n\n" + note
    return {"answer": text, "mode": "rules", "ids": sorted(set(ID_RE.findall(text)))}


# ------------------------------------------------------------------ claude
def answer_llm(state: dict, msgs: list[dict], lang: str = "en") -> str:
    from anthropic import beta_tool

    t = _tools(state)

    def run(name, **kw):
        try:
            return json.dumps(t[name](**{k: v for k, v in kw.items() if v is not None}), default=str, ensure_ascii=False)
        except NotFound as e:
            return json.dumps({"error": f"not found: {e}"})

    @beta_tool
    def get_consumer(consumer_id: str) -> str:
        """Full explanation of one consumer: risk score (0-100), band, reasons in English and Gujarati, the
        model features that moved the score most (drivers), estimated unbilled kWh and revenue, inspection status.

        Args:
            consumer_id: Consumer ID such as C01881.
        """
        return run("get_consumer", consumer_id=consumer_id)

    @beta_tool
    def top_consumers(feeder: str | None = None, dt: str | None = None, band: str | None = None, n: int = 10,
                      include_inspected: bool = False) -> str:
        """Highest-risk consumers (the inspection queue), optionally filtered. Returns ID, DT, feeder, category,
        risk score, band and the top reason.

        Args:
            feeder: Feeder ID such as FDR-05.
            dt: Distribution transformer ID such as DT-032.
            band: High, Medium or Low.
            n: How many to return (max 50).
            include_inspected: Also include consumers already inspected.
        """
        return run("top_consumers", feeder=feeder, dt=dt, band=band, n=n, include_inspected=include_inspected)

    @beta_tool
    def get_transformer(dt_id: str) -> str:
        """One distribution transformer (DT): feeder, capacity, consumers, high/medium-risk counts, baseline and recent
        unaccounted energy (% of input), excess kWh/day, area risk, and its 5 riskiest consumers.

        Args:
            dt_id: DT ID such as DT-032.
        """
        return run("get_transformer", dt_id=dt_id)

    @beta_tool
    def rank_transformers(by: str = "excess_kwh_day", n: int = 5, feeder: str | None = None) -> str:
        """Rank transformers. by: excess_kwh_day (unaccounted energy volume), loss_recent (recent loss %), risk
        (area risk score) or high (number of high-risk consumers).

        Args:
            by: Ranking field.
            n: How many (max 50).
            feeder: Only DTs on this feeder, e.g. FDR-05.
        """
        return run("rank_transformers", by=by, n=n, feeder=feeder)

    @beta_tool
    def feeder_summary() -> str:
        """Every feeder: DT count, consumers, high-risk count, excess loss %, area risk score."""
        return run("feeder_summary")

    @beta_tool
    def network_summary() -> str:
        """Network totals: consumers, DTs, feeders, high/medium counts, high-risk not yet inspected, revenue at risk
        per month (INR), meter data issues, data as-of date."""
        return run("network_summary")

    @beta_tool
    def model_summary() -> str:
        """Active model version, outcomes it was trained on, promotion-gate verdict, headline hit rate vs random and
        rule-based inspection on the holdout, calibration, drift status, version history."""
        return run("model_summary")

    @beta_tool
    def explain_term(term: str) -> str:
        """Meaning of a console term or metric: hit rate, risk score, band, PR-AUC, calibration, promotion gate,
        unaccounted energy, data issue, meter fault, drift.

        Args:
            term: The term to explain.
        """
        return run("explain_term", term=term)

    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    if lang == "gu":
        msgs = msgs + [{"role": "system", "content": "The console is set to Gujarati: reply in Gujarati."}]
    runner = _client().beta.messages.tool_runner(
        model=MODEL, max_tokens=4000, system=system, messages=msgs,
        tools=[get_consumer, top_consumers, get_transformer, rank_transformers, feeder_summary, network_summary,
               model_summary, explain_term],
        output_config={"effort": "low"}, max_iterations=6,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    final = runner.until_done()
    if final.stop_reason == "refusal":
        raise RuntimeError("refusal")
    text = "\n".join(b.text for b in final.content if b.type == "text").strip()
    if final.stop_reason == "max_tokens":
        text += "\n\n(Answer cut short.)"
    return text


# ------------------------------------------------------------------ rules
GU = {
    "risk": "જોખમ", "not_inspected": "હજી તપાસ બાકી", "theft_confirmed": "ચોરી પુષ્ટિ", "no_irregularity": "કોઈ ગેરરીતિ નથી",
    "meter_fault": "મીટર ખામી", "why": "કારણો", "unbilled": "અંદાજિત બિનબિલ વપરાશ", "top": "સૌથી વધુ જોખમી ગ્રાહકો",
    "on": "પર", "none": "આ ફિલ્ટર માટે કોઈ ગ્રાહક નથી.",
}
STATUS_EN = {"not_inspected": "not yet inspected", "theft_confirmed": "theft confirmed",
             "no_irregularity": "no irregularity found", "meter_fault": "meter fault"}


def _num(q: str, default: int) -> int:
    q2 = ID_RE.sub(" ", q)
    m = re.search(r"\b(\d{1,2})\b", q2)
    return max(1, min(int(m.group(1)), 50)) if m else default


def _help(lang: str) -> str:
    if lang == "gu":
        return ("હું જોખમ સ્કોર વિશેના પ્રશ્નોના જવાબ આપી શકું છું. ઉદાહરણ:\n• C01881 શા માટે ફ્લેગ થયો?\n"
                "• FDR-05 પર ટોચના 10 ગ્રાહકો\n• કયા DT પર સૌથી વધુ નુકસાન છે?\n• હિટ રેટ એટલે શું?")
    return ("I can answer questions about the live risk scores. Try:\n• Why was C01881 flagged?\n"
            "• Top 10 risky consumers on FDR-05\n• Which DT has the highest loss?\n• Tell me about DT-032\n"
            "• What does hit rate mean?\n• How good is the model?\n• How much revenue is at risk?")


def answer_rules(state: dict, q: str, lang: str = "en") -> str:
    t = _tools(state)
    gu = lang == "gu" or any("઀" <= ch <= "૿" for ch in q)
    ql = q.lower()
    ids = ID_RE.findall(q.upper())
    cons = [i for i in ids if i.startswith("C")]
    dts = [i for i in ids if i.startswith("DT-")]
    fdrs = [i for i in ids if i.startswith("FDR-")]
    asks_meaning = any(w in ql for w in ("what is", "what's", "what does", "meaning", "mean", "explain", "define",
                                         "એટલે", "શું છે", "અર્થ"))
    try:
        if cons:
            return "\n\n".join(_explain_consumer(t["get_consumer"](c), gu) for c in cons[:3])
        term = _match_term(q)
        if term and (asks_meaning or not (dts or fdrs)):
            if not any(w in ql for w in ("top", "list", "highest", "which")):
                return GLOSSARY[term]
        if any(w in ql for w in ("revenue", "how many high", "summary", "overview", "total", "kpi", "આવક")):
            k = t["network_summary"]()["kpis"]
            return (f"{k['high_uninspected']:,} high-risk consumers are not yet inspected (of {k['consumers']:,} on "
                    f"{k['dts']} DTs). Estimated revenue at risk: ₹{k['revenue_at_risk_inr_month']:,.0f} per month "
                    f"({k['leak_kwh_month_high']:,.0f} kWh unbilled). {k['data_issues']:,} meters have data issues "
                    "and go to the metering team.")
        if dts and not any(w in ql for w in ("top", "list", "risky", "consumers", "ગ્રાહક")):
            return _explain_dt(t["get_transformer"](dts[0]), gu)
        if any(w in ql for w in ("top", "risk", "list", "queue", "inspect", "visit", "highest risk", "consumers",
                                 "ગ્રાહક", "ટોચ", "જોખમ")) and not any(w in ql for w in ("loss", "નુકસાન", "transformer")):
            band = "High" if "high" in ql else "Medium" if "medium" in ql else None
            n = _num(q, 10)
            rows = t["top_consumers"](feeder=fdrs[0] if fdrs else None, dt=dts[0] if dts else None, band=band, n=n)
            return _list_consumers(rows, gu, fdrs[0] if fdrs else (dts[0] if dts else None))
        if any(w in ql for w in ("loss", "transformer", "dt", "unaccounted", "નુકસાન", "ટ્રાન્સફોર્મર")):
            by = "loss_recent" if "%" in ql or "percent" in ql else "excess_kwh_day"
            rows = t["rank_transformers"](by=by, n=_num(q, 5), feeder=fdrs[0] if fdrs else None)
            head = ("Transformers with the most unaccounted energy" if not gu else "સૌથી વધુ બિનહિસાબી ઊર્જાવાળા DT")
            return head + ":\n" + "\n".join(
                f"• {d['id']} ({d['feeder']}): loss {d['loss_recent']:.1f}% vs baseline {d['loss_base']:.1f}%, "
                f"≈{d['excess_kwh_day']:.0f} kWh/day unaccounted, {d['high']} high-risk consumers" for d in rows)
        if "feeder" in ql or fdrs:
            rows = sorted(t["feeder_summary"](), key=lambda f: -f["risk"])
            if fdrs:
                rows = [f for f in rows if f["id"] == fdrs[0]] or rows
            return "Feeders by area risk:\n" + "\n".join(
                f"• {f['id']}: risk {f['risk']:.0f}, {f['high']} high-risk of {f['consumers']} consumers on {f['dts']} DTs, "
                f"excess loss {f['excess_loss_pct']:.1f}%" for f in rows)
        if any(w in ql for w in ("model", "accura", "how good", "version", "perform", "મોડલ")):
            return _explain_model(t["model_summary"]())
        if term:
            return GLOSSARY[term]
    except NotFound as e:
        return (f"{e} ID મળ્યો નથી. ID તપાસો." if gu else f"I couldn't find {e}. Check the ID (e.g. C01881, DT-032, FDR-05).")
    return _help("gu" if gu else "en")


def _explain_consumer(d: dict, gu: bool) -> str:
    reasons = (d["reasons_gu"] if gu else d["reasons"]) or ["Overall consumption profile is a statistical outlier among peers"]
    status = GU.get(d["inspection_status"]) if gu else STATUS_EN.get(d["inspection_status"], d["inspection_status"])
    est = d["estimates"]
    if gu:
        head = (f"{d['consumer_id']}: {GU['risk']} {d['risk_score']:.0f}/100 ({d['band']}), {d['dt_id']} · {d['feeder_id']}. "
                f"{status}.\n{GU['why']}:")
        tail = f"{GU['unbilled']}: {est['leak_kwh_month']:,.0f} kWh/મહિનો (₹{est['revenue_inr_month']:,.0f})."
    else:
        head = (f"{d['consumer_id']} has risk {d['risk_score']:.0f}/100 ({d['band']}) on {d['dt_id']}, {d['feeder_id']}; "
                f"{status}.\nWhy it was flagged:")
        drivers = ", ".join(x["label"].lower() for x in d.get("drivers", [])[:3] if x["contribution"] > 0)
        tail = (f"Estimated unbilled use {est['leak_kwh_month']:,.0f} kWh/month (₹{est['revenue_inr_month']:,.0f})."
                + (f" Biggest model drivers: {drivers}." if drivers else "")
                + (" Note: meter data issue, check the meter first." if d["data_issue"] else ""))
    return head + "\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1)) + "\n" + tail


def _explain_dt(d: dict, gu: bool) -> str:
    top = ", ".join(f"{c['consumer_id']} ({c['risk_score']:.0f})" for c in d["top_consumers"])
    return (f"{d['id']} on {d['feeder']} ({d['kva']} kVA, {d['consumers']} consumers): unaccounted energy "
            f"{d['loss_recent']:.1f}% recently vs {d['loss_base']:.1f}% baseline, about {d['excess_kwh_day']:.0f} kWh/day. "
            f"{d['high']} high and {d['medium']} medium-risk consumers. Riskiest: {top}.")


def _list_consumers(rows: list[dict], gu: bool, where: str | None) -> str:
    if not rows:
        return GU["none"] if gu else "No uninspected consumers match that filter."
    head = (f"{GU['top']}{' (' + where + ')' if where else ''}:" if gu
            else f"Top {len(rows)} uninspected consumers by risk{' on ' + where if where else ''}:")
    return head + "\n" + "\n".join(
        f"{i}. {r['consumer_id']} · {r['risk_score']:.0f} {r['band']} · {r['dt_id']} · "
        f"{(r['top_reason_gu'] if gu and r['top_reason_gu'] else r['top_reason'])}" for i, r in enumerate(rows, 1))


def _explain_model(m: dict) -> str:
    h = m.get("headline") or {}
    s = f"Active model {m['active_model']}, trained on {m['trained_on_outcomes'] or 0:,} field outcomes."
    if h.get("hit_rate") is not None:
        s += (f" On the holdout area, {100 * h['hit_rate']:.0f}% of its top {h['k']} alerts were theft, vs "
              f"{100 * h['base_rate']:.1f}% for random checks and {100 * h['rule_hit_rate']:.0f}% for a >50%-drop rule.")
    if m.get("calibration_brier") is not None:
        s += f" Calibration Brier score {m['calibration_brier']:.4f}."
    if m.get("drift_status"):
        s += f" Drift: {m['drift_status']}."
    g = m.get("gate") or {}
    if g.get("reason"):
        s += f"\nLast promotion decision: {g['reason']}"
    return s
