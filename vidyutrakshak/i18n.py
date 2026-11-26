"""Reason-code text in English and Gujarati.

Reasons are stored as (code, params) and rendered per language, so a new
language is one more template table. Gujarati wording is a first draft and
should be reviewed by a native speaker and a vigilance officer before field use.
"""
from __future__ import annotations

CATEGORY = {
    "en": {"RES": "residential", "COM": "commercial", "AGR": "agricultural", "IND": "industrial"},
    "gu": {"RES": "રહેણાંક", "COM": "વાણિજ્યિક", "AGR": "કૃષિ", "IND": "ઔદ્યોગિક"},
}

TEMPLATES = {
    "en": {
        "drop": "Consumption fell {pct}% vs same-DT {cat} peers, starting ~{weeks} weeks ago",
        "low_days": "{pct}% of recent days read below 30% of expected",
        "flat": "Readings unusually flat: day-to-day variation down {pct}%",
        "erratic": "Erratic on/off pattern: day-to-day variation up {x}×",
        "cap": "Usage pinned at a ceiling on {pct}% of recent days",
        "tamper": "{n} meter tamper alarm(s) (cover-open / magnetic) in last 120 days",
        "low_load": "Uses only {pct}% of sanctioned {kw} kW load",
        "dt_rise": "{dt} unaccounted energy up {pp} pts vs baseline",
        "dt_step": "{dt} losses stepped up {pp} pts in the same weeks, matching ~{match}% of this shortfall",
        "outlier": "Overall consumption profile is a statistical outlier among peers",
    },
    "gu": {
        "drop": "સમાન DT પરના {cat} ગ્રાહકોની સરખામણીએ વપરાશ {pct}% ઘટ્યો, લગભગ {weeks} અઠવાડિયાથી",
        "low_days": "તાજેતરના {pct}% દિવસોમાં રીડિંગ અપેક્ષિત વપરાશના 30% કરતાં ઓછું",
        "flat": "રીડિંગ અસામાન્ય રીતે સપાટ: દૈનિક ફેરફાર {pct}% ઘટ્યો",
        "erratic": "અનિયમિત ચાલુ/બંધ પેટર્ન: દૈનિક ફેરફાર {x}× વધ્યો",
        "cap": "તાજેતરના {pct}% દિવસોમાં વપરાશ એક જ મર્યાદા પર અટકેલો",
        "tamper": "છેલ્લા 120 દિવસમાં {n} મીટર ટેમ્પર એલાર્મ (કવર ખોલવું / મેગ્નેટ)",
        "low_load": "મંજૂર {kw} kW લોડમાંથી માત્ર {pct}% વપરાશ",
        "dt_rise": "{dt} પર બિનહિસાબી ઊર્જા બેઝલાઇન કરતાં {pp} ટકા પોઇન્ટ વધી",
        "dt_step": "એ જ અઠવાડિયામાં {dt} નું નુકસાન {pp} ટકા પોઇન્ટ વધ્યું, જે આ ઘટાડાના લગભગ {match}% જેટલું છે",
        "outlier": "સમાન ગ્રાહકોની સરખામણીએ કુલ વપરાશ પેટર્ન આંકડાકીય રીતે અસામાન્ય",
    },
}

LANGS = tuple(TEMPLATES)


def render(code: str, params: dict, lang: str = "en") -> str:
    p = dict(params)
    if "cat" in p:
        p["cat"] = CATEGORY[lang].get(p["cat"], p["cat"])
    return TEMPLATES[lang][code].format(**p)
