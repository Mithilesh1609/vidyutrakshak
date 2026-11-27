"""Explainable PDF reports (fpdf2, no browser needed).

  consumer_report(state, cid, lang)   one consumer's case file for the field team
  route_report(state, items)          the day's inspection route sheet
  model_report(state)                 how the model decides and how well it does

Gujarati is rendered with Noto Sans Gujarati and HarfBuzz text shaping so
conjuncts and vowel signs come out right. Fonts ship in ./fonts (SIL OFL).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .i18n import TEMPLATES
from .views import DISCLAIMER, FEATURE_LABELS, consumer_detail, dt_summary, model_summary

FONTS = Path(__file__).parent / "fonts"
CAT = {"RES": "Residential", "COM": "Commercial", "AGR": "Agricultural", "IND": "Industrial"}
STATUS = {"not_inspected": "Not yet inspected", "theft_confirmed": "Theft confirmed",
          "no_irregularity": "No irregularity", "meter_fault": "Meter fault"}
INK, MUTED, LINE = (24, 32, 48), (100, 110, 125), (215, 220, 228)
ACCENT, EXPECTED = (30, 90, 180), (140, 150, 165)
BAND = {"High": (200, 50, 45), "Medium": (215, 140, 20), "Low": (60, 140, 90)}
UP, DOWN = (200, 50, 45), (60, 140, 90)


def _is_gu(s: str) -> bool:
    return any("઀" <= ch <= "૿" for ch in s)


class Report(FPDF):
    def __init__(self, title: str, meta: dict, orientation: str = "P"):
        super().__init__(orientation=orientation, format="A4")
        self.doc_title, self.meta = title, meta
        for style, w in (("", "Regular"), ("B", "Bold")):
            self.add_font("NS", style, str(FONTS / f"NotoSans-{w}.ttf"))
            self.add_font("NG", style, str(FONTS / f"NotoSansGujarati-{w}.ttf"))
        self.set_text_shaping(True)
        self.set_margins(14, 14, 14)
        self.set_auto_page_break(True, margin=16)
        self.set_title(title)
        self.set_author("VidyutRakshak")
        self.generated = datetime.now().strftime("%d %b %Y %H:%M")
        self.add_page()

    # -------------------------------------------------------------- frame
    def header(self):
        self.font("B", 9, ACCENT)
        self.cell(0, 5, "VidyutRakshak", new_x=XPos.LMARGIN, new_y=YPos.TOP)
        m = self.meta
        right = f"Model {m.get('active_model', '?')} · reads to {str(m.get('as_of', ''))[:10]}"
        if m.get("simulated", True) is not False:
            right += " · SIMULATED DATA"
        self.font("", 8, MUTED)
        self.cell(0, 5, right, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.font("", 7.5, MUTED)
        self.cell(0, 4, f"{self.doc_title} · generated {self.generated}", new_x=XPos.LMARGIN, new_y=YPos.TOP)
        self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="R")

    # -------------------------------------------------------------- text
    def font(self, style: str = "", size: float = 10, color=INK, gu: bool = False):
        self.set_font("NG" if gu else "NS", style, size)
        self.set_text_color(*color)

    def para(self, text: str, size: float = 10, style: str = "", color=INK, h: float | None = None, w: float = 0):
        self.font(style, size, color, gu=_is_gu(text))
        self.multi_cell(w, h or size * 0.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h1(self, text: str):
        self.para(text, 17, "B", h=8)
        self.ln(1)

    def h2(self, text: str):
        self.ln(3)
        self.para(text.upper(), 8.5, "B", MUTED, h=5)
        self.ln(0.5)

    def kv(self, items: list[tuple[str, str]], cols: int = 4):
        w = (self.w - self.l_margin - self.r_margin) / cols
        x0 = self.l_margin
        for n, (k, v) in enumerate(items):
            if n and n % cols == 0:
                self.ln(11)
            x = x0 + (n % cols) * w
            y = self.get_y()
            self.set_xy(x, y)
            self.font("", 7.5, MUTED)
            self.cell(w, 4, k.upper())
            self.set_xy(x, y + 4)
            self.font("B", 10.5, INK, gu=_is_gu(v))
            self.cell(w, 6, v)
            self.set_xy(x0, y)
        self.ln(12)

    # -------------------------------------------------------------- charts
    def line_chart(self, series: list[dict], labels: list[str], h: float = 42, y_fmt=lambda v: f"{v:.0f}",
                   ref: tuple[float, str] | None = None, legend: list[tuple[str, tuple, bool]] | None = None):
        """series: [{values, color, dash}] sharing one y scale; labels are ISO dates."""
        x0, y0 = self.l_margin + 10, self.get_y() + 2
        w = self.w - self.r_margin - x0
        vals = [v for s in series for v in s["values"] if v is not None] + ([ref[0]] if ref else [])
        hi = max(vals + [1e-9]) * 1.08
        step = _nice(hi / 4)
        top = step * max(1, np.ceil(hi / step))
        n = len(labels)
        X = lambda i: x0 + w * i / max(1, n - 1)  # noqa: E731
        Y = lambda v: y0 + h * (1 - v / top)  # noqa: E731
        self.set_line_width(0.15)
        self.set_draw_color(*LINE)
        self.font("", 6.5, MUTED)
        v = 0.0
        while v <= top + 1e-9:
            self.line(x0, Y(v), x0 + w, Y(v))
            self.set_xy(self.l_margin - 2, Y(v) - 2)
            self.cell(11, 4, y_fmt(v), align="R")
            v += step
        last_m = None
        for i, d in enumerate(labels):
            m = d[5:7]
            if m != last_m and i < n - 2:
                last_m = m
                self.set_xy(X(i) - 1, y0 + h + 0.5)
                self.cell(10, 3.5, datetime.strptime(d[:10], "%Y-%m-%d").strftime("%b"))
        if ref:
            self.set_draw_color(*EXPECTED)
            self.set_dash_pattern(dash=0.6, gap=1.2)
            self.line(x0, Y(ref[0]), x0 + w, Y(ref[0]))
            self.set_dash_pattern()
            self.set_xy(x0 + w - 50, Y(ref[0]) - 4)
            self.cell(50, 3.5, ref[1], align="R")
        for s in series:
            self.set_draw_color(*s["color"])
            self.set_line_width(s.get("width", 0.5))
            if s.get("dash"):
                self.set_dash_pattern(dash=1.2, gap=1)
            pts = [(X(i), Y(v)) for i, v in enumerate(s["values"]) if v is not None]
            for a, b in zip(pts, pts[1:]):
                self.line(*a, *b)
            self.set_dash_pattern()
        self.set_line_width(0.2)
        self.set_y(y0 + h + 5)
        if legend:
            self.font("", 7.5, MUTED)
            for label, color, dashed in legend:
                x, y = self.get_x(), self.get_y() + 2
                self.set_draw_color(*color)
                self.set_line_width(0.6)
                if dashed:
                    self.set_dash_pattern(dash=1.2, gap=1)
                self.line(x, y, x + 6, y)
                self.set_dash_pattern()
                self.set_x(x + 8)
                self.cell(self.get_string_width(label) + 6, 4, label)
            self.set_line_width(0.2)
            self.ln(6)

    def bars(self, rows: list[tuple[str, float]], signed: bool = True, fmt=lambda v: f"{v:+.2f}", label_w: float = 62):
        """Horizontal bars; signed ones grow right (raises risk) or left (lowers it)."""
        x0 = self.l_margin + label_w
        w = self.w - self.r_margin - x0 - 16
        mx = max([abs(v) for _, v in rows] + [1e-9])
        mid = x0 + w / 2 if signed else x0
        span = w / 2 if signed else w
        for label, v in rows:
            y = self.get_y()
            self.font("", 8.5, INK)
            self.set_xy(self.l_margin, y)
            self.cell(label_w - 2, 6, label, align="R")
            bw = span * abs(v) / mx
            self.set_fill_color(*((UP if v > 0 else DOWN) if signed else ACCENT))
            self.rect(mid if v >= 0 else mid - bw, y + 1.3, bw, 3.4, style="F")
            self.font("", 7.5, MUTED)
            tx = (mid + bw + 1) if v >= 0 else (mid - bw - 13)
            self.set_xy(tx, y)
            self.cell(12, 6, fmt(v), align="L" if v >= 0 else "R")
            self.set_y(y + 6)
        if signed:
            self.set_draw_color(*MUTED)
            self.line(mid, self.get_y() - 6 * len(rows), mid, self.get_y())

    def table(self, head: list[str], rows: list[list[str]], widths: list[float], size: float = 8,
              align: list[str] | None = None):
        align = align or ["L"] * len(head)

        def draw_head():
            self.set_fill_color(238, 241, 246)
            self.font("B", size - 0.5, MUTED)
            for t, w, a in zip(head, widths, align):
                self.cell(w, 6, t, fill=True, align=a)
            self.ln(6)

        draw_head()
        for r in rows:
            # Row height = tallest wrapped cell.
            lines = []
            for t, w in zip(r, widths):
                self.font("", size, gu=_is_gu(t))
                lines.append(len(self.multi_cell(w, 4.2, t, dry_run=True, output="LINES")) or 1)
            rh = 4.2 * max(lines) + 1.6
            if self.get_y() + rh > self.page_break_trigger:
                self.add_page()
                draw_head()
            y = self.get_y()
            x = self.l_margin
            for t, w, a in zip(r, widths, align):
                self.set_xy(x, y + 0.8)
                self.font("", size, gu=_is_gu(t))
                self.multi_cell(w, 4.2, t, align=a)
                x += w
            self.set_draw_color(*LINE)
            self.line(self.l_margin, y + rh, self.l_margin + sum(widths), y + rh)
            self.set_xy(self.l_margin, y + rh)

    def box(self, text: str, color=(246, 248, 251)):
        self.set_fill_color(*color)
        self.font("", 8.5, MUTED)
        self.multi_cell(0, 4.5, text, fill=True, padding=2.5, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def render(self) -> bytes:
        self.alias_nb_pages()
        return bytes(self.output())


def _nice(v: float) -> float:
    if v <= 0:
        return 1.0
    p = 10 ** np.floor(np.log10(v))
    m = v / p
    return float((1 if m <= 1 else 2 if m <= 2 else 5 if m <= 5 else 10) * p)


def _weekly(dates: list[str], values: list, week_starts: list[str]) -> tuple[list[str], list]:
    """Daily values -> weekly means aligned to the dashboard's week starts."""
    starts = [w for w in week_starts if w >= dates[0][:10]] or [dates[0][:10]]
    out = []
    for a, b in zip(starts, starts[1:] + ["9999"]):
        v = [x for d, x in zip(dates, values) if a <= d[:10] < b and x is not None]
        out.append(round(float(np.mean(v)), 2) if v else None)
    return starts, out


def _inr(x: float) -> str:
    return f"₹{x / 1e5:.1f} lakh" if x >= 1e5 else f"₹{x:,.0f}"


# ================================================================== reports
def consumer_report(state: dict, cid: str, lang: str = "en") -> bytes:
    d = consumer_detail(state, cid, days=min(364, state["ctx"]["X"].shape[1]), with_shap=True)
    meta = state["dash"]["meta"]
    pdf = Report(f"Case report {d['consumer_id']}", meta)

    # Title row with the score on the right
    y = pdf.get_y()
    pdf.font("", 8.5, MUTED)
    pdf.cell(0, 4, f"{CAT.get(d['category'], d['category'])} consumer · {d['dt_id']} on {d['feeder_id']}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.h1(f"Consumer {d['consumer_id']}")
    pdf.set_xy(pdf.w - pdf.r_margin - 40, y)
    pdf.font("B", 26, BAND.get(d["band"], INK))
    pdf.cell(40, 12, f"{d['risk_score']:.0f}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.w - pdf.r_margin - 40)
    pdf.font("", 7.5, MUTED)
    pdf.cell(40, 4, f"RISK / 100 · {d['band'].upper()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(max(pdf.get_y(), y + 20))

    pdf.kv([("Status", STATUS.get(d["inspection_status"], d["inspection_status"])),
            ("Unbilled estimate", f"{d['estimates']['leak_kwh_month']:,.0f} kWh/mo"),
            ("Revenue estimate", f"{_inr(d['estimates']['revenue_inr_month'])}/mo"),
            ("Tamper alarms", str(d["tamper_events"]))])
    if d["data_issue"]:
        pdf.box("Meter data issue: more than 20% of reads are missing. Route to the metering team before vigilance.",
                (253, 243, 225))

    first, second = (d["reasons_gu"], d["reasons"]) if lang == "gu" else (d["reasons"], d["reasons_gu"])
    pdf.h2("Why it was flagged" if lang != "gu" else "Why it was flagged · શા માટે ફ્લેગ થયું")
    if not d["reasons"]:
        first = ["Overall consumption profile is a statistical outlier among peers"]
        second = []
    for n, r in enumerate(first, 1):
        pdf.para(f"{n}.  {r}", 10.5, h=5.6)
        if n - 1 < len(second):
            pdf.set_x(pdf.l_margin + 5)
            pdf.para(second[n - 1], 9, color=MUTED, h=4.8)
        pdf.ln(0.8)

    pdf.h2("What moved the score (model contributions)")
    pdf.bars([(x["label"], x["contribution"]) for x in d["drivers"]])
    pdf.font("", 7.5, MUTED)
    pdf.multi_cell(0, 4, "Red bars push the risk up, green bars pull it down (TreeSHAP, log-odds units). "
                         "The reasons above are the red bars translated into plain language.",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    weeks = meta["week_starts"]
    lab, rec = _weekly(d["daily"]["dates"], d["daily"]["recorded_kwh"], weeks)
    _, exp = _weekly(d["daily"]["dates"], d["daily"]["expected_kwh"], weeks)
    pdf.h2("Metered vs expected use, weekly average kWh/day (last 12 months)")
    pdf.line_chart([{"values": exp, "color": EXPECTED, "dash": True, "width": 0.4},
                    {"values": rec, "color": ACCENT, "width": 0.6}], lab, h=38,
                   y_fmt=lambda v: f"{v:.0f}" if v >= 10 or v == 0 else f"{v:.1f}",
                   legend=[("Metered", ACCENT, False), ("Expected from same-DT peers", EXPECTED, True)])

    try:
        dt = dt_summary(state, d["dt_id"])
        series = next(x for x in state["dash"]["dts"] if x["id"] == d["dt_id"])["loss_series"]
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.h2(f"{d['dt_id']} unaccounted energy, % of DT input")
        pdf.line_chart([{"values": series, "color": UP, "width": 0.55}], weeks, h=30,
                       y_fmt=lambda v: f"{v:.0f}%", ref=(dt["loss_base"], f"baseline {dt['loss_base']:.1f}%"))
        pdf.font("", 8, MUTED)
        pdf.multi_cell(0, 4, f"Recent loss {dt['loss_recent']:.1f}% vs baseline {dt['loss_base']:.1f}%, "
                             f"about {dt['excess_kwh_day']:.0f} kWh/day unaccounted. "
                             f"{dt['high']} high-risk consumers on this DT.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    except Exception:
        pass

    if pdf.get_y() > 225:
        pdf.add_page()
    pdf.h2("Field inspection record")
    pdf.font("", 9.5)
    pdf.set_draw_color(*MUTED)
    for opt in ("Theft confirmed", "No irregularity", "Meter fault (to metering team)"):
        x, yy = pdf.get_x(), pdf.get_y()
        pdf.rect(x, yy + 1, 3.5, 3.5)
        pdf.set_x(x + 5.5)
        pdf.cell(55, 6, opt)
    pdf.ln(9)
    for label in ("Seal / meter no.", "What was found", "Officer name & signature", "Date"):
        pdf.font("", 8.5, MUTED)
        pdf.cell(42, 7, label)
        y2 = pdf.get_y() + 6
        pdf.line(pdf.l_margin + 42, y2, pdf.w - pdf.r_margin, y2)
        pdf.ln(8)
    pdf.ln(2)
    pdf.box(DISCLAIMER + " Gujarati wording is a draft pending review by a native speaker.")
    return pdf.render()


def route_report(state: dict, items: list[dict], title_note: str = "") -> bytes:
    meta = state["dash"]["meta"]
    pdf = Report("Inspection route sheet", meta, orientation="L")
    pdf.h1("Inspection route sheet")
    pdf.para(f"{len(items)} consumers, highest risk first within each DT. {title_note}".strip(), 9.5, color=MUTED)
    rank = {r["consumer_id"]: i + 1 for i, r in enumerate(items)}
    items = sorted(items, key=lambda r: (r["feeder_id"], r["dt_id"], -r["risk_score"]))
    widths = [11, 22, 20, 22, 16, 110, 34, 34]
    head = ["#", "Consumer", "DT", "Category", "Risk", "Top reasons", "Outcome", "Notes / seal no."]
    feeders = list(dict.fromkeys(r["feeder_id"] for r in items))
    for f in feeders:
        if pdf.get_y() > pdf.page_break_trigger - 30:  # don't strand a feeder heading at the page foot
            pdf.add_page()
        rows = [r for r in items if r["feeder_id"] == f]
        pdf.h2(f"Feeder {f} · {len(rows)} visits · {len({r['dt_id'] for r in rows})} DTs")
        pdf.table(head, [[str(rank[r["consumer_id"]]), r["consumer_id"], r["dt_id"],
                          CAT.get(r["category"], r["category"]), f"{r['risk_score']:.0f} {r['band'][0]}",
                          ("; ".join(r["reasons"][:2]) or "Statistical outlier among peers")
                          + (" · meter data issue" if r.get("data_issue") else ""),
                          "[ ] theft  [ ] clean\n[ ] meter fault", ""] for r in rows],
                  widths, size=7.8, align=["R", "L", "L", "L", "R", "L", "L", "L"])
    if not items:
        pdf.para("No uninspected consumers match these filters.", 10)
    pdf.ln(4)
    pdf.box(DISCLAIMER + " Record outcomes in the console (or the CSV route sheet) so the next model learns from them.")
    return pdf.render()


def model_report(state: dict) -> bytes:
    dash, meta = state["dash"], state["dash"]["meta"]
    ms = model_summary(state)
    hl = dash.get("headline") or {}
    pdf = Report("Model explainability summary", meta)
    pdf.h1("How the risk model decides")
    pdf.para("VidyutRakshak ranks consumers by the probability that an inspection finds theft. A gradient-boosted "
             "model (LightGBM) combines meter, billing and transformer energy-balance signals; each alert is "
             "explained by the features that pushed its score up (TreeSHAP).", 9.5, color=MUTED, h=4.8)
    gate = ms.get("gate") or {}
    pdf.ln(2)
    pdf.kv([("Active model", ms["active_model"]), ("Trained on", f"{ms['trained_on_outcomes'] or 0:,} outcomes"),
            ("Confirmed thefts", f"{ms['confirmed_thefts_in_training'] or 0:,}"),
            ("Score drift", str(ms.get("drift_status") or "n/a"))])
    if gate.get("reason"):
        pdf.box("Promotion gate: " + gate["reason"])

    if dash.get("ablation"):
        pdf.h2(f"Does it beat the status quo? Holdout area, top {hl.get('k', '')} alerts")
        rows = []
        for a in dash["ablation"]:
            rows.append([a["approach"], f"{100 * a['precision_at_k']:.0f}%", f"{100 * a['recall_at_k']:.0f}%",
                         f"{a['pr_auc']:.2f}", f"{a['legit_drop_false_alarms_at_k']:.0f}"])
        pdf.table(["Approach", "Hit rate", "Thefts caught", "PR-AUC", "Visits to legit drops"], rows,
                  [82, 22, 26, 20, 32], size=8.2, align=["L", "R", "R", "R", "R"])
        if hl.get("basis"):
            pdf.font("", 7.5, MUTED)
            pdf.multi_cell(0, 4, hl["basis"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    try:
        m = state["reg"].load()
        gain = m["booster"].feature_importance("gain")
        tot = float(gain.sum()) or 1.0
        imp = sorted(zip(m["features"], gain / tot), key=lambda t: -t[1])
        pdf.h2("What the model relies on overall (share of total split gain)")
        pdf.bars([(FEATURE_LABELS.get(f, f), float(v)) for f, v in imp], signed=False,
                 fmt=lambda v: f"{100 * v:.0f}%")
    except Exception:
        pass

    cal = dash.get("calibration")
    if cal:
        if pdf.get_y() > 215:
            pdf.add_page()
        pdf.h2("Does a score mean what it says? Calibration on the holdout")
        pdf.table(["Predicted range", "Consumers", "Avg predicted", "Observed theft rate"],
                  [[f"{100 * b['from']:.0f}–{100 * b['to']:.0f}%", f"{b['n']:,}", f"{100 * b['predicted']:.1f}%",
                    f"{100 * b['observed']:.1f}%"] for b in cal["bins"]], [40, 30, 34, 40], size=8.2,
                  align=["L", "R", "R", "R"])
        pdf.font("", 7.5, MUTED)
        pdf.multi_cell(0, 4, f"Brier score {cal['brier']:.4f} vs {cal['brier_base_rate']:.4f} for a constant base-rate "
                             "guess (lower is better).", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.h2("Reason codes shown to field officers")
    pdf.table(["Code", "Text template (English)"],
              [[k, v] for k, v in TEMPLATES["en"].items()], [24, 158], size=8)

    pdf.h2("Model versions")
    pdf.table(["Version", "Status", "Created", "Note"],
              [[v["version"], v["status"], v["created"], v["note"]] for v in ms["versions"]],
              [20, 24, 36, 102], size=8)
    pdf.ln(3)
    pdf.box(DISCLAIMER + " A new model goes live only if it is at least as good as the active one on field outcomes "
                         "the active model never saw.")
    return pdf.render()
