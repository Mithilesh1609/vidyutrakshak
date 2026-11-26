"""Build the narrated demo video: deck slides + a live screen recording of the console.

    scripts/demo_reset.sh          # in another terminal: clean demo at http://localhost:8000
    .venv/bin/python scripts/demo_video/make_video.py

Needs: ffmpeg, pdftoppm, LibreOffice (for the deck PDF), Chrome, and the Piper voice
en_US-ryan-high in ~/.local/share/piper-voices (see VOICE below for the download URL).

The recording records outcomes and retrains, so it changes the demo state.
Run scripts/demo_reset.sh again afterwards.

Outputs in dist/: VidyutRakshak_demo.mp4 (with subtitles), VidyutRakshak_demo_nosubs.mp4,
VidyutRakshak_demo.srt, VidyutRakshak_demo_script.txt.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("VR_VIDEO_WORK", ROOT / "dist" / "_video_build"))
OUT = ROOT / "dist"
URL = os.environ.get("VR_URL", "http://localhost:8000")
# https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx (+ .onnx.json)
VOICE = Path(os.environ.get("VR_VOICE", Path.home() / ".local/share/piper-voices/en_US-ryan-high.onnx"))
PIPER = ROOT / ".venv" / "bin" / "piper"
W, H = 1920, 1080
VIEW_W, VIEW_H = 1536, 864          # browser viewport, upscaled to 1080p (larger text on screen)
PAD = 0.6                           # silence after each scene, seconds

# ---------------------------------------------------------------------------
# Script. "say" is what the voice reads (spelled for the TTS); "sub" is the subtitle
# text when it differs. Slide numbers refer to the deck PDF (appendix A-D = 9-12).
SCENES = [
    dict(slide=1, say="Vidyut Rakshak. Explainable A.I. for electricity theft detection, built for G.U.V.N.L. "
                      "under the Seva First Innovation Challenge.",
         sub="VidyutRakshak: explainable AI for electricity theft detection, built for GUVNL "
             "under the Seva First Innovation Challenge."),
    dict(slide=2, say="Meter bypass, tampering and direct hooking hide among lakhs of consumers. "
                      "Inspections are picked from bill scrutiny and tip-offs, so only about five in a hundred visits find theft. "
                      "And simple rules that flag a big drop in consumption mostly send inspectors to vacant houses and rooftop solar homes."),
    dict(slide=3, say="Vidyut Rakshak compares every consumer, every day, with peers on the same transformer and tariff. "
                      "Then it checks the transformer's own energy balance. Theft makes energy vanish. A vacant home does not. "
                      "Every alert comes with reasons, and every inspection outcome teaches the model.",
         sub="VidyutRakshak compares every consumer, every day, with peers on the same transformer and tariff. "
             "Then it checks the transformer's own energy balance. Theft makes energy vanish. A vacant home does not. "
             "Every alert comes with reasons, and every inspection outcome teaches the model."),
    dict(slide=4, say="It runs on data G.U.V.N.L. already has: smart meter reads, billing, and transformer meters, "
                      "on a single C.P.U. server, with open source software only.",
         sub="It runs on data GUVNL already has: smart-meter reads, billing and transformer meters, "
             "on a single CPU server, with open-source software only."),
    dict(ui="kpis", say="Here is the working console, on a simulated network of five thousand consumers. "
                        "Seventy six high risk consumers are not yet inspected, with about sixty thousand rupees a month at risk. "
                        "Eighty five percent of the top alerts are real theft, against under five percent for random checks.",
         sub="Here is the working console, on a simulated network of 5,000 consumers. "
             "76 high-risk consumers are not yet inspected, with about ₹60,000 a month at risk. "
             "85% of the top alerts are real theft, against under 5% for random checks."),
    dict(ui="detail", say="Take the top alert. Three reasons, each a number an officer can check. "
                          "Consumption fell twenty three percent against commercial peers on the same transformer. "
                          "Usage is pinned at a ceiling. And there were two tamper alarms. "
                          "The chart shows the metered reading against what the peers suggest.",
         sub="Take the top alert. Three reasons, each a number an officer can check. "
             "Consumption fell 23% against commercial peers on the same transformer. "
             "Usage is pinned at a ceiling. And there were two tamper alarms. "
             "The chart shows the metered reading against what the peers suggest."),
    dict(ui="gujarati", say="Field officers can read the same reasons in Gujarati, with one click. "
                            "And today's inspection list downloads as a route sheet, grouped by feeder and transformer."),
    dict(ui="areas", say="The transformer view. D.T. zero zero two's losses rose from four point three to almost fifteen percent, "
                         "and here are the consumers behind it.",
         sub="The transformer view. DT-002's losses rose from 4.3% to almost 15%, "
             "and here are the consumers behind it."),
    dict(ui="ablation", say="How good is it? A rule that flags any drop above fifty percent sends inspectors to fifty three vacant or solar homes "
                            "in its top seventy one. Vidyut Rakshak sends them to six. "
                            "Where consumption fell, the transformer balance cuts those wasted visits by another nineteen percent.",
         sub="How good is it? A rule that flags any drop above 50% sends inspectors to 53 vacant or solar homes "
             "in its top 71. VidyutRakshak sends them to 6. "
             "Where consumption fell, the transformer balance cuts those wasted visits by another 19%."),
    dict(ui="calibration", say="And the scores mean what they say. On a holdout area, ninety six percent of consumers scored eighty to a hundred "
                               "were really stealing.",
         sub="And the scores mean what they say. On a holdout area, 96% of consumers scored 80–100 "
             "were really stealing."),
    dict(ui="outcome", say="When an officer inspects a consumer, the outcome goes into the vigilance log, with the officer's name."),
    dict(ui="retrain1", say="Now retrain. One new outcome is not enough evidence, so the promotion gate keeps version four live, "
                            "and saves the new model as a candidate.",
         sub="Now retrain. One new outcome is not enough evidence, so the promotion gate keeps v4 live, "
             "and saves the new model as a candidate."),
    dict(ui="retrain2", say="After a field round of sixty inspections, the new model is compared with version four, "
                            "on outcomes version four never saw. It holds up, so it is promoted. An admin can roll back at any time.",
         sub="After a field round of 60 inspections, the new model is compared with v4, "
             "on outcomes v4 never saw. It holds up, so it is promoted. An admin can roll back at any time."),
    dict(slide=9, say="We also tested the meter data path on real, public theft data: forty two thousand consumers from the S.G.C.C. dataset. "
                      "The top four percent of alerts are forty percent thieves, against an eight and a half percent base rate.",
         sub="We also tested the meter-data path on real, public theft data: 42,000 consumers from the SGCC dataset. "
             "The top 4% of alerts are 40% thieves, against an 8.5% base rate."),
    dict(slide=10, say="It holds up under stress: theft that began before the data starts, transformers with no meters, "
                       "and mapping errors. The hit rate stays between eighty one and eighty six percent. "
                       "Subtle bypass is the hardest case, at seventy three.",
         sub="It holds up under stress: theft that began before the data starts, transformers with no meters, "
             "and mapping errors. The hit rate stays between 81% and 86%. "
             "Subtle bypass is the hardest case, at 73%."),
    dict(slide=12, say="And it scales: one lakh consumers are scored in fifteen seconds on a laptop.",
         sub="And it scales: 1 lakh consumers are scored in 15 seconds on a laptop."),
    dict(slide=7, say="Explainable, in Gujarati, on premise, with pseudonymised data and no licence fees. "
                      "Vidyut Rakshak is ready for a one sub-division pilot with G.U.V.N.L.",
         sub="Explainable, in Gujarati, on-premise, with pseudonymised data and no licence fees. "
             "VidyutRakshak is ready for a one-sub-division pilot with GUVNL."),
    dict(slide=8, say="Thank you.", hold=2.5),
]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


# ---------------------------------------------------------------------------
def tts():
    (WORK / "audio").mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(SCENES):
        wav = WORK / "audio" / f"{i:02d}.wav"
        run([str(PIPER), "-m", str(VOICE), "-f", str(wav), "--length-scale", "1.05", "--sentence-silence", "0.25"],
            input=s["say"].encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        s["audio"] = wav
        s["dur"] = duration(wav)
    total = sum(s["dur"] + PAD for s in SCENES)
    print(f"narration: {len(SCENES)} scenes, {total:.0f} s")


def slides():
    pptx = ROOT / "VidyutRakshak_PS000205_Synopsis.pptx"
    pdf = ROOT / "VidyutRakshak_PS000205_Synopsis.pdf"
    if not pdf.exists() or pdf.stat().st_mtime < pptx.stat().st_mtime:
        sys.exit("The deck PDF is older than the .pptx: export it first (see deck/build_deck.js).")
    d = WORK / "slides"
    d.mkdir(parents=True, exist_ok=True)
    run(["pdftoppm", "-png", "-scale-to-x", str(W), "-scale-to-y", str(H), str(pdf), str(d / "s")])


# ---------------------------------------------------------------------------
CURSOR_JS = """
addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = '__cursor';
  c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:22px;height:22px;border-radius:50%;'
    + 'background:rgba(13,118,110,.35);border:2px solid #0d766e;z-index:99999;pointer-events:none;'
    + 'transform:translate(-50%,-50%);transition:width .12s,height .12s';
  document.body.appendChild(c);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', () => { c.style.width = c.style.height = '34px'; }, true);
  addEventListener('mouseup', () => { c.style.width = c.style.height = '22px'; }, true);
});
"""


class Ui:
    def __init__(self, page):
        self.p = page
        self.x, self.y = VIEW_W / 2, VIEW_H / 2

    def move(self, sel, steps=30):
        loc = self.p.locator(sel).first
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        self.p.mouse.move(x, y, steps=steps)
        self.x, self.y = x, y
        return loc

    def click(self, sel, pause=0.25):
        self.move(sel)
        time.sleep(pause)
        self.p.mouse.down(); time.sleep(0.08); self.p.mouse.up()

    def type(self, sel, text):
        self.click(sel)
        self.p.keyboard.type(text, delay=70)

    def scroll_to(self, sel, block="start"):
        self.p.evaluate("([s, b]) => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: b})", [sel, block])
        time.sleep(1.0)

    def top(self):
        self.p.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(0.6)

    def wait_toast(self, timeout=90):
        self.p.wait_for_selector(".toast", timeout=timeout * 1000)


def ui_actions(name, ui: Ui, dur):
    """Run the on-screen actions for one scene; sleep-based beats are spread over the narration."""
    p = ui.p
    if name == "kpis":
        time.sleep(0.8)
        for i in range(4):
            ui.move(f"#kpis .kpi:nth-child({i + 1})", steps=25)
            time.sleep(dur / 6)
    elif name == "detail":
        ui.click('#qlist .row[data-id="C01881"]')
        time.sleep(1.5)
        ui.move("#detail ol.reasons li", steps=25)
        time.sleep(dur * 0.45)
        ui.scroll_to("#detail .chartbox", block="center")
        ui.move("#detail .chartbox")
    elif name == "gujarati":
        ui.top()
        ui.click("#langchip")
        time.sleep(dur * 0.45)
        ui.click("#langchip")
        time.sleep(0.8)
        ui.click("#dl-list")
    elif name == "areas":
        ui.click("#t-areas")
        time.sleep(1.2)
        ui.click('.dtnode[data-dt="DT-002"]')
        time.sleep(dur * 0.3)
        ui.move("#dtdetail")
    elif name == "ablation":
        ui.click("#t-model")
        time.sleep(1.0)
        ui.scroll_to("#abl", block="center")
        ui.move("#abl tbody tr:nth-child(2)")
        time.sleep(dur * 0.35)
        ui.move("#abl tbody tr:last-child")
    elif name == "calibration":
        ui.scroll_to("#calpanel", block="center")
        ui.move("#caltable tbody tr:last-child")
    elif name == "outcome":
        ui.click("#t-queue")
        time.sleep(1.0)
        ui.click('#qlist .row[data-id="C03217"]')
        time.sleep(1.0)
        ui.scroll_to("#o-officer", block="center")
        ui.type("#o-officer", "R. Patel, Vigilance")
        time.sleep(0.4)
        ui.click('#detail [data-out="theft_confirmed"]')
        ui.wait_toast(20)
    elif name == "retrain1":
        ui.click("#t-model")
        time.sleep(1.0)
        ui.scroll_to("#retrainbox", block="center")
        ui.click("#retrain")
        ui.wait_toast()
        time.sleep(0.5)
        ui.move("#vtable tbody tr:first-child")
    elif name == "retrain2":
        ui.scroll_to("#retrainbox", block="center")
        ui.click("#fieldround")
        ui.wait_toast(60)
        time.sleep(2.0)
        p.wait_for_selector("#retrain", timeout=30000)
        ui.click("#retrain")
        p.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => /promoted|candidate/.test(t.textContent))",
                            timeout=120000)
        time.sleep(0.5)
        ui.move("#vtable tbody tr:first-child")


def record():
    from playwright.sync_api import sync_playwright
    (WORK / "rec").mkdir(parents=True, exist_ok=True)
    for f in (WORK / "rec").glob("*.webm"):
        f.unlink()
    ui_scenes = [s for s in SCENES if "ui" in s]
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context(viewport={"width": VIEW_W, "height": VIEW_H}, accept_downloads=True,
                            record_video_dir=str(WORK / "rec"), record_video_size={"width": VIEW_W, "height": VIEW_H})
        ctx.add_init_script(CURSOR_JS)
        page = ctx.new_page()
        t0 = time.monotonic()
        page.goto(URL + "/#queue")
        page.wait_for_selector("#qlist .row")
        page.wait_for_timeout(1500)
        ui = Ui(page)
        page.mouse.move(ui.x, ui.y)
        for s in ui_scenes:
            s["t_start"] = time.monotonic() - t0
            ui_actions(s["ui"], ui, s["dur"])
            left = s["t_start"] + s["dur"] + PAD - (time.monotonic() - t0)
            if left > 0:
                time.sleep(left)
            s["t_end"] = time.monotonic() - t0
            print(f"  {s['ui']:12s} {s['t_start']:6.1f} → {s['t_end']:6.1f} s (narration {s['dur']:.1f} s)")
        # fresh screenshot for the deck
        page.goto(URL + "/#queue")
        page.wait_for_selector("#qlist .row")
        page.wait_for_timeout(800)
        video = page.video.path()
        ctx.close()
        b.close()
    shutil.move(video, WORK / "rec" / "console.webm")


def screenshot():
    """Console screenshot for deck slide 3 (docs/console_queue.png), taken on the clean demo state."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(URL + "/#queue")
        page.wait_for_selector("#qlist .row")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(ROOT / "docs" / "console_queue.png"))
        b.close()


# ---------------------------------------------------------------------------
def build():
    seg = WORK / "seg"
    seg.mkdir(parents=True, exist_ok=True)
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    parts = []
    for i, s in enumerate(SCENES):
        out = seg / f"{i:02d}.mp4"
        hold = s.get("hold", 0)
        if "slide" in s:
            length = s["dur"] + PAD + hold
            img = WORK / "slides" / f"s-{s['slide']:02d}.png"
            run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(img),
                 "-i", str(s["audio"]), "-filter_complex",
                 f"[0:v]scale={W}:{H},format=yuv420p[v];[1:a]apad[a]", "-map", "[v]", "-map", "[a]",
                 "-t", f"{length:.3f}", *enc, str(out)])
        else:
            length = s["t_end"] - s["t_start"]
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s['t_start']:.3f}", "-i", str(WORK / "rec" / "console.webm"),
                 "-i", str(s["audio"]), "-filter_complex",
                 f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p[v];[1:a]apad[a]",
                 "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}", *enc, str(out)])
        s["len"] = duration(out)
        parts.append(out)
    lst = seg / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    nosubs = OUT / "VidyutRakshak_demo_nosubs.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(nosubs)])

    srt = OUT / "VidyutRakshak_demo.srt"
    srt.write_text(make_srt())
    style = "FontName=Lato,FontSize=11,PrimaryColour=&H00FFFFFF,OutlineColour=&H60000000,BackColour=&H60000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=14"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(nosubs), "-vf",
         f"subtitles={srt}:force_style='{style}'", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(OUT / "VidyutRakshak_demo.mp4")])
    (OUT / "VidyutRakshak_demo_script.txt").write_text(
        "\n\n".join(f"[{'slide ' + str(s['slide']) if 'slide' in s else 'console: ' + s['ui']}]\n{s.get('sub', s['say'])}"
                    for s in SCENES) + "\n")
    print(f"wrote {OUT / 'VidyutRakshak_demo.mp4'} ({duration(OUT / 'VidyutRakshak_demo.mp4'):.0f} s)")


def make_srt():
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    cues, t = [], 0.0
    for s in SCENES:
        text = s.get("sub", s["say"])
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        total = sum(len(x) for x in sents)
        at = t
        for x in sents:
            d = s["dur"] * len(x) / total
            cues.append((at, at + d, x))
            at += d
        t += s["len"]
    return "".join(f"{i}\n{ts(a)} --> {ts(b)}\n{x}\n\n" for i, (a, b, x) in enumerate(cues, 1))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["screenshot", "tts", "slides", "record", "build"]
    if "screenshot" in steps:
        screenshot()
        print("updated docs/console_queue.png; rebuild the deck and its PDF, then run: make_video.py tts slides record build")
        if steps == ["screenshot", "tts", "slides", "record", "build"]:
            sys.exit(0)
    if "tts" in steps:
        tts()
    if "slides" in steps:
        slides()
    if "record" in steps:
        record()
        json.dump([{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()} for s in SCENES],
                  open(WORK / "timeline.json", "w"), indent=1)
    if "build" in steps:
        if "record" not in steps:
            for s, saved in zip(SCENES, json.load(open(WORK / "timeline.json"))):
                s.update({k: v for k, v in saved.items() if k in ("t_start", "t_end", "dur")})
                s["audio"] = Path(saved["audio"])
        build()
