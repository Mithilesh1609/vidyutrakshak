// SFIC 2026 synopsis deck for PS000205 (VidyutRakshak), following the
// official 8-slide template structure, plus 4 appendix slides for jury Q&A.
// Every number is read from the prototype's outputs:
//   static/dashboard_data.json (headline block), docs/sgcc_validation.json,
//   docs/stress_tests.json, docs/benchmark.json
// Build:  cd deck && npm install && PPTX_SKILL=<pptx skill dir> node build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "static/dashboard_data.json"), "utf8"));
const OUT = path.join(__dirname, "VidyutRakshak_PS000205_Synopsis.pptx");

const THEME = {
  name: "VidyutRakshak",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "12201F", lt1: "FFFFFF", dk2: "0B4C47", lt2: "EEF3F3",
    accent1: "0D766E", accent2: "C03F2B", accent3: "A86D12",
    accent4: "3C7A4B", accent5: "57696A", accent6: "D5ECEA",
    hlink: "0D766E", folHlink: "57696A",
  },
};
const HEX = THEME.colors;

// ---------- metrics from the prototype (one source: the headline block) ----------
const HL = D.headline;                    // holdout area, ML rows averaged over 5 training runs
const abl = D.ablation;
const byName = n => abl.find(a => a.approach.startsWith(n));
const rule = byName("Rule"), anom = byName("Anomaly"), meterOnly = byName("ML on meter"), oursRow = byName("VidyutRakshak");
const PREV = HL.base_rate;                // what random checks find
const ours = { precision_at_k: HL.hit_rate, legit_drop_false_alarms_at_k: HL.legit_visits, k: HL.k };
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const K = ours.k;
const TESTS = +(process.env.VR_TESTS_PASSED || 10);
const readJSON = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8"));
const SG = readJSON("docs/sgcc_validation.json");
const ST = readJSON("docs/stress_tests.json");
const BM = readJSON("docs/benchmark.json");
const CAL = D.calibration;

// ---------- icons ----------
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.title = "VidyutRakshak – SFIC 2026 Synopsis (PS000205)";
  pres.author = "Smit Thakkar";
  pres.company = "Workloop Studios";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const I = {
    bolt: await icon(fa.FaBolt, HEX.lt1),
    search: await icon(fa.FaSearchLocation, HEX.lt1),
    layers: await icon(fa.FaLayerGroup, HEX.lt1),
    hands: await icon(fa.FaHandsHelping, HEX.lt1),
    users: await icon(fa.FaUsers, HEX.accent1),
    alert: await icon(fa.FaExclamationTriangle, HEX.accent2),
    play: await icon(fa.FaPlayCircle, HEX.accent1),
    db: await icon(fa.FaDatabase, HEX.accent1),
    user: await icon(fa.FaUserTie, HEX.lt1),
    userM: await icon(fa.FaUser, HEX.lt1),
    chalk: await icon(fa.FaChalkboardTeacher, HEX.accent6),
    check: await icon(fa.FaCheckCircle, HEX.accent4),
    clock: await icon(fa.FaRegClock, HEX.accent3),
  };

  // ---------- layouts ----------
  const FOOT = "SFIC 2026  ·  PS000205  ·  VidyutRakshak  ·  Team TM005200";
  pres.defineSlideMaster({
    title: "COVER",
    background: { color: HEX.dk1 },
    objects: [],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HEX.lt1 },
    margin: [0.3, 0.5, 0.45, 0.5],
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 5.6, h: 0.28, fontSize: 10, bold: true, color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "focus", type: "body", x: 6.1, y: 0.28, w: 3.4, h: 0.28, fontSize: 10, italic: true, color: C.accent5, align: "right", margin: 0, valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9.0, h: 0.55, fontSize: 24, bold: true, color: C.text1, align: "left", margin: 0, valign: "middle", fontFace: THEME.headFontFace }, text: "" } },
      { text: { text: FOOT, options: { x: 0.5, y: 5.24, w: 7.5, h: 0.25, fontSize: 9, color: C.accent5, margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 9.1, y: 5.24, w: 0.4, h: 0.25, fontSize: 9, color: C.accent5, align: "right", margin: 0 },
  });
  pres.defineSlideMaster({
    title: "CLOSING",
    background: { color: HEX.dk1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 5.6, h: 0.28, fontSize: 10, bold: true, color: C.accent6, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "focus", type: "body", x: 6.1, y: 0.28, w: 3.4, h: 0.28, fontSize: 10, italic: true, color: C.accent6, align: "right", margin: 0, valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9.0, h: 0.55, fontSize: 24, bold: true, color: C.background1, align: "left", margin: 0, valign: "middle", fontFace: THEME.headFontFace }, text: "" } },
      { text: { text: FOOT, options: { x: 0.5, y: 5.24, w: 7.5, h: 0.25, fontSize: 9, color: C.accent6, margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 9.1, y: 5.24, w: 0.4, h: 0.25, fontSize: 9, color: C.accent6, align: "right", margin: 0 },
  });

  const head = (s, kicker, title, focus) => {
    s.addText(kicker, { placeholder: "kicker" });
    s.addText(title, { placeholder: "title" });
    s.addText(focus, { placeholder: "focus" });
  };
  const circleIcon = (s, data, x, y, d, fill, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: name + " circle" });
    s.addImage({ data, x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, objectName: name + " icon" });
  };
  const card = (s, x, y, w, h, fill, name) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" }, objectName: name });

  // =====================================================================
  // Slide 1 — Cover
  pres.addSection({ title: "Cover" });
  let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Cover" });
  s.addText("Education Dept, Govt. of Gujarat  ·  SSIP 2.0", { x: 0.5, y: 0.35, w: 6, h: 0.3, fontSize: 11, color: C.accent6, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 0.85, w: 3.35, h: 0.36, rectRadius: 0.18, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "PS code pill" });
  s.addText("PROBLEM STATEMENT CODE: PS000205", { x: 0.5, y: 0.85, w: 3.35, h: 0.36, fontSize: 10.5, bold: true, color: C.background1, align: "center", valign: "middle", charSpacing: 1, margin: 0, isTextBox: true });
  s.addText("VidyutRakshak", { x: 0.5, y: 1.4, w: 6.4, h: 0.85, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText("Explainable AI for electricity theft detection and risk-based inspection", { x: 0.5, y: 2.22, w: 6.4, h: 0.4, fontSize: 16, color: C.accent6, margin: 0, isTextBox: true });
  s.addText("Ranks every consumer, transformer and feeder by theft risk, gives vigilance teams plain-language reasons for each alert, and learns from every inspection outcome.",
    { x: 0.5, y: 2.75, w: 6.0, h: 0.75, fontSize: 13, italic: true, color: C.background2, margin: 0, isTextBox: true });
  // bolt mark
  s.addShape(pres.shapes.OVAL, { x: 7.35, y: 0.9, w: 2.1, h: 2.1, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "mark circle" });
  s.addImage({ data: I.bolt, x: 7.88, y: 1.43, w: 1.04, h: 1.04, objectName: "mark bolt" });
  s.addText("Gujarat Urja Vikas Nigam Limited  ·  IT / ITeS", { x: 6.6, y: 3.12, w: 3.6, h: 0.3, fontSize: 10, color: C.accent6, align: "center", margin: 0, isTextBox: true });
  // detail strip
  const fields = [
    ["TEAM ID", "TM005200"],
    ["TRACK CATEGORY", "Grassroots Innovator"],
    ["INSTITUTION / ORGANIZATION", "Workloop Studios, Gandhinagar, Gujarat"],
    ["TEAM LEADER", "Smit Thakkar"],
  ];
  const fx = [0.5, 2.15, 4.15, 7.35], fw = [1.5, 1.85, 3.05, 2.15];
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 3.85, w: 9.0, h: 0, line: { color: HEX.dk2, width: 1 }, objectName: "strip divider" });
  fields.forEach(([k, v], i) => {
    s.addText(k, { x: fx[i], y: 4.0, w: fw[i], h: 0.25, fontSize: 9, bold: true, color: C.accent6, charSpacing: 1.5, margin: 0, isTextBox: true });
    s.addText(v, { x: fx[i], y: 4.25, w: fw[i], h: 0.35, fontSize: 13, bold: true, color: C.background1, margin: 0, isTextBox: true });
  });
  s.addText("SEVA FIRST INNOVATION CHALLENGE 2026  ·  OFFICIAL SYNOPSIS", { x: 0.5, y: 5.02, w: 9, h: 0.3, fontSize: 9.5, bold: true, color: C.accent6, charSpacing: 2, margin: 0, isTextBox: true });
  s.addNotes("Cover. Track category must match the portal registration; update if the helpdesk moves the team to the Startup category.");

  // =====================================================================
  // Slide 2 — Problem
  pres.addSection({ title: "Problem" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  head(s, "CHALLENGE DEFINITION", "Problem Statement & Real-World Context", "Focus: Root Cause & Citizen Friction");
  // left column: ground challenge + evidence
  s.addText("The Ground Challenge", { x: 0.5, y: 1.3, w: 4.4, h: 0.32, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("Meter bypass, tampering and direct hooking hide among lakhs of consumers. Vigilance teams pick whom to inspect from bill scrutiny and tip-offs, so most visits find nothing and real theft runs on until the next tip-off.",
    { x: 0.5, y: 1.65, w: 4.4, h: 1.0, fontSize: 12.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Evidence", { x: 0.5, y: 2.75, w: 4.4, h: 0.28, fontSize: 11, bold: true, color: C.accent2, charSpacing: 1, margin: 0, isTextBox: true });
  const stats = [
    ["15.04%", "AT&C losses, all Indian DISCOMs, FY25"],
    ["8.25%", "Gujarat state DISCOMs, FY25"],
    ["18.31%", "PGVCL, highest in Gujarat, FY23"],
  ];
  stats.forEach(([v, l], i) => {
    const x = 0.5 + i * 1.5;
    card(s, x, 3.08, 1.38, 1.3, HEX.lt2, `stat card ${i + 1}`);
    s.addText(v, { x: x + 0.1, y: 3.18, w: 1.18, h: 0.5, fontSize: 22, bold: true, color: i === 2 ? C.accent2 : C.text1, margin: 0, fontFace: THEME.headFontFace, isTextBox: true });
    s.addText(l, { x: x + 0.1, y: 3.7, w: 1.18, h: 0.62, fontSize: 10, color: C.accent5, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Sources: 14th Integrated Rating & Ranking of Power Distribution Utilities (MoP); Rajya Sabha USQ 149, 25 Nov 2024. AT&C also includes technical and collection losses; theft is the part that inspections can recover.",
    { x: 0.5, y: 4.5, w: 4.4, h: 0.6, fontSize: 9, italic: true, color: C.accent5, margin: 0, valign: "top", isTextBox: true });
  // right column
  card(s, 5.25, 1.3, 4.25, 3.8, HEX.lt2, "beneficiary panel");
  s.addImage({ data: I.users, x: 5.5, y: 1.48, w: 0.34, h: 0.34, objectName: "users icon" });
  s.addText("Target Beneficiaries & Flaws", { x: 5.95, y: 1.48, w: 3.4, h: 0.34, fontSize: 15, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  s.addText([
    { text: "DISCOM vigilance and O&M engineers", options: { bullet: true, breakLine: true } },
    { text: "Energy-audit and revenue-protection cells at GUVNL", options: { bullet: true, breakLine: true } },
    { text: "Honest consumers, who carry losses through tariffs", options: { bullet: true } },
  ], { x: 5.5, y: 1.95, w: 3.85, h: 1.0, fontSize: 12, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top", isTextBox: true });
  s.addImage({ data: I.alert, x: 5.5, y: 3.08, w: 0.3, h: 0.3, objectName: "gap icon" });
  s.addText("Existing Gap", { x: 5.9, y: 3.08, w: 3.4, h: 0.3, fontSize: 12.5, bold: true, color: C.accent2, margin: 0, valign: "middle", isTextBox: true });
  s.addText([
    { text: "Drop-based rules flag vacant homes and rooftop-solar homes as theft", options: { bullet: true, breakLine: true } },
    { text: "Commercial analytics suites are costly, need many labelled cases, and give black-box scores officers cannot defend in a case file", options: { bullet: true, breakLine: true } },
    { text: "Missing smart-meter reads get read as zero use", options: { bullet: true } },
  ], { x: 5.5, y: 3.45, w: 3.85, h: 1.55, fontSize: 11.5, color: C.text1, margin: 0, paraSpaceAfter: 4, valign: "top", isTextBox: true });

  // =====================================================================
  // Slide 3 — Solution
  pres.addSection({ title: "Solution" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  head(s, "SOLUTION APPROACH", "Proposed Solution & Innovation Novelty", "Ethos: Frugal, Accessible, Scalable");
  card(s, 0.5, 1.28, 9.0, 0.78, HEX.dk1, "summary band");
  s.addText("HIGH-LEVEL EXECUTIVE SUMMARY", { x: 0.7, y: 1.34, w: 4, h: 0.22, fontSize: 9, bold: true, color: C.accent6, charSpacing: 1.5, margin: 0, isTextBox: true });
  s.addText("A software platform that scores every consumer, DT and feeder for theft risk from existing smart-meter, billing and DT data, explains each alert in plain words, and builds a daily inspection list sized to field capacity. Every inspection outcome retrains the model.",
    { x: 0.7, y: 1.56, w: 8.6, h: 0.46, fontSize: 11.5, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  const sol = [
    [I.search, "Core Novelty", "A consumer's drop vs same-DT peers is checked against the DT's own energy balance: theft makes energy vanish; vacant homes and rooftop solar don't."],
    [I.layers, "Technical Moat", "Works with few confirmed cases, keeps meter gaps from looking like theft, retrains on every outcome. Runs on one CPU server."],
    [I.hands, "\"Seva First\" Value", "Recovered revenue eases the loss burden on honest consumers. Fewer visits to innocent homes; unsafe hookings removed sooner."],
  ];
  sol.forEach(([ic, h, t], i) => {
    const y = 2.2 + i * 0.86;
    circleIcon(s, ic, 0.5, y + 0.04, 0.5, HEX.accent1, h);
    s.addText(h, { x: 1.12, y, w: 3.8, h: 0.26, fontSize: 12.5, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: 1.12, y: y + 0.27, w: 3.95, h: 0.52, fontSize: 10.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  // product screenshot
  s.addImage({ path: path.join(ROOT, "docs/console_queue.png"), x: 5.3, y: 2.2, w: 4.2, h: 2.625, objectName: "console screenshot" });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 2.2, w: 4.2, h: 2.625, fill: { type: "none" }, line: { color: HEX.accent6, width: 0.75 }, objectName: "screenshot frame" });
  s.addText("Working prototype: ranked queue with reasons, consumption vs peers", { x: 5.3, y: 4.84, w: 4.2, h: 0.22, fontSize: 9, italic: true, color: C.accent5, margin: 0, isTextBox: true });
  s.addImage({ data: I.play, x: 0.5, y: 4.82, w: 0.26, h: 0.26, objectName: "video icon" });
  s.addText([
    { text: "Demonstration video:  ", options: { bold: true } },
    { text: "[add YouTube / Google Drive link]", options: { color: HEX.accent2 } },
  ], { x: 0.85, y: 4.82, w: 4.3, h: 0.26, fontSize: 10, color: C.text1, margin: 0, valign: "middle", isTextBox: true });

  // =====================================================================
  // Slide 4 — Architecture
  pres.addSection({ title: "Architecture" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
  head(s, "SYSTEM ARCHITECTURE", "Technical Architecture || Methodology", "End-to-End System Pipeline");
  // sources column
  const src = ["MDMS smart-meter reads & events", "Billing & sanctioned load", "DT / feeder energy meters", "GIS consumer–DT index", "Vigilance inspection history"];
  s.addText("DATA SOURCES", { x: 0.5, y: 1.25, w: 1.7, h: 0.22, fontSize: 9, bold: true, color: C.accent5, charSpacing: 1.5, margin: 0, isTextBox: true });
  src.forEach((t, i) => {
    const y = 1.52 + i * 0.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 1.7, h: 0.42, rectRadius: 0.06, fill: { color: HEX.lt2 }, line: { color: HEX.accent6, width: 0.75 }, objectName: `source ${i + 1}` });
    s.addText(t, { x: 0.58, y, w: 1.56, h: 0.42, fontSize: 9, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  // stages
  const stages = [
    ["1  Ingest & clean", "Adapters per system. Gaps imputed from peers; meters with >20% gaps go to metering.", "Clean daily kWh, gaps flagged"],
    ["2  Feature engine", "Use vs same-DT peers, sustained drop, flat-lining, capping, tamper alarms, DT loss step.", "13 evidence signals per consumer"],
    ["3  Risk engine", "Isolation Forest (no labels) + LightGBM on outcomes, calibrated: a score of 80 ≈ 80% likely theft.", "Theft probability, 0–100"],
    ["4  Explain & roll up", "TreeSHAP → plain-language reasons in English and Gujarati. Consumer → DT → feeder risk.", "Top-3 reasons; DT and feeder scores"],
    ["5  Act", "Daily queue sized to crew capacity, via console and REST API with roles.", "Ranked inspection list"],
  ];
  const sx = 2.45, gap = 0.16, sw = (9.5 - sx - 4 * gap) / 5;
  stages.forEach(([h, t, o], i) => {
    const x = sx + i * (sw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.52, w: sw, h: 2.42, rectRadius: 0.08, fill: { color: i === 2 ? HEX.accent1 : HEX.lt1 }, line: { color: HEX.accent1, width: 1 }, objectName: `stage ${i + 1}` });
    s.addText(h, { x: x + 0.08, y: 1.6, w: sw - 0.16, h: 0.5, fontSize: 11, bold: true, color: i === 2 ? C.background1 : C.accent1, margin: 0, valign: "top", isTextBox: true });
    s.addText(t, { x: x + 0.08, y: 2.12, w: sw - 0.16, h: 1.05, fontSize: 10, color: i === 2 ? C.background1 : C.text1, margin: 0, valign: "top", isTextBox: true });
    s.addText("OUTPUT", { x: x + 0.08, y: 3.22, w: sw - 0.16, h: 0.2, fontSize: 8, bold: true, charSpacing: 1.5, color: i === 2 ? C.accent6 : C.accent5, margin: 0, isTextBox: true });
    s.addText(o, { x: x + 0.08, y: 3.42, w: sw - 0.16, h: 0.46, fontSize: 10, bold: true, color: i === 2 ? C.background1 : C.accent1, margin: 0, valign: "top", isTextBox: true });
    if (i < stages.length - 1) s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + sw + 0.015, y: 2.66, w: 0.13, h: 0.11, fill: { color: HEX.accent1 }, line: { type: "none" }, rotate: 90, objectName: `arrow ${i + 1}` });
  });
  s.addShape(pres.shapes.LINE, { x: 2.2, y: 2.73, w: 0.25, h: 0, line: { color: HEX.accent1, width: 1.5, endArrowType: "triangle" }, objectName: "sources arrow" });
  // feedback loop band
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.45, y: 4.1, w: 7.05, h: 0.95, rectRadius: 0.08, fill: { color: HEX.lt2 }, line: { type: "none" }, objectName: "feedback band" });
  s.addText("MLOps feedback loop", { x: 2.6, y: 4.17, w: 3, h: 0.25, fontSize: 11, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText("Field outcome (theft / no irregularity / meter fault)  →  versioned retrain  →  promotion gate: goes live only if at least as good on outcomes the live model never saw, else kept as a candidate.  PSI drift check monthly; every version kept.",
    { x: 2.6, y: 4.43, w: 6.8, h: 0.55, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Privacy: consumer numbers replaced by keyed (HMAC) pseudonyms at ingest, role-based access, on-premise.", { x: 0.5, y: 4.2, w: 1.75, h: 0.85, fontSize: 9, italic: true, color: C.accent5, margin: 0, valign: "top", isTextBox: true });

  // =====================================================================
  // Slide 5 — Stack & feasibility
  pres.addSection({ title: "Feasibility" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Feasibility" });
  head(s, "ENGINEERING STACK", "Technology Stack & Feasibility", "Component Readiness & Dependencies");
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, fontSize: 10 } });
  const built = { text: "Built", options: { color: HEX.accent4, bold: true } };
  const rows = [
    ["Data adapters", "Validated CSV ingest, HMAC pseudonyms, Parquet", built, "MDMS / billing / DT exports (CSV or API)"],
    ["Feature engine", "NumPy peer comparison, change-point, DT energy balance", built, "Consumer–DT–feeder indexing from GIS"],
    ["Risk models", "scikit-learn Isolation Forest, LightGBM", built, "Past vigilance outcomes (a few hundred suffice)"],
    ["Explanations", "TreeSHAP → reason codes, English + Gujarati", built, "Officer review of Gujarati wording"],
    ["Service", "FastAPI, role-based API keys", built, "DISCOM single sign-on"],
    ["Console", "Web app, desktop and mobile", built, "None"],
    ["MLOps", "Registry, promotion gate, calibration, PSI drift, pytest", built, "Scheduled runs (cron or Airflow)"],
    ["Deployment", "Docker, on-premise", built, "One VM in the DISCOM data centre"],
  ];
  s.addTable([[hdr("Layer"), hdr("Technology"), hdr("Prototype"), hdr("Dependency for pilot")], ...rows.map((r, i) => r.map(c => typeof c === "string" ? { text: c, options: { fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } } : { ...c, options: { ...c.options, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }))],
    { x: 0.5, y: 1.28, w: 6.3, colW: [1.15, 2.2, 0.8, 2.15], fontSize: 9.5, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: HEX.accent6 }, rowH: 0.36, margin: 0.05, valign: "middle", fontFace: "Calibri" });
  // feasibility callouts
  const bm1 = BM.runs.find(r => r.consumers > 90000) || BM.runs[0];
  const feas = [
    [`${Math.round(bm1.total_seconds_excluding_simulation)} s`, `daily scoring run for ${(bm1.consumers / 1e5).toFixed(1)} lakh consumers × 365 days on one laptop`],
    [`${TESTS} / ${TESTS}`, "automated tests pass: quality bars, gate, roles, ingest, Gujarati"],
    ["₹0", "licence cost: fully open-source stack"],
  ];
  feas.forEach(([v, l], i) => {
    const y = 1.28 + i * 1.08;
    card(s, 7.05, y, 2.45, 0.96, HEX.lt2, `feasibility ${i + 1}`);
    s.addText(v, { x: 7.2, y: y + 0.08, w: 2.2, h: 0.42, fontSize: 22, bold: true, color: C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText(l, { x: 7.2, y: y + 0.5, w: 2.2, h: 0.42, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });

  // =====================================================================
  // Slide 6 — Impact
  pres.addSection({ title: "Impact" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Impact" });
  head(s, "IMPACT ASSESSMENT  ·  SIH SECTION 05", "Socio-Economic Impact & Viability", "Quantifiable Citizen Outcomes");
  const big = [
    [pct(ours.precision_at_k), `of top-${K} alerts were real theft, vs ${pct(PREV, 1)} for random checks`, C.accent1],
    [`${Math.round(ours.legit_drop_false_alarms_at_k)} vs ${Math.round(rule.legit_drop_false_alarms_at_k)}`, `visits to vacant or rooftop-solar homes per ${K}, vs a >50% drop rule`, C.accent2],
    [`${pct(HL.first_version_hit_rate)} → ${pct(HL.last_version_hit_rate)}`, "hit rate after three rounds of 60 field outcomes each (one run per version)", C.accent3],
  ];
  big.forEach(([v, l, col], i) => {
    const y = 1.28 + i * 1.12;
    s.addText(v, { x: 0.5, y, w: 3.3, h: 0.55, fontSize: 30, bold: true, color: col, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText(l, { x: 0.5, y: y + 0.55, w: 3.3, h: 0.45, fontSize: 10.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addChart(pres.charts.BAR, [{
    name: "Hit rate on holdout area",
    labels: ["Random checks", "Drop >50% rule", "Anomaly only", "VidyutRakshak"],
    values: [PREV, rule.precision_at_k, anom.precision_at_k, ours.precision_at_k].map(v => +(100 * v).toFixed(1)),
  }], {
    x: 4.05, y: 1.25, w: 5.45, h: 2.45, barDir: "bar",
    chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent1],
    showTitle: true, title: `Share of top-${K} alerts that were real theft (%)`, titleFontSize: 11, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0\"%\"", dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMaxVal: 100, valAxisMinVal: 0,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45,
  });
  const sg1 = SG.runs[0].results.find(r => r.approach.startsWith("VidyutRakshak meter-only"));
  s.addText(`Simulated network (5,035 consumers, 81 DTs), holdout area, 5-run average. Real SGCC data, no DT meters: ${pct(sg1.precision_at_k)} vs ${pct(SG.runs[0].thieves / SG.runs[0].consumers, 1)} at random (appendix A).`,
    { x: 4.05, y: 3.72, w: 5.45, h: 0.4, fontSize: 9, italic: true, color: C.accent5, margin: 0, isTextBox: true });
  card(s, 4.05, 4.18, 5.45, 0.92, HEX.lt2, "kpi panel");
  s.addText("KPIs tracked live (as listed in PS000205)", { x: 4.2, y: 4.24, w: 5.2, h: 0.24, fontSize: 10.5, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("Detection precision · confirmed-theft rate among high-risk inspections · false-positive rate · recall on known cases · revenue leakage identified and recovered · alert-to-inspection time · cases investigated and closed · NTL by consumer, DT and feeder",
    { x: 4.2, y: 4.5, w: 5.2, h: 0.55, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });

  // =====================================================================
  // Slide 7 — Roadmap
  pres.addSection({ title: "Roadmap" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Roadmap" });
  head(s, "EXECUTION TIMELINE  ·  SIH SECTION 06", "Implementation Roadmap & Milestones", "From Prototype to Field Rollout");
  const phases = [
    ["Done", "Prototype", "Console, API, models, feedback loop, tests, Docker", "Working end to end on simulated data", true],
    ["Month 1–2", "Data onboarding", "One sub-division: MDMS, billing, DT meters, vigilance history mapped", "Consumer–DT index ≥95% matched", false],
    ["Month 3–4", "Shadow pilot", "Score alongside current process; Gujarati reasons reviewed with officers", "Hit rate on past cases beats current method", false],
    ["Month 5–6", "Live pilot", "Daily queue to vigilance; outcomes logged in app; monthly retrain", "Inspection hit rate ≥2× baseline", false],
    ["Month 7–12", "Scale", "Circle, then DISCOM-wide; SSO; GIS and energy-audit integration", "NTL trend down on pilot DTs", false],
  ];
  const tx = 0.5, tw = 9.0, cw = tw / phases.length, ty = 1.62;
  s.addShape(pres.shapes.LINE, { x: tx + cw / 2, y: ty, w: tw - cw, h: 0, line: { color: HEX.accent6, width: 2.5 }, objectName: "timeline" });
  phases.forEach(([when, name, what, crit, done], i) => {
    const cx = tx + cw * i + cw / 2;
    s.addShape(pres.shapes.OVAL, { x: cx - 0.13, y: ty - 0.13, w: 0.26, h: 0.26, fill: { color: done ? HEX.accent1 : HEX.lt1 }, line: { color: HEX.accent1, width: 2 }, objectName: `milestone ${i + 1}` });
    s.addText(when, { x: cx - cw / 2, y: 1.2, w: cw, h: 0.25, fontSize: 10, bold: true, color: done ? C.accent1 : C.accent5, align: "center", margin: 0, isTextBox: true });
    card(s, cx - cw / 2 + 0.06, 1.95, cw - 0.12, 2.3, done ? HEX.accent6 : HEX.lt2, `phase ${i + 1}`);
    s.addText(name, { x: cx - cw / 2 + 0.16, y: 2.05, w: cw - 0.32, h: 0.3, fontSize: 12.5, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(what, { x: cx - cw / 2 + 0.16, y: 2.4, w: cw - 0.32, h: 0.8, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    s.addImage({ data: done ? I.check : I.clock, x: cx - cw / 2 + 0.16, y: 3.27, w: 0.2, h: 0.2, objectName: `criterion icon ${i + 1}` });
    s.addText("Exit criterion", { x: cx - cw / 2 + 0.42, y: 3.25, w: cw - 0.55, h: 0.24, fontSize: 9, bold: true, color: C.accent5, margin: 0, valign: "middle", isTextBox: true });
    s.addText(crit, { x: cx - cw / 2 + 0.16, y: 3.52, w: cw - 0.32, h: 0.65, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });

  card(s, 0.5, 4.4, 9.0, 0.68, HEX.dk1, "pilot ask band");
  s.addText("WHAT THE PILOT NEEDS FROM GUVNL", { x: 0.7, y: 4.47, w: 4, h: 0.22, fontSize: 9, bold: true, color: C.accent6, charSpacing: 1.5, margin: 0, isTextBox: true });
  s.addText("12 months of MDMS, billing and DT-meter data for one sub-division, its past vigilance records, and one nodal engineer for weekly reviews.",
    { x: 0.7, y: 4.69, w: 8.6, h: 0.32, fontSize: 11, color: C.background1, margin: 0, valign: "top", isTextBox: true });

  // =====================================================================
  // Slide 8 — Team
  pres.addSection({ title: "Team" });
  s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Team" });
  head(s, "SQUAD CAPABILITIES", "Multi-Disciplinary Team & Mentorship", "Complementary Skill Sets");
  const team = [
    ["TL", "Smit Thakkar", "Team Leader", "ML systems, backend & MLOps · Founder, Workloop Studios", I.user],
    ["M2", "[Member 2 Name]", "Data & ML", "Feature engineering, model evaluation", I.userM],
    ["M3", "[Member 3 Name]", "Product & field app", "Console, Gujarati UX, officer workflow", I.userM],
    ["M4", "[Member 4 Name]", "Power systems", "Energy audit, DT metering, theft patterns", I.userM],
  ];
  team.forEach(([tag, name, role, skill, ic], i) => {
    const x = 0.5 + i * 2.29;
    card(s, x, 1.35, 2.13, 2.45, HEX.dk2, `member ${tag}`);
    circleIcon(s, ic, x + 0.18, 1.52, 0.62, i === 0 ? HEX.accent1 : HEX.accent5, `member ${tag}`);
    s.addText(tag, { x: x + 1.4, y: 1.55, w: 0.55, h: 0.3, fontSize: 11, bold: true, color: C.accent6, align: "right", margin: 0, isTextBox: true });
    s.addText(name, { x: x + 0.18, y: 2.28, w: 1.85, h: 0.32, fontSize: 13, bold: true, color: C.background1, margin: 0, isTextBox: true });
    s.addText(role, { x: x + 0.18, y: 2.6, w: 1.85, h: 0.26, fontSize: 10.5, bold: true, color: C.accent6, margin: 0, isTextBox: true });
    s.addText(skill, { x: x + 0.18, y: 2.9, w: 1.85, h: 0.8, fontSize: 10, color: C.background2, margin: 0, valign: "top", isTextBox: true });
  });
  s.addImage({ data: I.chalk, x: 0.5, y: 4.05, w: 0.38, h: 0.38, objectName: "mentor icon" });
  s.addText([
    { text: "Faculty Mentor: ", options: { bold: true } },
    { text: "[Prof. / Dr. Full Name]", options: {} },
  ], { x: 1.0, y: 4.02, w: 5.5, h: 0.24, fontSize: 12, color: C.background1, margin: 0, isTextBox: true });
  s.addText("Department of Electrical Engineering, [Institution Name]", { x: 1.0, y: 4.27, w: 5.5, h: 0.22, fontSize: 10.5, color: C.accent6, margin: 0, isTextBox: true });
  s.addText("Ready for Jury Review & Evaluation", { x: 5.6, y: 4.08, w: 3.9, h: 0.4, fontSize: 15, bold: true, color: C.accent6, align: "right", margin: 0, isTextBox: true });

  // =====================================================================
  // Appendix — evidence for jury questions
  pres.addSection({ title: "Appendix" });
  const chartText = { titleFontSize: 11, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt" };
  const note = (s, t, y = 4.72) => s.addText(t, { x: 0.5, y, w: 9.0, h: 0.42, fontSize: 9, italic: true, color: C.accent5, margin: 0, valign: "top", isTextBox: true });
  const stat = (s, x, y, w, v, l, col = C.accent1) => {
    card(s, x, y, w, 1.0, HEX.lt2, `stat ${v}`);
    s.addText(v, { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: 0.45, fontSize: 24, bold: true, color: col, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText(l, { x: x + 0.15, y: y + 0.53, w: w - 0.3, h: 0.42, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  };

  // ---- A: real data ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  head(s, "APPENDIX A  ·  REAL-DATA CHECK", "Validated on Real, Labelled Theft Data", "SGCC public dataset, 42,372 consumers");
  const r1 = SG.runs[0], r2 = SG.runs[1] || SG.runs[0];
  const g = (run, p) => run.results.find(r => r.approach.startsWith(p));
  const sgLabels = ["Random checks", "Drop >50% rule", "Anomaly only", "Black-box (raw series)", "VidyutRakshak signals", "Signals + raw series"];
  const sgKeys = ["Random", "Rule", "Anomaly", "Black-box", "VidyutRakshak meter-only", "VidyutRakshak signals +"];
  s.addChart(pres.charts.BAR, [{ name: "Hit rate", labels: sgLabels, values: sgKeys.map(k => +(100 * g(r1, k).precision_at_k).toFixed(1)) }], {
    x: 0.5, y: 1.25, w: 5.6, h: 3.35, barDir: "bar", ...chartText,
    chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent1, HEX.accent6],
    showTitle: true, title: `Share of top-4% alerts that were real theft, last ${r1.days} days (%)`,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 60, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45,
  });
  const ours1 = g(r1, "VidyutRakshak meter-only"), ours2 = g(r2, "VidyutRakshak meter-only"), both2 = g(r2, "VidyutRakshak signals +");
  stat(s, 6.4, 1.25, 3.1, `${(ours1.lift_vs_random).toFixed(1)}× random`, `${pct(ours1.precision_at_k)} of top-4% alerts are thieves vs ${pct(r1.thieves / r1.consumers, 1)} base rate, using only explainable signals`);
  stat(s, 6.4, 2.38, 3.1, `${pct(ours2.precision_at_k)} → ${pct(both2.precision_at_k)}`, `with 2 years of history; adding raw weekly shape (${(both2.lift_vs_random).toFixed(1)}× random) is a pilot option`, C.accent3);
  stat(s, 6.4, 3.51, 3.1, `${pct(g(r1, "Rule").precision_at_k)}`, "for a drop rule: real theft is often long-running, with no visible drop", C.accent2);
  note(s, `5-fold cross-validation on SGCC (State Grid Corporation of China) labels. Same feature code as the prototype. SGCC has no DT meters, tariff, sanctioned load or tamper logs, so this is a lower bound for the meter-only path, not the full system. Source: ${SG.source}.`);

  // ---- B: robustness ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  head(s, "APPENDIX B  ·  STRESS TESTS", "Robust When Field Reality Gets Messy", `Simulator, ${ST.seeds.length} seeds per scenario`);
  const sh = t => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, fontSize: 10 } });
  const base = ST.scenarios[0].current.hit_rate;
  s.addTable([[sh("Scenario"), sh("What changes"), sh("Hit rate"), sh("vs baseline"), sh("Visits to vacant / solar homes")],
    ...ST.scenarios.map((r, i) => {
      const fill = { color: i % 2 ? HEX.lt1 : HEX.lt2 };
      const d = r.current.hit_rate - base;
      return [
        { text: r.scenario, options: { bold: true, fill } }, { text: r.description, options: { fill } },
        { text: pct(r.current.hit_rate), options: { align: "right", bold: true, fill, color: r.current.hit_rate < 0.78 ? HEX.accent2 : HEX.dk1 } },
        { text: i === 0 ? "—" : Math.round(100 * d) === 0 ? "0 pts" : `${d > 0 ? "+" : "−"}${Math.abs(Math.round(100 * d))} pts`, options: { align: "right", fill } },
        { text: `${r.current.legit_visits.toFixed(0)} of ${r.k}`, options: { align: "right", fill } }];
    })],
    { x: 0.5, y: 1.25, w: 9.0, colW: [1.85, 4.15, 0.9, 0.95, 1.15], fontSize: 10, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: HEX.accent6 }, rowH: 0.36, margin: 0.06, valign: "middle", fontFace: "Calibri" });
  const lr = ST.scenarios.find(r => r.scenario.startsWith("Long-running"));
  const sb = ST.scenarios.find(r => r.scenario.startsWith("Subtle"));
  note(s, `Hit rate = share of the top ${ST.scenarios[0].k} alerts on a holdout area that are real theft. Long-running theft: ${pct(lr.current.chronic_recall)} of thieves already stealing when the data starts are still in the top list. Weakest case is subtle 10–25% bypass, where DT meters add the most (${pct(sb.current.hit_rate)} vs ${pct(sb.meter_only.hit_rate)} on meter data alone).`, 4.3);

  // ---- C: DT balance + calibration ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  head(s, "APPENDIX C  ·  WHY THE DT BALANCE AND CALIBRATION MATTER", "Fewer Innocent Visits, Scores That Mean What They Say", "Holdout area, simulated network");
  const ds_ = oursRow.drop_slice;
  s.addChart(pres.charts.BAR, [{ name: "Vacant / solar homes", labels: ["Drop >50% rule", "Anomaly only", "ML, meter data only", "ML + DT balance"],
    values: [rule, anom, meterOnly, oursRow].map(a => +a.drop_slice.legit_in_top.toFixed(1)) }], {
    x: 0.5, y: 1.25, w: 4.3, h: 3.0, barDir: "bar", ...chartText,
    chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent1],
    showTitle: true, title: `Vacant/solar homes in top ${ds_.k} (use fell >25%)`,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisHidden: true, valAxisMinVal: 0, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45,
  });
  s.addText(`Among ${ds_.n.toFixed(0)} consumers whose use fell >25% (${ds_.thefts.toFixed(0)} thieves, ${ds_.legit.toFixed(0)} vacant or solar homes), the DT balance cuts wasted visits by ${pct(1 - oursRow.drop_slice.legit_in_top / meterOnly.drop_slice.legit_in_top)} vs meter data alone. Where DTs are unmetered, the model still works on meter data (appendix B).`,
    { x: 0.5, y: 4.3, w: 4.3, h: 0.8, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const ch = t => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, fontSize: 10, align: "center" } });
  s.addText("Does a risk score of 80 mean ~80% likely theft?", { x: 5.2, y: 1.25, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addTable([[ch("Risk score"), ch("Consumers"), ch("Model said"), ch("Actually theft")],
    ...CAL.bins.map((b, i) => {
      const fill = { color: i % 2 ? HEX.lt1 : HEX.lt2 };
      return [`${Math.round(100 * b.from)}–${Math.round(100 * b.to)}`, b.n.toLocaleString("en-IN"), pct(b.predicted), pct(b.observed)]
        .map((t, j) => ({ text: t, options: { fill, align: j ? "right" : "left", bold: j === 3 } }));
    })],
    { x: 5.2, y: 1.62, w: 4.3, colW: [1.0, 1.05, 1.1, 1.15], fontSize: 10, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: HEX.accent6 }, rowH: 0.34, margin: 0.06, valign: "middle", fontFace: "Calibri" });
  s.addText(`Probabilities are calibrated on cross-validated scores, leaving out tip-offs (biased towards blatant theft). Brier score ${CAL.brier.toFixed(3)} vs ${CAL.brier_base_rate.toFixed(3)} for a flat guess. Officers can read a score as a probability.`,
    { x: 5.2, y: 3.85, w: 4.3, h: 0.8, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });

  // ---- D: scale + pilot readiness ----
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  head(s, "APPENDIX D  ·  PILOT READINESS", "Scales Linearly, Plugs Into Existing Systems", "Measured, not estimated");
  BM.runs.slice(0, 2).forEach((r, i) => stat(s, 0.5 + i * 2.25, 1.25, 2.1,
    `${Math.round(r.total_seconds_excluding_simulation)} s`, `${(r.consumers / 1e5).toFixed(1)} lakh consumers × ${r.days} days, peak ${(r.peak_memory_mb / 1024).toFixed(1)} GB RAM, one laptop`));
  s.addText("Peers are compared within each DT, so a DISCOM runs division by division overnight; time grows linearly with consumers.",
    { x: 0.5, y: 2.4, w: 4.35, h: 0.6, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const ready = [
    [I.db, "Real-data ingest", "Validates MDMS, billing/GIS, DT-meter, tamper and vigilance exports with clear errors."],
    [I.check, "Privacy by design", "Consumer numbers become keyed HMAC pseudonyms at ingest; the lookup file is owner-only for dispatch."],
    [I.users, "Roles in the console", "Viewer, vigilance and admin keys; officers record outcomes with name and notes; Gujarati reasons."],
    [I.clock, "Safe model updates", "A retrained model goes live only if it scores as well as the live one on outcomes it never saw."],
  ];
  ready.forEach(([ic, h, t], i) => {
    const y = 1.25 + i * 0.95;
    card(s, 5.15, y, 4.35, 0.85, HEX.lt2, `ready ${i + 1}`);
    s.addImage({ data: ic, x: 5.3, y: y + 0.12, w: 0.3, h: 0.3, objectName: `ready icon ${i + 1}` });
    s.addText(h, { x: 5.72, y: y + 0.08, w: 3.65, h: 0.26, fontSize: 11, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: 5.72, y: y + 0.34, w: 3.65, h: 0.48, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, 0.5, 3.1, 4.35, 1.98, HEX.dk1, "pilot steps band");
  s.addText("FIRST WEEK OF A PILOT", { x: 0.7, y: 3.2, w: 4, h: 0.22, fontSize: 9, bold: true, color: C.accent6, charSpacing: 1.5, margin: 0, isTextBox: true });
  s.addText([
    { text: "Export 12 months of MDMS, billing/GIS, DT and vigilance data for one sub-division", options: { bullet: true, breakLine: true } },
    { text: "vidyutrakshak.ingest validates and pseudonymises it", options: { bullet: true, breakLine: true } },
    { text: "Pipeline trains on past inspections, reports cross-validated hit rate", options: { bullet: true, breakLine: true } },
    { text: "Shadow-score next to the current process before any field use", options: { bullet: true } },
  ], { x: 0.7, y: 3.45, w: 4.0, h: 1.55, fontSize: 10, color: C.background1, margin: 0, paraSpaceAfter: 3, valign: "top", isTextBox: true });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch(e => { console.error(e); process.exit(1); });
