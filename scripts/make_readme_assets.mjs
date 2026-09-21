#!/usr/bin/env node
/**
 * Renders every picture used by the README into docs/images/:
 *
 *   banner.png             hero banner
 *   architecture.png       system architecture (Mermaid)
 *   agent-workflow.png     how one question flows through the agent (Mermaid sequence diagram)
 *   screenshot-*.png       real screenshots of the running app (desktop + mobile)
 *
 * Sample chart PNGs are produced separately by scripts/make_sample_charts.py.
 *
 * Usage (the API + built frontend must be running, e.g. `make start`):
 *   cd scripts && npm install && BASE_URL=http://localhost:8000 npm run assets
 *
 * A Chromium/Chrome binary is resolved from CHROME_PATH, or downloaded once through
 * @sparticuz/chromium when the variable is not set.
 */
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(process.env.OUT || resolve(here, "..", "docs", "images"));
const BASE_URL = process.env.BASE_URL || "http://localhost:8000";
mkdirSync(OUT, { recursive: true });

async function resolveChrome() {
  if (process.env.CHROME_PATH) return { executablePath: process.env.CHROME_PATH, args: [] };
  const chromium = (await import("@sparticuz/chromium")).default;
  return { executablePath: await chromium.executablePath(), args: chromium.args };
}

const { executablePath, args } = await resolveChrome();
const browser = await puppeteer.launch({
  executablePath,
  args: [...args, "--font-render-hinting=none", "--hide-scrollbars"],
  headless: true,
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function newPage({ width, height, scale = 2, mobile = false }) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: scale, isMobile: mobile, hasTouch: mobile });
  page.setDefaultTimeout(120_000);
  return page;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Screenshots of the live application
// ─────────────────────────────────────────────────────────────────────────────
async function ask(page, question) {
  const before = await page.$$eval(".message.assistant:not(.thinking)", (els) => els.length);
  await page.click('textarea[aria-label="Ask the data scientist agent"]');
  await page.evaluate(() => {
    const el = document.querySelector('textarea[aria-label="Ask the data scientist agent"]');
    el.value = "";
  });
  await page.type('textarea[aria-label="Ask the data scientist agent"]', question, { delay: 5 });
  await page.keyboard.press("Enter");
  await page.waitForFunction((n) => document.querySelectorAll(".message.assistant:not(.thinking)").length > n, {}, before);
  // wait for chart images to load
  await page.evaluate(async () => {
    const imgs = [...document.querySelectorAll(".chart-card img")];
    await Promise.all(imgs.map((img) => (img.complete ? null : new Promise((r) => (img.onload = img.onerror = r)))));
  });
  await sleep(400);
}

async function screenshots() {
  const page = await newPage({ width: 1440, height: 1040 });
  // Start from a fresh conversation so the landing page is deterministic
  await page.goto(BASE_URL, { waitUntil: "networkidle0" });
  await page.evaluate(() => localStorage.clear());
  await page.goto(BASE_URL, { waitUntil: "networkidle0" });
  await page.waitForSelector(".intro h2");
  await sleep(500);
  await page.screenshot({ path: resolve(OUT, "screenshot-workspace.png") });
  console.log("✓ screenshot-workspace.png");

  await ask(page, "Total revenue by category");
  await page.evaluate(() => document.querySelector(".messages").scrollTo(0, 0));
  await sleep(200);
  await page.screenshot({ path: resolve(OUT, "screenshot-chat-chart.png") });
  console.log("✓ screenshot-chat-chart.png");

  await ask(page, "Predict revenue — which features matter?");
  // expand the tool trace of the last answer and bring charts + trace into view
  const toggles = await page.$$(".trace-toggle");
  await toggles[toggles.length - 1].click();
  await sleep(300);
  await page.evaluate(() => {
    const msgs = document.querySelectorAll(".message.assistant");
    const charts = msgs[msgs.length - 1].querySelector(".charts");
    (charts || msgs[msgs.length - 1]).scrollIntoView({ block: "start" });
    document.querySelector(".messages").scrollBy(0, -12);
  });
  await sleep(300);
  await page.screenshot({ path: resolve(OUT, "screenshot-model-trace.png") });
  console.log("✓ screenshot-model-trace.png");

  await page.close();

  // mobile
  const mobile = await newPage({ width: 390, height: 844, scale: 2, mobile: true });
  await mobile.goto(BASE_URL, { waitUntil: "networkidle0" });
  await mobile.waitForSelector(".intro h2, .message");
  await sleep(500);
  await mobile.evaluate(() => {
    const msgs = document.querySelectorAll(".message.assistant");
    if (msgs.length) msgs[0].scrollIntoView({ block: "start" });
  });
  await sleep(300);
  await mobile.screenshot({ path: resolve(OUT, "screenshot-mobile.png") });
  console.log("✓ screenshot-mobile.png");
  await mobile.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Mermaid diagrams
// ─────────────────────────────────────────────────────────────────────────────
const THEME = {
  theme: "base",
  themeVariables: {
    darkMode: true,
    background: "#0b0f18",
    primaryColor: "#182034",
    primaryTextColor: "#e6ebf7",
    primaryBorderColor: "#4f8cff",
    secondaryColor: "#1c1a3a",
    secondaryBorderColor: "#c77dff",
    tertiaryColor: "#122a26",
    tertiaryBorderColor: "#3ddc97",
    lineColor: "#8c98b8",
    textColor: "#e6ebf7",
    fontFamily: "Inter, Open Sans, DejaVu Sans, sans-serif",
    fontSize: "15px",
    clusterBkg: "#0f1420",
    clusterBorder: "#223052",
    edgeLabelBackground: "#131a2a",
    actorBkg: "#182034",
    actorBorder: "#4f8cff",
    actorTextColor: "#e6ebf7",
    actorLineColor: "#3a4a70",
    signalColor: "#c7d0e6",
    signalTextColor: "#e6ebf7",
    labelBoxBkgColor: "#1c1a3a",
    labelBoxBorderColor: "#c77dff",
    labelTextColor: "#e6ebf7",
    loopTextColor: "#e6ebf7",
    noteBkgColor: "#122a26",
    noteBorderColor: "#3ddc97",
    noteTextColor: "#e6ebf7",
    activationBkgColor: "#243050",
    activationBorderColor: "#4f8cff",
    sequenceNumberColor: "#0b0f18",
  },
  flowchart: { htmlLabels: true, curve: "basis", padding: 14, nodeSpacing: 36, rankSpacing: 70, useMaxWidth: false },
  sequence: { mirrorActors: false, actorMargin: 60, messageMargin: 42, boxMargin: 12, width: 190, height: 60, useMaxWidth: false },
};



const WORKFLOW = `
sequenceDiagram
  autonumber
  actor U as You
  participant W as React UI
  participant A as FastAPI /api/chat
  participant G as LangChain agent
  participant L as LLM (OpenAI · Anthropic · Groq · Ollama)
  participant T as Tools (pandas · sklearn · matplotlib)

  U->>W: "Predict revenue — which features matter?"
  W->>A: POST message + session_id
  A->>G: question + chat history + dataset
  loop ReAct loop (max 8 iterations)
    G->>L: system prompt · columns · tool schemas
    L-->>G: tool_call: train_baseline_model(target="revenue")
    G->>T: run tool on the dataframe
    T-->>G: metrics table · importance table · PNG charts
  end
  L-->>G: final answer (insight + evidence + caveats)
  G-->>A: answer · tool trace · chart URLs
  A-->>W: JSON
  W-->>U: Markdown answer + charts + expandable trace
  Note over G,T: No API key? The heuristic planner picks<br/>the same tools with keyword rules.
`;

async function renderMermaid(code, file, { width = 1600 } = {}) {
  const mermaidJs = pathToFileURL(resolve(here, "node_modules", "mermaid", "dist", "mermaid.min.js")).href;
  const html = `<!doctype html><html><head><meta charset="utf-8">
  <style>
    body { margin: 0; background: #0b0f18; padding: 28px; display: inline-block; }
    #d svg { font-family: Inter, "Open Sans", "DejaVu Sans", sans-serif; }
    #d .cluster-label span, #d .nodeLabel, #d .edgeLabel { font-weight: 500; }
    #d small { color: #aab4cc; font-size: 12px; }
    #d .cluster-label { font-weight: 700; }
    #d .cluster rect { rx: 14; ry: 14; }
    #d .node rect, #d .node polygon, #d .node path { rx: 10; ry: 10; }
  </style></head>
  <body><div id="d"></div><script src="${mermaidJs}"></script></body></html>`;
  const tmp = resolve(here, ".mermaid-tmp.html");
  writeFileSync(tmp, html);
  const page = await newPage({ width, height: 900, scale: 2 });
  await page.goto(pathToFileURL(tmp).href, { waitUntil: "load" });
  await page.evaluate(
    async (code, theme) => {
      window.mermaid.initialize({ startOnLoad: false, securityLevel: "loose", ...theme });
      const { svg } = await window.mermaid.render("diagram", code);
      document.getElementById("d").innerHTML = svg;
    },
    code,
    THEME,
  );
  await sleep(300);
  const el = await page.$("body");
  await el.screenshot({ path: resolve(OUT, file), omitBackground: false });
  console.log(`✓ ${file}`);
  await page.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b. Architecture diagram (hand-laid-out HTML for a predictable, readable picture)
// ─────────────────────────────────────────────────────────────────────────────
async function architecture() {
  const icon = (path, color) => `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
  const I = {
    browser: icon('<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M8 21h8M12 18v3"/>', "#4f8cff"),
    api: icon('<path d="M4 7h16M4 12h16M4 17h10"/><circle cx="19" cy="17" r="2"/>', "#4f8cff"),
    agent: icon('<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 8V4M8 4h8"/><circle cx="9" cy="14" r="1.2" fill="#c77dff" stroke="none"/><circle cx="15" cy="14" r="1.2" fill="#c77dff" stroke="none"/>', "#c77dff"),
    tools: icon('<path d="M4 20l6-6M14 4l6 6-8 8-6-6z"/><path d="M15 9l-1 1"/>', "#3ddc97"),
    llm: icon('<path d="M7 18a5 5 0 0 1-.9-9.9A6 6 0 0 1 17.8 9 4.5 4.5 0 0 1 17 18z"/>', "#ffd166"),
  };
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
  * { box-sizing: border-box; }
  body { margin: 0; width: 1680px; padding: 36px 40px 40px; background: #0b0f18; font-family: Inter, "Open Sans", "DejaVu Sans", sans-serif; color: #e6ebf7; }
  h2.title { margin: 0 0 4px; font-size: 26px; font-weight: 800; letter-spacing: -0.3px; }
  p.sub { margin: 0 0 26px; color: #8c98b8; font-size: 15px; }
  .row { display: grid; grid-template-columns: 300px 96px 300px 96px 340px 96px 372px; align-items: stretch; }
  .card { background: #131a2a; border: 1.5px solid var(--c, #223052); border-radius: 18px; padding: 18px 20px; position: relative; box-shadow: 0 16px 40px rgba(0,0,0,.35); }
  .card .head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  .card h3 { margin: 0; font-size: 18px; font-weight: 800; }
  .card .tag { font-size: 12px; color: #8c98b8; margin-bottom: 12px; }
  .card ul { margin: 0; padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: 6px; }
  .card li { font-size: 13.5px; color: #c7d0e6; padding-left: 14px; position: relative; }
  .card li::before { content: ""; position: absolute; left: 0; top: 8px; width: 6px; height: 6px; border-radius: 50%; background: var(--c); }
  .blue { --c: #4f8cff; } .purple { --c: #c77dff; } .green { --c: #3ddc97; } .yellow { --c: #ffd166; }
  .conn { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; color: #8c98b8; font-size: 12px; text-align: center; padding: 0 4px; }
  .conn svg { width: 100%; height: 26px; overflow: visible; }
  .split { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 4px; }
  .choice { display: flex; align-items: center; gap: 10px; }
  .diamond { width: 118px; height: 44px; background: #1c1a3a; border: 1.5px solid #c77dff; border-radius: 8px; display: grid; place-items: center; font-size: 12px; text-align: center; line-height: 1.15; }
  .branch { flex: 1; display: flex; flex-direction: column; gap: 8px; }
  .opt { border: 1.5px solid #c77dff; border-radius: 10px; padding: 8px 10px; font-size: 13px; background: #1c1a3a; }
  .opt b { display: block; font-size: 13.5px; }
  .opt small { color: #aab4cc; font-size: 11.5px; }
  .opt .lbl { position: absolute; margin-left: -34px; margin-top: 2px; color: #8c98b8; font-size: 11px; }
  .tool { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .tool code { display: block; background: #0f1420; border: 1px solid #1f4a3a; border-radius: 8px; padding: 6px 9px; font-size: 12px; color: #a6f0cf; font-family: ui-monospace, Menlo, monospace; }
  .tool code.wide { grid-column: 1 / -1; }
  .under { display: grid; grid-template-columns: 300px 96px 300px 96px 340px 96px 372px; margin-top: 0; }
  .vconn { grid-column: 5; display: flex; flex-direction: row; align-items: center; justify-content: center; color: #8c98b8; font-size: 12px; height: 62px; gap: 10px; }
  .vconn svg { height: 100%; width: 40px; overflow: visible; }
  .llm { grid-column: 5; }
  .providers { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 4px; }
  .prov { border: 1.5px solid #ffd166; background: #2a2314; border-radius: 10px; padding: 8px 10px; font-size: 13px; font-weight: 700; text-align: center; }
  .prov small { display: block; font-weight: 400; color: #d9c58f; font-size: 11px; }
  .foot { margin-top: 26px; display: flex; gap: 26px; flex-wrap: wrap; color: #8c98b8; font-size: 13px; }
  .foot b { color: #e6ebf7; }
  .legend { display: inline-flex; align-items: center; gap: 7px; }
  .legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  </style></head><body>
  <h2 class="title">Architecture</h2>
  <p class="sub">One question flows left → right. Every number comes from a tool; the LLM only plans and explains.</p>
  <div class="row">
    <div class="card blue"><div class="head">${I.browser}<h3>Browser</h3></div><div class="tag">React 18 + Vite + TypeScript</div>
      <ul><li>Upload a CSV or use the demo dataset</li><li>Chat with the agent in plain English</li><li>Charts, tables, tool trace</li><li>Download a Markdown report</li></ul></div>
    <div class="conn"><span>HTTP · JSON</span><svg viewBox="0 0 100 26"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#8c98b8"/></marker><marker id="b" markerWidth="8" markerHeight="8" refX="2" refY="4" orient="auto"><path d="M8,0 L0,4 L8,8 z" fill="#8c98b8"/></marker></defs><line x1="6" y1="13" x2="94" y2="13" stroke="#8c98b8" stroke-width="2" marker-end="url(#a)" marker-start="url(#b)"/></svg><span>question ⇄ answer + charts</span></div>
    <div class="card blue"><div class="head">${I.api}<h3>FastAPI backend</h3></div><div class="tag">Python 3.11 · Pydantic · Uvicorn</div>
      <ul><li><code>POST /api/datasets</code> — validate &amp; profile CSV</li><li><code>POST /api/datasets/{id}/chat</code></li><li><code>GET /api/artifacts/*.png</code> — charts</li><li>Bounded in-memory dataset store &amp; sessions</li></ul></div>
    <div class="conn"><span>question + history</span><svg viewBox="0 0 100 26"><line x1="6" y1="13" x2="94" y2="13" stroke="#8c98b8" stroke-width="2" marker-end="url(#a)"/></svg><span>&nbsp;</span></div>
    <div class="card purple"><div class="head">${I.agent}<h3>Data Scientist agent</h3></div><div class="tag">Same tools, two interchangeable brains</div>
      <div class="choice"><div class="diamond">LLM key<br/>configured?</div><div class="branch">
        <div class="opt"><span class="lbl">yes</span><b>LangChain tool-calling agent</b><small>plans → calls tools → observes → explains</small></div>
        <div class="opt"><span class="lbl">no</span><b>Heuristic planner</b><small>keyword rules · deterministic · offline</small></div></div></div></div>
    <div class="conn"><span>tool calls</span><svg viewBox="0 0 100 26"><line x1="6" y1="13" x2="94" y2="13" stroke="#8c98b8" stroke-width="2" marker-end="url(#a)"/></svg><span>Markdown + PNG back</span></div>
    <div class="card green"><div class="head">${I.tools}<h3>Analysis tools</h3></div><div class="tag">pandas · scikit-learn · matplotlib</div>
      <div class="tool"><code>dataset_overview</code><code>describe_columns</code><code>missing_values_report</code><code>value_counts</code><code>correlation_analysis</code><code>detect_outliers</code><code>group_aggregate</code><code>plot_chart</code><code>time_series_trend</code><code>train_baseline_model</code><code class="wide">run_pandas_expression (opt-in, AST allow-listed)</code></div></div>
  </div>
  <div class="under">
    <div class="vconn"><svg viewBox="0 0 40 60"><line x1="20" y1="4" x2="20" y2="56" stroke="#8c98b8" stroke-width="2" marker-end="url(#a)" marker-start="url(#b)"/></svg><span>plan · observe<br/>(tool schemas ⇄ tool calls)</span></div>
  </div>
  <div class="under" style="margin-top:-2px">
    <div class="card yellow llm"><div class="head">${I.llm}<h3>Any LLM provider</h3></div><div class="tag">switch with one variable: <code>DS_LLM_PROVIDER</code></div>
      <div class="providers"><div class="prov">OpenAI<small>gpt-4o-mini</small></div><div class="prov">Anthropic<small>claude-3-5-haiku</small></div><div class="prov">Groq<small>llama-3.3-70b</small></div><div class="prov">Ollama<small>local · llama3.1</small></div></div></div>
  </div>
  <div class="foot">
    <span class="legend"><i style="background:#4f8cff"></i> web layer</span>
    <span class="legend"><i style="background:#c77dff"></i> agent</span>
    <span class="legend"><i style="background:#3ddc97"></i> deterministic tools</span>
    <span class="legend"><i style="background:#ffd166"></i> optional LLM</span>
    <span><b>No key?</b> The app still works: the heuristic planner routes questions to the same tools.</span>
  </div>
  </body></html>`;
  const page = await newPage({ width: 1680, height: 900, scale: 2 });
  await page.setContent(html, { waitUntil: "load" });
  await sleep(200);
  const body = await page.$("body");
  await body.screenshot({ path: resolve(OUT, "architecture.png") });
  console.log("✓ architecture.png");
  await page.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Banner
// ─────────────────────────────────────────────────────────────────────────────
async function banner() {
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
  * { box-sizing: border-box; }
  body { margin: 0; width: 1600px; height: 520px; font-family: Inter, "Open Sans", "DejaVu Sans", sans-serif; color: #e6ebf7;
    background: radial-gradient(900px 500px at 12% 0%, #1b2a52 0%, rgba(11,15,24,0) 60%),
                radial-gradient(700px 400px at 95% 100%, #34205a 0%, rgba(11,15,24,0) 60%), #0b0f18; overflow: hidden; position: relative; }
  .grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(79,140,255,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(79,140,255,.07) 1px, transparent 1px); background-size: 40px 40px; }
  .left { position: absolute; left: 80px; top: 92px; width: 820px; }
  .logo { display: inline-flex; align-items: center; gap: 14px; margin-bottom: 26px; }
  .logo .box { width: 58px; height: 58px; border-radius: 16px; background: linear-gradient(135deg,#1a2440,#24184a); border: 1px solid #2c3b63; display: grid; place-items: center; }
  .kicker { font-size: 15px; letter-spacing: 3px; text-transform: uppercase; color: #8c98b8; font-weight: 600; }
  h1 { font-size: 62px; line-height: 1.05; margin: 0 0 18px; font-weight: 800; letter-spacing: -1px; }
  h1 span { background: linear-gradient(90deg,#4f8cff,#c77dff); -webkit-background-clip: text; color: transparent; }
  p { font-size: 22px; color: #aab4cc; margin: 0 0 30px; line-height: 1.45; max-width: 760px; }
  .chips { display: flex; gap: 10px; flex-wrap: wrap; }
  .chip { padding: 9px 16px; border-radius: 999px; border: 1px solid #2c3b63; background: rgba(19,26,42,.9); font-size: 16px; font-weight: 600; }
  .chip i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
  .card { position: absolute; right: 80px; top: 70px; width: 560px; background: rgba(19,26,42,.92); border: 1px solid #223052; border-radius: 22px; padding: 22px 24px; box-shadow: 0 30px 80px rgba(0,0,0,.5); }
  .q { display: flex; justify-content: flex-end; margin-bottom: 14px; }
  .q span { background: linear-gradient(135deg,#2a4b9b,#3a3f9b); padding: 10px 16px; border-radius: 14px; font-size: 17px; }
  .a { background: #0f1420; border: 1px solid #223052; border-radius: 14px; padding: 14px 16px; font-size: 15px; color: #c7d0e6; }
  .a b { color: #fff; font-size: 16px; }
  .bars { display: flex; align-items: flex-end; gap: 14px; height: 150px; margin-top: 14px; padding: 0 6px; }
  .bar { flex: 1; border-radius: 8px 8px 3px 3px; background: linear-gradient(180deg,#4f8cff,#3457b8); position: relative; }
  .bar::after { content: attr(data-l); position: absolute; top: 100%; left: 0; right: 0; text-align: center; font-size: 12px; color: #8c98b8; margin-top: 6px; }
  .bar.b2 { background: linear-gradient(180deg,#ff8a4c,#c05a2a); } .bar.b3 { background: linear-gradient(180deg,#3ddc97,#1f8f60); }
  .bar.b4 { background: linear-gradient(180deg,#c77dff,#7c45b3); } .bar.b5 { background: linear-gradient(180deg,#ffd166,#b58d2a); } .bar.b6 { background: linear-gradient(180deg,#ef476f,#a02a48); }
  .trace { margin-top: 30px; font-size: 13px; color: #8c98b8; }
  .trace code { background: #0f1420; border: 1px solid #223052; color: #9cc0ff; padding: 2px 8px; border-radius: 6px; margin-right: 4px; font-family: ui-monospace, Menlo, monospace; }
  </style></head><body>
  <div class="grid"></div>
  <div class="left">
    <div class="logo"><div class="box"><svg viewBox="0 0 64 64" width="38" height="38"><rect x="12" y="34" width="8" height="18" rx="2" fill="#4f8cff"/><rect x="24" y="24" width="8" height="28" rx="2" fill="#8f7dff"/><rect x="36" y="14" width="8" height="38" rx="2" fill="#c77dff"/><circle cx="50" cy="14" r="5" fill="#3ddc97"/></svg></div><span class="kicker">LangChain · FastAPI · React</span></div>
    <h1>Data Scientist<br/><span>LangChain AI</span></h1>
    <p>Upload a CSV and talk to an AI data scientist that profiles, visualises and models your data — with every number computed by real tools.</p>
    <div class="chips">
      <span class="chip"><i style="background:#4f8cff"></i>Natural-language EDA</span>
      <span class="chip"><i style="background:#3ddc97"></i>Charts &amp; baseline models</span>
      <span class="chip"><i style="background:#c77dff"></i>OpenAI · Anthropic · Groq · Ollama</span>
      <span class="chip"><i style="background:#ffd166"></i>Works offline, no API key</span>
    </div>
  </div>
  <div class="card">
    <div class="q"><span>Total revenue by category?</span></div>
    <div class="a"><b>Electronics leads with 116.7k</b> — 47% of all revenue. Books is lowest at 9.7k (a 12× gap).
      <div class="bars"><div class="bar" style="height:100%" data-l="Electronics"></div><div class="bar b2" style="height:41%" data-l="Home"></div><div class="bar b3" style="height:32%" data-l="Clothing"></div><div class="bar b4" style="height:21%" data-l="Sports"></div><div class="bar b5" style="height:12%" data-l="Beauty"></div><div class="bar b6" style="height:9%" data-l="Books"></div></div>
    </div>
    <div class="trace">▸ 1 tool call <code>group_aggregate</code> group_by="category", metric="revenue", aggregation="sum" · 0.2s</div>
  </div>
  </body></html>`;
  const page = await newPage({ width: 1600, height: 520, scale: 1.5 });
  await page.setContent(html, { waitUntil: "load" });
  await sleep(200);
  await page.screenshot({ path: resolve(OUT, "banner.png") });
  console.log("✓ banner.png");
  await page.close();
}

// ─────────────────────────────────────────────────────────────────────────────
const only = process.argv[2];
try {
  if (!only || only === "banner") await banner();
  if (!only || only === "diagrams") {
    await architecture();
    await renderMermaid(WORKFLOW, "agent-workflow.png", { width: 1500 });
  }
  if (!only || only === "screenshots") {
    if (!existsSync(resolve(here, "..", "frontend", "dist"))) console.warn("frontend/dist missing — run `npm run build` in frontend/ first");
    await screenshots();
  }
} finally {
  await browser.close();
}
