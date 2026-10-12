/* Gaze Lab controller.
 *
 * Flow: consent -> camera -> 9-point calibration -> stimulus -> results.
 *
 * Everything runs client-side. No frame, landmark, or gaze coordinate is ever sent
 * to the server or anywhere else.
 */
import { GazeTracker, CAL_POINTS } from "./gaze.js";
import { BrainView, REGIONS } from "./brain.js";
import { SYNTHETIC, DEMO_METRICS } from "./demo_session.js";

const $ = (s) => document.querySelector(s);
const el = (t, c, x) => { const n = document.createElement(t); if (c) n.className = c; if (x != null) n.textContent = x; return n; };

let tracker = null, brain = null, rafId = null;
let lastMetrics = null, lastNetworks = null;
let stimURL = "/static/sample_ad.png";
// True only for the canned demo. Drives the "these numbers are made up" banner and
// gets written into any exported JSON so a downloaded file is self-describing.
let isDemo = false;

// AOIs for the bundled sample, as fractions of the stimulus box.
const SAMPLE_AOIS = [
  { id: "headline", label: "Headline",      x: .04, y: .10, w: .40, h: .38 },
  { id: "badge",    label: "Circular badge", x: .70, y: .22, w: .26, h: .30 },
  { id: "footer",   label: "Footer bar",    x: 0,  y: .70, w: 1,   h: .30 },
  { id: "cta",      label: "Call to action", x: .70, y: .83, w: .28, h: .12 },
];

/**
 * Show exactly one step.
 *
 * We set BOTH the `active` class and the `hidden` attribute. The class alone is not
 * enough: `.gl .step.active { display:block }` is an author-origin rule, which beats
 * the UA stylesheet's `[hidden] { display:none }` regardless of specificity. Relying
 * on that accident means a step marked hidden-but-active would still render. Driving
 * both attributes keeps the two sources of truth in agreement.
 */
function step(name) {
  for (const s of document.querySelectorAll(".step")) {
    const on = s.id === "step-" + name;
    s.classList.toggle("active", on);
    s.hidden = !on;
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function fatal(html) {
  const f = $("#fatal");
  f.hidden = false;
  f.innerHTML = `<div><strong>We couldn't start the Gaze Lab.</strong><br><br>${html}</div>`;
}

function stopLoop() { if (rafId) cancelAnimationFrame(rafId); rafId = null; }

// ──────────────────────────── demo (no camera) ────────────────────────────
$("#btn-demo").addEventListener("click", () => {
  isDemo = true;
  lastMetrics = DEMO_METRICS;
  // The brain view needs no tracker, so a demo works with the camera never opened.
  step("results");
  renderResults();
});

$("#btn-demo-real").addEventListener("click", () => { isDemo = false; location.reload(); });

// ───────────────────────────────── consent ─────────────────────────────────
$("#btn-consent").addEventListener("click", async () => {
  const note = $("#consent-note");
  note.textContent = "Requesting camera access…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not expose a camera API. Camera capture needs HTTPS or localhost.");
    }
    step("cal");
    tracker = new GazeTracker($("#cam"));

    $("#cal-hint").textContent = "Loading the on-device face tracker (first run only)…";
    await tracker.start();

    $("#cal-hint").textContent = "Starting camera…";
    await tracker.openCamera();
    runCalibration();
  } catch (err) {
    const msg = String(err?.message || err);
    if (/NotAllowed|Permission/i.test(msg)) {
      fatal(`Camera permission was declined.<br><br>To enable it: click the camera icon in your
        browser's address bar &rarr; <em>Allow</em> for this site, then reload.<br><br>
        You can still use <a href="/" style="color:#7fc4e8">image analysis</a> without a camera.`);
    } else {
      fatal(`${msg}<br><br>Camera capture requires a secure context. <code>http://127.0.0.1:8000</code>
        counts as secure; opening the page via a raw <code>file://</code> path does not.`);
    }
    console.error(err);
  }
});

// ─────────────────────────────── calibration ───────────────────────────────
let calIndex = -1, calTimer = null, calCaptured = 0;

function runCalibration() {
  tracker.resetCalibration();
  calIndex = -1;
  calCaptured = 0;
  $("#btn-cal-retry").hidden = true;
  $("#btn-to-stim").hidden = true;
  $("#btn-to-stim").disabled = true;
  nextCalPoint();
  startLoop();
}

function nextCalPoint() {
  calIndex++;
  if (calIndex >= CAL_POINTS.length) {
    finishCalibration();
    return;
  }
  const [fx, fy] = CAL_POINTS[calIndex];
  const dot = $("#cal-dot");
  dot.style.left = `${fx * 100}%`;
  dot.style.top = `${fy * 100}%`;
  dot.classList.add("show");
  $("#cal-status").textContent =
    `Point ${calIndex + 1} of ${CAL_POINTS.length} — look at the dot and hold still`;

  tracker.beginCalibrationTarget([fx, fy]);
  clearTimeout(calTimer);
  calTimer = setTimeout(() => {
    if (tracker._calTarget === null) nextCalPoint();
  }, 2600);
}

function finishCalibration() {
  clearTimeout(calTimer);
  $("#cal-dot").classList.remove("show");
  const ok = tracker.finishCalibration();

  if (!ok) {
    $("#cal-status").textContent =
      "Not enough clean samples — some points had no face detected. Try again in better light.";
    $("#btn-cal-retry").hidden = false;
    return;
  }

  const pct = tracker.calErrorPct ?? 0;
  const verdict =
    pct < 2 ? "Excellent" : pct < 3.5 ? "Good" : pct < 5 ? "Usable" : "Poor";
  $("#cal-status").innerHTML =
    `Calibrated. <strong>${verdict}</strong> — expected error ` +
    `<strong>${pct.toFixed(2)}%</strong> of screen width (leave-one-out).`;
  $("#btn-to-stim").hidden = false;
  $("#btn-to-stim").disabled = false;
  $("#btn-cal-retry").hidden = false;
  step("cal");
}

$("#btn-cal-retry").addEventListener("click", runCalibration);

// The single animation loop drives calibration and recording alike.
//
// The loop is armed at page load but the tracker only exists after consent, so the
// tracker check must come first: without it the very first frames throw on a null
// tracker and spam the console before the user has granted any permission.
function loop() {
  rafId = requestAnimationFrame(loop);
  if (!tracker) return;
  const r = tracker.tick();
  if (!r) return;

  if (r.mode === "calibration" && tracker._calTarget === null) nextCalPoint();
  if (r.mode === "recording") {
    const p = r.point;
    const pip = $("#pip");
    pip.style.left = `${p.x * 100}%`;
    pip.style.top = `${p.y * 100}%`;
  }
  if (calIndex >= 0 && calIndex < CAL_POINTS.length) {
    $("#cal-bar").style.width = `${Math.round(tracker.calibrationProgress() * 100)}%`;
  }
}

/**
 * Arm the loop. Idempotent by design: loop() re-arms itself every frame, so a naive
 * "cancel then start" here would leave the old chain running and start a second one.
 * Two loops would double the sampling rate and duplicate every calibration sample.
 */
function startLoop() {
  if (rafId) return;
  rafId = requestAnimationFrame(loop);
}

$("#btn-to-stim").addEventListener("click", () => {
  step("stim");
  $("#pip").hidden = false;
  startLoop();
});

// ───────────────────────────────── stimulus ────────────────────────────────
$("#stim-file").addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  stimURL = URL.createObjectURL(f);
  $("#stim").src = stimURL;
  $("#res-stim").src = stimURL;
  $("#stim-name").textContent = f.name;
  // No hand-made AOIs for arbitrary images - metrics still work without them.
  tracker.setAOIs([]);
});

// Simple heuristic AOIs for a user image: split into a 3x3 grid so every region gets
// dwell stats. Clearly described in the UI as a generic grid, not semantic regions.
function gridAOIs() {
  const out = [];
  const names = ["Top-left", "Top-centre", "Top-right",
                 "Mid-left", "Centre", "Mid-right",
                 "Bottom-left", "Bottom-centre", "Bottom-right"];
  let i = 0;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out.push({
        id: `g${r}${c}`, label: names[i++],
        x: c / 3, y: r / 3, w: 1 / 3, h: 1 / 3,
      });
    }
  }
  return out;
}

$("#btn-rec").addEventListener("click", () => {
  if (!tracker.calibrated) {
    alert("Calibrate first — we need to know where your eyes are pointing.");
    return;
  }
  const aois = $("#stim-name").textContent === "bundled sample" ? SAMPLE_AOIS : gridAOIs();
  tracker.setAOIs(aois);
  tracker.startRecording();

  $("#rec-banner").hidden = false;
  $("#rec-timer").hidden = false;
  $("#pip").hidden = false;
  $("#btn-rec").disabled = true;
  $("#btn-to-results").hidden = true;
  $("#btn-restart").hidden = true;
  $("#stim-status").textContent = "Recording — look naturally. This stops itself after 12 seconds.";
  startLoop();

  clearInterval(window.__recT);
  window.__recT = setInterval(() => {
    const m = tracker.samples;
    const t = m.length ? m[m.length - 1].t : 0;
    $("#rec-timer").textContent = `${t.toFixed(1)}s`;
    if (t >= 12) stopRecording();
  }, 100);
});

function stopRecording() {
  clearInterval(window.__recT);
  const metrics = tracker.stopRecording();
  lastMetrics = metrics;

  $("#rec-banner").hidden = true;
  $("#rec-timer").hidden = true;
  $("#pip").hidden = true;
  $("#btn-rec").disabled = false;
  $("#btn-to-results").hidden = false;
  $("#btn-restart").hidden = false;
  $("#stim-status").textContent =
    `Done — ${metrics.fixation_count} fixations in ${metrics.duration_s}s.`;
  step("stim");
  renderResults();
}

$("#btn-restart").addEventListener("click", () => {
  tracker.startRecording();
  $("#rec-banner").hidden = false;
  $("#btn-to-results").hidden = true;
  $("#btn-restart").hidden = true;
  $("#stim-status").textContent = "Recording again…";
});
$("#btn-restart").addEventListener("click", () => { window.__recT2 = 1; });

$("#btn-to-results").addEventListener("click", () => { step("results"); renderResults(); });
$("#btn-again").addEventListener("click", () => location.reload());

$("#btn-save").addEventListener("click", () => {
  if (!lastMetrics) return;
  const blob = new Blob([JSON.stringify({
    recorded_at: new Date().toISOString(),
    // Provenance goes first in the file so it cannot be skimmed past.
    synthetic: isDemo,
    ...(isDemo ? { synthetic_reason: SYNTHETIC.source } : {}),
    stimulus: $("#stim-name").textContent,
    method: "webcam gaze estimation, 9-point bilinear calibration, I-DT fixation segmentation",
    disclaimer: "Gaze is measured. Brain activity is NOT measured; networks are implicated from literature.",
    calibration_error_pct_of_screen: tracker?.calErrorPct ?? null,
    metrics: lastMetrics,
    networks: lastNetworks,
  }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "neuroinsight-gaze-session.json";
  a.click();
});

// ────────────────────────────────── results ────────────────────────────────
const REGION_COLORS = Object.fromEntries(REGIONS.map((r) => [r.id, `#${r.color.toString(16).padStart(6, "0")}`]));

function renderResults() {
  const m = lastMetrics;
  if (!m) return;

  // Demo sessions are labelled before anything else is drawn, so the banner is never
  // left off by a later early-return.
  $("#demo-banner").hidden = !isDemo;

  $("#res-stim").src = stimURL;
  drawGazeOverlay();

  // AOI table
  const aoi = $("#aoi-table");
  aoi.innerHTML = "";
  const aois = m.aoi || [];
  if (!aois.length) {
    aoi.appendChild(el("p", "tiny", "No areas of interest were set for this image."));
  } else {
    const tbl = el("table", "aoi");
    tbl.innerHTML = `<thead><tr><th>Region</th><th>First look</th>
      <th>Fixations</th><th>Time</th></tr></thead>`;
    const tb = el("tbody");
    const maxShare = Math.max(...aois.map((a) => a.share), 0.0001);
    for (const a of aois) {
      if (!a.fixations && !a.dwell_s) continue;
      const tr = el("tr");
      tr.innerHTML =
        `<td>${a.label}</td>` +
        `<td>${a.first_fixation_s == null ? "—" : a.first_fixation_s.toFixed(2) + "s"}</td>` +
        `<td class="bar-cell"><div class="fill" style="width:${(a.share / maxShare) * 100}%"></div><span>${a.fixations}</span></td>` +
        `<td>${a.dwell_s.toFixed(2)}s</td>`;
      tb.appendChild(tr);
    }
    tbl.appendChild(tb);
    aoi.appendChild(tbl);
  }

  // headline stats
  const stats = $("#gaze-stats");
  stats.innerHTML = "";
  const add = (k, v, u, sub) => {
    const d = el("div", "stat");
    d.innerHTML = `<div class="k">${k}</div><div class="v">${v}<span class="u">${u || ""}</span></div>` +
      (sub ? `<div class="sub">${sub}</div>` : "");
    stats.appendChild(d);
  };
  add("Viewing time", m.duration_s, "s");
  add("Fixations", m.fixation_count, "", "sustained looks");
  add("Avg. fixation", (m.mean_fixation_ms / 1000).toFixed(2), "s", "per look");
  add("Longest look", m.longest_fixation_s, "s");
  add("Saccades", m.saccade_count, "", "rapid eye jumps");
  add("Scanpath rate", m.saccade_rate_hz, "/s", "more = more exploring");

  // Calibration note. A demo has no real calibration, so it says so rather than
  // borrowing a tracker that may not even exist yet.
  if (isDemo) {
    $("#cal-note").innerHTML =
      `This demo carries a made-up calibration error of <strong>` +
      `${(m.calibration_error_pct ?? 0).toFixed(2)}%</strong> of screen width. ` +
      `A real session reports the error actually measured on <em>your</em> eyes.`;
  } else {
    const errPct = tracker?.calErrorPct ?? 0;
    const q = errPct < 2 ? "good" : errPct < 3.5 ? "warn" : "bad";
    $("#cal-note").innerHTML =
      `Calibration error on this session: <strong class="${q}">${errPct.toFixed(2)}%</strong> of screen width. ` +
      `That sets the floor on every number above — a fixation is only located to within about that margin.`;
  }

  // brain + networks
  if (brain) { brain.dispose(); brain = null; }
  brain = new BrainView($("#brain-canvas"));
  lastNetworks = brain.update(m, aois);
  renderNetworks(lastNetworks);

  renderAccuracy(isDemo ? null : (tracker?.calErrorPct ?? null));
}

function drawGazeOverlay() {
  const c = $("#res-overlay");
  const img = $("#res-stim");
  const draw = () => {
    const w = img.clientWidth, h = img.clientHeight;
    if (!w || !h) return;
    c.width = w; c.height = h;
    const g = c.getContext("2d");
    g.clearRect(0, 0, w, h);

    // object-fit for an <img> in a box: recompute the drawn rect
    const ir = img.naturalWidth / img.naturalHeight || 16 / 9;
    const br = w / h;
    let dw = w, dh = h, dx = 0, dy = 0;
    if (ir > br) { dw = h * ir; dx = (w - dw) / 2; } else { dh = w / ir; dy = (h - dh) / 2; }
    g.save();
    g.beginPath(); g.rect(dx, dy, dw, dh); g.clip();

    const fx = lastMetrics.raw_fixations || [];

    // During a replay, only fixations up to the current index are drawn, and the
    // most recent one is highlighted, so the order of looking is legible.
    const upto = replay.i == null ? fx.length : Math.max(1, replay.i + 1);

    // Connect consecutive fixations with a faint line: the scanpath itself.
    if (upto > 1) {
      g.beginPath();
      for (let k = 0; k < upto; k++) {
        const px = dx + fx[k].x * dw, py = dy + fx[k].y * dh;
        k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
      }
      g.strokeStyle = "rgba(37,99,235,.34)";
      g.lineWidth = 1.6;
      g.setLineDash([4, 4]);
      g.stroke();
      g.setLineDash([]);
    }

    for (let k = 0; k < upto; k++) {
      const f = fx[k];
      const x = dx + f.x * dw, y = dy + f.y * dh;
      const active = replay.i != null && k === replay.i;
      const r = (Math.min(34, Math.max(7, f.dur * 26))) * (active ? 1.35 : 1);
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, active ? "rgba(37,99,235,.85)" : "rgba(47,111,143,.62)");
      grad.addColorStop(1, "rgba(47,111,143,0)");
      g.fillStyle = grad;
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();

      g.beginPath(); g.arc(x, y, active ? 5.4 : 3.4, 0, Math.PI * 2);
      g.fillStyle = active ? "rgba(29,78,216,.98)" : "rgba(20,60,90,.95)";
      g.fill();
      g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1.4; g.stroke();
    }

    // Replay cursor: a soft ring that sits on the fixation being "looked at".
    if (replay.i != null && fx[replay.i]) {
      const f = fx[replay.i];
      const x = dx + f.x * dw, y = dy + f.y * dh;
      g.beginPath();
      g.arc(x, y, 14 + 5 * Math.sin(replay.tick / 6), 0, Math.PI * 2);
      g.strokeStyle = "rgba(37,99,235,.55)";
      g.lineWidth = 2;
      g.stroke();
    }
    g.restore();
  };
  if (img.complete && img.naturalWidth) draw();
  else img.addEventListener("load", draw, { once: true });
  window.addEventListener("resize", draw);
  replay.redraw = draw;
}

/**
 * Replay the scanpath in the order the fixations actually happened.
 *
 * Uses the recorded start times, so pacing reflects real dwell rather than a fixed
 * interval per dot. Speed is deliberately compressed (4x) so a 12-second session
 * does not become tedious to watch on a results page nobody is waiting to finish.
 */
let replayRAF = null;
const replay = { i: null, tick: 0, redraw: null, playing: false };

function toggleReplay() {
  const btn = $("#btn-replay");
  if (replay.playing) { stopReplay(); return; }

  const fx = lastMetrics?.raw_fixations || [];
  if (fx.length < 2) {
    btn.textContent = "Not enough fixations to replay";
    btn.disabled = true;
    return;
  }
  replay.playing = true;
  replay.i = 0;
  btn.textContent = "■ Stop replay";

  const t0 = performance.now();
  const start = fx[0].start;

  const step = () => {
    if (!replay.playing) return;
    replay.tick++;
    const elapsed = ((performance.now() - t0) / 1000) * 4;
    // find the last fixation whose start time has passed
    let idx = 0;
    for (let k = 0; k < fx.length; k++) {
      if (fx[k].start - start <= elapsed) idx = k; else break;
    }
    if (idx !== replay.i) {
      replay.i = idx;
      replay.redraw && replay.redraw();
    }
    const last = fx[fx.length - 1];
    if (elapsed > last.start - start + last.dur + 0.6) { stopReplay(true); return; }
    replayRAF = requestAnimationFrame(step);
  };
  replayRAF = requestAnimationFrame(step);
}

function stopReplay(finished) {
  replay.playing = false;
  if (replayRAF) cancelAnimationFrame(replayRAF);
  replayRAF = null;
  replay.i = null;
  const btn = $("#btn-replay");
  if (btn) btn.textContent = finished ? "↻ Replay again" : "▶ Replay my scanpath";
  replay.redraw && replay.redraw();
}

$("#btn-replay").addEventListener("click", toggleReplay);

function renderNetworks(nets) {
  const host = $("#network-list");
  host.innerHTML = "";
  for (const n of nets) {
    const col = REGION_COLORS[n.id] || "#999";
    const d = el("div", "net" + (n.engaged ? " on" : ""));
    d.innerHTML =
      `<div class="net-head">
         <span class="net-dot" style="background:${col}"></span>
         <span class="net-name">${n.region}</span>
         <span class="net-tag ${n.engaged ? "on" : "off"}">${n.engaged ? "likely involved" : "not indicated"}</span>
       </div>
       <div class="net-fn">${n.network}</div>
       <div class="net-cite">${n.cite}</div>
       <div class="net-cite">Sources: ${n.refs.join(" · ")}</div>
       <div class="net-meter"><i style="width:${Math.round(n.strength * 100)}%"></i></div>`;
    d.addEventListener("mouseenter", () => brain && brain.focusRegion(n.id));
    d.addEventListener("mouseleave", () => brain && brain.clearFocus());
    host.appendChild(d);
  }
}

function renderAccuracy(errPct) {
  const host = $("#accuracy");
  const m = lastMetrics;
  const has = errPct != null && !Number.isNaN(errPct);
  const items = [];

  if (has) {
    items.push([errPct < 3.5 ? "good" : "warn", "≈",
      `Calibration error <strong>${errPct.toFixed(2)}%</strong> of screen width. ` +
      `Reliable for <em>which region</em> you looked at and <em>when</em>.`]);
    items.push(["warn", "!",
      `Not precise enough for single-pixel fixations or features smaller than ` +
      `~${(errPct * 2).toFixed(0)}% of the image. A research eye-tracker would be needed.`]);
  } else {
    items.push(["warn", "!",
      `<strong>This is a demo.</strong> No calibration was performed, because no eyes ` +
      `were involved. Real sessions report their own measured calibration error.`]);
  }

  items.push([isDemo ? "warn" : "good", isDemo ? "!" : "≈",
    isDemo
      ? `These figures are <strong>made up</strong>. They are a fixed example, not a recording.`
      : `Your gaze is <strong>measured</strong>, not predicted. ${m.fixation_count} fixations ` +
        `were recorded from your actual eye movements.`]);

  items.push(["bad", "×",
    `<strong>No brain activity is measured anywhere in this app.</strong> The brain ` +
    `shows which networks the literature links to this kind of looking — nothing more.`]);

  host.innerHTML = "";
  for (const [cls, mark, txt] of items) {
    const d = el("div", "acc-item");
    d.innerHTML = `<span class="mark2 ${cls}">${mark}</span><span>${txt}</span>`;
    host.appendChild(d);
  }
}

startLoop();