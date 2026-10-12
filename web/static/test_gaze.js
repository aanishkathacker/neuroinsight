// Verify the calibration solver and metrics on synthetic ground truth.
// Run: node test_gaze.js   (or via the .mjs loader)
import { GazeTracker, mapToNetworks } from "./gaze.js";

// Node has no DOM. The tracker only uses its `video` element inside detect(), which
// these tests never call, so a plain stub is enough and keeps the test dependency-free.
const fakeVideo = () => ({ readyState: 0, videoWidth: 0, videoHeight: 0, srcObject: null, play: async () => {} });

let pass = 0, fail = 0;
function check(label, got, want, tol) {
  const ok = Math.abs(got - want) <= tol;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label.padEnd(46)} got ${got.toFixed(4)}  want ${want} +-${tol}`);
  ok ? pass++ : fail++;
}

// --------------------------------------------------------------- solver check
console.log("\nLEAST-SQUARES SOLVER (recover known polynomial coefficients)");
{
  const t = new GazeTracker(fakeVideo());
  // Build samples where the TRUE map is: x' = 0.5 + 1.2x - 0.3y + 0.4x^2
  //                                       y' = 0.25 - 0.2x + 0.9y + 0.1xy
  const samples = [];
  let seed = 42;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 80; i++) {
    const x = rnd() * 0.1, y = rnd() * 0.1;
    samples.push({
      raw: { x, y },
      target: [
        0.5 + 1.2 * x - 0.3 * y + 0.4 * x * x,
        0.25 - 0.2 * x + 0.9 * y + 0.1 * x * y,
      ],
    });
  }
  t._fit(samples);
  const ev = (c, r) =>
    c[0] + c[1] * r.x + c[2] * r.y + c[3] * r.x ** 2 + c[4] * r.x * r.y + c[5] * r.y ** 2;

  check("recovered cx[0] (0.5)", ev(t.cx, { x: 0, y: 0 }), 0.5, 1e-3);
  check("recovered cx[1] (1.2)", t.cx[1], 1.2, 1e-3);
  check("recovered cx[3] (0.4)", t.cx[3], 0.4, 1e-2);
  check("recovered cy[1] (-0.2)", t.cy[1], -0.2, 1e-2);
  check("recovered cy[4] (0.1)", t.cy[4], 0.1, 1e-2);

  // Round-trip a raw point through the fitted map.
  const p = { x: 0.05, y: 0.07 };
  check("round-trip x", ev(t.cx, p), 0.5 + 1.2 * p.x - 0.3 * p.y + 0.4 * p.x ** 2, 1e-6);
  check("round-trip y", ev(t.cy, p), 0.25 - 0.2 * p.x + 0.9 * p.y + 0.1 * p.x * p.y, 1e-6);
}

// ------------------------------------------------------------- fixation logic
console.log("\nFIXATION / SACCADE SEGMENTATION (I-DT, 3.5% dispersion, 200ms dwell)");
{
  const t = new GazeTracker(fakeVideo());
  t.calibrated = true;
  t.cx = [0, 0, 0, 0, 0, 0];
  t.cy = [0, 0, 0, 0, 0, 0];
  t._map = function (raw) { return { x: raw.x, y: raw.y }; };

  t.startRecording();
  let clock = 0;
  const add = (x, y) => {
    // NB: use a local clock, not the tracker's own _last field, which starts as
    // null and would make `t._last ? ... : 0` collapse every timestamp to 0.
    const p = { t: clock, x, y };
    clock += 0.05;
    t.samples.push(p);      // tick() does this; we mirror it so sample_count is real
    t._classify(p);
  };

  // 0.50s dwell at (0.2,0.2), then a jump to (0.8,0.8), dwell 0.40s
  for (let i = 0; i < 10; i++) add(0.20, 0.20);
  for (let i = 0; i < 8; i++) add(0.80, 0.80);

  const m = t.computeMetrics();
  check("raw fixations kept", m.raw_fixations.length, 2, 0.5);
  check("saccade detected", m.saccade_count, 1, 0.5);
  check("saccade amplitude ~0.85", m.mean_saccade_amplitude, 0.85, 0.12);
  check("sample count", m.sample_count, 18, 0.5);
  check("duration ~0.85s", m.duration_s, 0.85, 0.06);
}

// ---------------------------------------------------------------- AOI mapping
console.log("\nAOI METRICS");
{
  const t = new GazeTracker(fakeVideo());
  t.calibrated = true;
  t._map = (r) => ({ x: r.x, y: r.y });
  t.setAOIs([
    { id: "logo", label: "Brand logo", x: 0.0, y: 0.0, w: 0.25, h: 0.25 },
    { id: "cta", label: "Call to action", x: 0.7, y: 0.7, w: 0.3, h: 0.3 },
  ]);
  t.startRecording();
  for (let i = 0; i < 12; i++) { const p = { t: i * 0.05, x: 0.10, y: 0.10 }; t.samples.push(p); t._classify(p); } // logo
  for (let i = 0; i < 10; i++) { const p={t:0.6+i*0.05,x:0.85,y:0.85}; t.samples.push(p); t._classify(p); }
  const m = t.computeMetrics();
  const logo = m.aoi.find((a) => a.id === "logo");
  const cta = m.aoi.find((a) => a.id === "cta");
  check("logo got a fixation", logo.fixations, 1, 0.1);
  check("cta got a fixation", cta.fixations, 1, 0.1);
  console.log(`  INFO  logo first-fixation latency = ${logo.first_fixation_s}s (expected ~0)`);
  console.log(`  INFO  cta  first-fixation latency = ${cta.first_fixation_s}s (expected ~0.6)`);
}

// ------------------------------------------------------------ network mapping
console.log("\nNETWORK MAPPING (must never claim measurement)");
{
  const calm = {
    duration_s: 10, fixation_count: 4, saccade_count: 3, saccade_rate_hz: 0.3,
    mean_saccade_amplitude: 0.10, longest_fixation_s: 3.0,
    aoi: [{ label: "Brand logo", fixations: 3, share: 0.8 }],
  };
  const busy = {
    duration_s: 10, fixation_count: 22, saccade_count: 26, saccade_rate_hz: 2.6,
    mean_saccade_amplitude: 0.40, longest_fixation_s: 0.4,
    aoi: [{ label: "Product shot", fixations: 9, share: 0.6 }],
  };
  const a = mapToNetworks(calm, calm.aoi);
  const b = mapToNetworks(busy, busy.aoi);
  console.log(`  calm viewer engages: ${a.filter((x) => x.engaged).map((x) => x.id).join(", ") || "(none)"}`);
  console.log(`  busy viewer engages: ${b.filter((x) => x.engaged).map((x) => x.id).join(", ") || "(none)"}`);
  console.log(`  INFO  every row carries a citation: ${a.every((x) => x.cite && x.refs.length > 0)}`);
  // Guard the project's core promise: no citation may read as a claim that we
  // measured this viewer's brain. Words like "measurable"/"activation" are fine in a
  // research context ("activation scales with saccade rate in fMRI") but must never
  // be phrased as something *we* observed.
  const forbidden = /\b(your|the) (brain|neural|cortex)|we (measured|detected) |blood oxygen|BOLD|is (active|engaged) in you/i;
  const offenders = a.filter((x) => forbidden.test(x.cite));
  console.log(`  ${offenders.length === 0 ? "PASS" : "FAIL"}  no citation claims we measured the viewer's brain`);
  if (offenders.length === 0) pass++; else { fail++; offenders.forEach((o) => console.log(`        -> ${o.id}: ${o.cite}`)); }
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);