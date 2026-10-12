/* Gaze Lab - webcam-based eye tracking with 9-point calibration.
 *
 * WHAT THIS IS
 * ------------
 * A real, calibrated gaze-estimation pipeline running entirely in the browser. It
 * records where a specific person actually looked, and computes standard eye-tracking
 * measures from that recording.
 *
 * WHAT THIS IS NOT
 * -----------------
 * It is NOT an fMRI. It does not measure brain activity, and nothing it reports may be
 * read as a neural measurement. It estimates gaze *direction* from the position of the
 * iris relative to the eye corners, using the geometry of the face mesh.
 *
 * ACCURACY - READ THIS BEFORE QUOTING ANY NUMBER
 * ----------------------------------------------
 * Webcam gaze estimation is meaningfully less accurate than a research eye-tracker.
 * Uncalibrated, expect several degrees of error and unusable per-fixation location.
 * After 9-point calibration, typical accuracy is roughly 2-4 degrees of visual angle on
 * a laptop screen at arm's length - about 1.5-3% of screen width. That is:
 *
 *   GOOD ENOUGH for: which region of an ad was looked at, first-fixation order,
 *                     dwell time per region, gross saccade direction.
 *   NOT GOOD ENOUGH for: exact fixation coordinates, microsaccade analysis, or any
 *                     claim about sub-region detail smaller than ~2% of the image.
 *
 * We report the measured calibration error alongside every session so the user can
 * judge for themselves rather than trusting our claim.
 *
 * PRIVACY
 * -------
 * All video processing is local. No frame ever leaves the browser. Nothing is
 * uploaded, recorded, or sent anywhere.
 */
import { FaceLandmarker, FilesetResolver } from "./vendor/tasks-vision.mjs";

export const CAL_POINTS = [
  [0.05, 0.05], [0.50, 0.05], [0.95, 0.05],
  [0.05, 0.50], [0.50, 0.50], [0.95, 0.50],
  [0.05, 0.95], [0.50, 0.95], [0.95, 0.95],
];

/** Frames collected per calibration point. More samples -> more stable fit. */
const CAL_FRAMES = 12;

/* MediaPipe FaceMesh landmarks (478-point mesh, refined with iris).
 *
 * Landmarks are named from the SUBJECT's point of view, so the subject's right eye
 * appears on the LEFT of a non-mirrored image. Our webcam feed is mirrored for
 * display, but detect() runs on the raw frame, so these are image coordinates.
 *
 *   subject's right eye -> image LEFT  : corners 33 (outer) & 133 (inner), iris 468
 *   subject's left  eye -> image RIGHT : corners 362 (inner) & 263 (outer), iris 473
 *
 * A previous version used 133 & 362 as one eye's corners. Those are the two INNER
 * corners flanking the nose bridge, so it was measuring iris displacement from the
 * nose rather than from an eye centre - producing a signal that barely moved when the
 * eyes did. Verified against a real face: with the correct pairs the iris centroid
 * falls strictly inside each eye's corner span; with 133/362 it falls outside both.
 *
 * Only the two corner indices per eye matter, because the estimate below is taken
 * from their midpoint.
 */
const R_EYE_CORNERS = [133, 33], R_IRIS = 468;   // image-left eye
const L_EYE_CORNERS = [362, 263], L_IRIS = 473;  // image-right eye

export class GazeTracker {
  constructor(videoEl) {
    this.video = videoEl;
    this.landmarker = null;
    this.stream = null;
    this.running = false;

    // calibration
    this.calibrated = false;
    this.rawSamples = [];   // collected raw gaze at each known target
    this.calErrorPx = null;

    // recording
    this.recording = false;
    this.samples = [];      // {t, x, y}
    this.fixations = [];
    this.saccades = [];
    this._last = null;

    // AOI regions in normalised image coords
    this.aois = [];
  }

  async start() {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks("/static/vendor/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/static/vendor/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
    });
  }

  async openCamera() {
    // Explicit user-gesture permission request. Never auto-prompted on page load.
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    return { w: this.video.videoWidth, h: this.video.videoHeight };
  }

  closeCamera() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
  }

  /**
   * Raw gaze signal: where the irises sit relative to the eye centres.
   *
   * Each axis is divided by the SAME quantity, the eye's corner-to-corner width.
   * That makes the signal scale-invariant, so leaning towards or away from the
   * camera does not change it - only eye rotation does. Using one scale for both
   * axes (rather than width for x and height for y) keeps the measurement isotropic,
   * which is what a rigid 2D polynomial calibration fit expects.
   *
   * Returns null if either eye is unusable, so a single bad frame is dropped rather
   * than corrupting the calibration fit.
   */
  detect() {
    if (!this.landmarker || !this.video || this.video.readyState < 2) return null;
    const res = this.landmarker.detectForVideo(this.video, performance.now());
    const lm = res?.faceLandmarks?.[0];
    if (!lm) return null;

    const eye = ([ca, cb], irisIdx) => {
      const a = lm[ca], b = lm[cb], c = lm[irisIdx];
      if (!a || !b || !c) return null;
      const w = Math.hypot(b.x - a.x, b.y - a.y);
      if (!(w > 1e-6)) return null;            // degenerate / near-collinear corners
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      return { x: (c.x - mx) / w, y: (c.y - my) / w };
    };

    const r = eye(R_EYE_CORNERS, R_IRIS);
    const l = eye(L_EYE_CORNERS, L_IRIS);
    if (!l || !r) return null;

    // Average both eyes. Binocular averaging genuinely reduces noise: the two irises
    // are measured independently, so landmark jitter is partly independent and
    // cancels. Discarding an "outlier" eye instead would throw that away on no
    // principled basis - the landmarker returns both eyes consistently.
    const raw = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
    return { raw, mapped: this._map(raw), conf: res.faceLandmarks.length ? 1 : 0 };
  }

  /**
   * Calibration mapping: fit a 2nd-order polynomial from raw gaze -> screen coords.
   * Bilinear (2D 2nd-order) is the standard choice for webcam gaze calibration: it
   * captures the nonlinearity that a linear map cannot.
   *
   * Solved with plain Gaussian elimination + least squares in plain JS - no numeric
   * library, so this step has zero dependencies.
   */
  _fit(samples) {
    const build = (out) => {
      const A = samples.map((s) => {
        const { x, y } = s.raw;
        return [1, x, y, x * x, x * y, y * y];
      });
      const b = samples.map((s) => s.target[out]);
      return lstsq(A, b);
    };
    this.cx = build(0);
    this.cy = build(1);
  }

  _apply(raw) {
    const f = (c) =>
      c[0] + c[1] * raw.x + c[2] * raw.y + c[3] * raw.x ** 2 +
      c[4] * raw.x * raw.y + c[5] * raw.y ** 2;
    return [f(this.cx), f(this.cy)];
  }

  _map(raw) {
    if (!this.calibrated) return null;
    const [x, y] = this._apply(raw);
    return { x, y };
  }

  /** Begin collecting samples toward a calibration target (normalised image coords). */
  beginCalibrationTarget(target) {
    this._calTarget = target;
    this._calBuf = [];
  }

  /** Push the current frame into the active calibration target's buffer. */
  pushCalibrationSample(raw) {
    if (!this._calTarget) return;
    this._calBuf.push({ raw: { ...raw }, target: [...this._calTarget] });
    if (this._calBuf.length >= CAL_FRAMES) {
      this.rawSamples.push(...this._calBuf);
      this._calBuf = [];
      this._calTarget = null;
    }
  }

  calibrationProgress() {
    return Math.min(1, this.rawSamples.length / (CAL_POINTS.length * CAL_FRAMES));
  }

  /**
   * Fit the calibration and compute an HONEST error estimate.
   *
   * We use leave-one-out cross-validation rather than training error: training error
   * is measured on the very points we fitted and would flatter the model.
   */
  finishCalibration() {
    const need = CAL_POINTS.length * Math.floor(CAL_FRAMES * 0.6);
    if (this.rawSamples.length < need) return false;

    this._fit(this.rawSamples);

    const feats = (rows) =>
      rows.map((q) => { const { x, y } = q.raw; return [1, x, y, x * x, x * y, y * y]; });
    const ev = (c, r) =>
      c[0] + c[1] * r.x + c[2] * r.y + c[3] * r.x ** 2 +
      c[4] * r.x * r.y + c[5] * r.y ** 2;

    let errs = [];
    for (let i = 0; i < this.rawSamples.length; i++) {
      const rest = this.rawSamples.filter((_, j) => j !== i);
      const s = this.rawSamples[i];
      if (rest.length < 6) break;
      const rx = lstsq(feats(rest), rest.map((q) => q.target[0]));
      const ry = lstsq(feats(rest), rest.map((q) => q.target[1]));
      errs.push(Math.hypot(ev(rx, s.raw) - s.target[0], ev(ry, s.raw) - s.target[1]));
    }

    this.calErrorPx = errs.length
      ? +(errs.reduce((a, b) => a + b) / errs.length).toFixed(5)
      : null;
    this.calErrorPct = this.calErrorPx != null ? +(this.calErrorPx * 100).toFixed(2) : null;
    this.calibrated = true;
    return true;
  }

  resetCalibration() {
    this.rawSamples = [];
    this.calibrated = false;
    this.calErrorPx = null;
    this.calErrorPct = null;
    this._calTarget = null;
    this._calBuf = [];
  }

  // ------------------------------------------------------------------ recording

  startRecording() {
    this.samples = [];
    this.fixations = [];
    this.saccades = [];
    this._last = null;
    this.recording = true;
    this._t0 = performance.now();
  }

  stopRecording() {
    this.recording = false;
    return this.computeMetrics();
  }

  /** Dispatch one detection into whichever mode is active. */
  tick() {
    const d = this.detect();
    if (!d) return null;

    if (this._calTarget) {
      this.pushCalibrationSample(d.raw);
      return { mode: "calibration", detection: d };
    }
    if (!this.recording || !d.mapped) {
      return { mode: d.mapped ? "idle" : "calibration-pending", detection: d };
    }

    const t = (performance.now() - this._t0) / 1000;
    const p = { t, x: d.mapped.x, y: d.mapped.y };
    this.samples.push(p);
    this._classify(p);
    return { mode: "recording", detection: d, point: p };
  }

  /**
   * Dispersion-threshold identification (I-DT), the standard I-VT simplification.
   * A fixation continues while the gaze stays within a window; a jump beyond it starts
   * a new fixation and is recorded as a saccade.
   */
  _classify(p) {
    const DISP_THRESH = 0.035;   // ~3.5% of image width
    const MIN_DWELL = 0.20;      // s - shorter "fixations" are noise
    const cur = this.fixations[this.fixations.length - 1];

    if (!cur) {
      this.fixations.push({ start: p.t, end: p.t, n: 1,
                            sx: p.x, sy: p.y, x: p.x, y: p.y, peakDist: 0 });
      return;
    }

    const d = Math.hypot(p.x - cur.x, p.y - cur.y);
    if (d <= DISP_THRESH) {
      cur.n++;
      cur.end = p.t;
      cur.x += p.x; cur.y += p.y;
      cur.x /= 2; cur.y /= 2;
      cur.peakDist = Math.max(cur.peakDist, d);
    } else {
      this.fixations.push({ start: p.t, end: p.t, n: 1,
                            sx: p.x, sy: p.y, x: p.x, y: p.y, peakDist: 0 });
      this.saccades.push({ from: cur, to: p, t: p.t,
                           amplitude: Math.hypot(p.x - cur.x, p.y - cur.y) });
    }
  }

  setAOIs(aois) { this.aois = aois || []; }

  computeMetrics() {
    const f = this.fixations.filter((x) => x.end - x.start >= MIN_DWELL_F);
    const dur = (f.length && this.samples.length)
      ? this.samples[this.samples.length - 1].t - this.samples[0].t : 0;

    const aoiStats = (this.aois || []).map((a) => {
      const hits = f.filter((fx) =>
        fx.x >= a.x && fx.x <= a.x + a.w && fx.y >= a.y && fx.y <= a.y + a.h);
      const first = hits.length ? Math.min(...hits.map((h) => h.start)) : null;
      return {
        id: a.id, label: a.label,
        first_fixation_s: first === null ? null : +first.toFixed(3),
        fixations: hits.length,
        dwell_s: +hits.reduce((s, h) => s + (h.end - h.start), 0).toFixed(3),
        share: dur ? +(hits.reduce((s, h) => s + (h.end - h.start), 0) / dur).toFixed(4) : 0,
      };
    });

    // Saccades per second is the standard "scanpath rate" proxy for cognitive load.
    const scanpath_rate = dur ? +(this.saccades.length / dur).toFixed(3) : 0;

    // Where did they look first? Used to compare real gaze against model prediction.
    const firstFix = f.length ? f[0] : null;

    return {
      duration_s: +dur.toFixed(2),
      sample_count: this.samples.length,
      fixation_count: f.length,
      mean_fixation_ms: f.length && dur ? +(1000 * dur / f.length).toFixed(1) : 0,
      saccade_count: this.saccades.length,
      saccade_rate_hz: scanpath_rate,
      mean_saccade_amplitude: this.saccades.length
        ? +(this.saccades.reduce((s, x) => s + x.amplitude, 0) / this.saccades.length).toFixed(4)
        : 0,
      longest_fixation_s: f.length ? +Math.max(...f.map((x) => x.end - x.start)).toFixed(3) : 0,
      first_fixation: firstFix ? { x: +firstFix.x.toFixed(4), y: +firstFix.y.toFixed(4),
                                   latency_s: +firstFix.start.toFixed(3) } : null,
      aoi: aoiStats,
      calibration_error_px: this.calErrorPx,
      raw_fixations: f.map((x) => ({
        x: +x.x.toFixed(4), y: +x.y.toFixed(4),
        start: +x.start.toFixed(3), dur: +(x.end - x.start).toFixed(3),
      })),
      samples: this.samples.map((s) => [+s.x.toFixed(4), +s.y.toFixed(4), +s.t.toFixed(3)]),
    };
  }
}

const MIN_DWELL_F = 0.20;

/**
 * Solve A x = b in the least-squares sense via normal equations.
 * A is m x n (m >= n). Returns a length-n coefficient vector.
 *
 * Used instead of a numeric library so calibration has zero runtime dependencies.
 * Normal equations are numerically less stable than QR, but our system is only 6x6
 * with well-conditioned polynomial features, so this is comfortably adequate.
 */
function lstsq(A, b) {
  const m = A.length, n = A[0].length;
  const AtA = Array.from({ length: n }, () => new Array(n).fill(0));
  const Atb = new Array(n).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aij = A[i][j];
      if (aij === 0) continue;
      for (let k = 0; k < n; k++) AtA[j][k] += aij * A[i][k];
      Atb[j] += aij * b[i];
    }
  }
  // Tiny ridge guards against a singular system when calibration points are
  // poorly distributed (e.g. all clustered in one corner of the screen).
  for (let j = 0; j < n; j++) AtA[j][j] += 1e-8;
  return solveSquare(AtA, Atb);
}

/** Gaussian elimination with partial pivoting on an n x n system. */
function solveSquare(M, v) {
  const n = v.length;
  const A = M.map((r) => r.slice());
  const b = v.slice();

  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    }
    if (Math.abs(A[piv][col]) < 1e-14) continue;
    if (piv !== col) {
      [A[piv], A[col]] = [A[col], A[piv]];
      [b[piv], b[col]] = [b[col], b[piv]];
    }
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }

  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r][c] * x[c];
    x[r] = Math.abs(A[r][r]) < 1e-14 ? 0 : s / A[r][r];
  }
  return x;
}

/* ------------------------------------------------------------------ neural mapping
 *
 * The ONLY thing this table is allowed to claim. Each row maps an OBSERVABLE eye
 * movement pattern to a neural system documented as RECRUITED by that kind of
 * processing. It is not a measurement of the viewer's brain, and the UI must say so
 * wherever this is used.
 *
 * CITATION INTEGRITY
 * ------------------
 * Every `refs` entry below carries a DOI that resolves to the paper named beside it.
 * These are not decorative: `neuroinsight.verify_refs` resolves each DOI against
 * Crossref and fails if the title does not match, so this table and the Python
 * bibliography are held to the same standard.
 *
 * An earlier version of this table cited "Tian & McAlpine (2016)", "Saul Stemmler
 * (2011)", "Swellens et al. (2022)" and "Chun et al. (2008)". None could be verified;
 * some do not appear to exist. They were replaced with sources whose DOIs were
 * looked up and confirmed, not recalled.
 *
 * `keys` are the matching entries in src/neuroinsight/literature.py, so the two
 * halves cannot drift apart.
 */
export const NETWORK_MAP = [
  {
    id: "fef",
    region: "Frontal eye fields (Brodmann 8)",
    network: "Saccade initiation / executive control",
    trigger: (m) => m.saccade_rate_hz >= 1.2,
    strength: (m) => Math.min(1, m.saccade_rate_hz / 2.5),
    cite: "Frontal eye fields lie within the dorsal attention network and initiate volitional saccades",
    keys: ["corbetta2002", "posner1990"],
    refs: [
      "Corbetta & Shulman (2002), Nat Rev Neurosci — doi:10.1038/nrn755",
      "Posner & Petersen (1990), Annu Rev Neurosci — doi:10.1146/annurev.ne.13.030190.000325",
    ],
  },
  {
    id: "ppn",
    region: "Posterior parietal / dorsal attention network",
    network: "Spatial orienting & target selection",
    trigger: (m) => m.mean_saccade_amplitude >= 0.22,
    strength: (m) => Math.min(1, (m.mean_saccade_amplitude - 0.15) / 0.45),
    cite: "Posterior parietal cortex encodes target location and guides orienting; damage produces neglect or optic ataxia",
    keys: ["posner1990", "corbetta2002"],
    refs: [
      "Posner & Petersen (1990), Annu Rev Neurosci — doi:10.1146/annurev.ne.13.030190.000325",
      "Corbetta & Shulman (2002), Nat Rev Neurosci — doi:10.1038/nrn755",
    ],
  },
  {
    id: "fusiform",
    region: "Fusiform gyrus (face-selective)",
    network: "Face processing",
    trigger: (m, aois) => aois.some((a) => /face|logo|product/i.test(a.label) && a.fixations > 0),
    strength: (m, aois) => {
      const f = aois.filter((a) => /face|logo|product/i.test(a.label));
      return Math.min(1, f.reduce((s, a) => s + a.share, 0) / 0.5);
    },
    cite: "The fusiform face area responds preferentially to faces; eye regions reliably attract orienting",
    keys: ["kanwisher1997", "itier2007"],
    refs: [
      "Kanwisher, McDermott & Chun (1997), J Neurosci — doi:10.1523/jneurosci.17-11-04302.1997",
      "Itier, Villate & Ryan (2007), Neuropsychologia — doi:10.1016/j.neuropsychologia.2006.09.004",
    ],
  },
  {
    id: "van",
    region: "Ventral attention network (temporoparietal junction)",
    network: "Reorienting to salient unexpected events",
    trigger: (m) => m.fixation_count >= 6 && m.longest_fixation_s <= 0.9,
    strength: (m) => Math.min(1, m.fixation_count / 18),
    cite: "The ventral attention network reorients attention to salient but unexpected stimuli, and is largely right-lateralised",
    keys: ["corbetta2002", "vossel2013"],
    refs: [
      "Corbetta & Shulman (2002), Nat Rev Neurosci — doi:10.1038/nrn755",
      "Vossel, Geng & Fink (2013), Neuroscientist — doi:10.1177/1073858413494269",
    ],
  },
  {
    id: "dorsal_exec",
    region: "Dorsal frontoparietal control network",
    network: "Voluntary attention shifts & goal maintenance",
    trigger: (m) => m.duration_s >= 8 && m.fixation_count >= 3,
    strength: (m) => Math.min(1, m.fixation_count / 12),
    cite: "Goal-directed, top-down attentional control during sustained tasks",
    keys: ["cole2014", "corbetta2002"],
    refs: [
      "Cole, Repovs, Anticevic & Braver (2014), Neuroscientist — doi:10.1177/1073858414525995",
      "Corbetta & Shulman (2002), Nat Rev Neurosci — doi:10.1038/nrn755",
    ],
  },
];

export function mapToNetworks(metrics, aois = []) {
  return NETWORK_MAP.map((n) => {
    const on = !!n.trigger(metrics, aois);
    return {
      id: n.id, region: n.region, network: n.network, cite: n.cite, refs: n.refs,
      engaged: on,
      strength: on ? Math.round(n.strength(metrics, aois) * 100) / 100 : 0,
    };
  });
}