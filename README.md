<div align="center">

# NeuroInsight by Aanishka

**Turning visual stimuli into neuroscience-informed insights**

An AI-assisted computational neuromarketing platform.
Upload an image · get a research-grounded report on its likely influence on
visual attention, perception and consumer interpretation.

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-Milestone%201%20prototype-orange)](PROJECT_PLAN.md)

</div>

---

## Screenshots

**Upload**
![Upload screen](docs/screenshots/01-upload.png)

**Results** — provenance badges, measured features, predicted-attention map, overlay,
live literature, pipeline timings, downloadable PDF.
![Results dashboard](docs/screenshots/02-results.png)

**Generated PDF report** (5 pages, from a real run)
![PDF report page 1](docs/screenshots/03-report-page1.png)

A complete sample report is committed at
[`docs/screenshots/sample-report.pdf`](docs/screenshots/sample-report.pdf).

> Screenshots are regenerated from live output by `scripts/screenshots.sh`. None are
> mocked up.

---

## The problem this project is about

Commercial "neuromarketing" tools routinely publish numbers like *engagement score*,
*emotional intensity* or *subliminal persuasion index* derived from a single image.
None of those numbers are recoverable from a picture. They require participant data —
eye-tracking, surveys, or brain recording — and a model trained and validated on that
specific outcome.

NeuroInsight is built around a simple refusal: **it will not produce those numbers.**
Instead it computes what genuinely *is* measurable from an image, separates that from
what a model merely predicts, and cites the neuroscience that connects the two.

## The core design decision

Every single finding in the report carries one of five provenance tags. This is enforced
in code — a finding cannot be created without one:

| Tag | Meaning | Example |
|:---|:---|:---|
| 🟢 **Measurement** | Computed directly from your pixels. Recomputable and exact. | "Contrast ratio 17.4:1" |
| 🟡 **Model output** | A prediction from a pretrained algorithm. An informed guess. | "Attention peaks in region R2C1" |
| 🔵 **Literature** | A mechanism established by published research. True in general, not about your image. | "Feature-integration theory predicts..." |
| 🩷 **Hypothesis** | Our reading. Plausible, unproven. | "Secondary elements may be missed at brief exposure" |
| ⚪ **Not determinable** | Cannot be established from a static image. We say so rather than guess. | "The viewer's emotional response" |

The last category is the point. Every report ends by explicitly declining to answer
what it cannot:

> **Cannot be determined: purchase intent or conversion probability**
> *This would require participant study with a predefined outcome measure and a trained
> model. We do not estimate it.*

Implemented in [`src/neuroinsight/provenance.py`](src/neuroinsight/provenance.py) — about
80 lines, and the file worth reading first.

## Deploy

The app is a stateless FastAPI service plus static files, so it deploys as a container.

### One-click (Render)

1. Sign in at <https://render.com>
2. **New → Blueprint**, select this repo, **Apply**

`render.yaml` provisions everything: it builds `./Dockerfile`, runs on `$PORT`, and polls
`/api/health`. You get a real HTTPS certificate — which matters, because browsers refuse
camera access on plain HTTP.

The free plan sleeps after 15 minutes idle and takes ~1–2 minutes to wake. Fine for
showing a link to someone; for anything that must be reliably up, change `plan: free` to
`plan: starter` in `render.yaml`.

Two things to know before scaling past one instance:

- **The result store is in-memory** (capped at 24 analyses). A permalink issued by
  instance A 404s if a later request lands on instance B. Needs a shared store before
  `numInstances: 2`.
- **OCR needs a system binary.** `pytesseract` is only a Python wrapper, and the
  preferred engine (Apple Vision) is macOS-only. The Dockerfile installs
  `tesseract-ocr`; a deployment that skips it reports OCR as *unavailable* and the text
  analysis comes back empty.

### Temporary public link (demos only)

Camera access requires a secure context, so a bare LAN address will not work. To hand
someone a link from your own machine:

```bash
./scripts/tunnel.sh start    # prints an https://*.trycloudflare.com URL
./scripts/tunnel.sh status
./scripts/tunnel.sh stop
```

Dies when you stop it, changes every restart, needs your laptop awake. Do not put it in
a portfolio — use the Render path above for that.

## Quick start

```bash
git clone https://github.com/aanishkathacker/neuroinsight.git
cd neuroinsight

# macOS / Linux
uv venv && uv pip install -e .        # or: python -m venv .venv && pip install -e .

./scripts/serve.sh start              # → http://127.0.0.1:8000
```

No upload of yours leaves your machine. There are no API keys, no accounts and no
external model calls except a text query to Crossref for paper titles.

**One-off image analysis from the terminal:**

```bash
uv run neuroinsight analyse path/to/ad.jpg --pdf report.pdf
```

### Platform notes

| Feature | macOS | Linux | Windows |
|:---|:---|:---|:---|
| Saliency, faces, colour, contrast, PDF | ✅ | ✅ | ✅ |
| OCR | ✅ Apple Vision (built in) | ⚠️ needs `tesseract-ocr` | ⚠️ needs Tesseract on PATH |
| Sample image | ✅ bundled | ✅ bundled | ✅ bundled |

If no OCR engine is available the report says so explicitly rather than pretending the
image contains no text.

## Architecture

```
browser ──POST /api/analyse──►  FastAPI (uvicorn)
                                    │
                    ┌───────────────┴───────────────┐
                    │        vision/                │
                    │  features.py   (measurements)│
                    │  saliency.py   (prediction)  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  provenance.py  (5-level tags) │
                    │  literature.py  (live Crossref)│
                    │  interpret.py   (findings)     │
                    └───────────────┬───────────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │  report/pdf.py  →  ReportLab PDF     │
                 │  web/static/    →  vanilla JS UI      │
                 └─────────────────────────────────────┘
```

* **Backend** — FastAPI, Python 3.13
* **Frontend** — vanilla JS, **no build step** (nothing to `npm install`, nothing to break)
* **Results store** — in-memory, capped at 24 analyses (local prototype only)
* **Dependencies** — numpy · scipy · OpenCV · ReportLab · Crossref API

The Gaze Lab adds MediaPipe FaceLandmarker (Apache-2.0) and three.js (MIT), both
**vendored into the repo** so nothing is fetched from a CDN at runtime. Provenance for
every vendored file — version, source, licence, purpose, I/O and limits — is in
[`web/static/vendor/PROVENANCE.md`](web/static/vendor/PROVENANCE.md).

## The Gaze Lab — `/gazelab`

The image pipeline above predicts *where most people would look*. The Gaze Lab measures
*where you actually looked*, using your webcam and a 9-point calibration.

```
consent ──► camera ──► 9-point calibration ──► view the ad ──► results
 (opt-in)   (explicit)   (leave-one-out error)   (12 s)         │
                                                                 ▼
                                        fixations · saccades · dwell time
                                        per AOI · 3D brain of *implicated*
                                        attentional networks
```

What comes out is genuinely measured — **your** fixations, **your** dwell times, and a
calibration error computed on **your** eyes and shown to you.

What does **not** come out is any brain activity. A webcam cannot detect neural signals.
The 3D brain shows which attentional networks the published literature associates with
the *kind* of looking your eye movements imply, each with its citation, labelled
**"implicated — not measured."** See [EVALUATION.md](EVALUATION.md) Part 2.

**All video processing is local.** There is no upload path in the code — no frame,
landmark or gaze coordinate reaches the server.

Run the tests:

```bash
python -m neuroinsight.verify_refs          # every DOI resolves to the paper we name
python -m neuroinsight.verify_refs --offline # structural check, no network
node web/static/test_gaze.js                 # 15 checks: solver, I-DT, AOI, mapping
open http://127.0.0.1:8000/static/test_landmarks.html   # 8 checks: landmark geometry
```

There is also a **demo session** (`View a demo session` on the first screen) that renders
the whole results view with a fixed, clearly-labelled synthetic recording, so the app can
be evaluated without a webcam. It is flagged `synthetic: true` in any exported JSON.

### Citation integrity

Every reference in this project — in the Python bibliography *and* in the citations the
Gaze Lab shows to users — is checked against Crossref by `neuroinsight.verify_refs`,
which resolves each DOI and compares the returned title to the one we claim.

This check earned its place immediately. It found that **six of eight** bibliography
entries had DOIs that resolved to nothing at all, one pointed at an entirely different
paper, and one was filed under a key (`heng2012`) that named a paper other than the one
it described. Five entries appear to have been fabricated. All twelve references now
resolve to the paper they cite, and the verifier's detection power was confirmed by
planting the original bad DOI and watching it fail.

Every DOI was obtained by searching Crossref and reading the result — including while
writing the verifier itself, where two DOIs I produced from memory (`10.1038/nrb755`,
`10.1523/JNEUROSCI.17-18-09662`) were both wrong and had to be looked up properly.

### Measured pipeline timing

Per 1200×800 image on an Apple M4. `literature` is the live Crossref lookup that
backs every citation; it is memoised per process, so only the first analysis in a
session pays for it.

```
cold   ocr 0.82s · literature 5.4s · everything else 0.5s  → total 6.8s
warm   ocr 0.82s · literature 0.00s · everything else 0.5s → total 1.4s
```

Before caching, literature was 8.5 s of a 10.5 s pipeline — 79% of the runtime spent
waiting on the network for references that never change.

## Models used

| Role | Model | Source | Licence |
|:---|:---|:---|:---|
| Saliency | Spectral Residual + Fine-Grained | Hou & Zhang, CVPR 2007 (OpenCV contrib) | Apache-2.0 |
| Face detection | Viola-Jones Haar cascades | Viola & Jones, 2001 (OpenCV) | Apache-2.0 |
| OCR | Apple Vision / Tesseract | Apple; Tesseract project | Apple / Apache-2.0 |
| Face mesh + iris | MediaPipe FaceLandmarker | Google, tasks-vision 0.10.21 | Apache-2.0 |
| 3D rendering | three.js | r160 | MIT |
| Literature | Crossref REST API | crossref.org | free API |

All are **pretrained or published algorithms** — nothing here was trained from scratch,
and nothing is claimed to be validated beyond its original publication.

See [`MODEL_CARD.md`](MODEL_CARD.md) for limitations of each, and
[`web/static/vendor/PROVENANCE.md`](web/static/vendor/PROVENANCE.md) for the vendored
browser assets.

## What this tool cannot do

**About the image analysis (the `/` half):**

- ❌ Measure emotion, purchase intent or subconscious response
- ❌ Measure anyone's eye movements — the image path has no gaze data at all
- ❌ Tell you whether your design will sell

**About the Gaze Lab (the `/gazelab` half):**

- ❌ Estimate brain activity, by any means, ever
- ❌ Be clinically validated, or a substitute for a research eye-tracker
- ❌ Claim your dorsolateral prefrontal cortex is lit up because you looked at a logo
- ❌ Survive with no webcam — use the demo session instead

The image path analyses **pictures**. The Gaze Lab analyses **one person's recorded eye
movements**. Neither one analyses a brain, and neither is a clinical instrument.

## Roadmap

- [x] **M1** — End-to-end prototype, 10-stage pipeline, PDF report
- [ ] **M2** — Visual hierarchy, object detection, AOI mapping, real sample gallery
- [x] **M3a** — **Statistical evaluation on MIT1003** (1,003 images): AUC 0.532, CC 0.208, **NSS 2.243**
- [ ] **M3b** — Pretrained deep saliency model, benchmarked against this baseline
- [x] **M3c** — **Gaze Lab**: webcam eye tracking, 9-point calibration, AOI metrics, 3D brain
- [x] **M5** — **Citation audit**: all 22 references DOI-verified against Crossref (`verify_refs`)
- [ ] **M3d** — Compare recorded gaze against predicted saliency *(the first real bridge between the two halves)*
- [ ] **M4** — Java validation component, Rust coordinate utility, public release, poster

## Project documentation

| File | Purpose |
|:---|:---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Milestones and progress |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design |
| [`RESEARCH_FOUNDATIONS.md`](RESEARCH_FOUNDATIONS.md) | Neuroscience evidence base |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Models, licences, limitations |
| [`EVALUATION.md`](EVALUATION.md) | Evaluation protocol and **measured results** (MIT1003, n=1003; Gaze Lab Part 2) |
| [`LEARNING_LOG.md`](LEARNING_LOG.md) | Concepts learned while building |
| [`GLOSSARY.md`](GLOSSARY.md) | Terms explained |

## Author

**Aanishka Thacker** — Biology undergraduate, Ashoka University.
Built this as a learning project to explore the intersection of computational biology,
visual attention research and responsible ML practice.

## Licence

MIT — see [`LICENSE`](LICENSE).
Third-party models and data retain their own licences; see [`MODEL_CARD.md`](MODEL_CARD.md).

> **Disclaimer.** Research prototype. Not clinically validated, not a commercial
> neuromarketing system, and not a substitute for fMRI, laboratory eye-tracking or
> controlled field experiments. Predicted attention should always be treated as a
> hypothesis to be tested. The Gaze Lab's webcam eye tracking is far less precise than a
> research eye-tracker, and the 3D brain shows which attentional networks are
> *implicated* by your gaze pattern — it is not a brain scan.