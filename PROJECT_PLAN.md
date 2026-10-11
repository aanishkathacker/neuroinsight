# Project plan

## Milestone 1 — End-to-end prototype ✅ COMPLETE

- [x] Vision layer: colour, WCAG contrast, luminance statistics
- [x] Saliency prediction (Hou & Zhang 2007, OpenCV contrib)
- [x] Face detection (Viola–Jones cascades, vendored)
- [x] OCR (Apple Vision primary, Tesseract fallback)
- [x] Five-level provenance system
- [x] Curated literature + live Crossref retrieval
- [x] Interpretation engine (10 findings on the test image)
- [x] FastAPI backend + no-build frontend
- [x] ReportLab PDF report with provenance badges
- [x] CLI (`serve`, `analyse`)
- [x] Cross-platform OCR fallback

**Verified:** 3.07 s per image, 10/10 stages green, 5-page PDF, error handling tested
(415 unsupported type, 404 unknown id, 413 oversize).

## Milestone 2 — Depth and breadth

- [ ] Visual hierarchy (detected ordering of salient elements)
- [ ] Object detection (generalised objects, not just faces)
- [ ] AOI (area-of-interest) mapping and coverage metrics
- [ ] Real sample gallery with attribution
- [ ] Unit tests for the vision layer

## Milestone 3 — Science and models

- [ ] Pretrained deep saliency model, benchmarked against the baseline
- [ ] R/RStudio evaluation of model performance
- [ ] Webcam-based eye-tracking extension (opt-in, consent-first)
- [ ] `EVALUATION.md` with real measured results

## Milestone 4 — Polish and release

- [ ] Java: report-schema validation component with tests
- [ ] Rust: coordinate-processing utility, benchmarked vs Python
- [ ] Public GitHub release + poster
- [ ] Demo script and presentation
