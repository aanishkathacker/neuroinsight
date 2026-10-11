# Model card

Each component, what it predicts, and where it is known to fail.

## 1. Visual saliency — Spectral Residual + Fine-Grained

- **Source:** Hou, X. & Zhang, L. (2007) *Saliency Detection: A Spectral Residual
  Approach*, CVPR 2007. doi:10.1109/CVPR.2007.383289
- **Implementation:** OpenCV contrib `cv2.saliency`
- **Licence:** Apache-2.0 (OpenCV); the algorithm is a published academic method
- **Trained?** No learned weights required — it is a closed-form spectral method.
  Trained, fine-tuned or validated by us? **No**, and we never claim otherwise.
- **Input:** BGR uint8 image → **Output:** float32 saliency map in [0,1]
- **Cost:** ~60 ms per image, CPU only, no download
- **Limitations:**
  - Classical method; no learned representation of faces, text or semantics
  - Optimised for natural photographs, **not** designed advertisements
  - Strongly contrast-driven; can miss semantically salient low-contrast regions
  - Predicted attention, **not** measured gaze. Never present it as eye-tracking.
- **Known failure mode to watch:** flat brand-coloured backgrounds can produce
  spurious high saliency at element boundaries.

## 2. Face detection — Viola-Jones Haar cascades

- **Source:** Viola & Jones (2001); OpenCV implementation
- **Licence:** Apache-2.0 (OpenCV)
- **Note:** `opencv-contrib-python-headless` 5.x ships **no** cascade XML files, so the
  two we use are vendored into `data/cascades/`.
- **Input:** grayscale BGR → **Output:** bounding boxes
- **Cost:** ~150 ms
- **Limitations:**
  - Biased to frontal, well-lit faces; under-detects profile and angled faces
  - Fixed-scale training; degrades on small/low-res faces
  - Higher false positives on textured backgrounds
  - Bounding boxes only — no identity, no expression, no demographics. We do not
    identify anyone.

## 3. OCR — Apple Vision (primary) / Tesseract (fallback)

- **Source:** macOS Vision framework; Tesseract project (Apache-2.0)
- **Input:** BGR → RGB uint8 → **Output:** text blocks with confidence and geometry
- **Cost:** ~820 ms (Apple Vision, accurate mode)
- **Limitations:**
  - Confidence measures legibility, **not** persuasive intent
  - Stylised display type is harder than body copy
  - Tesseract is materially weaker than Vision on display typography
- **Platform:** macOS uses Vision automatically; elsewhere install Tesseract. If
  neither is available the report states `unavailable` — it never returns a silent
  empty result.

## 4. Literature retrieval — Crossref REST API

- **Source:** api.crossref.org, public API, no key required
- **Output:** real bibliographic records
- **Limitations:** relevance is not screened automatically; live results are marked
  as unscreened in the report. Curated primaries are the citable backbone.

## Explicitly NOT models

There is **no** emotion classifier, purchase-intent predictor or neural-decoding model
anywhere in this project. Building one would require labelled participant data and a
validated model; we have neither, so producing those outputs would be fabrication.
See `src/neuroinsight/provenance.py`.
