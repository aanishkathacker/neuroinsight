"""Directly-measured image features.

Everything in this module is a MEASUREMENT in the provenance sense: a number computed
deterministically from the pixel data. No model, no training, no opinion. These are
the facts about the image that the interpretation layer is later allowed to reason over.

Analyses performed
------------------
  Colour        mean/median hue, saturation, value; dominant palette; colour harmony;
                contrast against background (WCAG-style relative luminance ratio)
  Luminance     distribution, dynamic range, RMS contrast, edge density
  Hierarchy     estimated reading order by saliency-weighted position and size
  Faces         OpenCV Haar cascade detection (frontal + profile)
  Text          macOS Vision OCR with per-box confidence and bounding geometry
  Complexity    colour count, edge density, entropy, occupied-area ratio
"""

from __future__ import annotations

import colorsys
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------- cascade handling

# opencv-contrib-python-headless 5.x ships NO haar cascade XML files (verified: its
# cv2/data directory is empty). Rather than depend on a system OpenCV or silently
# report zero faces, we vendor the two cascade files we need into the repository at
# data/cascades/. They are ~1.7 MB of static XML from the OpenCV project.
_CASCADE_DIR = Path(__file__).resolve().parents[3] / "data" / "cascades"
_CASCADES: dict[str, cv2.CascadeClassifier] = {}
_CASCADE_ERROR: str | None = None


def _load_cascade(name: str) -> cv2.CascadeClassifier | None:
    if name in _CASCADES:
        return _CASCADES[name]
    path = _CASCADE_DIR / f"haarcascade_{name}.xml"
    global _CASCADE_ERROR
    if not path.exists():
        _CASCADE_ERROR = (
            f"Haar cascade missing at {path}. Fetch with: curl -fsSL -o '{path}' "
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
            f"haarcascade_{name}.xml"
        )
        return None
    clf = cv2.CascadeClassifier(str(path))
    if clf.empty():
        _CASCADE_ERROR = f"Haar cascade at {path} failed to parse."
        return None
    _CASCADES[name] = clf
    return clf


# --------------------------------------------------------------------------- colour


def colour_features(bgr: np.ndarray) -> dict:
    """Colour statistics in HSV, plus an OKLab-style palette extraction."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

    mean_h = float(np.mean(h)) if h.size else 0.0
    mean_s = float(np.mean(s)) if s.size else 0.0
    mean_v = float(np.mean(v)) if v.size else 0.0

    # Circular mean for hue: a naive mean of 350 and 10 must not give 180.
    ang = h.astype(np.float64) * (2 * math.pi / 180.0)
    mean_h = float((math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360)

    # Dominant palette by hue binning (ignore near-grey pixels, they carry no hue info)
    mask = s > 40
    palette = []
    if mask.sum() > 50:
        bins = h[mask] // 15
        counts = Counter(bins.tolist())
        total = int(mask.sum())
        for bin_idx, n in counts.most_common(5):
            hue = float(bin_idx * 15 + 7.5)
            frac = n / total
            if frac < 0.02:
                continue
            rgb = colorsys.hsv_to_rgb(hue / 360.0, min(1.0, mean_s / 255.0), min(1.0, mean_v / 255.0))
            palette.append(
                {
                    "hue_deg": round(hue, 1),
                    "share": round(frac, 4),
                    "rgb": [int(c * 255) for c in rgb],
                    "hex": "#%02X%02X%02X" % tuple(int(c * 255) for c in rgb),
                }
            )

    return {
        "mean_hue_deg": round(mean_h, 1),
        "mean_saturation": round(mean_s, 1),
        "mean_value": round(mean_v, 1),
        "saturation_pct": round(100 * float(np.mean(s > 40)), 1),
        "palette": palette,
        "distinct_colour_bins": int(len(Counter((h // 15)[mask].tolist()).keys())) if mask.sum() > 50 else 0,
    }


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def relative_luminance(bgr: np.ndarray) -> np.ndarray:
    """WCAG 2.1 relative luminance per pixel (0=black, 1=white)."""
    b, g, r = (_srgb_to_linear(bgr[..., i].astype(np.float64)) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(bgr: np.ndarray) -> dict:
    """WCAG contrast ratio between the brightest and darkest regions.

    This is a real, standardised accessibility metric (WCAG 2.1 §1.4.3), which means
    we can report it against published thresholds rather than inventing one.
    """
    lum = relative_luminance(bgr)
    lmax, lmin = float(lum.max()), float(lum.min())
    ratio = (lmax + 0.05) / (lmin + 0.05)

    def band(r: float) -> str:
        if r >= 7:
            return "AAA (>=7:1) — passes WCAG for normal text"
        if r >= 4.5:
            return "AA (>=4.5:1) — passes WCAG for normal text"
        if r >= 3:
            return "AA Large (>=3:1) — passes only for large text"
        return "Fail (<3:1) — below WCAG minimum"

    return {
        "ratio": round(ratio, 2),
        "max_luminance": round(lmax, 4),
        "min_luminance": round(lmin, 4),
        "dynamic_range": round(lmax - lmin, 4),
        "wcag_band": band(ratio),
        "standard": "WCAG 2.1 relative luminance contrast ratio",
    }


# ------------------------------------------------------------------------ luminance


def luminance_features(bgr: np.ndarray) -> dict:
    """Luminance distribution and structural edge statistics."""
    lum = relative_luminance(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    rms = float(np.sqrt(np.mean((gray.astype(np.float64) / 255.0 - 0.5) ** 2)))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() or 1.0)
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    return {
        "mean_luminance": round(float(lum.mean()), 4),
        "std_luminance": round(float(lum.std()), 4),
        "rms_contrast": round(rms, 4),
        "edge_density": round(edge_density, 5),
        "luminance_entropy_bits": round(entropy, 3),
        "dynamic_range_pct": round(
            100 * float((lum > 0.95).sum() + (lum < 0.05).sum()) / lum.size, 2
        ),
    }


def composition_features(bgr: np.ndarray) -> dict:
    """Spatial composition: where visual weight sits, and how much space is empty.

    Everything here is a direct computation on the pixels, so these are
    MEASUREMENT-grade facts about the image - not claims about how anyone will
    react to it. That distinction is the whole point of reporting them separately.

    Visual weight is approximated as local edge energy (a proxy for how much
    "stuff" is in a region), because amount of detail is what makes one area
    perceptually heavier than another. It is an approximation of a perceptual
    quantity, not a measurement of it, and the finding text says so.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    area = float(h * w)

    # --- local detail energy, heavily blurred so it measures "mass", not texture
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    energy = cv2.GaussianBlur(detail, (0, 0), sigmaX=max(h, w) / 12.0)

    # --- thirds: the two vertical and two horizontal rule-of-thirds lines
    third_x = [w / 3.0, 2 * w / 3.0]
    third_y = [h / 3.0, 2 * h / 3.0]
    tol_x, tol_y = w * 0.10, h * 0.10
    third_weight = float(
        sum(
            energy[:, int(x - tol_x) : int(x + tol_x)].sum()
            for x in third_x
        )
        + sum(
            energy[int(y - tol_y) : int(y + tol_y), :].sum()
            for y in third_y
        )
    )
    # Each of the 4 bands is 20% of the width/height, so the maximum possible
    # share is ~40%. Report it as a fraction of the achievable maximum.
    band_area = 2 * (2 * tol_x * h) + 2 * (2 * tol_y * w)
    third_share = third_weight / (energy.sum() or 1.0)
    third_ratio = third_share / ((band_area / area) * 2 or 1.0) if band_area else 0.0

    # --- visual centre of gravity of detail
    ys, xs = np.mgrid[0:h, 0:w]
    total_e = float(energy.sum())
    cog_x = float((energy * xs).sum() / total_e) / w if total_e > 0 else 0.5
    cog_y = float((energy * ys).sum() / total_e) / h if total_e > 0 else 0.5
    # Distance from frame centre, normalised so 1.0 == fully in a corner.
    off_centre = float(np.hypot(cog_x - 0.5, cog_y - 0.5) * 2)

    # --- empty space: how much of the frame carries almost no detail
    low = energy <= max(float(np.percentile(energy, 25)), 1e-6)
    whitespace = float(low.sum()) / area

    # --- quadrant balance: is detail spread evenly, or piled into one corner?
    q = {
        "tl": float(energy[: h // 2, : w // 2].sum()),
        "tr": float(energy[: h // 2, w // 2 :].sum()),
        "bl": float(energy[h // 2 :, : w // 2].sum()),
        "br": float(energy[h // 2 :, w // 2 :].sum()),
    }
    qtot = sum(q.values()) or 1.0
    qshare = {k: v / qtot for k, v in q.items()}
    # 1.0 = perfectly even across 4 quadrants; 0 = all of it in one.
    quadrant_evenness = float(sum(qshare.values()) ** 2 / 4)

    return {
        "visual_weight_centre_x": round(cog_x, 4),
        "visual_weight_centre_y": round(cog_y, 4),
        "visual_weight_offcentre": round(off_centre, 4),
        "thirds_concentration": round(float(min(1.0, max(0.0, third_ratio))), 4),
        "whitespace_pct": round(100 * whitespace, 2),
        "quadrant_evenness": round(quadrant_evenness, 4),
        "heaviest_quadrant": max(qshare, key=qshare.get),
        "edge_density": round(float(np.count_nonzero(cv2.Canny(gray.astype(np.uint8), 100, 200))) / area, 5),
    }


def focal_alignment(saliency: np.ndarray) -> dict:
    """How well does predicted attention sit on the rule-of-thirds intersections?

    Takes the saliency map produced by `saliency.py` (0-1, float). This measures
    the geometry of the prediction only - it says nothing about whether the
    prediction is correct, which is what the MIT1003 benchmark is for.
    """
    s = np.asarray(saliency, dtype=np.float32)
    if s.ndim != 2 or s.max() <= 0:
        return {"available": False, "reason": "empty saliency map"}
    h, w = s.shape[:2]
    total = float(s.sum()) or 1.0

    peak = np.unravel_index(int(np.argmax(s)), s.shape)
    peak_x, peak_y = peak[1] / w, peak[0] / h

    # Mass within 10% of each of the four thirds intersections.
    r = 0.10
    score = 0.0
    for tx in (1 / 3, 2 / 3):
        for ty in (1 / 3, 2 / 3):
            xs, xe = int(max(0, (tx - r) * w)), int(min(w, (tx + r) * w))
            ys, ye = int(max(0, (ty - r) * h)), int(min(h, (ty + r) * h))
            if xe > xs and ye > ys:
                score += float(s[ys:ye, xs:xe].sum())
    on_thirds = score / total
    # 4 windows of (2r x 2r) = (0.2*0.2)*4 = 16% of the area is the max possible.
    thirds_ratio = on_thirds / 0.16

    # Centroid of predicted attention - the average "place" people are drawn to.
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((s * xs).sum() / total) / w
    cy = float((s * ys).sum() / total) / h

    return {
        "available": True,
        "peak_x": round(peak_x, 4),
        "peak_y": round(peak_y, 4),
        "attention_centroid_x": round(cx, 4),
        "attention_centroid_y": round(cy, 4),
        "on_thirds_share": round(100 * min(1.0, on_thirds), 2),
        "thirds_alignment_ratio": round(min(1.0, thirds_ratio), 4),
        "peak_is_centred": bool(abs(peak_x - 0.5) < 0.12 and abs(peak_y - 0.5) < 0.12),
    }


def text_hierarchy(blocks: dict) -> dict:
    """Is there a clear typographic hierarchy, or is everything shouting?

    Works from the OCR blocks only. With few or no detected blocks it reports
    `available: False` rather than inventing a hierarchy - text detection on
    stylised display type is unreliable, and a confident number here would be
    exactly the kind of fabricated precision this project refuses.
    """
    items = (blocks or {}).get("blocks") or []
    if len(items) < 2:
        return {"available": False, "reason": "fewer than 2 text blocks detected",
                "block_count": len(items)}

    # OCR blocks expose their extent as "h" (with "height_px" accepted as an alias,
    # since the two OCR backends have historically differed in field naming).
    def _h(b):
        v = b.get("h")
        if v is None:
            v = b.get("height_px")
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    heights = sorted((_h(b) for b in items), reverse=True)
    if not heights or heights[0] <= 0:
        return {"available": False, "reason": "no usable block heights",
                "block_count": len(items)}

    largest = heights[0]
    ratios = [h / largest for h in heights if h > 0]
    # Size contrast between the biggest and second-biggest block.
    step = (largest - heights[1]) / largest if len(heights) > 1 else 0.0
    mean_ratio = sum(ratios) / len(ratios)

    if step >= 0.35:
        verdict = "clear"
    elif step >= 0.18:
        verdict = "moderate"
    else:
        verdict = "flat"

    return {
        "available": True,
        "block_count": len(items),
        "largest_height_px": round(largest, 1),
        "top_step_ratio": round(step, 3),
        "mean_size_ratio": round(mean_ratio, 3),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- faces


_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def detect_faces(bgr: np.ndarray) -> dict:
    """Detect faces with OpenCV's Viola-Jones Haar cascades.

    Honest framing: Haar cascades are fast and dependency-free but dated. They are
    biased toward frontal, well-lit, roughly frontal-pose faces and will under-detect
    profile faces, small faces, and faces at unusual angles. This is a screening tool,
    not a face-recognition system, and we do not identify anyone.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape[:2]
    faces = []
    scale = max(1.0, (300.0 / max(1, min(h, w))))  # upscale small images; helps detection

    for name in ("frontalface_default", "profileface"):
        clf = _load_cascade(name)
        if clf is None:
            continue
        for (x, y, fw, fh) in clf.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        ):
            x, y, fw, fh = int(x), int(y), int(fw), int(fh)
            if x < 0 or y < 0 or x + fw > w or y + fh > h:
                continue
            faces.append(
                {
                    "x": x, "y": y, "w": fw, "h": fh,
                    "area_pct": round(100 * (fw * fh) / (w * h), 2),
                    "orientation": "frontal" if "frontal" in name else "profile",
                }
            )

    # Deduplicate frontal/profile overlap on the same region.
    dedup: list[dict] = []
    for f in sorted(faces, key=lambda d: -d["area_pct"]):
        if any(
            abs(f["x"] - k["x"]) < 0.4 * f["w"] and abs(f["y"] - k["y"]) < 0.4 * f["h"]
            for k in dedup
        ):
            continue
        dedup.append(f)

    return {
        "count": len(dedup),
        "faces": dedup,
        "largest_area_pct": round(max((f["area_pct"] for f in dedup), default=0.0), 2),
        "total_area_pct": round(sum(f["area_pct"] for f in dedup), 2),
        "model": "OpenCV Viola-Jones Haar cascade",
        "licence": "Apache-2.0 (OpenCV); original algorithm by Viola & Jones (2001)",
        "availability_error": _CASCADE_ERROR,
        "limitations": [
            "Biased toward frontal, well-lit faces; under-detects profile and angled faces.",
            "Fixed-scale training; accuracy drops on small or low-resolution faces.",
            "Produces bounding boxes only — no identity, no expression, no demographics.",
            "Higher false-positive rate on textured backgrounds.",
        ],
    }


# ----------------------------------------------------------------------------- text


def extract_text(bgr: np.ndarray) -> dict:
    """OCR with automatic engine selection.

    Engine priority:
      1. **Apple Vision** (macOS only) - native, fast, no install needed.
      2. **Tesseract** via pytesseract (any OS) - requires the user to install the
         `tesseract` binary (macOS: `brew install tesseract`, Debian/Ubuntu:
         `sudo apt install tesseract-ocr`).
      3. Neither - return an explicit `unavailable` status. We never return an empty
         result as though the image genuinely contained no text.
    """
    vision_error = tess_error = None

    if _vision_available():
        try:
            return _ocr_vision(bgr)
        except Exception as exc:  # pragma: no cover - platform dependent
            vision_error = f"{type(exc).__name__}: {exc}"

    if _tesseract_available():
        try:
            result = _ocr_tesseract(bgr)
            result["fallback_note"] = (
                "Apple Vision unavailable on this platform; used Tesseract instead. "
                "Tesseract is typically weaker on stylised display type."
            )
            return result
        except Exception as exc:  # pragma: no cover
            tess_error = f"{type(exc).__name__}: {exc}"

    reasons = []
    if vision_error:
        reasons.append(f"macOS Vision failed ({vision_error})")
    if tess_error:
        reasons.append(f"Tesseract failed ({tess_error})")

    return {
        "available": False,
        "status": "unavailable",
        "reason": "; ".join(reasons) or "no OCR engine available on this platform",
        "install_hint": (
            "macOS: built in. Linux: `sudo apt install tesseract-ocr` then "
            "`uv pip install pytesseract`. Windows: install Tesseract, add to PATH."
        ),
        "blocks": [],
        "block_count": 0,
        "total_words": 0,
        "mean_confidence": 0.0,
    }


def _vision_available() -> bool:
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except ImportError:
        return False
    return True


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_tesseract(bgr: np.ndarray) -> dict:
    """Tesseract fallback for platforms without Apple Vision."""
    import pytesseract
    from pytesseract import Output

    h, w = bgr.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.0
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]

    data = pytesseract.image_to_data(
        bgr, output_type=Output.DICT, lang="eng", config="--psm 11"
    )

    blocks = []
    for i, text in enumerate(data["text"]):
        if not text or not str(text).strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        bw, bh = int(data["width"][i]), int(data["height"][i])
        blocks.append({
            "text": str(text).strip(),
            "confidence": round(conf / 100.0, 4),
            "x": round(x, 1), "y": round(y, 1),
            "w": round(bw, 1), "h": round(bh, 1),
            "area_pct": round(100 * (bw * bh) / (w * h), 3),
        })

    blocks.sort(key=lambda b: -b["area_pct"])
    return {
        "available": True,
        "status": "ok",
        "engine": "Tesseract (via pytesseract)",
        "licence": "Apache-2.0 (Tesseract)",
        "block_count": len(blocks),
        "total_words": sum(len(b["text"].split()) for b in blocks),
        "mean_confidence": round(
            float(np.mean([b["confidence"] for b in blocks])) if blocks else 0.0, 4
        ),
        "blocks": blocks[:40],
        "limitations": [
            "Confidence reflects OCR legibility, not persuasive intent.",
            "Tesseract is weaker than Apple Vision on stylised display type.",
            "Small or low-contrast text is frequently missed.",
        ],
    }


def _ocr_vision(bgr: np.ndarray) -> dict:
    """Apple Vision OCR - the preferred engine on macOS."""
    import Quartz
    import Vision

    h, w = bgr.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.0
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)

    try:
        ns = rgba.tobytes()
        cfdata = Quartz.CFDataCreate(None, ns, len(ns))
        provider = Quartz.CGDataProviderCreateWithCFData(cfdata)
        # CoreGraphics signature is:
        #   CGImageCreate(width, height, bitsPerComponent, bitsPerPixel, bytesPerRow,
        #                 colorSpace, bitmapInfo, provider, decode, shouldInterpolate, intent)
        # The provider is the 8th argument, not the first. RGBA (32 bpp) with
        # kCGImageAlphaNoneSkipLast is the combination CoreGraphics accepts reliably;
        # tightly-packed 24-bit RGB requires kCGImageAlphaNone instead and is fussier.
        cg = Quartz.CGImageCreate(
            w,                          # width
            h,                          # height
            8,                          # bits per component
            8 * rgba.shape[2],          # bits per pixel (RGBA = 32)
            rgba.strides[0],            # bytes per row
            Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaNoneSkipLast,
            provider,
            None,                       # decode array
            False,                      # shouldInterpolate
            Quartz.kCGRenderingIntentDefault,
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        return {
            "available": False,
            "status": "error",
            "reason": f"could not build CGImage: {type(exc).__name__}: {exc}",
            "blocks": [],
        }

    if cg is None:
        return {
            "available": False,
            "status": "error",
            "reason": (
                "CoreGraphics refused the image buffer "
                f"(w={w}, h={h}, stride={rgba.strides[0]})."
            ),
            "blocks": [],
        }

    if cg is None:
        return {"available": False, "status": "error", "reason": "CGImage creation failed", "blocks": []}

    req = Vision.VNRecognizeTextRequest.alloc().init()
    # 0 = VNRequestTextRecognitionLevelAccurate, 1 = ...Fast.
    # Accurate mode costs ~1s more but gives usable confidence scores, which we need
    # in order to honestly distinguish legible copy from decorative noise.
    req.setRecognitionLevel_(0)
    req.setRecognitionLanguages_(["en-US"])
    req.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    results = req.results() or []

    blocks = []
    for obs in results:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        top = cand[0]
        bb = obs.boundingBox()
        blocks.append(
            {
                "text": str(top.string()),
                "confidence": round(float(top.confidence()), 4),
                # Vision's origin is bottom-left; convert to top-left pixels.
                "x": round(bb.origin.x * w, 1),
                "y": round((1 - bb.origin.y - bb.size.height) * h, 1),
                "w": round(bb.size.width * w, 1),
                "h": round(bb.size.height * h, 1),
                "area_pct": round(100 * bb.size.width * bb.size.height, 3),
            }
        )

    blocks.sort(key=lambda b: -b["area_pct"])
    return {
        "available": True,
        "status": "ok",
        "engine": "Apple Vision VNRecognizeTextRequest",
        "licence": "macOS system framework",
        "block_count": len(blocks),
        "total_words": sum(len(b["text"].split()) for b in blocks),
        "mean_confidence": round(
            float(np.mean([b["confidence"] for b in blocks])) if blocks else 0.0, 4
        ),
        "blocks": blocks[:40],
        "limitations": [
            "Confidence reflects OCR legibility, not persuasive intent.",
            "Small or low-contrast text is frequently missed.",
            "Stylised display type is harder than body copy.",
        ],
    }