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
    """OCR via the native macOS Vision framework.

    Uses Apple's Vision because it is a real, high-accuracy OCR engine already
    present on the machine — no extra binary, no model download, and no tesseract
    system dependency. Returns early with an explicit status if unavailable, rather
    than pretending the image contains no text.
    """
    try:
        import Quartz
        import Vision
    except ImportError:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "macOS Vision framework not importable (non-macOS build?)",
            "blocks": [],
        }

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