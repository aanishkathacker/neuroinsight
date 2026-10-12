"""Analysis pipeline: bytes in, structured report out.

One call, `analyse(bytes)`, runs the full chain. Each stage is independent and the
pipeline records wall-clock timings and any partial failures rather than aborting, so
a single unavailable model degrades one section instead of the whole report.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

import cv2
import numpy as np

from . import interpret
from .literature import LITERATURE, retrieve_live
from .vision import features, saliency

MAX_BYTES = 12 * 1024 * 1024  # 12 MB
MAX_PIXELS = 40_000_000        # guards against decompression bombs


@dataclass
class Stage:
    name: str
    seconds: float
    status: str = "ok"
    error: str | None = None


@dataclass
class Analysis:
    image_id: str
    width: int
    height: int
    findings: list[dict] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    saliency_stats: dict = field(default_factory=dict)
    saliency_model: dict = field(default_factory=dict)
    palette: list = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    literature: list[dict] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "dimensions": {"width": self.width, "height": self.height},
            "created": self.created,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "literature": self.literature,
            "palette": self.palette,
            "features": self.features,
            "saliency": {
                "stats": self.saliency_stats,
                "model": self.saliency_model,
            },
            "stages": [
                {"name": s.name, "seconds": round(s.seconds, 3),
                 "status": s.status, "error": s.error}
                for s in self.stages
            ],
        }


def decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Could not decode this file as an image. Supported formats: JPEG, PNG, "
            "WebP, BMP, TIFF."
        )
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError(
            f"Image is {w}x{h} pixels ({h*w/1e6:.1f} MP), above the "
            f"{MAX_PIXELS/1e6:.0f} MP limit. Please downscale it."
        )
    return img


def analyse(data: bytes) -> tuple[Analysis, np.ndarray, np.ndarray]:
    """Run the full analysis.

    Returns (analysis, original_bgr, saliency_map) so callers can render visuals.
    """
    t_all = time.perf_counter()
    image_id = hashlib.sha256(data).hexdigest()[:16]
    img = decode(data)
    h, w = img.shape[:2]

    a = Analysis(
        image_id=image_id, width=w, height=h,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    # ------------------------------------------------------------- measurements
    def stage(name, fn, default):
        t = time.perf_counter()
        try:
            val = fn()
            a.stages.append(Stage(name, time.perf_counter() - t))
            return val
        except Exception as exc:
            a.stages.append(
                Stage(name, time.perf_counter() - t, status="failed",
                      error=f"{type(exc).__name__}: {exc}")
            )
            return default

    col = stage("colour", lambda: features.colour_features(img), {})
    con = stage("contrast", lambda: features.contrast_ratio(img), {})
    lum = stage("luminance", lambda: features.luminance_features(img), {})
    comp = stage("composition", lambda: features.composition_features(img), {})
    faces = stage("faces", lambda: features.detect_faces(img),
                  {"count": 0, "faces": [], "total_area_pct": 0.0,
                   "largest_area_pct": 0.0})
    text = stage("ocr", lambda: features.extract_text(img),
                 {"status": "failed", "block_count": 0, "blocks": [],
                  "total_words": 0, "mean_confidence": 0.0})

    a.features = {
        "colour": col, "contrast": con, "luminance": lum,
        "composition": comp, "faces": faces, "text": text,
    }
    a.palette = col.get("palette", [])

    # Text hierarchy depends on OCR, so it is derived rather than measured directly.
    th = stage("text_hierarchy", lambda: features.text_hierarchy(text),
               {"available": False, "reason": "ocr stage failed", "block_count": 0})
    a.features["text_hierarchy"] = th

    # ------------------------------------------------------------------ saliency
    def _sal():
        s, meta = saliency.compute_saliency(img)
        return s, meta

    sal, meta = stage("saliency", _sal, (np.zeros((h, w), np.float32), {}))

    focal = stage("focal_alignment", lambda: features.focal_alignment(sal),
                  {"available": False, "reason": "saliency stage failed"})
    a.features["focal_alignment"] = focal

    def _stats():
        return saliency.saliency_statistics(sal, img)

    a.saliency_stats = stage("saliency_statistics", _stats, {"regions": [], "cells": 0})
    a.saliency_model = meta

    # ------------------------------------------------------------ interpretation
    def _interp():
        return interpret.interpret(a.features, meta, a.saliency_stats)

    a.findings = stage("interpretation", _interp, [])

    def _recs():
        return interpret.recommendations(a.features, a.saliency_stats)

    a.recommendations = stage("recommendations", _recs, [])

    # ---------------------------------------------------------------- references
    def _lit():
        live = retrieve_live("visual saliency attention advertisement eye tracking", limit=5)
        curated = [
            {
                "title": r.citation, "authors": r.citation.split("(")[0].strip("."),
                "year": r.citation.split("(")[1][:4] if "(" in r.citation else "",
                "doi": r.doi, "venue": "", "url": f"https://doi.org/{r.doi}" if r.doi else "",
                "supports": r.finding, "source": "curated",
            }
            for r in LITERATURE.values()
        ]
        for x in live:
            x["supports"] = "Retrieved live from Crossref; relevance not yet screened."
            x["source"] = "crossref_live"
        return curated, live

    lit = stage("literature", _lit, ([], []))
    a.literature = lit[0] if isinstance(lit, tuple) else []

    a.stages.append(Stage("total", time.perf_counter() - t_all))
    return a, img, sal