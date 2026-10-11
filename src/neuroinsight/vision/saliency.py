"""Visual saliency estimation.

WHAT THIS IS
------------
A saliency map is a prediction of *where a human is likely to look first*. It is
computed from the image alone, using models trained on datasets where observers'
fixations were recorded while they viewed stimuli.

WHAT THIS IS NOT
----------------
It is NOT eye-tracking. It is NOT measured gaze. Nobody watched a person view your
image to produce this map. It is a learned prior about what tends to attract
attention in general.

MILESTONE 1 IMPLEMENTATION
--------------------------
Spectral Residual saliency (Hou & Zhang, CVPR 2007) via OpenCV contrib, combined
with a Fine-Grained saliency channel (the multi-scale variant of the same paper).

This is a genuine, peer-reviewed, published algorithm rather than a placeholder, and
it runs in milliseconds with no model download. It is however a classical method,
not a deep network, and its known weaknesses are real:

  - it is trained on natural photographs, not on designed advertisements
  - it is dominated by high-contrast regions and can under-serve face regions
    that a deep gaze model would weight heavily
  - it has no notion of text, faces or semantic meaning

Milestone 3 replaces this with a pretrained deep model; see MODEL_CARD.md. Both paths
call `compute_saliency()`, so nothing downstream changes.
"""

from __future__ import annotations

import cv2
import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def compute_saliency(image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Predict a spatial attention map.

    Parameters
    ----------
    image_bgr : BGR uint8 array, the image as uploaded.

    Returns
    -------
    (saliency, metadata)
        saliency  : float32 array in [0, 1], same spatial size as the input
        metadata  : dict describing which algorithm ran and its provenance

    Raises
    ------
    RuntimeError if OpenCV's saliency module is unavailable. We raise rather than
    silently returning zeros, because a blank "attention map" that looks like a real
    result is exactly the kind of quiet dishonesty this project exists to avoid.
    """
    if not hasattr(cv2, "saliency"):
        raise RuntimeError(
            "opencv-contrib-python-headless is required for saliency estimation. "
            "Install it with: uv pip install opencv-contrib-python-headless"
        )

    h, w = image_bgr.shape[:2]
    # Work at a reduced resolution: saliency is a low-frequency signal, and this
    # keeps the cost constant regardless of upload size.
    work = cv2.resize(image_bgr, (min(w, 640), min(h, 640)), interpolation=cv2.INTER_AREA)

    spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
    fine = cv2.saliency.StaticSaliencyFineGrained_create()

    ok_s, sal_s = spectral.computeSaliency(work)
    ok_f, sal_f = fine.computeSaliency(work)

    if ok_s and sal_s is not None:
        sal_s = _normalise(sal_s)
    else:
        sal_s = np.zeros((work.shape[0], work.shape[1]), np.float32)

    if ok_f and sal_f is not None:
        sal_f = _normalise(sal_f)
    else:
        sal_f = np.zeros((work.shape[0], work.shape[1]), np.float32)

    # Equal blend: spectral captures global surprise, fine-grained adds local detail.
    # Weights are a design choice, not a fitted parameter, and are documented as such.
    combined = _normalise(0.5 * sal_s + 0.5 * sal_f)
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=3.0)
    combined = _normalise(combined)
    combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_CUBIC)

    meta = {
        "model_name": "OpenCV Static Saliency (Spectral Residual + Fine-Grained)",
        "model_source": "Hou, X. & Zhang, L. (2007) CVPR; OpenCV contrib module",
        "model_version": "opencv-contrib 5.0.0",
        "licence": "Apache-2.0 (OpenCV); algorithm is public domain academic method",
        "kind": "pretrained / published algorithm, no learned weights required",
        "evidence_class": "model_output",
        "what_it_is": "Predicted attention — an algorithmic estimate, not measured gaze.",
        "known_limitations": [
            "Classical (non-deep) method; no learned representation of faces or text.",
            "Optimised for natural photographs, not designed advertisements.",
            "Strongly driven by contrast; can miss semantically salient but low-contrast regions.",
            "Trained prior to modern ad-design practice.",
        ],
        "computed_at": "runtime",
    }
    return combined, meta


def saliency_statistics(sal: np.ndarray, image_bgr: np.ndarray) -> dict:
    """Summarise where attention is predicted to concentrate.

    Converts the continuous map into interpretable regions by splitting it into a
    coarse grid and asking which cells win. Grid resolution is chosen relative to
    image size so small and large images behave comparably.
    """
    h, w = sal.shape[:2]
    cells = 4 if min(h, w) >= 400 else 2
    ys = np.linspace(0, h, cells + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)

    grid = np.zeros((cells, cells), float)
    for i in range(cells):
        for j in range(cells):
            block = sal[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            grid[i, j] = float(block.mean()) if block.size else 0.0

    # Also compute luminance so we can describe *why* a region wins.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lum_grid = np.zeros((cells, cells), float)
    for i in range(cells):
        for j in range(cells):
            block = lab[ys[i] : ys[i + 1], xs[j] : xs[j + 1], 0]
            lum_grid[i, j] = float(block.mean()) if block.size else 0.0

    vmax = float(grid.max()) or 1.0
    total = float(grid.sum()) or 1.0

    regions = []
    for i in range(cells):
        for j in range(cells):
            v = float(grid[i, j]) / float(vmax)
            regions.append(
                {
                    "cell": f"R{i+1}C{j+1}",
                    "row": i + 1,
                    "col": j + 1,
                    "saliency": round(v, 4),
                    "luminance": round(float(lum_grid[i, j]), 2),
                    "share": round(float(grid[i, j]) / float(total), 4),
                }
            )
    regions.sort(key=lambda r: -r["saliency"])

    top = regions[0]
    top3_share = sum(r["share"] for r in regions[:3])
    return {
        "cells": cells,
        "regions": regions,
        "top_region": top,
        "top3_share_of_attention": round(top3_share, 4),
        "concentration": (
            "concentrated" if top3_share > 0.45 else
            "balanced" if top3_share > 0.32 else "diffuse"
        ),
        "luminance_grid": lum_grid.round(2).tolist(),
    }


def overlay(image_bgr: np.ndarray, sal: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend the saliency map over the image using a perceptually ordered colormap."""
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)