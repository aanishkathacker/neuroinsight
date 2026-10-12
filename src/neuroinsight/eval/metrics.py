"""Saliency-prediction evaluation metrics.

Standard metrics from the visual-saliency literature, following the protocol of Judd
et al. (ICCV 2009) and the MIT saliency benchmark code.

METRICS
-------
AUC  Area under the ROC curve. How well predicted saliency ranks true fixation
     locations above non-fixated pixels. 0.5 = chance, >= ~0.7 is a working model.

CC   Pearson correlation coefficient between the predicted map and the ground-truth
     fixation density map. 0 = no linear relationship. This is the metric most
     sensitive to *global* shape agreement.

NSS  Normalised scanpath salience = mean(predicted saliency at fixation points) /
     std(predicted saliency). Measures how many standard deviations above the mean
     your predicted-saliency values are, on average, at locations people actually
     looked. 0 = chance, 1 is decent, >2 is strong, >3 is very strong.

WHY NOT JUST ONE
----------------
These metrics disagree by construction, and that disagreement is informative:

  - AUC and NSS care about *where* fixations land (rank / local contrast).
  - CC cares about whether the overall map *shape* matches.

A method can score well on CC and poorly on NSS if it produces a correctly-shaped but
flat map. Reporting one metric alone would hide that, so we report all three.
"""

from __future__ import annotations

import cv2
import numpy as np

# Fixation "size": the standard evaluation treats each fixation as a small Gaussian
# blob rather than a single pixel, because fixation estimates are noisy. Judd et al.
# use a 2-degree Gaussian; at ~30 px/degree on a 768x1024 image that is sigma ~24 px.
# We use a slightly smaller sigma (9 px) matching the MIT benchmark code's default.
FIX_SIGMA = 9.0


def _fixation_points(gt: np.ndarray, sigma: float = FIX_SIGMA) -> list[tuple[int, int]]:
    """Extract discrete fixation coordinates from a continuous fixation density map.

    The MIT dataset ships fixation maps already blurred with a Gaussian. We recover
    approximate fixation locations by taking local maxima that exceed a threshold
    derived from the map's own statistics.
    """
    g = gt.astype(np.float64)
    if g.max() <= 0:
        return []
    g = g / g.max()

    # Threshold: mean + 1 std of the map. Fixation blobs sit well above the
    # diffuse background, so this isolates peaks without hardcoding a magic number.
    thresh = float(g.mean() + g.std())
    peak = g >= thresh

    # Non-maximum suppression: keep only local maxima so one blob yields one point.
    dilated = cv2.dilate(g.astype(np.float32), np.ones((3, 3), np.uint8))
    coords = np.argwhere(peak & (g >= dilated - 1e-9))

    if len(coords) > 400:  # cap for tractability on pathological maps
        idx = np.argsort(-g[coords[:, 0], coords[:, 1]])[:400]
        coords = coords[idx]
    return [(int(y), int(x)) for y, x in coords]


def auc(sal: np.ndarray, gt: np.ndarray, ds: int = 4) -> float:
    """Continuous AUC-Judd: area under the ROC between predicted saliency and
    human fixation locations.

    For each fixation we build a small Gaussian "positives" kernel, then ask how well
    the predicted saliency ranks that region above the rest of the image, weighted by
    the Gaussian. Averaged over fixations. 0.5 = chance.

    We delegate the rank arithmetic to `sklearn.metrics.roc_auc_score`, which handles
    ties and weighting correctly. A hand-rolled Mann-Whitney here is a reliable way to
    ship a plausible-looking number that is simply wrong - ties in a mostly-flat
    saliency map are everywhere, and getting them wrong inflates AUC past 1.0.

    ``ds`` downsamples both maps before scoring (default 4x). These metrics are
    spatially coarse (sigma ~9 px), so 4x costs almost no accuracy and buys ~16x speed.
    """
    from sklearn.metrics import roc_auc_score

    pts = _fixation_points(gt)
    if not pts:
        return float("nan")

    s = sal.astype(np.float64)
    if ds > 1:
        s = cv2.resize(s, (s.shape[1] // ds, s.shape[0] // ds),
                       interpolation=cv2.INTER_AREA)
    h, w = s.shape

    pts = [(y // ds, x // ds) for (y, x) in pts]
    pts = [(y, x) for (y, x) in pts if 0 <= y < h and 0 <= x < w]
    if not pts:
        return float("nan")

    r = max(1, int(3 * FIX_SIGMA / ds))
    kernel = cv2.getGaussianKernel(2 * r + 1, FIX_SIGMA / ds)

    # Neighbourhood the Gaussian occupies, so we only score that patch plus the
    # whole background - not the Gaussian's own soft ramp, which is neither
    # "at the fixation" nor "away from it".
    pad = r + 1

    aucs = []
    for (y, x) in pts:
        y0, y1 = max(0, y - pad), min(h, y + pad + 1)
        x0, x1 = max(0, x - pad), min(w, x + pad + 1)
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue

        sub = s[y0:y1, x0:x1]
        blob = np.zeros_like(sub)
        blob[y - y0, x - x0] = 1.0
        blob = cv2.filter2D(blob, -1, kernel)

        inside = blob > (blob.max() * 0.5)
        if not inside.any():
            continue

        # Negatives: a sample of well-separated background pixels, so the ROC is not
        # dominated by millions of near-identical flat-background values.
        mask = np.ones_like(sub, dtype=bool)
        mask &= ~inside
        bg_idx = np.argwhere(mask)
        if len(bg_idx) > 4000:
            step = len(bg_idx) // 4000 + 1
            bg_idx = bg_idx[::step]

        pos_scores = sub[inside].ravel()
        neg_scores = sub[bg_idx[:, 0], bg_idx[:, 1]].ravel()
        if pos_scores.size == 0 or neg_scores.size == 0:
            continue

        y_true = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)])
        y_score = np.concatenate([pos_scores, neg_scores])
        # Weight positives by Gaussian proximity: the centre of the fixation counts
        # most, the rim least. Negatives are unweighted.
        # NOTE: must not be called `w` - that shadows the image width used above.
        sw = np.concatenate([
            blob[inside].ravel() / max(blob[inside].max(), 1e-9),
            np.ones(neg_scores.size),
        ])

        try:
            aucs.append(roc_auc_score(y_true, y_score, sample_weight=sw))
        except ValueError:
            continue

    return float(np.mean(aucs)) if aucs else float("nan")


def cc(sal: np.ndarray, gt: np.ndarray) -> float:
    """Pearson correlation between predicted saliency and ground-truth density."""
    s = sal.astype(np.float64).ravel()
    g = gt.astype(np.float64).ravel()
    if s.std() < 1e-12 or g.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(s, g)[0, 1])


def nss(sal: np.ndarray, gt: np.ndarray) -> float:
    """Normalised scanpath salience: mean of saliency at fixations / its global std."""
    pts = _fixation_points(gt)
    if not pts:
        return float("nan")
    h, w = gt.shape[:2]
    vals = [
        sal[y, x]
        for (y, x) in pts
        if 0 <= y < h and 0 <= x < w
    ]
    if not vals or sal.std() < 1e-12:
        return float("nan")
    return float(np.mean(vals) / sal.std())


def score(sal: np.ndarray, gt: np.ndarray) -> dict:
    """All three metrics for one image. `sal` must be float in [0, 1]."""
    s = sal.astype(np.float64)
    return {
        "auc": auc(s, gt),
        "cc": cc(s, gt),
        "nss": nss(s, gt),
    }


def normalise_saliency(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s, dtype=np.float64)
    return (s - lo) / (hi - lo)


# Interpretation bands, stated conservatively. These are conventions from the
# saliency literature, not thresholds we validated ourselves.
INTERPRETATION = {
    "auc": {0.5: "chance", 0.65: "weak", 0.70: "usable", 0.75: "good", 0.80: "strong"},
    "cc": {0.0: "none", 0.1: "weak", 0.2: "usable", 0.3: "good", 0.4: "strong"},
    "nss": {0.0: "chance", 0.5: "weak", 1.0: "usable", 2.0: "good", 3.0: "strong"},
}


def interpret(metric: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    label = "chance"
    for th, name in sorted(INTERPRETATION[metric].items()):
        if value >= th:
            label = name
    return label