"""Epistemic provenance model.

Every substantive statement NeuroInsight makes must declare what kind of claim it is.
This module is the single source of truth for that distinction, and the report layer
refuses to render a finding that lacks a provenance tag.

The five levels, weakest claim first:

  MEASUREMENT  - computed directly from the pixels by our own code
  MODEL_OUTPUT - produced by a pretrained third-party model (a prediction, not a fact)
  LITERATURE   - a general mechanism established in the published literature
  HYPOTHESIS   - our speculation about how a specific image might affect a viewer
  UNVERIFIABLE - cannot be determined from a static image at all

The distinction between MEASUREMENT and MODEL_OUTPUT matters most: a pixel-contrast
value we computed is a fact about the image, whereas a saliency map is a model's
opinion about where people will look. Presenting the second as the first is the most
common failure mode in applied "neuromarketing" and this project refuses to commit it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Evidence(str, Enum):
    """How do we know this? Ordered from strongest to weakest claim."""

    MEASUREMENT = "measurement"
    MODEL_OUTPUT = "model_output"
    LITERATURE = "literature"
    HYPOTHESIS = "hypothesis"
    UNVERIFIABLE = "unverifiable"

    @property
    def label(self) -> str:
        return {
            "measurement": "Directly calculated from your image",
            "model_output": "Predicted by a pretrained model",
            "literature": "Established mechanism in the literature",
            "hypothesis": "Hypothesis (not established)",
            "unverifiable": "Not determinable from a static image",
        }[self.value]

    @property
    def confidence(self) -> str:
        """How strongly this should be weighted in the report."""
        return {
            "measurement": "High — recomputable from the image",
            "model_output": "Moderate — model-dependent, validated on saliency datasets",
            "literature": "Moderate-High — replicated, but not specific to this image",
            "hypothesis": "Low — a plausible reading, not a demonstrated effect",
            "unverifiable": "None — stated explicitly as not answerable",
        }[self.value]


class Finding(dict):
    """One reportable observation.

    A plain dict so it serialises straight to JSON for the frontend and the PDF
    renderer, but constructed through `Finding.make` so provenance cannot be
    forgotten — you cannot create a Finding without supplying evidence.
    """

    @classmethod
    def make(
        cls,
        *,
        title: str,
        detail: str,
        evidence: Evidence,
        value: Any = None,
        unit: str | None = None,
        refs: list[str] | None = None,
    ) -> "Finding":
        refs = refs or []
        if evidence in (Evidence.HYPOTHESIS, Evidence.UNVERIFIABLE) and not refs:
            raise ValueError(
                f"Hypothesis/unverifiable finding '{title}' must cite at least one "
                "reference so the reader can check the reasoning."
            )
        return cls(
            title=title,
            detail=detail,
            evidence=evidence.value,
            evidence_label=evidence.label,
            confidence=evidence.confidence,
            value=value,
            unit=unit,
            references=refs,
        )


def never_claim(what: str, needs: str) -> dict:
    """Standard phrasing for something a static image simply cannot answer.

    We surface these rather than omitting them, because a reader needs to know the
    boundary of the tool. Pretending we can estimate these would be the worst
    possible failure for a project whose premise is scientific integrity.
    """
    return {
        "title": f"Cannot be determined: {what}",
        "detail": (
            f"A static image cannot support a claim about {what}. Doing so would "
            f"require {needs}. We report this gap explicitly instead of estimating it, "
            f"because a fabricated figure would be worse than no figure."
        ),
        "evidence": Evidence.UNVERIFIABLE.value,
        "evidence_label": Evidence.UNVERIFIABLE.label,
        "confidence": "None — stated explicitly as not answerable",
        "value": None,
        "unit": None,
        "references": [],
    }


# Claims we are structurally forbidden from making without real participant data.
FORBIDDEN_WITHOUT_PARTICIPANTS = {
    "emotion": (
        "emotional response, purchase intent, brain activation, subconscious "
        "manipulation, or conversion probability"
    ),
    "note": (
        "These require participant-level data (eye-tracking, survey or neural "
        "recording) and a model trained and validated on that outcome. We do not "
        "estimate them, and no number in this report should be read as such."
    ),
}