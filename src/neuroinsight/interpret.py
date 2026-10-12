"""Turn measurements + model output into evidence-tagged findings.

Rules enforced here:
  1. Every finding carries a provenance tag (see provenance.py).
  2. No numeric claim about emotion, intent, or neural activation is ever emitted.
  3. Image-specific statements cite the measurement they were derived from.
  4. Anything we cannot support is surfaced as UNVERIFIABLE, not omitted.
"""

from __future__ import annotations

import statistics

from .literature import LITERATURE, refs
from .provenance import Evidence, Finding, never_claim


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def interpret(features: dict, sal_meta: dict, sal_stats: dict) -> list[dict]:
    """Build the full finding list for one analysed image."""
    out: list[dict] = []
    lum = features["luminance"]
    col = features["colour"]
    con = features["contrast"]
    faces = features["faces"]
    text = features["text"]
    comp = features.get("composition") or {}
    focal = features.get("focal_alignment") or {}
    thier = features.get("text_hierarchy") or {}

    # ---------------------------------------------------------- attention / saliency
    top = sal_stats["top_region"]
    conc = sal_stats["concentration"]
    out.append(
        Finding.make(
            title="Predicted attention concentrates in a "
            + ("narrow region" if conc == "concentrated" else "moderate region"
               if conc == "balanced" else "diffuse region"),
            detail=(
                f"The saliency model places its peak in region {top['cell']} "
                f"(row {top['row']}, column {top['col']}), which carries "
                f"{_pct(top['share'])} of total predicted attention and sits at a mean "
                f"luminance of {top['luminance']:.0f}/255. "
                + (
                    "Attention is predicted to converge on a small part of the image, "
                    "which increases the chance that surrounding content is never "
                    "consciously registered."
                    if conc == "concentrated"
                    else "Attention is predicted to spread over most of the frame, so "
                    "no single element can be assumed to dominate."
                )
            ),
            evidence=Evidence.MODEL_OUTPUT,
            value=top["share"],
            unit="fraction of predicted attention",
            refs=refs("itti2001", "henderson1999"),
        )
    )

    out.append(
        Finding.make(
            title="This is predicted attention, not measured gaze",
            detail=(
                "The map above is a computational estimate from "
                f"{sal_meta['model_name']}. No person was observed viewing this image. "
                "Predicted saliency is a useful prior about where attention tends to "
                "fall, and it correlates with fixation data, but it is not a substitute "
                "for eye-tracking and should never be reported as measured gaze."
            ),
            evidence=Evidence.LITERATURE,
            # hou2007 is the algorithm actually being run, so it belongs on the
            # finding that names the algorithm; itti2001/henderson1999 cover why a
            # saliency prior is meaningful at all.
            refs=refs("hou2007", "itti2001", "henderson1999"),
        )
    )

    # ------------------------------------------- composition (brand-facing)
    # These are all direct computations on the pixels, so they stay MEASUREMENT.
    # The wording is deliberately concrete and design-shop vocabulary ("visual
    # weight", "breathing room") rather than perceptual-science jargon, because
    # the audience for this half of the report is designers and marketers.
    if comp:
        off = comp.get("visual_weight_offcentre", 0.0)
        even = comp.get("quadrant_evenness", 0.0)
        ws = comp.get("whitespace_pct", 0.0)
        quad = comp.get("heaviest_quadrant", "")
        quadrant_name = {"tl": "top-left", "tr": "top-right",
                         "bl": "bottom-left", "br": "bottom-right"}.get(quad, quad)

        out.append(Finding.make(
            title=(
                "The design's visual weight sits " + (
                    "almost dead centre" if off < 0.15 else
                    "slightly off-centre" if off < 0.35 else
                    "well off-centre")
            ),
            detail=(
                f"Measuring where detail and contrast actually accumulate, the centre "
                f"of visual mass sits at {100 * comp.get('visual_weight_centre_x', 0.5):.0f}% "
                f"across and {100 * comp.get('visual_weight_centre_y', 0.5):.0f}% down the "
                f"frame — a distance of {off:.2f} from dead centre (0.00 = perfectly "
                f"centred, 1.00 = pushed into a corner). "
                + ("Centred layouts feel stable and formal; an off-centre layout reads as "
                   "more energetic. This is a property of the image, not a prediction "
                   "about how anyone will feel about it."
                   if off >= 0.15 else
                   "A perfectly centred layout reads as stable and formal.")
            ),
            evidence=Evidence.MEASUREMENT,
            value=round(off, 4),
            unit="distance from centre (0 = centred, 1 = corner)",
            refs=refs("treisman1980"),
        ))

        out.append(Finding.make(
            title=(
                f"Detail is piled into the {quadrant_name} quarter" if even < 0.28
                else "Detail is spread fairly evenly across the frame"
            ),
            detail=(
                f"The four quarters of the image carry {100 * even:.0f}% even "
                f"distribution of visual detail (100% would be perfectly balanced across "
                f"all four; {100 * 0.25:.0f}% means everything is in a single quarter). "
                f"The heaviest quarter is the {quadrant_name}. "
                + ("Clumped layouts direct the eye to one place but can leave large areas "
                   "unnoticed. Even layouts give every element a fighting chance."
                   if even < 0.28 else
                   "An even spread means no single element dominates by sheer position.")
            ),
            evidence=Evidence.MEASUREMENT,
            value=round(even, 4),
            unit="quadrant evenness (0.25 = one corner, 1.0 = even)",
            refs=refs("rosbergen1997"),
        ))

        out.append(Finding.make(
            title=(
                f"Breathing room is {'generous' if ws >= 30 else 'tight' if ws < 15 else 'moderate'}"
                f" ({ws:.0f}% of the frame is quiet)"
            ),
            detail=(
                f"{ws:.0f}% of the image contains almost no detail or contrast — the "
                f"equivalent of empty space in a layout. "
                + ("Generous whitespace gives the eye somewhere to rest and makes the "
                   "busy elements read as deliberate. Too little and everything competes."
                   if ws >= 30 else
                   "Very little empty space means many elements are competing for the "
                   "same attention. Whitespace is the cheapest way to create hierarchy."
                   if ws < 15 else
                   "A moderate amount of empty space is typical for a busy layout.")
            ),
            evidence=Evidence.MEASUREMENT,
            value=ws,
            unit="% of frame with minimal detail",
            refs=refs("rosbergen1997"),
        ))

    # ------------------------------------------------- focal point / thirds
    if focal.get("available"):
        ratio = focal.get("thirds_alignment_ratio", 0.0)
        on_thirds = focal.get("on_thirds_share", 0.0)
        centred = focal.get("peak_is_centred")
        out.append(Finding.make(
            title=(
                "Predicted attention lands on the rule-of-thirds grid"
                if ratio > 0.75 else
                "Predicted attention sits between the rule-of-thirds lines"
            ),
            detail=(
                f"The saliency peak falls at {100 * focal.get('peak_x', 0):.0f}% across, "
                f"{100 * focal.get('peak_y', 0):.0f}% down. {on_thirds:.0f}% of the "
                f"predicted attention falls within 10% of the four thirds intersections. "
                + ("Laying the main message on a thirds intersection is the classic "
                   "layout rule and the model agrees with it here — which means the "
                   "composition is working with the eye's habits, not against them."
                   if ratio > 0.75 else
                   "The eye's likely landing point is not on a thirds intersection. That "
                   "is not an error — plenty of strong designs centre deliberately — but "
                   "if the goal is effortless scanning, moving the focal element onto a "
                   "thirds line usually helps."
                   if not centred else
                   "The design is centred rather than thirds-aligned, which reads as "
                   "deliberate and stable rather than dynamic.")
            ),
            evidence=Evidence.MODEL_OUTPUT,
            value=round(ratio, 4),
            unit="thirds alignment (0-1)",
            refs=refs("treisman1980"),
        ))

    # ------------------------------------------------------- text hierarchy
    if thier.get("available"):
        verdict = thier.get("verdict", "flat")
        out.append(Finding.make(
            title={
                "clear": "There's a clear typographic hierarchy",
                "moderate": "Type sizes are only slightly differentiated",
                "flat": "All the text is roughly the same size — no clear hierarchy",
            }.get(verdict, "Text hierarchy assessed"),
            detail=(
                f"{thier.get('block_count')} text blocks were detected. The largest is "
                f"{thier.get('largest_height_px')}px tall; the next differs by only "
                f"{100 * thier.get('top_step_ratio', 0):.0f}%. "
                + ("A strong size jump means the reader's eye knows instantly what to "
                   "read first. This design has that."
                   if verdict == "clear" else
                   "When every element is close to the same size, nothing reads as more "
                   "important than anything else and the reader has to work out the "
                   "hierarchy themselves. Bolding, scaling up the headline, or cutting "
                   "the number of competing blocks all fix this."
                   if verdict == "flat" else
                   "There is some size differentiation, but a bigger gap would make the "
                   "reading order unambiguous.")
            ),
            evidence=Evidence.MEASUREMENT,
            value=round(thier.get("top_step_ratio", 0.0), 3),
            unit="size difference between largest and second text block",
            refs=refs("rosbergen1997"),
        ))

    # ------------------------------------------------------------------- contrast
    out.append(
        Finding.make(
            title=f"Luminance contrast ratio {con['ratio']}: {con['wcag_band']}",
            detail=(
                f"Brightest and darkest regions differ by a WCAG relative-luminance "
                f"ratio of {con['ratio']}:1 (dynamic range "
                f"{_pct(con['dynamic_range'])} of the 0–1 luminance scale). "
                + (
                    "This comfortably exceeds the 7:1 threshold at which WCAG defines "
                    "maximum contrast compliance, so the image's most and least "
                    "luminous elements should be readily distinguishable. Note this "
                    "measures the extremes of the image, not whether specific text is "
                    "legible in context."
                    if con["ratio"] >= 7
                    else "Contrast is moderate; adjacent elements of similar brightness "
                    "may not separate cleanly, particularly on small displays or for "
                    "viewers with reduced contrast sensitivity."
                )
            ),
            evidence=Evidence.MEASUREMENT,
            value=con["ratio"],
            unit="WCAG contrast ratio",
            refs=[],
        )
    )

    # ------------------------------------------------------------------ complexity
    ed = lum["edge_density"]
    ent = lum["luminance_entropy_bits"]
    complexity = (
        "high" if (ed > 0.09 and ent > 7.0)
        else "moderate" if (ed > 0.04 or ent > 6.5)
        else "low"
    )
    out.append(
        Finding.make(
            title=f"Visual complexity is {complexity}",
            detail=(
                f"Edge density is {_pct(ed)} of pixels and the luminance distribution "
                f"has an entropy of {ent:.2f} bits (8.0 = perfectly uniform). "
                + (
                    "Dense edge content and a broad luminance distribution both indicate "
                    "many competing elements. High complexity raises cognitive load, and "
                    "high load is associated with reduced processing of brand and "
                    "product information."
                    if complexity == "high"
                    else "The image presents a moderate number of elements. Attention is "
                    "less likely to be swamped, though some regions will still be missed "
                    "in brief exposure."
                )
            ),
            evidence=Evidence.MEASUREMENT,
            value=ed,
            unit="fraction of pixels on edges",
            refs=refs("rosbergen1997"),
        )
    )

    # ---------------------------------------------------------------------- faces
    if faces["count"] > 0:
        out.append(
            Finding.make(
                title=f"{faces['count']} face{'s' if faces['count'] > 1 else ''} detected "
                f"({faces['total_area_pct']}% of frame area)",
                detail=(
                    f"The largest detected face occupies {faces['largest_area_pct']}% of "
                    "the image. Faces capture fixation disproportionately to their "
                    "physical area, so face presence typically increases the share of "
                    "attention directed at that region relative to its size in the frame."
                ),
                evidence=Evidence.MODEL_OUTPUT,
                value=faces["total_area_pct"],
                unit="percent of frame area",
                refs=refs("itier2007", "kanwisher1997"),
            )
        )
    else:
        out.append(
            Finding.make(
                title="No faces detected",
                detail=(
                    "The cascade detector found no faces. This is a negative detection, "
                    "not proof of absence: the Viola-Jones cascades are biased toward "
                    "frontal, well-lit faces and miss profile, small and angled faces."
                ),
                evidence=Evidence.MODEL_OUTPUT,
                refs=[],
            )
        )

    # ----------------------------------------------------------------------- text
    if text.get("status") == "ok" and text["block_count"] > 0:
        big = text["blocks"][0]
        out.append(
            Finding.make(
                title=f"{text['block_count']} text block{'s' if text['block_count'] > 1 else ''} "
                f"detected, largest is \"{big['text'][:40]}\"",
                detail=(
                    f"OCR recovered {text['total_words']} words at a mean confidence of "
                    f"{text['mean_confidence']:.2f}. The largest block covers "
                    f"{big['area_pct']}% of the frame. Copy occupying a larger share of "
                    "the field is more likely to be read, and typographic weight is one "
                    "of the stimulus properties that drives initial orienting of "
                    "attention."
                ),
                evidence=Evidence.MODEL_OUTPUT,
                value=big["area_pct"],
                unit="percent of frame area",
                refs=refs("rosbergen1997", "pieters2007"),
            )
        )
    else:
        out.append(
            Finding.make(
                title="No text detected",
                detail=(
                    "OCR returned no readable text blocks"
                    + (f" ({text.get('reason')})" if text.get("reason") else "")
                    + ". Either the image contains no copy, or the copy is too small, "
                    "stylised or low-contrast for reliable recognition."
                ),
                evidence=Evidence.MODEL_OUTPUT,
                refs=[],
            )
        )

    # --------------------------------------------------------------------- colour
    if col["palette"]:
        dominant = col["palette"][0]
        out.append(
            Finding.make(
                title=f"Dominant hue at {dominant['hue_deg']:.0f}° covering "
                f"{_pct(dominant['share'])} of chromatic pixels",
                detail=(
                    f"Mean saturation is {col['mean_saturation']:.0f}/255 and "
                    f"{_pct(col['saturation_pct'])} of pixels carry meaningful chroma. "
                    "A dominant hue concentrates visual identity and makes an image "
                    "easier to group as a single object; strong colour contrast between "
                    "an element and its background raises the probability that the "
                    "element is separated early during visual search."
                ),
                evidence=Evidence.MEASUREMENT,
                value=dominant["hue_deg"],
                unit="degrees",
                refs=refs("treisman1980"),
            )
        )

    # ------------------------------------------- honest boundaries (always surfaced)
    out.append(never_claim(
        "the viewer's emotional response",
        "participant-level data such as a validated affect rating scale",
    ))
    out.append(never_claim(
        "purchase intent or conversion probability",
        "participant study with a predefined outcome measure and a trained model",
    ))
    out.append(never_claim(
        "neural activity in the brain",
        "fMRI, MEG or EEG recording of a person viewing the stimulus",
    ))

    return out


def recommendations(features: dict, sal_stats: dict) -> list[dict]:
    """Evidence-informed hypotheses to test, never promises."""
    recs = []
    lum = features["luminance"]
    faces = features["faces"]
    text = features["text"]

    if sal_stats["concentration"] == "concentrated" and sal_stats["top3_share_of_attention"] > 0.45:
        recs.append({
            "priority": "high",
            "hypothesis": (
                "Attention is predicted to converge on one area, so secondary elements "
                "may be missed entirely at normal viewing duration."
            ),
            "test": (
                "Produce a variant that reduces top-three predicted-attention share "
                "below ~0.40, then compare with eye-tracking on 15+ participants. "
                "Primary metric: fixation share on the secondary element."
            ),
            "basis": refs("rosbergen1997", "pieters2007"),
        })

    if lum["edge_density"] > 0.09:
        recs.append({
            "priority": "medium",
            "hypothesis": (
                "High edge density suggests competing elements that raise cognitive "
                "load, which is associated with reduced processing of product and brand "
                "information."
            ),
            "test": (
                "A/B test a decluttered variant. Primary metric: unaided brand recall "
                "after a fixed 3-second exposure, not self-reported liking."
            ),
            "basis": refs("rosbergen1997"),
        })

    if faces["count"] == 0 and text["status"] == "ok" and text["block_count"] > 0:
        recs.append({
            "priority": "low",
            "hypothesis": (
                "With no face detected and text carrying the attention, the visual "
                "hierarchy may be carried by typography alone, which limits the "
                "social-affective route to attention."
            ),
            "test": (
                "Compare a typographic-only variant against one adding a face. "
                "Measure fixation latency and dwell time on the product."
            ),
            "basis": refs("kanwisher1997", "rosbergen1997"),
        })

    if not recs:
        recs.append({
            "priority": "low",
            "hypothesis": (
                "Measured complexity and predicted attention spread are both moderate, "
                "so no single structural change is clearly indicated."
            ),
            "test": (
                "The informative next step is participant data, not further image "
                "metrics: run an eye-tracking or first-fixation study to establish "
                "whether predictions hold for this specific stimulus."
            ),
            "basis": refs("henderson1999"),
        })

    return recs