"""PDF report generation with ReportLab.

Produces a self-contained A4 document: the analysed image, the saliency overlay, every
finding with its provenance badge, the limitations, and the reference list.

The provenance tags are rendered visually, because a PDF that reads as uniformly
authoritative is exactly the failure mode this project is built to avoid.
"""

from __future__ import annotations

import io
from datetime import datetime

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#12161C")
MUTED = colors.HexColor("#5B6672")
ACCENT = colors.HexColor("#2F6F8F")
RULE = colors.HexColor("#D8DEE4")

# Colour coding mirrors the on-screen provenance badges.
BADGE = {
    "measurement": colors.HexColor("#1F6F43"),
    "model_output": colors.HexColor("#8A5A00"),
    "literature": colors.HexColor("#2F5F8F"),
    "hypothesis": colors.HexColor("#8B3A62"),
    "unverifiable": colors.HexColor("#6B6B6B"),
}


def _styles() -> dict:
    s = getSampleStyleSheet()
    base = dict(textColor=INK, leading=13.5, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=25, textColor=INK, spaceAfter=2),
        "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=10.5, leading=14,
                              textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=16,
                             textColor=ACCENT, spaceBefore=13, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                             textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, **base),
        "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=8.2, leading=11,
                                textColor=MUTED),
        "badge": ParagraphStyle("bd", fontName="Helvetica-Bold", fontSize=7.6,
                                leading=9.6, textColor=colors.white),
        "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.4, leading=11,
                               textColor=INK),
    }


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 12 * mm,
                      "NeuroInsight by Aanishka - computational prototype, not a "
                      "validated clinical or commercial instrument")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.restoreState()


def _as_png(img: np.ndarray, max_w: float) -> RLImage:
    ok, buf = cv2.imencode(".png", img)
    h, w = img.shape[:2]
    ratio = max_w / w
    return RLImage(io.BytesIO(buf.tobytes()), width=max_w, height=h * ratio)


def build_pdf(data: dict, original: np.ndarray, sal: np.ndarray) -> bytes:
    from ..vision import saliency as sal_mod

    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="NeuroInsight report", author="NeuroInsight by Aanishka",
    )
    W = doc.width
    E: list = []

    E.append(Paragraph("NeuroInsight report", st["title"]))
    E.append(Paragraph(
        f"Turning visual stimuli into neuroscience-informed insights &nbsp;|&nbsp; "
        f"{data['dimensions']['width']}x{data['dimensions']['height']} px &nbsp;|&nbsp; "
        f"analysis {data.get('created','')}", st["sub"]))

    # ------------------------------------------------------------ exec summary
    meas = [f for f in data["findings"] if f["evidence"] == "measurement"]
    model = [f for f in data["findings"] if f["evidence"] == "model_output"]

    E.append(Paragraph("Executive summary", st["h2"]))
    con = data["features"]["contrast"]
    lum = data["features"]["luminance"]
    top = data["saliency"]["stats"].get("top_region", {})
    face_n = data["features"]["faces"]["count"]
    txt_n = data["features"]["text"].get("block_count", 0)

    E.append(Paragraph(
        f"Computed directly from this image: a luminance contrast ratio of "
        f"<b>{con['ratio']}:1</b> ({con['wcag_band'].split(' ')[0]}), an edge density of "
        f"<b>{lum['edge_density']*100:.1f}%</b>, and a mean luminance of "
        f"<b>{lum['mean_luminance']:.2f}</b>. Predicted attention peaks in region "
        f"<b>{top.get('cell','-')}</b> holding <b>{top.get('share',0)*100:.1f}%</b> of the "
        f"modelled attention. Face detection returned <b>{face_n}</b> face(s); OCR "
        f"recovered <b>{txt_n}</b> text block(s).", st["body"]))
    E.append(Spacer(1, 3))
    E.append(Paragraph(
        "Everything below is tagged by how it was established. Measured values are "
        "facts about the image; model outputs are predictions; literature entries are "
        "general mechanisms that are not specific to this image; hypotheses are our "
        "reading. No claim is made about any individual's response.",
        st["small"]))
    E.append(Spacer(1, 8))

    # ---------------------------------------------------------------- visuals
    E.append(Paragraph("Attention map", st["h2"]))
    try:
        E.append(_as_png(sal_mod.overlay(original, sal), W))
        E.append(Spacer(1, 4))
        E.append(Paragraph(
            "Predicted saliency overlay (turbo colour scale, warm = higher predicted "
            "attention). This is a computational estimate from "
            f"{data['saliency']['model'].get('model_name','-')}. "
            "<b>No person was observed viewing this image.</b> It is not eye-tracking "
            "and must not be presented as measured gaze.", st["small"]))
    except Exception:
        E.append(Paragraph("Saliency overlay unavailable.", st["small"]))

    # --------------------------------------------------------------- findings
    E.append(PageBreak())
    E.append(Paragraph("Findings", st["h2"]))
    for f in data["findings"]:
        col = BADGE.get(f["evidence"], MUTED)
        badge = Table(
            [[Paragraph(f["evidence_label"].upper(), st["badge"])]],
            colWidths=[min(W, 62 * mm)],
        )
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), col),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        E.append(badge)
        E.append(Spacer(1, 3))
        E.append(Paragraph(f["title"], st["h3"]))
        E.append(Paragraph(f["detail"], st["body"]))
        if f.get("references"):
            names = ", ".join(f["references"])
            E.append(Paragraph(
                f"<i>Confidence: {f['confidence']} &nbsp;|&nbsp; refs: {names}</i>",
                st["small"]))
        else:
            E.append(Paragraph(f"<i>Confidence: {f['confidence']}</i>", st["small"]))
        E.append(Spacer(1, 7))

    # ------------------------------------------------------- recommendations
    E.append(Paragraph("Recommendations (hypotheses to test)", st["h2"]))
    E.append(Paragraph(
        "These are testable predictions, not proven improvements. Each requires a "
        "controlled comparison with participant data before it could be called an "
        "improvement.", st["small"]))
    E.append(Spacer(1, 5))
    for r in data["recommendations"]:
        E.append(Paragraph(
            f"<b>[{r['priority'].upper()}] {r['hypothesis']}</b>", st["h3"]))
        E.append(Paragraph(f"<b>How to test it:</b> {r['test']}", st["body"]))
        E.append(Spacer(1, 5))

    # -------------------------------------------------------------- limitations
    E.append(Paragraph("Limitations", st["h2"]))
    lims = list(data["saliency"]["model"].get("known_limitations", []))
    lims += list(data["features"]["faces"].get("limitations", []))
    lims += list(data["features"]["text"].get("limitations", []))
    for l in dict.fromkeys(lims):
        E.append(Paragraph(f"&bull; {l}", st["body"]))
    E.append(Spacer(1, 5))
    E.append(Paragraph(
        "This tool estimates properties of <i>images</i>. It cannot measure a person. "
        "It is a research prototype: it is not clinically validated, not a commercial "
        "neuromarketing system, and not a substitute for fMRI, eye-tracking or field "
        "experiments. Predicted attention should be treated as a hypothesis to be "
        "tested, never as a finding about an audience.", st["small"]))

    # -------------------------------------------------------------- references
    E.append(PageBreak())
    E.append(Paragraph("References", st["h2"]))
    for r in data.get("literature", []):
        txt = r.get("title", "")
        if r.get("doi"):
            txt += f" &nbsp;<font color='#2F6F8F'>doi:{r['doi']}</font>"
        E.append(Paragraph(txt, st["cell"]))
        if r.get("supports"):
            E.append(Paragraph(f"<i>Supports:</i> {r['supports']}", st["small"]))
        E.append(Spacer(1, 4))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        f"Generated by NeuroInsight by Aanishka v{data.get('version','0.1.0')} on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.", st["small"]))

    doc.build(E, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()