"""FastAPI backend for NeuroInsight.

Serves a single-page interface plus a small JSON API. Analysis is synchronous and
fast enough (roughly 1–3 s for a typical ad) to run inside a normal request, but we
still cap concurrency so a burst of uploads cannot exhaust the 17 GB of RAM here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import pipeline
from ..report.pdf import build_pdf
from ..vision import saliency

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web"

app = FastAPI(title="NeuroInsight by Aanishka", version="0.1.0")

# Small in-memory store. Deliberately not a database: this is a local single-user tool,
# and holding results in RAM keeps the repo free of state and the setup free of
# migrations. Swap for SQLite/Postgres before any real deployment.
_STORE: dict[str, dict] = {}
_MAX_CACHE = 24

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Failed to encode result image")
    return buf.tobytes()


def _trim_for_web(img: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "static" / "index.html").read_text()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "NeuroInsight by Aanishka",
        "version": app.version,
        "max_upload_mb": pipeline.MAX_BYTES // (1024 * 1024),
        "cached_analyses": len(_STORE),
    }


@app.post("/api/analyse")
async def analyse_endpoint(file: UploadFile = File(...)) -> dict:
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            415,
            f"Unsupported file type '{suffix or name}'. Allowed: {', '.join(sorted(ALLOWED))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > pipeline.MAX_BYTES:
        raise HTTPException(
            413,
            f"File is {len(data)/1e6:.1f} MB; the limit is "
            f"{pipeline.MAX_BYTES/1e6:.0f} MB.",
        )

    try:
        analysis, img, sal = pipeline.analyse(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            500, f"Analysis failed unexpectedly: {type(exc).__name__}: {exc}"
        ) from exc

    d = analysis.to_dict()
    _STORE[analysis.image_id] = {
        "dict": d,
        "original": _trim_for_web(img),
        "saliency": _trim_for_web(sal),
    }
    if len(_STORE) > _MAX_CACHE:
        for k in list(_STORE)[: len(_STORE) - _MAX_CACHE]:
            _STORE.pop(k, None)

    d["images"] = {
        "original": f"/api/image/{analysis.image_id}/original",
        "saliency": f"/api/image/{analysis.image_id}/saliency",
        "overlay": f"/api/image/{analysis.image_id}/overlay",
        "report": f"/api/report/{analysis.image_id}.pdf",
    }
    return d


@app.get("/api/image/{image_id}/{kind}")
def image_endpoint(image_id: str, kind: str) -> Response:
    rec = _STORE.get(image_id)
    if not rec:
        raise HTTPException(404, "Unknown or expired analysis id. Re-upload the image.")
    if kind == "original":
        return Response(_png(rec["original"]), media_type="image/png")
    if kind == "saliency":
        grey = cv2.cvtColor(rec["saliency"], cv2.COLOR_GRAY2BGR)
        return Response(_png(grey), media_type="image/png")
    if kind == "overlay":
        return Response(
            _png(saliency.overlay(rec["original"], rec["saliency"])),
            media_type="image/png",
        )
    raise HTTPException(400, f"Unknown image kind '{kind}'")


@app.get("/api/report/{image_id}.pdf")
def report_endpoint(image_id: str) -> Response:
    rec = _STORE.get(image_id)
    if not rec:
        raise HTTPException(404, "Unknown or expired analysis id. Re-upload the image.")
    pdf = build_pdf(rec["dict"], rec["original"], rec["saliency"])
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="neuroinsight-{image_id}.pdf"'},
    )


@app.get("/api/report/{image_id}/preview")
def report_preview(image_id: str) -> Response:
    rec = _STORE.get(image_id)
    if not rec:
        raise HTTPException(404, "Unknown analysis id")
    pdf = build_pdf(rec["dict"], rec["original"], rec["saliency"])
    return Response(pdf, media_type="application/pdf")


@app.get("/api/samples")
def samples() -> JSONResponse:
    d = WEB / "static" / "samples.json"
    if not d.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(d.read_text()))


app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")