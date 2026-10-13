"""FastAPI backend for NeuroInsight.

Serves a single-page interface plus a small JSON API. Analysis is synchronous and
fast enough (roughly 1–3 s for a typical ad) to run inside a normal request, but we
still cap concurrency so a burst of uploads cannot exhaust the 17 GB of RAM here.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import pipeline
from ..report.pdf import build_pdf
from ..vision import saliency

def _find_web_root() -> Path:
    """Locate the repo's `web/` directory, which holds the HTML/CSS/JS.

    The obvious `parents[3]` works for a source checkout and for `pip install -e .`,
    but silently breaks for a normal (non-editable) install: the package then lives in
    site-packages, so parents[3] is some directory inside the interpreter rather than
    the repo, and every page 404s with a confusing "directory not found" from
    StaticFiles instead of an obvious error.

    Rather than trust the install mode, check the plausible locations in order and
    return the first one that actually contains static/index.html. If none do, raise
    immediately and say so - a clear failure beats a blank page in production.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "web",              # source checkout / editable install
        here.parents[2] / "web",
        here.parents[1] / "web",
        Path.cwd() / "web",                    # container WORKDIR /app
        Path("/app/web"),                      # Dockerfile layout
    ]
    for c in candidates:
        try:
            if (c / "static" / "index.html").is_file():
                return c
        except OSError:
            continue
    raise RuntimeError(
        "Could not locate the web/ directory. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Run the app from the repository root, or set NEUROINSIGHT_WEB to the "
          "directory containing static/index.html."
    )


WEB = Path(os.environ["NEUROINSIGHT_WEB"]).resolve() if os.environ.get(
    "NEUROINSIGHT_WEB"
) else _find_web_root()

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


@app.get("/gazelab", response_class=HTMLResponse)
def gazelab() -> str:
    """Webcam gaze-tracking lab. Everything runs client-side; nothing is uploaded."""
    return (WEB / "static" / "gazelab.html").read_text()


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


@app.get("/api/analysis/{image_id}")
def analysis_endpoint(image_id: str) -> dict:
    """Fetch a stored analysis by id, enabling permalinks (/?id=<id>).

    The store is in-memory, so a permalink only survives until the server restarts.
    """
    rec = _STORE.get(image_id)
    if not rec:
        raise HTTPException(404, "Unknown or expired analysis id. Re-upload the image.")
    d = dict(rec["dict"])
    d["images"] = {
        "original": f"/api/image/{image_id}/original",
        "saliency": f"/api/image/{image_id}/saliency",
        "overlay": f"/api/image/{image_id}/overlay",
        "report": f"/api/report/{image_id}.pdf",
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
        # rec["saliency"] is float32 in [0,1]. cv2.imencode expects uint8/uint16, so
        # encoding it raw silently yields a BLACK image (values truncated to 0).
        # Scale to 8-bit first, then colour-map with the same turbo scale the overlay
        # uses, so the two panels are visually comparable.
        u8 = np.clip(rec["saliency"] * 255.0, 0, 255).astype(np.uint8)
        return Response(
            _png(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)), media_type="image/png"
        )
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


# MediaPipe's runtime fetches its .wasm with streaming compilation, which browsers
# only accept when the response is typed application/wasm. Without this the tracker
# fails with an opaque "CompileError" rather than an obvious MIME-type complaint.
# Starlette resolves mimetypes through the stdlib, so we register them there.
import mimetypes as _mimetypes

_mimetypes.add_type("application/wasm", ".wasm")
_mimetypes.add_type("application/octet-stream", ".task")

app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")