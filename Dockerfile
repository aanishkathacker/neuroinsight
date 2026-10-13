# NeuroInsight — container image
#
# Render, Railway and Fly all build this directly. The important detail here is
# tesseract-ocr: `pytesseract` is only a Python wrapper around the Tesseract
# BINARY, which is not part of any Python image. Without the apt package, OCR
# silently degrades to "unavailable" on every Linux host - which is exactly where
# this app would be deployed - and the text-analysis half of the product goes dark.
# Apple Vision, the preferred engine, is macOS-only.
#
# Build:  docker build -t neuroinsight .
# Run:    docker run -p 8000:8000 -e PORT=8000 neuroinsight

FROM python:3.13-slim

# Tesseract for OCR. libglib2.0-0 / libgl1 are needed by some OpenCV paths even
# in the headless build; curl is used by the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      libglib2.0-0 \
      libgl1 \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Fail fast on a broken dependency rather than at first request.
RUN python -c "import cv2, numpy, fastapi, reportlab; print('deps ok', cv2.__version__)"

WORKDIR /app

# src/ is small and pyproject declares no build-time codegen, so there is no layer-caching
# win to be had from splitting metadata from source here - COPY the tree in one go.
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web

# Regular install (not -e): see the CMD note below for why.
RUN pip install --no-cache-dir .

# Unprivileged runtime user. Nothing here writes to the image.
RUN useradd --create-home --uid 10001 neuro \
 && chown -R neuro:neuro /app
USER neuro

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Results live in memory only, so the container is stateless and scales
    # horizontally without a shared store.
    NEUROINSIGHT_PORT=8000

EXPOSE 8000

# The app serves its own /health endpoint; Render polls it during rollout.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Not an editable install on purpose. `pip install -e .` makes the package resolve to
# /app/src, which is what the source tree expects, but it also means the venv depends on
# the build tree staying put. A regular install copies the package into site-packages,
# so the app locates web/ through its own search plus this explicit env var. Verify at
# build time so a bad image fails here rather than serving 404s in production.
ENV NEUROINSIGHT_WEB=/app/web
RUN python -c "from neuroinsight.web.app import WEB; \
assert (WEB/'static'/'index.html').is_file(), f'web assets missing at {WEB}'; \
print('web assets ok:', WEB)"

CMD ["sh", "-c", "python -m uvicorn neuroinsight.web.app:app --host 0.0.0.0 --port ${PORT}"]