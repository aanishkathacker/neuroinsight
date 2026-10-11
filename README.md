# NeuroInsight by Aanishka

**Turning visual stimuli into neuroscience-informed insights.**

An AI-assisted computational neuromarketing platform. Upload an image, get a
research-grounded report on its likely influence on visual attention, perception and
consumer interpretation.

> **Status: Milestone 1 — end-to-end working prototype.**
> See `PROJECT_PLAN.md` for milestones and progress.

---

## Quick start

```bash
cd ~/Documents/projects/neuroinsight
uv sync                      # or: uv venv && uv pip install -e .
uv run neuroinsight serve    # http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000> and drop in an image.

## Requirements

- macOS on Apple Silicon (developed on M4, 17 GB RAM)
- [uv](https://docs.astral.sh/uv/) for Python 3.13
- Internet access on first run for model weights and live literature retrieval

## What it does

Computes real, measurable features from the uploaded image and combines them with
established findings from the attentional neuroscience literature. Every claim in the
report is tagged by epistemic status — see `docs/METHODS.md`.

## Licence

MIT. See `LICENSE`. Third-party models retain their own licences — see `MODEL_CARD.md`.