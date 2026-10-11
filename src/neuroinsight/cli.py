"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .pipeline import MAX_BYTES, analyse
from .report.pdf import build_pdf


def cmd_serve(args) -> int:
    import uvicorn

    print("\n  NeuroInsight by Aanishka")
    print("  Turning visual stimuli into neuroscience-informed insights")
    print(f"\n  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(
        "neuroinsight.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def cmd_analyse(args) -> int:
    p = Path(args.image)
    if not p.exists():
        print(f"error: no such file: {p}", file=sys.stderr)
        return 1
    data = p.read_bytes()
    if len(data) > MAX_BYTES:
        print(f"error: {len(data)/1e6:.1f} MB exceeds the {MAX_BYTES/1e6:.0f} MB limit",
              file=sys.stderr)
        return 1

    a, img, sal = analyse(data)
    d = a.to_dict()

    print(f"\n  {p.name}  {d['dimensions']['width']}x{d['dimensions']['height']}")
    print(f"  stages: " + ", ".join(
        f"{s['name']}={s['seconds']:.2f}s" for s in d["stages"]))
    print(f"\n  {len(d['findings'])} findings "
          f"({sum(1 for f in d['findings'] if f['evidence']=='measurement')} measured, "
          f"{sum(1 for f in d['findings'] if f['evidence']=='model_output')} predicted)\n")
    for f in d["findings"]:
        print(f"    [{f['evidence']:12s}] {f['title']}")

    if args.pdf:
        out = Path(args.pdf)
        out.write_bytes(build_pdf(d, img, sal))
        print(f"\n  PDF written to {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="neuroinsight",
                                 description="NeuroInsight by Aanishka")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web application")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("analyse", help="analyse a single image from the terminal")
    a.add_argument("image")
    a.add_argument("--pdf", help="also write a PDF report to this path")
    a.set_defaults(func=cmd_analyse)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())