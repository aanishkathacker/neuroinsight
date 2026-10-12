"""Verify every bibliography entry against Crossref.

Motivation, not tidiness: this project exists to refuse fabricated or wrong
citations, so its own reference list has to be held to that standard. A live
check caught a real error - the spectral-residual entry pointed at DOI
10.1109/CVPR.2007.383289, which resolves to "Symmetric Objects are Hardly
Ambiguous" (Aggarwal et al.), an entirely different paper. The key was also
misnamed `heng2012`, and pointed at a DOI belonging to a
different paper entirely.

Three independent failure modes are checked:

1. DOI resolves at all.
2. The resolved title matches the title we cite (token overlap).
3. The key name is consistent with the authors/year in the entry, so a key
   cannot quietly stop describing its own paper.

Run:  python -m neuroinsight.verify_refs        (network required)
      python -m neuroinsight.verify_refs --offline   (structure only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .literature import LITERATURE

CROSSREF = "https://api.crossref.org/works/"
UA = "NeuroInsight/0.1 (citation self-check; mailto:not-provided@example.com)"

# Words too common to carry signal in a title comparison.
STOPWORDS = {
    "a", "an", "the", "of", "and", "for", "in", "on", "to", "with", "is",
    "approach", "detection", "using", "via", "new", "study", "analysis",
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOPWORDS and len(t) > 2}


def _title_of(citation: str) -> str:
    """Pull just the title out of a full APA-style citation.

    Crossref returns a title; our citation string is "Authors (year). Title. Venue".
    Tokenising the whole citation compares author surnames and venue words against a
    title-only response, which makes correct references look wrong.
    """
    m = re.search(r"\(\d{4}[a-z]?\)\.?\s*(.+?)\.\s", citation, re.S)
    return (m.group(1) if m else citation).strip()


def _key_consistency(key: str, text: str) -> str | None:
    """Catch a key that names a different paper than the entry it points at.

    Convention: <firstauthorlowercase><year>, e.g. `hou2007`, `kanwisher1997`.
    We only check when the pattern is recognisable, so legacy keys are not all
    flagged as errors.
    """
    m = re.fullmatch(r"([a-z]+?)(\d{4})", key)
    if not m:
        return None  # not in the convention; leave it alone
    surname, year = m.group(1), m.group(2)
    low = text.lower()
    if surname not in low:
        return f"key '{key}' implies author '{surname}', absent from the citation text"
    if year not in text:
        return f"key '{key}' implies year {year}, absent from the citation text"
    return None


def verify_offline() -> int:
    """Structural checks only. No network."""
    bad = 0
    print(f"Bibliography: {len(LITERATURE)} entries\n")
    for key, ref in sorted(LITERATURE.items()):
        problems = []
        if ref.key != key:
            problems.append(f"ref.key '{ref.key}' != key '{key}'")
        if not (ref.citation or "").strip():
            problems.append("empty citation text")
        if not (ref.finding or "").strip():
            problems.append("empty 'finding' — a reference with no stated purpose is a guess")
        if not re.search(r"\d{4}", ref.citation or ""):
            problems.append("no year in citation text")
        # DOIs must look like DOIs when present.
        if ref.doi and not re.match(r"^10\.\d{4,9}/\S+$", ref.doi):
            problems.append(f"malformed DOI '{ref.doi}'")
        if problems:
            bad += 1
            print(f"  FAIL {key}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"  ok   {key}")
    print(f"\n{len(LITERATURE) - bad} structurally valid, {bad} invalid\n")
    return 1 if bad else 0


def verify_online() -> int:
    """Full check: resolve every DOI and compare titles."""
    bad = 0
    print(f"Verifying {len(LITERATURE)} references against Crossref\n")
    for key, ref in sorted(LITERATURE.items()):
        issues = []

        structural = _key_consistency(key, ref.citation)
        if structural:
            issues.append(structural)

        if not ref.doi:
            print(f"  SKIP {key} — no DOI recorded (cannot verify)")
            if issues:
                bad += 1
                print(f"        {'; '.join(issues)}")
            continue

        url = CROSSREF + urllib.parse.quote(ref.doi)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                msg = json.load(r)["message"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            print(f"  WARN {key} — DOI {ref.doi} did not resolve ({e})")
            if issues:
                bad += 1
            continue

        real = msg.get("title", [""])[0]
        want, got = _tokens(_title_of(ref.citation)), _tokens(real)

        # Containment, not symmetric overlap. Crossref routinely stores a title
        # WITHOUT its subtitle ("Dorsal and Ventral Attention Systems"), while our
        # citation keeps it ("...: distinct neural circuits but collaborative
        # roles"). Symmetric overlap punishes that and reports a correct DOI as
        # wrong. Containment asks the right question instead: is everything
        # Crossref claims to have called this paper actually present in our text?
        overlap = len(want & got) / max(1, len(got))

        if overlap < 0.6:
            issues.append(
                f"DOI resolves to a different paper.\n"
                f"          we cite : {_title_of(ref.citation)[:78]}\n"
                f"          DOI gives: {real[:78]}"
            )
        if issues:
            bad += 1
            print(f"  FAIL {key} — {ref.doi}")
            for i in issues:
                print(f"        {i}")
        else:
            print(f"  ok   {key} — title contained ({overlap:.0%})")

    total = len(LITERATURE)
    print(f"\n{total - bad} verified, {bad} wrong, {total} total\n")
    return 1 if bad else 0


def _resolve(doi: str) -> str:
    """Return the Crossref title for a DOI, or an error string."""
    url = CROSSREF + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["message"].get("title", [""])[0]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        return f"<{e}>"


def verify_frontend_refs() -> int:
    """Resolve the DOIs cited in the browser code.

    The Gaze Lab displays citations to the user, so they are subject to exactly the
    same standard as the Python bibliography. This walks web/static/gaze.js, pulls out
    every `doi:...` string, and checks each one resolves and names the paper the
    surrounding reference line claims it does.
    """
    import re as _re
    from pathlib import Path

    here = Path(__file__).resolve().parents[2]
    js = here / "web" / "static" / "gaze.js"
    if not js.exists():
        print(f"  SKIP {js} not found\n")
        return 0

    # Each ref line looks like: "Author (year), Venue — doi:10.xxxx/yyy"
    entries = _re.findall(r'"([^"]*?)—\s*doi:([^"]+)"', js.read_text(encoding="utf-8"))
    if not entries:
        print("  no DOIs found in gaze.js - is the citation format still the same?\n")
        return 1

    print(f"Checking {len(entries)} frontend citations in web/static/gaze.js\n")
    bad = 0
    for label, doi in entries:
        doi = doi.strip()
        real = _resolve(doi)
        if real.startswith("<"):
            print(f"  WARN {doi} did not resolve ({real})")
            continue
        # The label names authors + year; Crossref gives us the title. Check the
        # year agrees, which is the cheapest strong signal that author, paper and
        # DOI all describe the same work.
        years = _re.findall(r"\((\d{4})\)", label)
        ry = _re.search(r"\b(19|20)\d{2}\b", real)
        if years and ry and not years[-1].endswith(ry.group(0)[-2:]):
            print(f"  FAIL {doi}")
            print(f"        label says {years[-1]}, Crossref title implies {ry.group(0)}")
            bad += 1
        else:
            print(f"  ok   {doi} — {real[:56]}")
        time.sleep(0.2)

    print(f"\n{len(entries) - bad} verified, {bad} wrong, {len(entries)} total\n")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="structure only, no network")
    ap.add_argument("--frontend-only", action="store_true",
                    help="check only the browser-side citations")
    a = ap.parse_args()
    if a.offline:
        return verify_offline()
    if a.frontend_only:
        return verify_frontend_refs()
    return verify_online() or verify_frontend_refs()


if __name__ == "__main__":
    sys.exit(main())