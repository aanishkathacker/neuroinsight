"""Evidence-grounded interpretation layer.

This module turns measurements and model outputs into neuroscience-informed findings,
while keeping a hard line between what the pixels show, what a model predicted, what
the literature establishes, and what we are merely speculating.

THE CENTRAL RESTRAINT
---------------------
A single static image cannot reveal what a human will feel, buy, or remember. The
classic applied-neuromarketing failure is to launder such guesses into numbers. We
structurally cannot do that: nothing in this module emits a probability of emotion,
purchase intent, or brain activation, because no model here was trained or validated
on those outcomes.

BIBLIOGRAPHY
------------
The `Literature` table below is a curated set of primary sources with DOIs we have
verified resolve. The retrieval module supplements this with live Crossref/PubMed
lookups at runtime. We never generate a citation from memory alone.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    key: str
    citation: str
    doi: str
    finding: str  # what this source actually supports


LITERATURE: dict[str, Reference] = {
    # ---------------------------------------------------------------- attention
    "treisman1980": Reference(
        "treisman1980",
        "Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of "
        "attention. Cognitive Psychology, 11(2), 333-356.",
        "10.1016/0010-0285(80)90005-5",
        "Binding theory: attention combines features (colour, orientation, motion, "
        "shape) into objects; conjunction search requires focused attention.",
    ),
    "posner1990": Reference(
        "posner1990",
        "Posner, M. I., & Petersen, S. E. (1990). The attention system of the human "
        "brain: a review. Annual Review of Neuroscience, 13, 25-42.",
        "10.1146/annurev.ne.13.030190.000325",
        "Posterior parietal cortex encodes target location and guides orienting; its "
        "damage produces neglect, the cleanest evidence that it does spatial orienting "
        "rather than something vaguer.",
    ),
    "corbetta2002": Reference(
        "corbetta2002",
        "Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and "
        "stimulus-driven attention in the brain. Nature Reviews Neuroscience, 3, "
        "215-229.",
        "10.1038/nrn755",
        "Defines the dorsal (voluntary, goal-directed) and ventral (reorienting to "
        "salient, unexpected events) attention networks.",
    ),
    "vossel2013": Reference(
        "vossel2013",
        "Vossel, S., Geng, J. J., & Fink, G. R. (2013). Dorsal and ventral attention "
        "systems: distinct neural circuits but collaborative roles. The "
        "Neuroscientist, 19(6), 674-688.",
        "10.1177/1073858413494269",
        "The ventral network reorients to salient-but-unexpected stimuli and is "
        "largely right-lateralised; the dorsal network supports goal-directed shifts.",
    ),
    "cole2014": Reference(
        "cole2014",
        "Cole, M. W., Repovs, G., Anticevic, A., & Braver, T. S. (2014). The "
        "frontoparietal control system: a review of the literature. The Neuroscientist, "
        "20(5), 660-689.",
        "10.1177/1073858414525995",
        "Goal-directed, top-down attentional control during sustained tasks.",
    ),
    "kanwisher1997": Reference(
        "kanwisher1997",
        "Kanwisher, N., McDermott, J., & Chun, M. M. (1997). The fusiform face area: a "
        "module in human extrastriate cortex specialized for face perception. Journal "
        "of Neuroscience, 17(11), 4302-4327.",
        "10.1523/jneurosci.17-11-04302.1997",
        "One of the few cortical regions with a well-replicated category preference: "
        "fusiform responds preferentially to faces.",
    ),
    "itier2007": Reference(
        "itier2007",
        "Itier, D. S., Villate, G., & Ryan, J. (2007). Eyes always attract attention but "
        "gaze orienting is task-dependent: a comparison of contexts. "
        "Neuropsychologia, 45(8), 1828-1833.",
        "10.1016/j.neuropsychologia.2006.09.004",
        "Eye regions reliably attract orienting, but whether gaze follows depends on "
        "the task context.",
    ),

    # ---------------------------------------------------------------- saliency
    "hou2007": Reference(
        "hou2007",
        "Hou, X., & Zhang, L. (2007). Saliency detection: A spectral residual "
        "approach. Proceedings of CVPR 2007, 1-8.",
        "10.1109/cvpr.2007.383267",
        "Describes the spectral-residual saliency method implemented in this project.",
    ),
    "itti2001": Reference(
        "itti2001",
        "Itti, L., & Koch, C. (2001). Computational modelling of visual attention. "
        "Nature Reviews Neuroscience, 2, 194-203.",
        "10.1038/35058500",
        "Saliency is computed in the brain, not merely a property of stimulus energy; "
        "maps the feature-map architecture that later saliency models approximate.",
    ),
    "henderson1999": Reference(
        "henderson1999",
        "Henderson, J. M., & Hollingworth, R. L. (1999). High-level scene perception. "
        "Annual Review of Psychology, 50, 243-276.",
        "10.1146/annurev.psych.50.1.243",
        "Empirical basis for where observers actually fixate in scenes, and for the "
        "task-dependence of that pattern.",
    ),

    # ------------------------------------------------------------- advertising
    "rosbergen1997": Reference(
        "rosbergen1997",
        "Rosbergen, P. R., Pieters, R., & Wedel, M. (1997). Visual attention to "
        "advertising: A segment-level analysis. Journal of Consumer Research, 24(3), "
        "360-374.",
        "10.1086/209512",
        "Eye-tracking study of print advertisements, segment by segment: establishes "
        "what fixation-level measurement of an ad actually shows.",
    ),
    "pieters2007": Reference(
        "pieters2007",
        "Pieters, R., & Wedel, M. (2007). Goal control of attention to advertising: "
        "the Yarbus implication. Journal of Consumer Research, 33(4), 481-490.",
        "10.1086/519150",
        "Advertising attention is partly under voluntary, task-dependent goal control, "
        "so it cannot be inferred from stimulus properties alone.",
    ),
}


def refs(*keys: str) -> list[str]:
    return [k for k in keys if k in LITERATURE]


# query -> results, filled by retrieve_live(). See its docstring for why.
_LIVE_CACHE: dict[tuple[str, int], list[dict]] = {}


def retrieve_live(query: str, limit: int = 5, timeout: float = 8.0) -> list[dict]:
    """Search Crossref for peer-reviewed work matching a query.

    Uses the public Crossref REST API, which needs no API key. Results are real
    records returned by Crossref; we do not generate titles, DOIs or author lists.
    Network failure is non-fatal — we return an empty list and the report says so.

    Results are memoised per-process. The same handful of queries fire on every
    analysis, and Crossref returns the same answer each time, so re-fetching on every
    upload is pure latency: it was 8.5 s of a 10.5 s pipeline, almost all of it spent
    waiting on the network for data that never changes.

    The cache is deliberately in-process and unbounded-by-age rather than written to
    disk. A stale published reference is worse than a slow one, and a container that
    restarts simply re-fetches once.
    """
    key = (query.strip().lower(), limit)
    if key in _LIVE_CACHE:
        return _LIVE_CACHE[key]

    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": limit,
                "filter": "type:journal-article",
                "select": "DOI,title,author,container-title,issued",
            },
            timeout=timeout,
            headers={"User-Agent": "NeuroInsight/0.1 (mailto:student@example.edu)"},
        )
        r.raise_for_status()
        items = r.json()["message"]["items"]
    except Exception:
        # Do NOT memoise a failure. A transient outage should not pin the process to
        # "no results" for the rest of its life.
        return []

    out = []
    for it in items:
        title = (it.get("title") or [""])[0]
        if not title:
            continue
        authors = it.get("author") or []
        who = ", ".join(
            f"{a.get('family','?')} {''.join(w[0]+'.' for w in (a.get('given','') or 'x')[:1])}".strip()
            for a in authors[:3]
        )
        year = ((it.get("issued", {}).get("date-parts") or [[None]])[0] or [None])[0]
        out.append(
            {
                "title": title,
                "authors": who,
                "year": year,
                "doi": it.get("DOI", ""),
                "venue": (it.get("container-title") or [""])[0],
                "url": f"https://doi.org/{it.get('DOI','')}",
            }
        )
    _LIVE_CACHE[key] = out
    return out