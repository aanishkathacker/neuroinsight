# LEARNING_LOG

Written incrementally as the project develops. Each entry records what was built, what
was verified, what broke, and what that teaches — including my own mistakes.

---

## Milestone 1 — Core pipeline (image → report)

FastAPI service, 10-stage pipeline, saliency via spectral residual + fine-grained
(Hou & Zhang 2007), OCR, findings with explicit provenance, 5-page PDF. 3.07 s/image.

**Learned:** separating *measured* from *predicted* from *hypothesis* from *literature*
should be a data structure (`provenance.Evidence`), not a paragraph of prose. Once it
is typed, the system can *refuse* to emit an unsupported finding instead of relying on
my discipline.

---

## Milestone 3a — Evaluation (MIT1003)

1,003 images benchmarked. AUC 0.532, CC 0.208, NSS 2.243. R analysis for distribution,
correlation and figures.

**Learned:** reporting an AUC *worse than the published range* is more valuable than
quoting a flattering number. The honest framing — "we are not comparable to canonical
scores because our ground truth is the authors' pre-blurred map, not raw coordinates" —
is what makes the rest of the project trustworthy.

---

## Milestone 5 — Citation integrity audit

Not a feature: a reckoning. `neuroinsight.verify_refs` resolves every DOI against
Crossref and compares the returned title to what we claim it is.

```
python -m neuroinsight.verify_refs              # live: 12 Python + 10 browser citations
python -m neuroinsight.verify_refs --offline    # structural only, no network
```

Result: 12/12 Python references and 10/10 Gaze Lab references now resolve to the paper
they name. Before this audit, 6 of 8 Python references failed to resolve and one pointed
at an unrelated paper. See Bug 7 below for the full account — it is the most important
entry in this log.

## Milestone 4 — Gaze Lab (webcam eye tracking)

Built the camera path, 9-point calibration, real fixation/saccade segmentation, AOI
metrics, and a 3D brain. Kept honest framing throughout: **gaze is measured, brain
activity is not**.

### Decisions worth recording

**Node.js was missing.** Installing it was not incidental — it is the only practical way
to unit-test browser code outside a browser, and it caught three bugs in the gaze engine
before a single frame was ever rendered.

**A demo session is a feature, not a test hack.** Anyone without a webcam — or unwilling
to grant camera access — can still see what the output looks like. It also gave me a way
to render-test the results view headlessly. The catch: synthetic data must be loudly
labelled, so the banner is a blocking element, and the flag is written into any exported
JSON so a file can never be misread later.

**Refused the "brain regions light up" framing as literally asked.** Eye movements
imply attentional networks; they do not measure them. The mapping from gaze pattern to
network is literature-grounded and every row carries citations — but it is labelled
*implicated, not measured* everywhere it appears. The alternative was a product that
looks more impressive and is worth less.

### Bugs found, and what each teaches

**1. `aoIs` vs `aois` — a case-mismatch that returned `undefined`.**
Two identifiers differing only in capital `I`. The symptom was a `TypeError` in
`computeMetrics`. *Teaches:* a language that treats `undefined` as a first-class value
(this is JS) will silently return it through several layers before failing. Grepping for
both spellings found a second copy in `setAOIs` that the stack trace never showed.

**2. Two concurrent `requestAnimationFrame` loops.**
The helper was named as if it armed a loop from scratch, but `loop()` re-arms itself
every frame, so "cancel then start" left the old chain running. Two loops meant double
the sampling rate and duplicated calibration samples. *Teaches:* self-rescheduling loops
need an idempotent starter. The bug was found by reading the code, not by running it —
the symptom (slightly noisier calibration) looks like normal noise.

**3. CSS class mismatch — `.gazelab` in the stylesheet, `gl` in the HTML.**
Every page-specific rule silently matched nothing. The worst of them was
`.step { display: none }`, so all four wizard steps rendered stacked on top of each
other. *Teaches:* a selector that matches nothing is not an error in CSS. It is a
silent no-op that looks identical to "working" if you never look closely.

**4. The `hidden` attribute trap — the most instructive bug of the project.**
The error overlay was styled `display: flex` and toggled with `el.hidden = true`.
It never hid. Author-origin styles beat the browser's built-in `[hidden] { display: none }`
**regardless of specificity** — cascade origin is checked before specificity. So a 96 %
opaque overlay sat permanently over the page, swallowing every click. The app looked
frozen but reported no errors anywhere.

*Teaches:* this is not a CSS quirk, it is a rule about the cascade, and the failure is
silent. I wrote an audit script that cross-references every `hidden` element against
every author rule that sets `display`, which caught a second instance
(`#rec-banner`) at the same time. The durable fix is the one line in `normalize.css`:
`[hidden] { display: none !important; }`. I also made the wizard set both the class and
the attribute, so visibility no longer depends on which rule wins.

**5. Wrong MediaPipe landmark indices — the worst bug, because it produced plausible
numbers.**
I used landmarks 133 and 362 as the two corners of one eye. They are in fact the two
*inner* corners, flanking the nose bridge. Every frame I measured iris displacement from
the centre of the nose. The output looked fine — plausible floats, a plausible
distribution — and was close to meaningless: the signal barely moved when the eyes moved.

*Teaches:* this is the strongest argument in the project for writing the assertion you
*think* is already true. It took a real face photo and one question — "does the iris
centroid fall inside the span of its own eye's corners?" — to expose it. Neither the
stack trace nor any smoke test would have. "It returns numbers" is not "it is correct."
It is now a permanent regression test (`/static/test_landmarks.html`) asserting exactly
that property, so the indices cannot drift apart again.

While fixing it I also made the measurement scale-invariant by dividing both axes by eye
width, so leaning toward the camera no longer shifts the reported gaze point. Verified:
zero drift (1.2e-15) across a 0.7×–1.6× change in apparent face size.

**6. WebGL canvas sized to its parent, not itself.**
`BrainView.resize()` read `parentElement.clientHeight`, but the canvas is a fixed 340 px
tall inside a much taller card. The renderer allocated a buffer taller than the canvas,
so the brain rendered off-centre with its lower half cut off. *Teaches:* in the DOM,
`clientHeight` of a parent says nothing about a child's box. Resize observers should
observe and measure the same element.

### Bug 7 — five fabricated citations, in the module whose whole job is citation integrity

The worst find of the project, and the one I am most glad I checked.

`literature.py` opens with: *"We never generate a citation from memory alone."* Then I
resolved every DOI against Crossref. **Six of eight failed to resolve at all**, and the
two that resolved included one pointing at a completely different paper.

| Entry | Problem |
|:---|:---|
| `heng2012` → key was Hou & Zhang **2007**; DOI `…383289` resolved to *"Symmetric Objects are Hardly Ambiguous"* (Aggarwal et al.) | wrong key **and** wrong DOI |
| `itzel2017` | no such paper is findable; appears fabricated |
| `koehler2016` "Saliency of natural scenes revisited" | no such paper; the real Koehler saliency work is *"What do saliency models predict?"* (2014) |
| `swellens2022` | no trace of this paper or these authors in this literature |
| `zhou2015` | container-title search for that journal+year returned nothing |
| `bellantoni2008` | no trace of this paper |

The Gaze Lab's network table had the same disease: *"Tian & McAlpine (2016)"*,
*"Saul Stemmler (2011)"*, *"Swellens et al. (2022)"* and *"Chun et al. (2008)"* — none
verifiable.

*Teaches:* the most dangerous failure mode in this project is not a crash, it is a
citation that looks completely plausible. These entries have real-sounding authors,
real journals, correctly-formatted DOIs in the right shape. Nothing in the code flags
them. I wrote them into a project whose entire premise is refusing to fabricate
citations, and I did it silently.

The fix was not to delete the uncomfortable entries quietly — it was to **look each one
up and replace it**. Twelve references now resolve to the paper they name, verified by
`neuroinsight.verify_refs`, which resolves every DOI against Crossref and compares the
returned title. The test was validated by planting the original bad DOI and confirming
it still fails.

A lesson about my own process: while *building the verifier* I guessed DOIs from memory
(`10.1038/nrb755` for Corbetta & Shulman, `10.1523/JNEUROSCI.17-18-09662` for Kanwisher)
and **both were wrong** — the real ones are `nrn755` and `17-11-04302`. I introduced
exactly the error I was hunting, in the tool meant to catch it. Every DOI in this
project was obtained by searching Crossref and reading the result.

Two verifier bugs found while validating the verifier itself:
- Tokenising the whole APA citation against Crossref's title-only response compared
  author surnames and venue words against a title, so correct references looked wrong.
- Symmetric overlap then failed on subtitles, because Crossref stores *"Dorsal and
  Ventral Attention Systems"* without the subtitle we keep. Containment is the right
  test: is everything Crossref claims present in our text?

### Bug 8 — `never_claim()` bypasses the invariant it exists to protect

`Finding.make` raises if a HYPOTHESIS or UNVERIFIABLE finding has no reference — that is
the module's central guarantee. `never_claim()` returned a plain dict, so all three
"Cannot be determined: …" findings shipped with empty references, skipping the one check
that mattered most, in the one place it mattered most.

Not fixed by forcing citations onto them. A boundary statement ("this cannot be
determined from a static image") is a claim about *our own capability*, not about the
world; bolting a literature citation onto it would be an unearned appeal to authority —
the exact move this project refuses. The checkable requirement is that `needs` is
non-empty, because that is the reasoning a reader can check.

### What I got wrong on the first pass

- I wrote `np.linalg.lstsq` into browser JavaScript, where numpy does not exist. I then
  hand-rolled a normal-equations solver instead of checking whether the calibration even
  needed one.
- Two of my own tests failed for test-harness reasons, not product reasons
  (`t._last = 0` is falsy, collapsing every timestamp to zero; a regex matching the word
  "measurable" in a citation that was not claiming a measurement). Worth separating
  carefully — it is easy to "fix" working code to satisfy a broken test.
- I over-engineered a binocular outlier-rejection scheme that would have *increased*
  noise by discarding the averaging that reduces it. Removed.

### Honest limitations, carried forward

- Webcam gaze is meaningfully less accurate than a research eye-tracker. Usable for
  *which region*, not for precise fixation coordinates.
- The 3D brain is a schematic built from primitives, labelled as such on screen. It is
  not a registered anatomical mesh.
- No gaze data has been collected from real participants yet. Every number in the demo
  session is synthetic, and the demo path exists partly so that this stays visibly true.