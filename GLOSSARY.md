# Glossary

Terms as they come up, explained when they matter.

**Saliency map** — a prediction of *where a person will likely look first*, computed
from the image alone. Not eye-tracking.

**Measured gaze** — actual eye movements recorded with an eye-tracker while a real
person views a stimulus. NeuroInsight does **not** produce this.

**Asymmetric (chiral) carbon** — a carbon bonded to four *different* groups. Makes a
molecule chiral (non-superimposable on its mirror image).

**Bottom-up attention** — attention captured automatically by stimulus properties
(contrast, motion, onset). Exogenous.

**Top-down attention** — attention directed by goals and expectation. Endogenous.

**Feature-integration theory** — Treisman & Gelade (1980): the visual system combines
separate features (colour, orientation, motion) into coherent objects. The basis for
why a high-contrast element pops out.

**Conjunction search** — finding a target defined by a *combination* of features.
Much slower than feature search, and requires focused rather than pop-out attention.

**WCAG contrast ratio** — a standardised accessibility metric (WCAG 2.1 §1.4.3)
comparing relative luminance of two regions. ≥4.5:1 for normal text, ≥7:1 for AAA.

**Relative luminance** — perceptual brightness computed from linearised sRGB with
coefficients 0.2126 R + 0.7152 G + 0.0722 B.

**Provenance (in this project)** — the epistemic status of a claim. Which of
measurement / model output / literature / hypothesis / unverifiable.

**Vitreal / dorsal / ventral streams** — visual pathways; ventral ("what") handles
object and face identity, dorsal ("where/how") handles spatial relationships and action.

**fMRI** — functional MRI. Measures blood-oxygen response as a proxy for neural
activity. NeuroInsight does **not** do this and cannot.

**Dorsal-stream bias in design** — why high-contrast text is predicted to attract
attention before any semantic processing occurs.

**Epistemic humility** — stating the boundary of what you know. The organising
principle of this project.

---

## Gaze Lab terms

**Gaze estimation** — inferring where someone is looking from an image of their eyes.
A webcam gives *direction*, not precision. Calibrated, typical error is a few degrees
of visual angle; this is far coarser than a research eye-tracker at 1000 Hz.

**9-point calibration** — the user looks at nine known screen positions so the software
can learn the mapping from raw eye appearance to screen position. Without it, webcam
gaze is close to unusable for region-level analysis.

**I-DT (dispersion-threshold identification)** — the standard way to split a continuous
gaze stream into *fixations* (looking) and *saccades* (jumping). This project uses a
3.5 % dispersion threshold and a 200 ms minimum dwell.

**Fixation** — a sustained look, typically 200–400 ms. Its centre is the closest thing
to "where they were looking", and its position carries most of the meaning.

**Saccade** — a rapid, ballistic jump between fixations. Not seen as visual detail;
the brain suppresses perception during them.

**Dwell time** — total time fixations land inside one area of interest. The most
robust single measure of where attention went.

**First-fixation latency** — time from stimulus onset to the first fixation in a
region. Indicates salience and priority.

**Scale invariance** — a measurement that does not change when the face gets bigger or
smaller. Achieved here by expressing iris displacement in units of eye width, so leaning
toward the camera does not move the reported gaze point.

**Areas of interest (AOIs)** — defined regions of the stimulus (a headline, a logo, a
button) used to attribute fixations to meaningful parts of a design.

**Implicated, not measured** — the framing this project uses for everything brain-related.
Eye movements and attention share neural circuitry, so a pattern of looking legitimately
*suggests* which systems are recruited. It is not an observation of anyone's neurons.

**Frontal eye fields (FEF, BA8)** — frontal cortex that initiates voluntary saccades.
Recruited by rapid, goal-directed eye movements.

**Posterior parietal / dorsal attention network** — encodes where targets are and guides
the eyes to them. Damage causes optic ataxia, which is the cleanest evidence that it is
doing spatial orienting rather than something vaguer.

**Fusiform gyrus** — ventral temporal cortex with preferential responses to faces. One
of the few regions with a well-replicated category preference.

**Ventral attention network (temporoparietal junction)** — reorients attention to salient
but *unexpected* events; largely right-lateralised.

**Dorsal frontoparietal control network** — goal-directed, top-down attentional control.

**WebGL / Three.js** — the browser technology behind the 3D brain. The brain is a
schematic built from primitives, labelled as such; it is not a registered MRI mesh.
