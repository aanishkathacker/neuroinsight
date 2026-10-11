# Evaluation

## Status: not yet established

Milestone 1 demonstrates that the pipeline **runs**. It does **not** demonstrate that
the saliency predictions are *accurate*. Those are different claims and we keep them
separate.

## What has been verified so far

| Property | Method | Result |
|:---|:---|:---|
| Pipeline completes | Timed end-to-end run | 3.07 s/image, 10/10 stages `ok` |
| Sequences are correct | Byte comparison vs source | N/A (no training involved) |
| OCR detects known text | Synthetic image with known strings | 3/3 blocks, correct geometry |
| Contrast metric is correct | Hand-computed WCAG value | matches reference |
| Error handling | Invalid type, unknown id, oversize | 415 / 404 / 413 as designed |
| PDF renders | Opened and inspected | 5 pages, images and badges present |

## What has NOT been verified

- **Saliency accuracy.** No comparison against real fixation data yet.
- **Face detection accuracy.** No ground truth; only smoke-tested.
- **Cross-method agreement.** Only one method (spectral residual) is implemented.

## Planned protocol (Milestone 3)

1. Assemble a labelled stimulus set with published fixation maps
   (e.g. MIT1003/SALICON).
2. Compute predicted saliency for each stimulus.
3. Score against ground-truth fixations using:
   - **AUC** (area under the ROC of predicted-vs-actual fixation locations)
   - **CC** (correlation coefficient between maps)
   - **NSS** (normalised scanpath salience — the standard saliency metric)
4. Report the **baseline metrics of the currently published methods** on the same data
   as a reference point, so our number is interpretable rather than impressive-looking.
5. Repeat for the deep model in M3 and compare directly.

## Honest note

A high AUC here would mean the implementation matches the literature — not that
the method predicts *this* audience. Individual differences in attention are large
and a fixation dataset from 2005 does not represent a 2026 consumer. Predicted
attention remains a hypothesis until tested on the population that matters.
