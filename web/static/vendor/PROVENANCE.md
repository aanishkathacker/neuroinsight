# Vendored third-party assets

NeuroInsight ships these files **in the repository rather than loading them from a CDN**.
That is a deliberate choice, not an accident:

1. **Reproducibility.** A cloned repo behaves identically offline and in six months.
   A CDN URL is a single point of failure that can change or vanish without notice.
2. **Privacy.** The Gaze Lab claims that camera frames never leave the browser. Fetching
   executable code from a third-party origin at runtime would weaken that claim, since a
   compromised CDN could serve modified code into the camera pipeline.
3. **The public-repo requirement.** Anyone can run it on any OS without a network round
   trip to a vendor.

Total: ~24 MB, largest single file 9.1 MB (GitHub's per-file limit is 100 MB).

Every file below is pinned to an exact version. Do not "upgrade to latest" without
re-running `../../test_gaze.js` and `/static/test_landmarks.html`.

---

## 1. `tasks-vision.mjs` + `wasm/`

| | |
|:---|:---|
| **Name** | `@mediapipe/tasks-vision` |
| **Version** | 0.10.21 (pinned, not floating) |
| **Source** | <https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/> |
| **Licence** | Apache-2.0 (declared in `package.json`) |
| **Purpose** | Runs the FaceLandmarker graph in WebAssembly. Provides the 478-point face mesh with iris refinement. |
| **Input** | A `HTMLVideoElement`, `HTMLImageElement` or canvas. |
| **Output** | Normalised landmark array per face: `x`, `y` in `[0,1]`; `z` depth relative to the face centre. |
| **Limits** | Trained for faces at typical webcam distance and lighting. Strong performance degradation on profile views beyond ~45°, heavy occlusion, sunglasses, and low light. `z` is a *relative* depth estimate, not a metric distance. |

Why the version is pinned: `0.10.22` returns HTTP 404 on jsDelivr, so an unpinned
`@latest` range resolves to a build that cannot be fetched. That was found the hard way.

`wasm/` holds both the SIMD and non-SIMD builds. MediaPipe selects at runtime; shipping
only one would break browsers lacking WebAssembly SIMD.

---

## 2. `face_landmarker.task`

| | |
|:---|:---|
| **Name** | MediaPipe FaceLandmarker (float16) |
| **Source** | <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task> |
| **Licence** | Distributed by Google under the MediaPipe terms; the accompanying tasks-vision code is Apache-2.0. Review Google's model terms before commercial redistribution. |
| **Purpose** | The trained weights for the face mesh, including iris refinement. |
| **Input** | Raw pixel data (video frame, image or canvas). |
| **Output** | 478 3-D landmarks. Indices 468–477 are the iris ring; **468** and **473** are the two iris centroids this project uses. |
| **Limits** | Detects **faces only**. Produces no gaze, no emotion, no attention and no brain-state estimate — gaze in this project is derived from iris geometry by our own code, not predicted by this model. |

---

## 3. `three.module.js`

| | |
|:---|:---|
| **Name** | three.js |
| **Version** | 0.160.0 |
| **Source** | <https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js> |
| **Licence** | MIT — Copyright © 2010-2023 three.js authors |
| **Purpose** | Renders the 3D brain visualisation on the results screen. |
| **Input** | Scene graph of primitives (`SphereGeometry`, `TorusGeometry`, `CylinderGeometry`) constructed in `brain.js`. |
| **Output** | WebGL pixels on `<canvas id="brain-canvas">`. |
| **Limits** | Requires WebGL. The brain here is a **schematic built from primitives** — it is not a registered anatomical mesh and is labelled as such in the UI. |

---

## Fixture

`../fixture_face.jpg` is OpenCV's official `lena.jpg` sample, fetched from
<https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg>
(SHA-256 `7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07`).

It is used only by `test_landmarks.html`, as a known face with a deterministic landmark
output. It is not shipped to any end user and is not part of the product surface.