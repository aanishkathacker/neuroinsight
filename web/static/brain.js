/* 3D brain visualisation - which attentional networks are RECRUITED by this task.
 *
 * WHAT THIS SHOWS
 * ---------------
 * A schematic brain where regions light up according to which neural systems are
 * documented in the literature as being engaged by the KIND of cognitive operation
 * your gaze pattern implies (e.g. many long saccades -> frontal eye fields, which
 * initiate saccades).
 *
 * WHAT THIS IS NOT
 * ----------------
 * This is NOT a brain scan. It does not show YOUR brain activity. Nothing here is
 * derived from measuring neural signals, because no such measurement is possible from
 * a webcam. Regions are placed anatomically for orientation only; the geometry is a
 * schematic, not a registered anatomical model, and we say so on screen.
 *
 * The honest framing, used verbatim in the UI:
 *   "Implicated by this gaze pattern - not measured."
 */
import * as THREE from "./vendor/three.module.js";
import { mapToNetworks } from "./gaze.js";

// Region definitions: anatomical position (unit-sphere-ish coords), radius, colour,
// and the network id they correspond to.
const REGIONS = [
  {
    id: "fef", label: "Frontal eye fields",
    pos: [0.42, 0.30, 0.78], r: 0.20, color: 0xffb454,
    short: "Saccade initiation",
  },
  {
    id: "ppn", label: "Posterior parietal",
    pos: [-0.62, 0.12, 0.52], r: 0.24, color: 0x6fd3c4,
    short: "Spatial orienting",
  },
  {
    id: "fusiform", label: "Fusiform gyrus",
    pos: [-0.48, -0.26, 0.30], r: 0.20, color: 0xf58fb0,
    short: "Face processing",
  },
  {
    id: "van", label: "Ventral attention net.",
    pos: [0.70, -0.10, 0.18], r: 0.19, color: 0x8fb8f5,
    short: "Reorienting",
  },
  {
    id: "dorsal_exec", label: "Dorsal frontoparietal",
    pos: [0.10, 0.42, 0.66], r: 0.22, color: 0xb9e48a,
    short: "Voluntary control",
  },
];

export class BrainView {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({ autoRotate: true }, opts);
    this.active = new Set();

    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    this.camera.position.set(0, 0.1, 3.4);

    this.group = new THREE.Group();
    this.scene.add(this.group);
    this._build();

    this.clock = new THREE.Clock();
    this._animate = this._animate.bind(this);
    this._raf = requestAnimationFrame(this._animate);
    this._ro = new ResizeObserver(() => this.resize());
    // Observe the canvas itself for the reason given in resize().
    this._ro.observe(canvas);
    this.resize();
  }

  _build() {
    // Cerebrum: a squashed sphere, so it reads as a brain silhouette.
    const brainGeo = new THREE.SphereGeometry(1, 48, 32);
    brainGeo.scale(1.18, 0.86, 1.02);
    // Lighter than a true brain would be: on a near-black canvas a literal
    // anatomical tone renders as an unreadable silhouette.
    const brainMat = new THREE.MeshStandardMaterial({
      color: 0x51687f, roughness: 0.72, metalness: 0.05,
      transparent: true, opacity: 0.97,
    });
    this.brain = new THREE.Mesh(brainGeo, brainMat);
    this.group.add(this.brain);

    // Subtle longitudinal fissure so the two hemispheres read separately.
    const fissure = new THREE.Mesh(
      new THREE.TorusGeometry(1.12, 0.016, 8, 64, Math.PI),
      new THREE.MeshStandardMaterial({ color: 0x1d2836, roughness: 0.9 })
    );
    fissure.rotation.y = Math.PI / 2;
    fissure.scale.set(1, 0.86, 1);
    this.group.add(fissure);

    // Cerebellum, lower-posterior.
    const cb = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 28, 20),
      new THREE.MeshStandardMaterial({ color: 0x44596e, roughness: 0.85 })
    );
    cb.scale.set(1.1, 0.72, 0.9);
    cb.position.set(0, -0.72, -0.62);
    this.group.add(cb);

    // Brainstem.
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.14, 0.19, 0.68, 20),
      new THREE.MeshStandardMaterial({ color: 0x44596e, roughness: 0.85 })
    );
    stem.position.set(0, -0.85, -0.12);
    stem.rotation.x = 0.22;
    this.group.add(stem);

    // Lighting: ambient fill plus key and rim so the form reads without textures.
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(3, 4, 4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x8fc0ff, 1.0);
    rim.position.set(-4, 1.5, -3);
    this.scene.add(rim);

    // Region markers. Each is a solid core plus an additive halo. The halo is what
    // makes an engaged region legible at a glance against the dark canvas; emissive
    // alone barely lifts off a dark surround.
    this.regionMeshes = {};
    for (const r of REGIONS) {
      const mat = new THREE.MeshStandardMaterial({
        color: r.color, emissive: r.color,
        emissiveIntensity: 0.0, roughness: 0.3,
        transparent: true, opacity: 0.95,
      });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r.r, 32, 24), mat);
      mesh.position.set(...r.pos);
      mesh.userData = r;
      this.group.add(mesh);

      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(r.r * 1.9, 24, 18),
        new THREE.MeshBasicMaterial({
          color: r.color, transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false,
        })
      );
      halo.position.copy(mesh.position);
      mesh.add(halo);
      mesh.userData.halo = halo;

      this.regionMeshes[r.id] = mesh;
    }
  }

  /** Drive the visual from real gaze metrics. */
  update(metrics, aois = []) {
    const mapped = mapToNetworks(metrics, aois);
    const byId = Object.fromEntries(mapped.map((m) => [m.id, m]));
    this.active = new Set(mapped.filter((m) => m.engaged).map((m) => m.id));

    for (const r of REGIONS) {
      const mesh = this.regionMeshes[r.id];
      const info = byId[r.id] || { engaged: false, strength: 0 };
      mesh.userData.info = info;
      mesh.userData.target = info.engaged ? 0.35 + 0.65 * info.strength : 0.0;
    }
    this._t0 = performance.now();
    this._animStart = true;
    return mapped;
  }

  /** Accessibility: keyboard users can step through regions. */
  focusRegion(id) {
    for (const r of REGIONS) {
      this.regionMeshes[r.id].userData.focused = r.id === id;
    }
  }

  clearFocus() {
    for (const r of REGIONS) this.regionMeshes[r.id].userData.focused = false;
  }

  /**
   * Match the drawing buffer to the canvas's own CSS box.
   *
   * This must read the CANVAS, not its parent. The canvas is styled to a fixed
   * 340px tall inside a much taller results card; sizing to the parent made the
   * renderer allocate a taller buffer than the canvas, so the brain rendered
   * off-centre with its lower half cut off.
   *
   * setSize(w, h, false) leaves the CSS size alone, so the stylesheet stays in
   * control of layout and we only set the backing resolution.
   */
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width || this.canvas.clientWidth || 1));
    const h = Math.max(1, Math.round(rect.height || this.canvas.clientHeight || 1));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Keep the whole brain in frame at any aspect: narrow/tall viewports otherwise
    // crop the cerebellum and brainstem off the bottom.
    const vFov = (this.camera.fov * Math.PI) / 180;
    const fitH = 2 * Math.tan(vFov / 2);
    const fitW = fitH * this.camera.aspect;
    const need = 2.55;                       // half-extent of the brain group + margin
    this.camera.position.set(0, 0.05, Math.max(need / fitH, need / fitW));
    this.camera.lookAt(0, -0.12, 0);
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate);
    const t = this.clock.getElapsedTime();

    // Oscillate rather than spin. A continuous 360-degree rotation turns whichever
    // region is lit away from the viewer, so the most important thing on screen -
    // which networks are implicated - becomes unreadable half the time.
    if (this.opts.autoRotate && !this._dragging) {
      this.group.rotation.y = Math.sin(t * 0.32) * 0.55;
    }
    // Gentle float so it does not look static.
    this.group.position.y = Math.sin(t * 0.7) * 0.03;

    const ease = (cur, target) => cur + (target - cur) * 0.10;
    for (const r of REGIONS) {
      const m = this.regionMeshes[r.id];
      const target = m.userData.target ?? 0;
      m.material.emissiveIntensity = ease(m.material.emissiveIntensity, target);

      // Dimmed regions stay faintly visible so the viewer can see the whole set,
      // but recede hard compared to anything engaged.
      m.material.opacity = 0.30 + m.material.emissiveIntensity * 0.68;

      const halo = m.userData.halo;
      if (halo) {
        halo.material.opacity = ease(halo.material.opacity, m.material.emissiveIntensity * 0.30);
        halo.scale.setScalar(1 + m.material.emissiveIntensity * 0.22);
      }

      let s = 1 + m.material.emissiveIntensity * 0.16;
      if (m.userData.focused) s *= 1.2;
      m.scale.setScalar(ease(m.scale.x, s));
    }

    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.renderer.dispose();
  }
}

export { REGIONS };