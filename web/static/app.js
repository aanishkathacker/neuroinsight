/* NeuroInsight frontend. No framework, no build step: the API returns a
   fully-rendered set of numbers and this file only decides how to show them. */
(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, txt) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  };
  const esc = (s) => String(s ?? "");

  const LABEL = {
    measurement: "Measured",
    model_output: "Predicted",
    literature: "Literature",
    hypothesis: "Hypothesis",
    unverifiable: "Not determinable",
  };

  // ------------------------------------------------------------------ views
  function show(name) {
    for (const v of document.querySelectorAll(".view")) {
      const on = v.id === "view-" + name;
      v.hidden = !on;
      v.classList.toggle("active", on);
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.querySelectorAll(".navbtn[data-view]").forEach((b) =>
    b.addEventListener("click", () => show(b.dataset.view))
  );
  $("#back-btn").addEventListener("click", () => show("upload"));

  // ------------------------------------------------------------------- drop
  const drop = $("#drop"), input = $("#file");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hot"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) upload(f);
  });
  input.addEventListener("change", () => {
    if (input.files?.[0]) upload(input.files[0]);
  });

  document.querySelectorAll("[data-sample]").forEach((b) =>
    b.addEventListener("click", () => loadSample(b.dataset.sample))
  );

  async function loadSample(key) {
    if (key !== "synthetic") return;
    setStatus("Loading synthetic demo image", "fetching sample");
    try {
      const blob = await fetch("/static/sample_ad.png").then((r) => r.blob());
      if (!blob.size) throw new Error("Sample image is missing from web/static/.");
      await upload(new File([blob], "sample_ad.png", { type: "image/png" }));
    } catch (err) {
      fail(err.message || String(err));
    }
  }

  // --------------------------------------------------------------- status
  function setStatus(main, sub) {
    $("#status").hidden = false;
    $("#status .status-main").textContent = main;
    $("#status-sub").textContent = sub || "";
  }
  function stopStatus() { $("#status").hidden = true; }
  function fail(msg) {
    stopStatus();
    const box = $("#error");
    box.hidden = false;
    box.innerHTML = "<strong>Could not analyse that file.</strong><br>" + esc(msg);
  }

  // ---------------------------------------------------------------- upload
  async function upload(file) {
    $("#error").hidden = true;
    setStatus("Analysing your image…", "uploading");

    const steps = [
      [700, "computing colour and contrast"],
      [1500, "detecting faces and reading text"],
      [2400, "predicting attention"],
      [3400, "retrieving literature"],
      [5200, "writing findings"],
    ];
    const t = steps.map(([ms, msg]) => setTimeout(() => {
      if (!$("#status").hidden) $("#status-sub").textContent = msg;
    }, ms));

    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/analyse", { method: "POST", body: fd });
      const ct = res.headers.get("content-type") || "";
      if (!res.ok) {
        let msg = `Server returned ${res.status}.`;
        if (ct.includes("application/json")) {
          const j = await res.json().catch(() => null);
          if (j && j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        }
        throw new Error(msg);
      }
      render(await res.json());
      show("results");
    } catch (err) {
      fail(err.message || String(err));
    } finally {
      t.forEach(clearTimeout);
      stopStatus();
      input.value = "";
    }
  }

  // ---------------------------------------------------------------- render
  function render(d) {
    $("#rdims").textContent = `${d.dimensions.width} × ${d.dimensions.height} px`;
    const total = d.stages.find((s) => s.name === "total");
    $("#rtime").textContent = total ? `analysed in ${total.seconds.toFixed(2)} s` : "";

    $("#img-original").src = d.images.original;
    $("#img-saliency").src = d.images.saliency;
    $("#img-overlay").src = d.images.overlay;
    $("#dl").href = d.images.report;

    renderSummary(d);
    renderLegend(d);
    renderFindings(d);
    renderRecs(d);
    renderRefs(d);
    renderSide(d);
    renderAbout(d);
  }

  function renderSummary(d) {
    const con = d.features.contrast, lum = d.features.luminance;
    const top = d.saliency.stats.top_region || {};
    const nMeas = d.findings.filter((f) => f.evidence === "measurement").length;
    const nPred = d.findings.filter((f) => f.evidence === "model_output").length;

    $("#exec-summary").innerHTML =
      `Measured from your image: a luminance contrast ratio of <b>${con.ratio}:1</b> ` +
      `(${esc(con.wcag_band.split(" ")[0])}), edge density of ` +
      `<b>${(lum.edge_density * 100).toFixed(1)}%</b> and mean luminance of ` +
      `<b>${lum.mean_luminance.toFixed(2)}</b>. Predicted attention peaks in region ` +
      `<b>${esc(top.cell || "-")}</b>, holding <b>${((top.share || 0) * 100).toFixed(1)}%</b> ` +
      `of the modelled attention. Face detection returned <b>${d.features.faces.count}</b>, ` +
      `OCR recovered <b>${d.features.text.block_count || 0}</b> text block(s). ` +
      `This report contains <b>${nMeas}</b> measured and <b>${nPred}</b> predicted findings.`;
  }

  function renderLegend(d) {
    const box = $("#legend");
    box.innerHTML = "";
    const present = [...new Set(d.findings.map((f) => f.evidence))];
    for (const k of present) {
      const s = el("span");
      const i = el("i");
      i.style.background = `var(--${k})`;
      s.appendChild(i);
      s.appendChild(document.createTextNode(LABEL[k] || k));
      box.appendChild(s);
    }
  }

  function renderFindings(d) {
    $("#find-count").textContent = `(${d.findings.length})`;
    const host = $("#findings");
    host.innerHTML = "";

    for (const f of d.findings) {
      const card = el("div", "finding");
      const b = el("span", "badge " + f.evidence, LABEL[f.evidence] || f.evidence);
      card.appendChild(b);
      card.appendChild(el("h4", null, f.title));
      card.appendChild(el("p", null, f.detail));

      const conf = el("p", "conf");
      conf.textContent = "Confidence: " + f.confidence;
      if (f.references && f.references.length) {
        conf.appendChild(document.createTextNode(" · refs: "));
        f.references.forEach((r, i) => {
          const s = el("span", "refs-inline", r);
          s.title = "See the reference list below";
          conf.appendChild(s);
          if (i < f.references.length - 1) conf.appendChild(document.createTextNode(", "));
        });
      }
      card.appendChild(conf);
      host.appendChild(card);
    }
  }

  function renderRecs(d) {
    const host = $("#recs");
    host.innerHTML = "";
    for (const r of d.recommendations) {
      const card = el("div", "rec p-" + r.priority);
      card.appendChild(el("span", "pri", r.priority + " priority"));
      card.appendChild(el("h4", null, r.hypothesis));
      const p = el("p");
      p.innerHTML = "<b>How to test it:</b> " + esc(r.test);
      card.appendChild(p);
      host.appendChild(card);
    }
  }

  function renderRefs(d) {
    const host = $("#refs");
    host.innerHTML = "";
    for (const r of d.literature) {
      const li = el("li");
      li.textContent = r.title;
      if (r.supports) {
        const s = el("span", "sup");
        s.textContent = r.doi ? `Supports: ${r.supports} · doi:${r.doi}`
                              : `Supports: ${r.supports}`;
        li.appendChild(s);
      }
      host.appendChild(li);
    }
  }

  function renderSide(d) {
    // palette
    const pal = $("#palette");
    pal.innerHTML = "";
    for (const p of d.palette || []) {
      const s = el("div", "sw");
      const chip = el("div", "sw-chip");
      chip.style.background = p.hex;
      chip.title = `${p.hex} — hue ${p.hue_deg}°, ${(p.share * 100).toFixed(1)}% of chromatic pixels`;
      s.appendChild(chip);
      s.appendChild(el("div", "sw-lab", (p.share * 100).toFixed(0) + "%"));
      pal.appendChild(s);
    }
    if (!(d.palette || []).length) pal.appendChild(el("p", "tiny", "No chromatic content."));

    // measured features
    const kv = $("#features");
    kv.innerHTML = "";
    const f = d.features, s = d.saliency.stats;
    const rows = [
      ["Contrast ratio", `${f.contrast.ratio}:1`],
      ["WCAG band", f.contrast.wcag_band.split(" ")[0]],
      ["Mean luminance", f.luminance.mean_luminance.toFixed(3)],
      ["RMS contrast", f.luminance.rms_contrast.toFixed(3)],
      ["Edge density", `${(f.luminance.edge_density * 100).toFixed(2)}%`],
      ["Luminance entropy", `${f.luminance.luminance_entropy_bits} bits`],
      ["Mean saturation", `${f.colour.mean_saturation}/255`],
      ["Colour bins", f.colour.distinct_colour_bins],
      ["Faces", f.faces.count],
      ["Text blocks", f.text.block_count || 0],
      ["OCR words", f.text.total_words || 0],
      ["Attention spread", s.concentration || "—"],
      ["Top-3 attention", `${((s.top3_share_of_attention || 0) * 100).toFixed(1)}%`],
    ];
    for (const [k, v] of rows) {
      const tr = el("tr");
      tr.appendChild(el("td", null, k));
      tr.appendChild(el("td", null, String(v)));
      kv.appendChild(tr);
    }

    // saliency grid
    const g = $("#salgrid");
    g.innerHTML = "";
    if (s.cells) {
      g.style.gridTemplateColumns = `repeat(${s.cells},1fr)`;
      for (let i = 0; i < s.cells * s.cells; i++) {
        const c = el("div", "salcell");
        c.style.background = `rgb(${Math.round(240 - 190 * i0(s, i))},${Math.round(244 - 120 * i0(s, i))},${Math.round(248 - 30 * i0(s, i))})`;
        g.appendChild(c);
      }
    }
    function i0(st, idx) { const r = st.regions[idx]; return r ? r.saliency : 0; }

    const top = s.top_region || {};
    $("#salnote").textContent =
      `Peak in ${top.cell || "-"} at mean luminance ${(top.luminance || 0).toFixed(0)}/255. ` +
      `Concentration: ${s.concentration || "—"}.`;

    // pipeline timings
    const st2 = $("#stages");
    st2.innerHTML = "";
    for (const st of d.stages) {
      const tr = el("tr");
      const a = el("td", null, st.name);
      if (st.status !== "ok") a.style.color = "var(--model_output)";
      tr.appendChild(a);
      tr.appendChild(el("td", null, `${st.seconds.toFixed(3)}s`));
      st2.appendChild(tr);
    }
  }

  function renderAbout(d) {
    const m = d.saliency.model || {};
    $("#about-models").textContent =
      `${m.model_name || "n/a"} — ${m.model_source || ""}. ` +
      `Licence: ${m.licence || "unknown"}. ` +
      `This is predicted attention, not measured gaze.`;

    const lims = $("#about-lims");
    lims.innerHTML = "";
    const all = [
      ...(m.known_limitations || []),
      ...(d.features.faces.limitations || []),
      ...(d.features.text.limitations || []),
      "The tool analyses images, not people. It cannot measure emotion, purchase "
        + "intent or neural activity.",
      "Predicted saliency is correlated with, but not identical to, real gaze data.",
    ];
    for (const l of [...new Set(all)]) lims.appendChild(el("li", null, l));
  }

  // ------------------------------------------------------------------ boot
  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("#foot-ver").textContent = `v${h.version} · ${h.status}`;
  }).catch(() => { $("#foot-ver").textContent = "offline"; });
})();