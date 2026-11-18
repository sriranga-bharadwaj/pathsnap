// PathSnap evaluator UI (Laptop 2).
// Contains NO model, NO data and NO ground truth: it only calls the API on Laptop 1
//   GET /trips              -> list of trip IDs
//   GET /predict/{trip_id}  -> GPS points, predicted route, metrics, events
(function () {
  "use strict";
  const CFG = window.PATHSNAP_CONFIG || {};
  const LOW = CFG.LOW_CONFIDENCE ?? 0.5;
  const $ = (id) => document.getElementById(id);

  // ---------- API base URL (config.js default, editable box, remembered per browser) ----------
  let saved = null;
  try { saved = localStorage.getItem("pathsnap_api"); } catch (e) { /* storage blocked: ignore */ }
  $("apiUrl").value = saved || CFG.API_BASE_URL || "http://127.0.0.1:8000";
  $("lowConf").textContent = LOW;
  const apiBase = () => $("apiUrl").value.trim().replace(/\/+$/, "");

  function setStatus(el, text, cls) {
    el.textContent = text;
    el.className = "status" + (cls ? " " + cls : "") + (el.id === "basemapStatus" ? " small" : "");
  }

  async function getJSON(path, timeoutMs) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs || 15000);
    try {
      const r = await fetch(apiBase() + path, { signal: ctrl.signal });
      let body = null;
      try { body = await r.json(); } catch (e) { /* non-JSON body */ }
      if (!r.ok) throw new Error((body && body.message) || ("HTTP " + r.status));
      return body;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("request timed out");
      if (e instanceof TypeError) throw new Error("API unreachable at " + apiBase() +
        " (is Laptop 1 running run_api, same Wi-Fi, firewall port 8000 open?)");
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  // ---------- map (basemap optional; vectors draw on a blank map when offline) ----------
  const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
  let tileErrors = 0;
  const basemap = L.tileLayer(CFG.BASEMAP_URL || "", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" });
  basemap.on("tileerror", () => {
    tileErrors += 1;
    if (tileErrors === 1) setStatus($("basemapStatus"), "Basemap unavailable (offline?) - drawing on a blank map.", "err");
  });
  basemap.on("load", () => { if (!tileErrors) setStatus($("basemapStatus"), "Basemap loaded.", "ok"); });
  if (CFG.BASEMAP_URL) basemap.addTo(map);
  $("basemapChk").checked = !!CFG.BASEMAP_URL;
  $("basemapChk").addEventListener("change", (e) => {
    if (e.target.checked) { tileErrors = 0; basemap.addTo(map); } else { map.removeLayer(basemap); setStatus($("basemapStatus"), "Basemap hidden."); }
  });

  // predicted route is drawn in its own pane ABOVE the GPS points so it stays visible
  map.createPane("routePane");
  map.getPane("routePane").style.zIndex = 450;
  const routeRenderer = L.svg({ pane: "routePane" });
  const layers = { gps: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map) };
  L.control.layers(null, { "Predicted route": layers.route, "GPS points": layers.gps }, { collapsed: false }).addTo(map);

  // ---------- trips ----------
  async function connect() {
    try { localStorage.setItem("pathsnap_api", apiBase()); } catch (e) { /* ignore */ }
    setStatus($("apiStatus"), "Connecting to " + apiBase() + " ...");
    $("tripList").innerHTML = "";
    try {
      const h = await getJSON("/health");
      const t = await getJSON("/trips");
      setStatus($("apiStatus"), `Connected: model ${h.model}${h.tuned ? " (tuned)" : ""}, ${t.count} trips.`, "ok");
      $("tripCount").textContent = "(" + t.count + ")";
      t.trips.forEach((id) => {
        const li = document.createElement("li");
        li.textContent = id;
        li.title = "Run the prediction pipeline on Laptop 1 for trip " + id;
        li.addEventListener("click", () => predict(id, li));
        $("tripList").appendChild(li);
      });
      if (!t.count) setStatus($("apiStatus"), "Connected, but the API found no trips in its DATA_DIR.", "err");
    } catch (e) {
      setStatus($("apiStatus"), "Error: " + e.message, "err");
    }
  }

  const fmt = (v, pct) => (v === null || v === undefined) ? "n/a" : (pct ? (100 * v).toFixed(2) + " %" : v);

  function table(el, rows) {
    el.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }

  async function predict(id, li) {
    document.querySelectorAll("#tripList li").forEach((x) => x.classList.remove("active"));
    li.classList.add("active", "busy");
    setStatus($("apiStatus"), `Running prediction for trip ${id} on Laptop 1 ...`);
    try {
      const t0 = performance.now();
      const r = await getJSON("/predict/" + encodeURIComponent(id), CFG.PREDICT_TIMEOUT_MS || 180000);
      const ms = performance.now() - t0;
      draw(r);
      const m = r.metrics || {}, ev = r.events || {};
      $("resultBox").hidden = false;
      $("resultTrip").textContent = id;
      table($("metricsTable"), [
        ["Top-1 accuracy (server-side)", fmt(m.top1_accuracy, true)],
        ["Route continuity", fmt(m.route_continuity, true)],
        ["Candidate miss rate", fmt(m.candidate_miss_rate, true)],
        ["Inference time (API)", fmt(m.inference_time_ms) + " ms"],
        ["Round trip (this browser)", ms.toFixed(0) + " ms"],
        ["GPS points", r.gps_points.length],
        ["Model", r.model || "n/a"],
      ]);
      table($("eventsTable"), [
        ["Unmatched points", fmt(ev.unmatched_points)],
        ["Relaxed transitions", fmt(ev.relaxed_transitions)],
        ["Multi-hop transitions", fmt(ev.multi_hop_transitions)],
        ["Direct transitions", fmt(ev.direct_transitions)],
        ["Reversal transitions", fmt(ev.reversal_transitions)],
        ["Low-confidence points", fmt(ev.low_confidence_points)],
      ]);
      setStatus($("apiStatus"), `Trip ${id}: status ${r.status}.`, "ok");
    } catch (e) {
      setStatus($("apiStatus"), `Trip ${id}: ${e.message}`, "err");
    } finally {
      li.classList.remove("busy");
    }
  }

  function draw(r) {
    layers.gps.clearLayers();
    layers.route.clearLayers();
    (r.route_geometry || []).forEach((line) =>
      L.polyline(line, { color: "#1a9850", weight: 4, opacity: 0.9, pane: "routePane", renderer: routeRenderer })
        .addTo(layers.route));
    const preds = r.predictions || [];
    r.gps_points.forEach((p, i) => {
      const pr = preds[i] || {};
      let color = "#2166ac", radius = 2.5;
      if (pr.fallback || pr.arc_id === null) { color = "#d62728"; radius = 5; }
      else if (pr.confidence < LOW) { color = "#f08c00"; radius = 3.5; }
      L.circleMarker(p, { radius, color, weight: 0, fillOpacity: 0.7 })
        .bindTooltip(`obs ${i}<br>arc ${pr.arc_id}<br>conf ${pr.confidence}<br>${pr.transition_type}` +
                     (pr.fallback ? "<br><b>fallback</b>" : ""))
        .addTo(layers.gps);
    });
    if (r.gps_points.length) map.fitBounds(L.latLngBounds(r.gps_points), { padding: [20, 20] });
  }

  $("connectBtn").addEventListener("click", connect);
  $("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  connect();
})();
