/* M.AX dataset editor - frame-accurate viewer + segment labeller. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const S = {
  dataset: null, detail: null, episodes: [],
  epId: null, ep: null, edits: null,
  frame: 0, numFrames: 0, fps: 30,
  selSeg: -1, dirty: false, playing: false, videos: [],
  loadToken: 0, autoplay: true,
  hiddenGroups: new Set(),
  subtasks: {}, objects: [], targets: [],
  drag: null,
};

/* ------------------------------------------------------------------ toast */
let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), 1800);
}

function setDirty(on) {
  S.dirty = on;
  $("dirtyFlag").textContent = on ? "● unsaved" : "";
}

/* --------------------------------------------------------------- datasets */
async function loadDatasets() {
  const { root, datasets, default_dataset } = await api("/api/datasets");
  $("rootPath").value = root;
  const sel = $("dsSelect");
  sel.innerHTML = datasets
    .map((d) => `<option value="${d.name}">${d.name} (${d.num_episodes})</option>`)
    .join("");
  if (!datasets.length) return toast("no datasets found");
  const preferred = S.dataset
    || (datasets.some((d) => d.name === default_dataset) ? default_dataset : null)
    || datasets[0].name;
  sel.value = preferred;
  await loadDataset(sel.value);
}

async function loadDataset(name) {
  S.dataset = name;
  S.detail = await api(`/api/datasets/${encodeURIComponent(name)}`);
  S.episodes = S.detail.episodes;
  $("fVocab").value = (S.detail.label_vocabulary || []).join(", ");
  refreshVocabList();
  const labeled = S.episodes.filter((e) => e.num_segments > 0).length;
  const rej = S.episodes.filter((e) => e.rejected).length;
  const frames = S.episodes.reduce((a, e) => a + e.kept_frames, 0);
  $("dsStats").textContent =
    `${S.episodes.length} eps · ${labeled} labeled · ${rej} rejected · ${frames} frames kept`;
  renderEpList();
  if (S.episodes.length) await selectEpisode(S.episodes[0].id);
}

function renderEpList() {
  const q = $("epFilter").value.trim().toLowerCase();
  const items = S.episodes.filter((e) => {
    if (!q) return true;
    if (q === "unlabeled") return e.num_segments === 0;
    if (q === "rejected") return e.rejected;
    return e.id.includes(q) || (e.main_prompt || "").toLowerCase().includes(q);
  });
  $("epList").innerHTML = items
    .map((e) => {
      const badge = e.rejected
        ? '<span class="badge rej">rej</span>'
        : e.num_segments
        ? `<span class="badge ok">${e.num_segments} seg</span>`
        : '<span class="badge">–</span>';
      const score = Number(e.score ?? DEFAULT_SCORE).toFixed(2);
      return `<div class="ep ${e.id === S.epId ? "active" : ""}" data-id="${e.id}">
        <div class="t"><span>${e.id.replace("episode_", "ep ")}</span>${badge}</div>
        <div class="s">${e.kept_frames}/${e.num_frames}f · score ${score}</div></div>`;
    })
    .join("");
  [...document.querySelectorAll(".ep")].forEach((el) =>
    el.onclick = () => selectEpisode(el.dataset.id)
  );
}

/* --------------------------------------------------------------- episode */
async function selectEpisode(id) {
  if (S.dirty && !confirm("Unsaved changes will be lost. Continue?")) return;
  S.epId = id;
  const url = `/api/datasets/${encodeURIComponent(S.dataset)}/episodes/${id}`;
  S.ep = await api(url);
  S.edits = S.ep.edits;
  S.fps = S.ep.fps || 30;
  S.numFrames = S.ep.series.num_frames || 0;
  S.frame = S.edits.trim?.start || 0;
  S.selSeg = -1;
  setDirty(false);
  $("epTitle").textContent = id;
  if (S.ep.series.error) toast("timeseries unavailable: " + S.ep.series.error);
  buildVideos();
  fillForm();
  renderSegments();
  renderKeyframes();
  renderEpList();
  renderChannelToggles();
  drawPlot();
  updateFrameLabel();
}

function mediaUrl(file, kind) {
  return `/api/datasets/${encodeURIComponent(S.dataset)}/episodes/${S.epId}/${kind}/${file}`;
}

/** Detach in-flight media loads before the elements are discarded.
 *  Replacing innerHTML alone leaves the browser aborting a partial download,
 *  which then fires `error` on the dead element and looks like a codec fault. */
function teardownVideos() {
  S.videos.forEach((v) => {
    v.onerror = v.ontimeupdate = v.onended = v.onloadeddata = null;
    try {
      v.pause();
      v.removeAttribute("src");
      v.load();          // cancels the pending fetch
    } catch (_) { /* element already gone */ }
  });
  S.videos = [];
}

function buildVideos() {
  const token = ++S.loadToken;   // stale callbacks from a previous episode are ignored
  teardownVideos();
  $("videos").innerHTML = S.ep.cameras
    .map(
      (c) => `<div class="vwrap"><label>${c}</label>
      <video data-cam="${c}" preload="auto" muted playsinline></video></div>`
    )
    .join("");
  S.videos = [...document.querySelectorAll("#videos video")];

  let ready = 0;
  S.videos.forEach((v) => {
    v.onerror = () => {
      if (token !== S.loadToken) return;        // superseded: not a real failure
      const err = v.error;
      // MEDIA_ERR_ABORTED(1)/NETWORK(2) happen on fast episode switching.
      if (!err || err.code === 1 || err.code === 2) return;
      toast(`${v.dataset.cam}: 영상 재생 실패 (${err.message || "decode error"})`);
    };
    v.onloadeddata = () => {
      if (token !== S.loadToken) return;
      v.currentTime = S.frame / S.fps;
      if (++ready === S.videos.length && S.autoplay) setPlaying(true);
    };
    // Assign src only after the handlers exist, so nothing is missed.
    v.src = mediaUrl(v.dataset.cam + ".mp4", "video");
  });

  // The first camera drives the shared playhead; the rest follow it.
  if (S.videos[0]) {
    S.videos[0].ontimeupdate = () => {
      if (!S.playing || token !== S.loadToken) return;
      S.frame = clamp(Math.round(S.videos[0].currentTime * S.fps), 0, S.numFrames - 1);
      syncFollowers();
      if (!selectAtPlayhead()) drawPlot();
      updateFrameLabel();
    };
    S.videos[0].onended = () => setPlaying(false);
  }
}

function syncFollowers() {
  const t = S.frame / S.fps;
  for (let i = 1; i < S.videos.length; i++) {
    if (Math.abs(S.videos[i].currentTime - t) > 0.12) S.videos[i].currentTime = t;
  }
}

function seekTo(frame) {
  S.frame = clamp(Math.round(frame), 0, Math.max(0, S.numFrames - 1));
  const t = S.frame / S.fps;
  S.videos.forEach((v) => { v.currentTime = t; });
  updateFrameLabel();
  // renderHighlight() redraws, so only draw here when nothing was selected.
  if (!selectAtPlayhead()) drawPlot();
}

function updateFrameLabel() {
  $("frameLabel").textContent =
    `f ${S.frame} / ${S.numFrames}  (${(S.frame / S.fps).toFixed(2)}s)`;
}

function setPlaying(on) {
  S.playing = on;
  const rate = parseFloat($("rate").value) || 1;
  S.videos.forEach((v) => { v.playbackRate = rate; });
  if (on) S.videos.forEach((v) => v.play().catch(() => {}));
  else S.videos.forEach((v) => v.pause());
  $("btnPlay").textContent = on ? "⏸ Pause" : "▶ Play";
}

function renderKeyframes() {
  $("keyframes").innerHTML = S.ep.keyframes
    .map((k) => `<a href="${mediaUrl(k, "file")}" target="_blank" style="color:var(--accent);margin-right:8px">${k}</a>`)
    .join("");
}

/* ------------------------------------------------------------------ form */
function fillForm() {
  $("fPrompt").value = S.edits.main_prompt || "";
  $("fScore").value = S.edits.score ?? DEFAULT_SCORE;
  $("fReject").checked = !!S.edits.rejected;
  $("fNotes").value = S.edits.notes || "";
  $("fTrimA").value = S.edits.trim?.start ?? 0;
  $("fTrimB").value = S.edits.trim?.end ?? S.numFrames;
  syncTrimToSegments();
  applyTrimAutoState();
}

/** Grey out the trim boxes while they are being derived, so it is obvious
 *  why typing in them does nothing. */
function applyTrimAutoState() {
  const auto = $("fTrimAuto").checked;
  $("fTrimA").disabled = auto;
  $("fTrimB").disabled = auto;
  $("trimHint").textContent = auto ? "= segment 범위" : "";
}

/** Trim actually saved: derived from the segments when auto is on. */
function trimRange() {
  const list = segs();
  if ($("fTrimAuto").checked && list.length) {
    return { start: Math.min(...list.map((s) => s.start)),
             end: Math.max(...list.map((s) => s.end)) };
  }
  return {
    start: parseInt($("fTrimA").value || "0", 10),
    end: parseInt($("fTrimB").value || String(S.numFrames), 10),
  };
}

function collectForm() {
  const score = $("fScore").value;
  return {
    main_prompt: $("fPrompt").value,
    notes: $("fNotes").value,
    // Empty means unrated, which is DEFAULT_SCORE -- same rule as segments.
    score: score === "" ? DEFAULT_SCORE : round2(parseFloat(score)),
    rejected: $("fReject").checked,
    trim: trimRange(),
    segments: S.edits.segments || [],
  };
}

async function save() {
  if (!S.epId) return;
  const body = collectForm();
  // `score: null` is meaningful (clear it), so send the payload as-is.
  const url = `/api/datasets/${encodeURIComponent(S.dataset)}/episodes/${S.epId}/edits`;
  const saved = await api(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  S.edits = saved;
  setDirty(false);
  fillForm();
  renderSegments();
  drawPlot();
  const row = S.episodes.find((e) => e.id === S.epId);
  if (row) {
    row.main_prompt = saved.main_prompt;
    row.score = saved.score;
    row.rejected = saved.rejected;
    row.num_segments = (saved.segments || []).length;
    row.kept_frames = Math.max(0, (saved.trim?.end ?? 0) - (saved.trim?.start ?? 0));
    renderEpList();
  }
  toast("saved");
}

/* -------------------------------------------------------------- segments */
/* Mirrors DEFAULT_SCORE in store.py: an unrated episode or subtask counts as
   good, so only a downgrade needs typing. */
const DEFAULT_SCORE = 1;
/* Coarse enough that the arrow keys walk 1 -> .8 -> .6 -> .4 -> .2 -> 0. */
const SCORE_STEP = 0.2;

/** Stepping by 0.2 produces values like 0.6000000000000001; keep two decimals. */
const round2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : v);

function segs() {
  if (!S.edits.segments) S.edits.segments = [];
  return S.edits.segments;
}

function addSegment(start, end) {
  segs().push({ start, end, label: "", prompt: "", score: DEFAULT_SCORE });
  segs().sort((a, b) => a.start - b.start);
  S.selSeg = segs().findIndex((s) => s.start === start && s.end === end);
  setDirty(true);
  renderSegments();
  drawPlot();
}

function splitAtPlayhead() {
  const i = segs().findIndex((s) => S.frame > s.start && S.frame < s.end);
  if (i < 0) return toast("playhead is not inside a segment");
  const s = segs()[i];
  const tail = { ...s, start: S.frame };
  s.end = S.frame;
  segs().splice(i + 1, 0, tail);
  setDirty(true);
  renderSegments();
  drawPlot();
}

function renderSegments() {
  const list = segs();
  // Every segment mutation ends here, so this is the one place the derived
  // trim needs to be recomputed.
  syncTrimToSegments();
  $("segCount").textContent = list.length ? `(${list.length})` : "";
  if (!list.length) {
    $("segList").innerHTML = '<div class="hint">No segments. Use Set In/Out or Auto-segment.</div>';
    return;
  }
  $("segList").innerHTML = list
    .map((s, i) => {
      const dur = ((s.end - s.start) / S.fps).toFixed(2);
      // The numeric row is labelled: in a half-width pane three bare number
      // boxes are impossible to tell apart.
      return `<div class="seg ${i === S.selSeg ? "sel" : ""}" data-i="${i}">
        <div class="hd"><span>#${i + 1} · ${dur}s</span>
          <span><button data-act="go" data-i="${i}" title="seek here">↦</button>
                <button data-act="del" data-i="${i}" class="danger" title="delete">×</button></span></div>
        <input data-f="label" data-i="${i}" list="vocabList" placeholder="subtask label" value="${esc(s.label)}">
        <input data-f="prompt" data-i="${i}" placeholder="subtask prompt (optional)" value="${esc(s.prompt)}" style="margin-top:4px">
        <div class="segnums">
          <label>score<input data-f="score" data-i="${i}" type="number" min="0" max="1" step="${SCORE_STEP}"
                 placeholder="${DEFAULT_SCORE}" value="${s.score ?? DEFAULT_SCORE}"></label>
          <label>start<input data-f="start" data-i="${i}" type="number" value="${s.start}"></label>
          <label>end<input data-f="end" data-i="${i}" type="number" value="${s.end}"></label>
        </div></div>`;
    })
    .join("");

  $("segList").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      const i = +b.dataset.i;
      if (b.dataset.act === "del") {
        segs().splice(i, 1);
        S.selSeg = -1;
      } else {
        S.selSeg = i;
        seekTo(segs()[i].start);
      }
      setDirty(true);
      renderSegments();
      drawPlot();
    };
  });
  $("segList").querySelectorAll("input").forEach((inp) => {
    inp.onfocus = () => { S.selSeg = +inp.dataset.i; renderHighlight(); };
    inp.onchange = () => {
      const s = segs()[+inp.dataset.i];
      const f = inp.dataset.f;
      // Blank means "unscored" -- the server stores DEFAULT_SCORE, so
      // show that rather than leaving the field looking empty-but-saved.
      if (f === "score") {
        const raw = inp.value === "" ? DEFAULT_SCORE : parseFloat(inp.value);
        s.score = Number.isFinite(raw) ? round2(raw) : DEFAULT_SCORE;
        inp.value = s.score;
      }
      else if (f === "start" || f === "end") s[f] = clamp(parseInt(inp.value || "0", 10), 0, S.numFrames);
      else s[f] = inp.value;
      setDirty(true);
      drawPlot();
    };
  });
}

function renderHighlight(scroll = false) {
  $("segList").querySelectorAll(".seg").forEach((el) => {
    const on = +el.dataset.i === S.selSeg;
    el.classList.toggle("sel", on);
    // Auto-selection is pointless if the card is off-screen, but never yank
    // the list while the user is typing in one of its inputs.
    if (on && scroll && !el.contains(document.activeElement)) {
      el.scrollIntoView({ block: "nearest" });
    }
  });
  drawPlot();
}

const esc = (s) => String(s || "").replace(/"/g, "&quot;").replace(/</g, "&lt;");

/* -------------------------------------------------------------- subtasks */
/* `subtasks` in editor_config.yaml maps a short label key to a prompt
   template over {object}/{target}. Applying one fills BOTH segment fields:
   the key becomes `label` (stable for stats/filtering) and the expanded
   sentence becomes `prompt` (what a VLA model is conditioned on). */
async function loadSubtasks() {
  try {
    const r = await api("/api/subtasks");
    S.subtasks = r.subtasks || {};
    S.objects = r.objects || [];
    S.targets = r.targets || [];
  } catch (_) { /* older server: the free-text vocabulary still works */ }
  fillSlot("fObject", S.objects);
  fillSlot("fTarget", S.targets);
  renderSubtaskButtons();
}

function fillSlot(id, items) {
  $(id).innerHTML = items.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  $(id).disabled = !items.length;
}

function subtaskKeys() {
  return Object.keys(S.subtasks);
}

/** Template with {object}/{target} replaced by the current dropdown picks. */
function expandSubtask(key) {
  const tpl = S.subtasks[key];
  if (!tpl) return "";
  return tpl
    .replace(/{object}/g, $("fObject").value || "{object}")
    .replace(/{target}/g, $("fTarget").value || "{target}");
}

function renderSubtaskButtons() {
  const keys = subtaskKeys();
  const el = $("subtaskBtns");
  if (!keys.length) {
    el.innerHTML = '<div class="hint">No subtasks defined in editor_config.yaml.</div>';
    $("slotRow").style.display = "none";
    return;
  }
  // Hide a slot nothing references, so the panel only shows real choices.
  const used = (field) => keys.some((k) => S.subtasks[k].includes(`{${field}}`));
  $("fObject").parentElement.style.display = used("object") ? "" : "none";
  $("fTarget").parentElement.style.display = used("target") ? "" : "none";

  el.innerHTML = keys
    .map((k, i) => `<button data-k="${esc(k)}">${i < 9 ? `<span class="k">${i + 1}</span>` : ""}${esc(k)}
        <span class="p">${esc(expandSubtask(k))}</span></button>`)
    .join("");
  el.querySelectorAll("button").forEach((b) => {
    b.onclick = () => applySubtask(b.dataset.k);
  });
}

function applySubtask(key) {
  if (!(key in S.subtasks)) return;
  if (S.selSeg < 0) return toast("select a segment first");
  const seg = segs()[S.selSeg];
  seg.label = key;
  seg.prompt = expandSubtask(key);
  setDirty(true);
  renderSegments();
  drawPlot();
  toast(`#${S.selSeg + 1} = ${seg.prompt}`);
}

/* ------------------------------------------------------------------ plot */
const GROUP_PALETTE = ["#5aa9ff", "#4ec98a", "#e5b567", "#e5697a", "#7c6bd8",
                       "#54c4cf", "#d98cc4", "#9dc45a", "#e08a4b", "#6fa8dc", "#c47ee0"];
const MIN_ROW_H = 34;  // absolute floor; rows grow to fill the box above this
const PLOT_TOP = 10;
const PLOT_PAD = 22;   // room under the last row for stale markers
const SEG_LABEL_PX = 13;  // subtask label drawn centred on each segment
const SEG_LABEL_TOP = 30; // its offset below PLOT_TOP

/** Channels bundled by their `group`, honouring the hidden-group set. */
function plotGroups() {
  const chans = (S.ep && S.ep.series.channels) || [];
  const order = [];
  const byName = new Map();
  chans.forEach((c) => {
    const name = c.group || c.label;
    if (!byName.has(name)) { byName.set(name, []); order.push(name); }
    byName.get(name).push(c);
  });
  return order.map((name, i) => ({
    name,
    color: GROUP_PALETTE[i % GROUP_PALETTE.length],
    channels: byName.get(name),
    hidden: S.hiddenGroups.has(name),
  }));
}

/** Row height that makes every visible group fill the container exactly.
 *  Falls back to MIN_ROW_H (and a scrollbar) only when the window is too
 *  short to give each group a legible row. */
function rowHeight() {
  const visible = Math.max(1, plotGroups().filter((gr) => !gr.hidden).length);
  const box = $("plotScroll");
  const avail = (box ? box.clientHeight : 0) - PLOT_TOP - PLOT_PAD;
  if (avail <= 0) return MIN_ROW_H;
  return Math.max(MIN_ROW_H, Math.floor(avail / visible));
}

function plotHeight(rowH) {
  const visible = Math.max(1, plotGroups().filter((gr) => !gr.hidden).length);
  return PLOT_TOP + visible * rowH + PLOT_PAD;
}

function drawPlot() {
  const cv = $("plot");
  const w = cv.clientWidth;
  const rowH = rowHeight();
  const h = plotHeight(rowH);
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr;
    cv.height = h * dpr;
    cv.style.height = h + "px";
  }
  const box = $("plotScroll");
  if (box) box.style.overflowY = h > box.clientHeight + 1 ? "auto" : "hidden";
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (!S.ep || !S.numFrames) return;

  const x = (f) => (f / S.numFrames) * w;
  const groups = plotGroups().filter((gr) => !gr.hidden);
  const H = Math.max(1, groups.length) * rowH;

  // trimmed-away regions
  const tr = S.edits.trim || { start: 0, end: S.numFrames };
  g.fillStyle = "rgba(0,0,0,.45)";
  g.fillRect(0, PLOT_TOP, x(tr.start), H);
  g.fillRect(x(tr.end), PLOT_TOP, w - x(tr.end), H);

  // segments
  segs().forEach((s, i) => {
    g.fillStyle = i === S.selSeg ? "rgba(90,169,255,.26)" : "rgba(124,107,216,.18)";
    g.fillRect(x(s.start), PLOT_TOP, x(s.end) - x(s.start), H);
    g.strokeStyle = i === S.selSeg ? "#5aa9ff" : "#7c6bd8";
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(x(s.start), PLOT_TOP); g.lineTo(x(s.start), PLOT_TOP + H); g.stroke();
    g.beginPath(); g.moveTo(x(s.end), PLOT_TOP); g.lineTo(x(s.end), PLOT_TOP + H); g.stroke();
    if (s.label) {
      const a = x(s.start), b = x(s.end);
      g.save();
      // Clip to the segment so a label never bleeds over its neighbours.
      g.beginPath();
      g.rect(a, PLOT_TOP, b - a, H);
      g.clip();
      g.font = `bold ${SEG_LABEL_PX}px ui-sans-serif`;
      g.textAlign = "center";
      g.textBaseline = "top";
      // Dark halo: the label sits over plot traces of every colour, and white
      // alone disappears against the light ones.
      g.lineWidth = 3;
      g.strokeStyle = "rgba(0,0,0,.75)";
      g.strokeText(s.label, (a + b) / 2, PLOT_TOP + SEG_LABEL_TOP);
      g.fillStyle = "#fff";
      g.fillText(s.label, (a + b) / 2, PLOT_TOP + SEG_LABEL_TOP);
      g.restore();
    }
  });

  // One row per group. Components of a group share a y-scale so joint1..7
  // stay directly comparable; brightness separates them within the row.
  groups.forEach((gr, gi) => {
    const y0 = PLOT_TOP + gi * rowH;
    const lo = Math.min(...gr.channels.map((c) => c.min));
    const hi = Math.max(...gr.channels.map((c) => c.max));
    const span = hi - lo || 1;

    g.strokeStyle = "rgba(255,255,255,.07)";
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, y0); g.lineTo(w, y0); g.stroke();

    const head = Math.max(14, Math.round(rowH * 0.26));   // label gutter
    gr.channels.forEach((c, ci) => {
      g.strokeStyle = gr.color;
      g.globalAlpha = gr.channels.length === 1 ? 1 : 0.45 + 0.55 * (ci / (gr.channels.length - 1));
      g.lineWidth = rowH >= 90 ? 1.5 : 1;
      g.beginPath();
      c.values.forEach((v, i) => {
        if (v == null) return;
        const px = (i / (c.values.length - 1 || 1)) * w;
        const py = y0 + rowH - 4 - ((v - lo) / span) * (rowH - head - 4);
        i === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
      });
      g.stroke();
    });
    g.globalAlpha = 1;

    g.fillStyle = gr.color;
    const fs = Math.max(11, Math.min(15, Math.round(rowH * 0.22)));
    g.font = `${fs}px ui-sans-serif`;
    const n = gr.channels.length;
    g.fillText(`${gr.name}${n > 1 ? ` (${n})` : ""}  [${lo.toFixed(3)}, ${hi.toFixed(3)}]`,
               4, y0 + fs + 1);
  });

  // stale frame markers
  g.fillStyle = "#e5697a";
  (S.ep.series.stale_frames || []).forEach((f) => g.fillRect(x(f), PLOT_TOP + H, 1, 6));

  // playhead
  g.strokeStyle = "#fff";
  g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(x(S.frame), PLOT_TOP - 6); g.lineTo(x(S.frame), PLOT_TOP + H + 8); g.stroke();
}

/** Per-group show/hide chips under the timeline. */
function renderChannelToggles() {
  const groups = plotGroups();
  const el = $("chanToggles");
  if (!groups.length) { el.innerHTML = ""; return; }
  el.innerHTML = groups
    .map((gr) => `<button class="chip ${gr.hidden ? "off" : ""}" data-g="${esc(gr.name)}"
        style="--c:${gr.color}">${esc(gr.name)}${gr.channels.length > 1 ? ` ${gr.channels.length}` : ""}</button>`)
    .join("") +
    `<button class="chip" data-g="__all">all</button><button class="chip" data-g="__none">none</button>`;
  el.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      const name = b.dataset.g;
      if (name === "__all") S.hiddenGroups.clear();
      else if (name === "__none") plotGroups().forEach((gr) => S.hiddenGroups.add(gr.name));
      else if (S.hiddenGroups.has(name)) S.hiddenGroups.delete(name);
      else S.hiddenGroups.add(name);
      renderChannelToggles();
      drawPlot();
    };
  });
}

/* ------------------------------------------------------------------ trim */
/* With "trim to segments" on, the kept range is exactly the labelled span:
   start = first segment's start, end = last segment's end. Everything before
   the first subtask and after the last is lead-in/lead-out the build should
   drop. Turn the checkbox off to trim by hand. */
function syncTrimToSegments() {
  if (!$("fTrimAuto").checked) return false;
  const list = segs();
  if (!list.length) return false;   // nothing to derive from; leave trim alone
  const start = Math.min(...list.map((s) => s.start));
  const end = Math.max(...list.map((s) => s.end));
  const cur = S.edits.trim || {};
  if (cur.start === start && cur.end === end) return false;
  S.edits.trim = { start, end };
  $("fTrimA").value = start;
  $("fTrimB").value = end;
  return true;
}

/** Re-derive the trim and mark the episode dirty if it actually moved. */
function refreshTrim() {
  if (syncTrimToSegments()) setDirty(true);
}

/* ------------------------------------------------- segment selection */
/** Index of the segment containing `frame`, or -1. Later segments win on an
 *  exact boundary so a playhead sitting on a shared edge selects the one it
 *  is entering, which is what you want while scrubbing forwards. */
function segmentAtFrame(frame) {
  const list = segs();
  for (let i = list.length - 1; i >= 0; i--) {
    if (frame >= list[i].start && frame <= list[i].end) return i;
  }
  return -1;
}

/** Follow the playhead with the selection. Leaving every segment keeps the
 *  last one selected, so the subtask buttons and nudges stay aimed at
 *  something while you scrub just outside its edge. */
function selectAtPlayhead() {
  const i = segmentAtFrame(S.frame);
  if (i < 0 || i === S.selSeg) return false;
  S.selSeg = i;
  renderHighlight(true);
  return true;
}

/* ------------------------------------------- segment boundary dragging */
/* Grabbing within GRAB_PX of a segment edge resizes it; anywhere else the
   click just seeks, so scrubbing keeps working as before. */
const GRAB_PX = 6;
const MIN_SEG_FRAMES = 2;

function frameAtClientX(clientX) {
  const r = $("plot").getBoundingClientRect();
  return clamp(Math.round(((clientX - r.left) / r.width) * S.numFrames), 0, S.numFrames);
}

/** Nearest segment edge under the cursor, or null. */
function edgeAt(clientX) {
  if (!S.numFrames) return null;
  const r = $("plot").getBoundingClientRect();
  const px = (f) => (f / S.numFrames) * r.width;
  const cx = clientX - r.left;
  let best = null;
  segs().forEach((seg, i) => {
    for (const side of ["start", "end"]) {
      const d = Math.abs(px(seg[side]) - cx);
      if (d <= GRAB_PX && (!best || d < best.dist)) best = { i, side, dist: d };
    }
  });
  return best;
}

/** Segment sharing this boundary, for ripple drags. Exact match only --
 *  a gap between segments is intentional and must not be closed silently. */
function neighbourAt(i, side, frame) {
  const list = segs();
  const j = side === "start" ? i - 1 : i + 1;
  const other = side === "start" ? "end" : "start";
  if (j < 0 || j >= list.length || list[j][other] !== frame) return null;
  return { j, other };
}

/** How far one edge of segment `i` may travel.
 *
 *  Bounded by its own opposite edge (a segment keeps MIN_SEG_FRAMES) and by
 *  the nearest neighbour on that side, so a drag can meet a neighbour exactly
 *  but never overlap it. Under a ripple the neighbour is being pushed rather
 *  than run into, so the limit comes from the far side of that neighbour.
 */
function dragBounds(i, side, rippling) {
  const list = segs();
  const seg = list[i];
  if (side === "start") {
    const nb = list[i - 1];
    // Rippling: we drag the neighbour's end along, so we may go as far as
    // its own start allows. Otherwise we must stop at its end.
    const floor = !nb ? 0
      : rippling ? nb.start + MIN_SEG_FRAMES
      : nb.end;
    return { lo: floor, hi: seg.end - MIN_SEG_FRAMES };
  }
  const nb = list[i + 1];
  const ceil = !nb ? S.numFrames
    : rippling ? nb.end - MIN_SEG_FRAMES
    : nb.start;
  return { lo: seg.start + MIN_SEG_FRAMES, hi: ceil };
}

function applyDrag(frame) {
  const d = S.drag;
  const seg = segs()[d.i];
  const { lo, hi } = dragBounds(d.i, d.side, !!d.link);
  // A neighbour already flush against this one leaves no room at all; clamp
  // order keeps `lo` winning so the edge parks on the boundary instead of
  // jumping past it.
  const f = clamp(frame, lo, Math.max(lo, hi));
  seg[d.side] = f;
  if (d.link) segs()[d.link.j][d.link.other] = f;
  S.selSeg = d.i;
  setDirty(true);
  drawPlot();
}

$("plot").addEventListener("mousedown", (e) => {
  if (!S.numFrames) return;
  const hit = edgeAt(e.clientX);
  if (!hit) return;   // plain scrub -- handled by the click listener
  e.preventDefault();
  const frame = segs()[hit.i][hit.side];
  S.drag = {
    i: hit.i,
    side: hit.side,
    link: $("fRipple").checked ? neighbourAt(hit.i, hit.side, frame) : null,
  };
  document.body.style.cursor = "ew-resize";
});

window.addEventListener("mousemove", (e) => {
  if (S.drag) return applyDrag(frameAtClientX(e.clientX));
  // Hover feedback so the grab zone is discoverable.
  $("plot").style.cursor = edgeAt(e.clientX) ? "ew-resize" : "crosshair";
});

window.addEventListener("mouseup", () => {
  if (!S.drag) return;
  S.drag = null;
  document.body.style.cursor = "";
  // No re-sort: dragBounds() keeps every edge inside its neighbours, so the
  // order is already correct and S.selSeg stays pointing at the same segment.
  renderSegments();
  drawPlot();
});

$("plot").addEventListener("click", (e) => {
  if (!S.numFrames || S.drag) return;
  if (edgeAt(e.clientX)) return;   // that was a resize, not a seek
  const r = e.currentTarget.getBoundingClientRect();
  const frame = ((e.clientX - r.left) / r.width) * S.numFrames;
  // Select whatever was clicked, including a re-click on the segment the
  // playhead already sits in -- seekTo() alone would treat that as no change.
  const i = segmentAtFrame(Math.round(frame));
  if (i >= 0) S.selSeg = i;
  seekTo(frame);
  renderHighlight(true);
});

/** Move one edge of the selected segment by `delta` frames. */
function nudge(side, delta) {
  if (S.selSeg < 0) return toast("select a segment first");
  const seg = segs()[S.selSeg];
  S.drag = {
    i: S.selSeg, side,
    link: $("fRipple").checked ? neighbourAt(S.selSeg, side, seg[side]) : null,
  };
  applyDrag(seg[side] + delta);
  S.drag = null;
  renderSegments();
}

$("btnGrowA").onclick = () => nudge("start", -1);
$("btnShrinkA").onclick = () => nudge("start", 1);
$("btnShrinkB").onclick = () => nudge("end", -1);
$("btnGrowB").onclick = () => nudge("end", 1);

/* ------------------------------------------------------------- shortcuts */
document.addEventListener("keydown", async (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
  if (e.ctrlKey && e.key.toLowerCase() === "s") { e.preventDefault(); return save(); }
  if (typing) return;
  const jump = e.shiftKey ? 30 : 10;
  switch (e.key) {
    case "j": seekTo(S.frame - 1); break;
    case "l": seekTo(S.frame + 1); break;
    case "k": case " ": e.preventDefault(); setPlaying(!S.playing); break;
    case "ArrowLeft": seekTo(S.frame - jump); break;
    case "ArrowRight": seekTo(S.frame + jump); break;
    case "i": markIn(); break;
    case "o": markOut(); break;
    case "s": splitAtPlayhead(); break;
    // [ / ] move the segment start; shift+[ / ] move the end.
    case "[": nudge("start", -1); break;
    case "]": nudge("start", 1); break;
    case "{": nudge("end", -1); break;
    case "}": nudge("end", 1); break;
    case "Delete": case "Backspace":
      if (S.selSeg >= 0) { segs().splice(S.selSeg, 1); S.selSeg = -1; setDirty(true); renderSegments(); drawPlot(); }
      break;
    default:
      if (/^[1-9]$/.test(e.key)) {
        const keys = subtaskKeys();
        if (keys.length) applySubtask(keys[+e.key - 1]);
        else applyVocab(+e.key - 1);
      }
  }
});

let pendingIn = null;
function markIn() { pendingIn = S.frame; toast(`in = ${S.frame}`); }
function markOut() {
  if (pendingIn == null) return toast("set In first [I]");
  const a = Math.min(pendingIn, S.frame), b = Math.max(pendingIn, S.frame);
  if (b - a < 2) return toast("segment too short");
  addSegment(a, b);
  pendingIn = null;
}

function vocab() {
  return $("fVocab").value.split(",").map((s) => s.trim()).filter(Boolean);
}
function refreshVocabList() {
  $("vocabList").innerHTML = vocab().map((v) => `<option value="${esc(v)}">`).join("");
}
function applyVocab(idx) {
  const v = vocab()[idx];
  if (!v || S.selSeg < 0) return;
  segs()[S.selSeg].label = v;
  setDirty(true);
  renderSegments();
  drawPlot();
  toast(`#${S.selSeg + 1} = ${v}`);
}

/* ----------------------------------------------------------------- wiring */
$("dsSelect").onchange = (e) => loadDataset(e.target.value);

async function applyRoot() {
  const root = $("rootPath").value.trim();
  if (!root) return;
  if (S.dirty && !confirm("Unsaved changes will be lost. Continue?")) return;
  try {
    await api("/api/root", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root }),
    });
    S.dataset = null;
    setDirty(false);
    await loadDatasets();
    toast("root: " + root);
  } catch (err) {
    toast("경로 열기 실패: " + err.message);
  }
}
$("btnRoot").onclick = applyRoot;
$("rootPath").addEventListener("keydown", (e) => { if (e.key === "Enter") applyRoot(); });
$("epFilter").oninput = renderEpList;
$("btnPlay").onclick = () => setPlaying(!S.playing);
$("fAutoplay").onchange = (e) => {
  S.autoplay = e.target.checked;
  try { localStorage.setItem("maxedit.autoplay", S.autoplay ? "1" : "0"); } catch (_) { /* private mode */ }
};
$("rate").onchange = () => { const r = parseFloat($("rate").value); S.videos.forEach((v) => (v.playbackRate = r)); };
$("btnIn").onclick = markIn;
$("btnOut").onclick = markOut;
$("btnSplit").onclick = splitAtPlayhead;
$("btnSave").onclick = () => save().catch((err) => toast("save failed: " + err.message));
$("btnReload").onclick = () => { setDirty(false); selectEpisode(S.epId); };
["fPrompt", "fScore", "fReject", "fNotes", "fTrimA", "fTrimB"].forEach((id) => {
  $(id).addEventListener("input", () => { setDirty(true); if (id.startsWith("fTrim")) { S.edits.trim = collectForm().trim; drawPlot(); } });
});

$("fTrimAuto").onchange = () => {
  applyTrimAutoState();
  refreshTrim();
  drawPlot();
  try { localStorage.setItem("maxedit.trimAuto", $("fTrimAuto").checked ? "1" : "0"); }
  catch (_) { /* storage unavailable */ }
};

$("fObject").onchange = renderSubtaskButtons;
$("fTarget").onchange = renderSubtaskButtons;

$("fVocab").addEventListener("change", async () => {
  refreshVocabList();
  await api(`/api/datasets/${encodeURIComponent(S.dataset)}/vocabulary`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ labels: vocab() }),
  });
  toast("vocabulary saved");
});

$("btnSuggest").onclick = async () => {
  if (!S.epId) return;
  if (segs().length && !confirm("Replace existing segments with auto-detected ones?")) return;
  const url = `/api/datasets/${encodeURIComponent(S.dataset)}/episodes/${S.epId}/suggest`;
  try {
    const r = await api(url);
    S.edits.segments = r.segments;
    S.edits.trim = r.suggested_trim;
    fillForm();
    setDirty(true);
    renderSegments();
    drawPlot();
    toast(`${r.segments.length} segments suggested`);
  } catch (err) { toast("suggest failed: " + err.message); }
};

$("btnCopyPrev").onclick = () => {
  const i = S.episodes.findIndex((e) => e.id === S.epId);
  if (i <= 0) return toast("no previous episode");
  const prev = S.episodes[i - 1];
  api(`/api/datasets/${encodeURIComponent(S.dataset)}/episodes/${prev.id}`).then((p) => {
    S.edits.segments = JSON.parse(JSON.stringify(p.edits.segments || []));
    S.edits.main_prompt = p.edits.main_prompt;
    fillForm();
    setDirty(true);
    renderSegments();
    drawPlot();
    toast(`copied from ${prev.id}`);
  });
};

window.addEventListener("resize", drawPlot);
window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });

try {
  const saved = localStorage.getItem("maxedit.autoplay");
  if (saved !== null) S.autoplay = saved === "1";
} catch (_) { /* storage unavailable */ }
$("fAutoplay").checked = S.autoplay;
try {
  const t = localStorage.getItem("maxedit.trimAuto");
  if (t !== null) $("fTrimAuto").checked = t === "1";
} catch (_) { /* storage unavailable */ }
applyTrimAutoState();

loadSubtasks();
loadDatasets().catch((err) => toast("load failed: " + err.message));
