/* ================================================================
   SteelSight — app.js
   All API calls, canvas drawing, animations, state management
   ================================================================ */

'use strict';

const API = 'http://localhost:8000';

/* queue action state (local UI only)
   TODO: wire to backend endpoint (e.g., POST /predictions/{id}/approve or /flag) */
const queueState = {};

/* ──────────────────────────────────────────
   DOM helpers
   ────────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ──────────────────────────────────────────
   CLOCK
   ────────────────────────────────────────── */
function updateClock() {
  const el = $('current-time');
  if (!el) return;
  const s = new Date().toISOString().replace('T', '  ').substring(0, 21) + ' UTC';
  el.textContent = s;
}

/* ──────────────────────────────────────────
   TOAST NOTIFICATIONS
   ────────────────────────────────────────── */
const TOAST_ICONS = { success: '✓', warning: '⚠', error: '✕', info: '·' };

function showToast(message, type = 'success', duration = 3800) {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', 'alert');
  toast.innerHTML = `<span class="toast-icon">${TOAST_ICONS[type] ?? '·'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('hiding');
    setTimeout(() => toast.remove(), 220);
  }, duration);
}

/* ──────────────────────────────────────────
   HEALTH CHECK
   ────────────────────────────────────────── */
async function checkHealth() {
  const dot   = $('status-dot');
  const label = $('status-label');
  try {
    const res  = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (data.status === 'ok') {
      dot.className   = 'status-dot online';
      label.className = 'status-label online mono';
      label.textContent = 'ONLINE';
    } else { throw new Error('backend not ok'); }
  } catch {
    dot.className   = 'status-dot offline';
    label.className = 'status-label offline mono';
    label.textContent = 'OFFLINE';
  }
}

/* ──────────────────────────────────────────
   PREDICTIONS — KPIs + TABLE + QUEUE
   ────────────────────────────────────────── */
async function loadPredictions() {
  try {
    const res = await fetch(`${API}/predictions`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const predictions = await res.json();
    updateKPIs(predictions);
    renderActivityTable(predictions);
    renderQueue(predictions);
    return predictions;
  } catch (err) {
    console.warn('[SteelSight] Could not load predictions:', err.message);
    renderActivityTable([]);
    renderQueue([]);
    return [];
  }
}

function updateKPIs(preds) {
  const total     = preds.length;
  const defective = preds.filter(p => p.detections?.length > 0).length;
  const uncertain = preds.filter(p => p.is_uncertain).length;
  const rate      = total > 0 ? ((defective / total) * 100).toFixed(1) : '0.0';

  $('kpi-inspected').textContent    = total;
  $('kpi-defect-rate').textContent  = `${rate}%`;
  $('kpi-uncertain').textContent    = uncertain;

  $('kpi-uncertain-card').style.borderColor =
    uncertain > 0 ? 'rgba(232,162,61,0.5)' : '';
}

/* ──────────────────────────────────────────
   RECENT ACTIVITY TABLE
   ────────────────────────────────────────── */
function formatTs(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toISOString().replace('T', ' ').substring(0, 19);
  } catch { return ts; }
}

function renderActivityTable(preds) {
  const tbody = $('activity-body');
  const recent = preds.slice(0, 10);

  if (!recent.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:1.75rem;color:var(--text-3);font-size:0.82rem">
      No activity yet — upload an image to begin.
    </td></tr>`;
    return;
  }

  tbody.innerHTML = recent.map(p => {
    const dets    = p.detections ?? [];
    const topDet  = dets.length ? dets.reduce((a, b) => a.confidence > b.confidence ? a : b) : null;
    const cls     = topDet ? topDet.class_name : '—';
    const conf    = topDet ? (topDet.confidence * 100).toFixed(1) + '%' : '—';
    const status  = p.is_uncertain ? 'review' : (dets.length ? 'pass' : 'none');
    const sLabel  = p.is_uncertain ? 'NEEDS REVIEW' : (dets.length ? 'PASS' : 'NO DEFECT');

    return `<tr>
      <td class="mono" style="font-size:0.75rem;white-space:nowrap;color:var(--text-1)">${formatTs(p.timestamp)}</td>
      <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-1);font-size:0.82rem">${p.image_filename}</td>
      <td style="font-family:var(--font-h);font-size:0.82rem;font-weight:600">${cls}</td>
      <td class="mono" style="font-size:0.82rem">${conf}</td>
      <td><span class="status-pill ${status}">${sLabel}</span></td>
    </tr>`;
  }).join('');
}

/* ──────────────────────────────────────────
   REVIEW QUEUE
   ────────────────────────────────────────── */
function renderQueue(preds) {
  const uncertain = preds.filter(p => p.is_uncertain);
  const container = $('queue-list');
  $('queue-count').textContent = uncertain.length;

  if (!uncertain.length) {
    container.innerHTML = `<div class="queue-empty">
      <strong>Queue is clear</strong>
      No items awaiting review — all recent scans were high-confidence.
    </div>`;
    return;
  }

  container.innerHTML = uncertain.slice(0, 20).map(p => {
    const id      = p.id;
    const dets    = p.detections ?? [];
    const topDet  = dets.length ? dets.reduce((a, b) => a.confidence > b.confidence ? a : b) : null;
    const cls     = topDet ? topDet.class_name : 'No detection';
    const conf    = topDet ? (topDet.confidence * 100).toFixed(1) + '%' : '—';
    const st      = queueState[id] ?? {};

    return `<div class="queue-card" id="qcard-${id}">
      <div class="q-filename">${p.image_filename}</div>
      <div class="q-ts">${formatTs(p.timestamp)}</div>
      <div class="q-det">${cls} · ${conf}</div>
      <div class="q-actions">
        <button class="action-btn approve ${st.approved ? 'active' : ''}"
          onclick="handleApprove(${id}, this)"
          aria-pressed="${st.approved ? 'true' : 'false'}"
          aria-label="Approve prediction ${id}">
          Approve
        </button>
        <button class="action-btn flag ${st.flagged ? 'active' : ''}"
          onclick="handleFlag(${id}, this)"
          aria-pressed="${st.flagged ? 'true' : 'false'}"
          aria-label="Flag prediction ${id} for retraining">
          Flag
        </button>
      </div>
    </div>`;
  }).join('');
}

/* Queue action handlers — called via inline onclick */
function handleApprove(id, btn) {
  // TODO: wire to backend endpoint (e.g., PATCH /predictions/{id} with {status:"approved"})
  const st = queueState[id] ?? {};
  st.approved = !st.approved;
  st.flagged  = false;
  queueState[id] = st;

  btn.classList.toggle('active', st.approved);
  btn.setAttribute('aria-pressed', st.approved ? 'true' : 'false');

  const card = $(`qcard-${id}`);
  if (card) {
    const flagBtn = card.querySelector('.action-btn.flag');
    if (flagBtn) { flagBtn.classList.remove('active'); flagBtn.setAttribute('aria-pressed', 'false'); }
  }

  showToast(st.approved ? `Approved — Prediction #${id}` : `Approval cleared — #${id}`, 'success');
}

function handleFlag(id, btn) {
  // TODO: wire to backend endpoint (e.g., PATCH /predictions/{id} with {status:"flagged"})
  const st = queueState[id] ?? {};
  st.flagged  = !st.flagged;
  st.approved = false;
  queueState[id] = st;

  btn.classList.toggle('active', st.flagged);
  btn.setAttribute('aria-pressed', st.flagged ? 'true' : 'false');

  const card = $(`qcard-${id}`);
  if (card) {
    const approveBtn = card.querySelector('.action-btn.approve');
    if (approveBtn) { approveBtn.classList.remove('active'); approveBtn.setAttribute('aria-pressed', 'false'); }
  }

  showToast(st.flagged ? `Flagged for retraining — #${id}` : `Flag cleared — #${id}`, 'warning');
}

/* ──────────────────────────────────────────
   DROP ZONE SETUP
   ────────────────────────────────────────── */
function setupDropZone() {
  const dropZone  = $('drop-zone');
  const fileInput = $('file-input');
  const btnSelect = $('btn-select-file');
  const btnRescan = $('btn-rescan');
  const btnRefresh= $('btn-refresh');

  dropZone.addEventListener('click', () => fileInput.click());
  btnSelect.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) processFile(file);
  });

  /* Drag events */
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', e => {
    if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) { showToast('Please drop a valid image file', 'error'); return; }
    processFile(file);
  });

  btnRescan.addEventListener('click',  resetToDropZone);
  btnRefresh.addEventListener('click', () => { loadPredictions(); showToast('Activity refreshed', 'info'); });
}

function resetToDropZone() {
  $('drop-zone').hidden       = false;
  $('canvas-wrapper').hidden  = true;
  $('detections-panel').hidden= true;
  $('scan-overlay').hidden    = true;
  $('result-stamp').hidden    = true;
  $('result-stamp').className = 'result-stamp';
  $('file-input').value       = '';
  $('drop-idle').hidden       = false;
}

/* ──────────────────────────────────────────
   PROCESS FILE — scan + predict + render
   ────────────────────────────────────────── */
async function processFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Invalid file. Please choose a JPG, PNG, or BMP image.', 'error');
    return;
  }

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const dropZone       = $('drop-zone');
  const scanOverlay    = $('scan-overlay');
  const scanLine       = $('scan-line');

  /* ── Show scan UI ── */
  $('drop-idle').hidden       = true;
  $('canvas-wrapper').hidden  = true;
  $('detections-panel').hidden= true;
  scanOverlay.hidden           = false;

  /* ── Trigger scan-line sweep ── */
  if (!prefersReduced) {
    scanLine.className = 'scan-line';
    void scanLine.offsetWidth;         /* force reflow to restart animation */
    scanLine.className = 'scan-line sweeping';
  }

  /* ── Run API call and scan animation concurrently ── */
  const formData = new FormData();
  formData.append('file', file);

  const apiPromise  = fetch(`${API}/predict`, { method: 'POST', body: formData });
  const scanPromise = prefersReduced
    ? Promise.resolve()
    : new Promise(r => setTimeout(r, 850));  /* ensure animation always finishes */

  let res, data;
  try {
    [res] = await Promise.all([apiPromise, scanPromise]);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (err) {
    scanOverlay.hidden  = true;
    $('drop-idle').hidden = false;
    showToast(`Scan failed: ${err.message}`, 'error');
    return;
  }

  /* ── Render results ── */
  scanOverlay.hidden = true;
  dropZone.hidden    = true;

  await drawCanvas(file, data);
  $('canvas-wrapper').hidden = false;

  /* Stamp appears briefly on canvas then fades */
  requestAnimationFrame(() => showStamp(data, prefersReduced));

  renderDetections(data);
  $('detections-panel').hidden = false;

  /* Toast summary */
  const n = data.detections?.length ?? 0;
  if (n === 0)            showToast('Scan complete — no defects detected', 'success');
  else if (data.is_uncertain) showToast(`Scan complete — ${n} detection(s), needs review`, 'warning');
  else                    showToast(`Scan complete — ${n} defect(s) found`, 'error');

  /* Reload sidebar/table without blocking UI */
  loadPredictions();
}

/* ──────────────────────────────────────────
   CANVAS — draw image + bounding boxes
   ────────────────────────────────────────── */
const CLASS_COLORS = {
  defect_1: '#4fd1c5',  /* cyan   */
  defect_2: '#e8a23d',  /* amber  */
  defect_3: '#d9534f',  /* red    */
  defect_4: '#b794f4',  /* purple */
};

function detColor(className) {
  return CLASS_COLORS[className] ?? '#4fd1c5';
}

async function drawCanvas(file, data) {
  const canvas = $('result-canvas');
  const ctx    = canvas.getContext('2d');

  const img = new Image();
  const url = URL.createObjectURL(file);

  await new Promise((ok, fail) => {
    img.onload  = ok;
    img.onerror = fail;
    img.src     = url;
  });

  /* Scale to container width, cap height at 450px */
  const maxW  = (canvas.parentElement?.clientWidth || 800);
  const scale = Math.min(1, maxW / img.naturalWidth, 450 / img.naturalHeight);
  canvas.width  = Math.round(img.naturalWidth  * scale);
  canvas.height = Math.round(img.naturalHeight * scale);

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);

  const dets = data.detections ?? [];
  if (!dets.length) return;

  const sx = canvas.width  / data.image_width;
  const sy = canvas.height / data.image_height;

  ctx.font = '11px "IBM Plex Mono"';

  for (const det of dets) {
    const { x_min, y_min, x_max, y_max } = det.bbox;
    const color = detColor(det.class_name);

    const x = Math.round(x_min * sx);
    const y = Math.round(y_min * sy);
    const w = Math.round((x_max - x_min) * sx);
    const h = Math.round((y_max - y_min) * sy);

    /* Tinted fill */
    ctx.fillStyle   = color + '22';   /* ~13 % opacity hex */
    ctx.fillRect(x, y, w, h);

    /* Box stroke */
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(x, y, w, h);

    /* Label */
    const label  = `${det.class_name}  ${(det.confidence * 100).toFixed(1)}%`;
    ctx.font     = '11px "IBM Plex Mono"';
    const tw     = ctx.measureText(label).width;
    const lh     = 18;
    const ly     = y >= lh + 2 ? y - 1 : y + h + 1;

    ctx.fillStyle = color;
    ctx.fillRect(x, ly - lh + 2, tw + 10, lh);

    ctx.fillStyle = '#1e1c1a';
    ctx.fillText(label, x + 5, ly - 4);
  }
}

/* ──────────────────────────────────────────
   INK STAMP
   ────────────────────────────────────────── */
function showStamp(data, prefersReduced) {
  const stamp = $('result-stamp');

  let cls, text;
  if (data.is_uncertain) {
    cls  = 'stamp-review'; text = 'NEEDS REVIEW';
  } else if ((data.detections?.length ?? 0) > 0) {
    cls  = 'stamp-fail';   text = 'DEFECT DETECTED';
  } else {
    cls  = 'stamp-pass';   text = 'PASS';
  }

  stamp.textContent = text;
  stamp.className   = `result-stamp ${cls}`;
  stamp.hidden      = false;

  if (!prefersReduced) {
    void stamp.offsetWidth;   /* force reflow */
    stamp.classList.add('stamp-thud');
  } else {
    /* Skip animation; just show final state */
    stamp.style.transform = 'translate(-50%,-50%) rotate(-6deg) scale(1)';
    stamp.style.opacity   = '0.9';
  }

  /* Fade out after 2.6 s */
  setTimeout(() => {
    if (!prefersReduced) {
      stamp.style.transition = 'opacity 0.45s ease';
      stamp.style.opacity    = '0';
      setTimeout(() => {
        stamp.hidden           = true;
        stamp.style.transition = '';
        stamp.style.opacity    = '';
      }, 460);
    } else {
      stamp.hidden           = true;
      stamp.style.transform  = '';
      stamp.style.opacity    = '';
    }
  }, 2600);
}

/* ──────────────────────────────────────────
   DETECTIONS RESULT PANEL
   ────────────────────────────────────────── */
function renderDetections(data) {
  const dets = data.detections ?? [];
  const list = $('detections-list');

  let badgeClass, badgeText;
  if (data.is_uncertain) {
    badgeClass = 'review'; badgeText = '⚠ NEEDS REVIEW — Confidence below threshold';
  } else if (dets.length) {
    badgeClass = 'fail';   badgeText = `✕ ${dets.length} DEFECT${dets.length > 1 ? 'S' : ''} DETECTED`;
  } else {
    badgeClass = 'pass';   badgeText = '✓ PASS — No anomalies detected';
  }

  let html = `<div class="result-badge ${badgeClass}">${badgeText}</div>`;

  if (!dets.length) {
    html += `<div class="no-defects-msg">Surface passed visual inspection. No anomalies detected above the confidence threshold.</div>`;
  } else {
    html += dets.map(det => {
      const color   = detColor(det.class_name);
      const confPct = (det.confidence * 100).toFixed(1);
      return `<div class="detection-item">
        <div class="det-class" style="color:${color}">${det.class_name}</div>
        <div class="det-bar-wrap">
          <div class="det-bar">
            <div class="det-bar-fill" style="width:${confPct}%;background:${color}"></div>
          </div>
          <span class="det-conf mono">${confPct}%</span>
        </div>
      </div>`;
    }).join('');
  }

  list.innerHTML = html;
}

/* ──────────────────────────────────────────
   BOOTSTRAP
   ────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
  updateClock();
  checkHealth();
  loadPredictions();

  setInterval(updateClock,  1_000);   /* 1 s  — clock tick      */
  setInterval(checkHealth, 10_000);   /* 10 s — backend poll     */
  setInterval(loadPredictions, 15_000); /* 15 s — refresh KPIs   */
});
