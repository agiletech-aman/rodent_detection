'use strict';
/* Face Detection web UI - plain JS single page app (no build step). */

// ------------------------------------------------------------------ helpers

const view = document.getElementById('view');
let pageTimers = [];
let pageCleanups = [];

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'multiple') e[k] = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return e;
}

async function api(method, url, body) {
  const opt = { method, headers: { 'X-Requested-With': 'fetch' } };
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const r = await fetch(url, opt);
  if (r.status === 204) return null;
  const data = await r.json().catch(() => null);
  if (!r.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map(d => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join('; ');
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return data;
}

function toast(msg, kind = 'info', ms = 4000) {
  const t = h('div', { class: `toast ${kind}`, role: 'status' }, msg);
  document.getElementById('toasts').append(t);
  setTimeout(() => t.remove(), ms);
}
const fail = e => toast(e.message || String(e), 'error', 6000);

function every(ms, fn) { fn(); pageTimers.push(setInterval(fn, ms)); }

function fmtTime(epochOrIso) {
  const d = typeof epochOrIso === 'number' ? new Date(epochOrIso * 1000) : new Date(epochOrIso);
  return isNaN(d) ? '–' : d.toLocaleString();
}
function ago(epoch) {
  if (!epoch) return 'never';
  const s = Math.max(0, Date.now() / 1000 - epoch);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
const num = (v, d = 1) => (typeof v === 'number' ? v.toFixed(d) : '–');
const enc = encodeURIComponent;

function zoom(src) {
  const z = document.getElementById('zoom');
  z.firstElementChild.src = src;
  z.hidden = false;
}
document.getElementById('zoom').addEventListener('click', e => {
  e.currentTarget.hidden = true;
  e.currentTarget.firstElementChild.src = '';
});

function thumb(url, zoomUrl) {
  return url ? h('img', { class: 'thumb', src: url, alt: 'face', loading: 'lazy', onclick: () => zoom(zoomUrl || url) })
             : h('div', { class: 'thumb ph' });
}

// On phones, tables become stacked cards; each cell shows its column name (CSS uses data-label).
function labelCells(table) {
  const heads = [...table.querySelectorAll('thead th')].map(th => th.textContent);
  table.querySelectorAll('tbody tr').forEach(tr => [...tr.children].forEach((td, i) => {
    if (heads[i] && !td.hasAttribute('colspan')) td.dataset.label = heads[i];
  }));
}

function badge(state) { return h('span', { class: `badge ${state || ''}` }, state || '–'); }

function field(label, input, help) {
  return h('label', { class: 'field' }, h('span', {}, label), input, help ? h('div', { class: 'help' }, help) : null);
}

function modal(title, body, actions = [], opts = {}) {
  const root = document.getElementById('modal-root');
  const close = () => { bg.remove(); document.removeEventListener('keydown', onKey); opts.onClose && opts.onClose(); };
  const onKey = e => { if (e.key === 'Escape') close(); };
  const bg = h('div', { class: 'modal-bg', onmousedown: e => { if (e.target === bg) close(); } },
    h('div', { class: `modal ${opts.wide ? 'wide' : ''}`, role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
      h('header', {}, h('h2', {}, title), h('button', { class: 'icon-btn', 'aria-label': 'Close', onclick: close }, '×')),
      h('div', { class: 'body' }, body),
      actions.length ? h('footer', {}, actions) : null));
  document.addEventListener('keydown', onKey);
  root.append(bg);
  const first = bg.querySelector('input:not([disabled]), select, textarea');
  if (first) first.focus();
  return { close, el: bg };
}

function confirmBox(text, okLabel = 'Delete') {
  return new Promise(resolve => {
    let done = false;
    const m = modal('Please confirm', h('p', {}, text), [
      h('button', { class: 'btn', onclick: () => { done = true; m.close(); resolve(false); } }, 'Cancel'),
      h('button', { class: 'btn primary', onclick: () => { done = true; m.close(); resolve(true); } }, okLabel),
    ], { onClose: () => { if (!done) resolve(false); } });
  });
}

async function busy(btn, fn) {
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Working…';
  try { return await fn(); } finally { btn.disabled = false; btn.textContent = label; }
}

// ------------------------------------------------------------------ router

const pages = { live: pageLive, cameras: pageCameras, persons: pagePersons, detections: pageDetections,
                rodents: pageRodents, 'rodent-frames': pageRodentFrames, images: pageImages, settings: pageSettings };

function route() {
  pageTimers.forEach(clearInterval); pageTimers = [];
  pageCleanups.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } }); pageCleanups = [];
  const [path, query] = (location.hash.slice(2) || 'live').split('?');
  const name = pages[path] ? path : 'live';
  const navName = name === 'rodent-frames' ? 'rodents' : name;
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.page === navName));
  view.textContent = '';
  pages[name](new URLSearchParams(query || ''));
}
window.addEventListener('hashchange', route);

async function refreshSys() {
  try {
    const s = await api('GET', '/api/system');
    document.getElementById('sys').textContent =
      `${s.device.toUpperCase()} · ${s.model} · ${s.cameras} cameras · ${s.persons} persons`;
    return s;
  } catch (e) { return null; }
}

// ------------------------------------------------------------------ Live

function pageLive() {
  const pauseBtn = h('button', { class: 'btn', hidden: true, onclick: async () => {
    const paused = !pauseBtn.dataset.paused;
    pauseBtn.disabled = true;
    try {
      await api('PUT', '/api/detection', { paused });
      syncPause({ engine_running: true, detection_paused: paused });
      toast(paused ? 'Detection stopped. Live video keeps running.' : 'Detection started', 'ok');
    } catch (e) { fail(e); } finally { pauseBtn.disabled = false; }
  } });
  const stats = h('div', { class: 'stats' });
  const grid = h('div', { class: 'cam-grid' });
  const feed = h('div', { class: 'feed' }, h('div', { class: 'muted' }, 'No detections yet'));
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Live'), pauseBtn),
    stats,
    h('div', { class: 'live-layout' }, grid, h('aside', {}, h('h2', { class: 'small muted' }, 'RECENT DETECTIONS'), feed)));

  const cards = new Map();
  pageCleanups.push(() => cards.forEach(c => { c.img.src = ''; }));  // closes the MJPEG connections

  const stat = (k, v) => h('div', { class: 'stat' }, h('div', { class: 'v' }, v), h('div', { class: 'k' }, k));

  function syncPause(s) {
    pauseBtn.hidden = !s.engine_running;
    pauseBtn.dataset.paused = s.detection_paused ? '1' : '';
    pauseBtn.textContent = s.detection_paused ? '▶ Start detection' : '■ Stop detection';
    pauseBtn.className = s.detection_paused ? 'btn primary' : 'btn danger';
  }

  every(5000, async () => {
    const s = await refreshSys();
    if (!s) return;
    syncPause(s);
    const rodentStat = s.rodent_model ? stat('Rodent alerts today', s.rodents_today.alerts) : null;
    if (rodentStat && s.rodents_today.alerts) rodentStat.classList.add('alert');
    stats.replaceChildren(stat('Faces today', s.today.faces), stat('Known today', s.today.known),
      stat('People seen today', s.today.people), rodentStat, stat('Cameras', s.cameras), stat('Device', s.device.toUpperCase()));
  });

  every(1000, async () => {
    let list;
    try { list = await api('GET', '/api/status'); } catch (e) { return; }
    const ids = new Set(list.map(c => c.id));
    for (const [id, c] of cards) if (!ids.has(id)) { c.img.src = ''; c.el.remove(); cards.delete(id); }
    if (!list.length) {
      grid.replaceChildren(h('div', { class: 'empty' }, 'No cameras running. ', h('a', { href: '#/cameras' }, 'Add a camera')));
      return;
    }
    grid.querySelector('.empty')?.remove();
    for (const cam of list) {
      let c = cards.get(cam.id);
      if (!c) {
        const streamUrl = () => `/api/stream/${enc(cam.id)}?t=${Date.now()}`;
        const img = h('img', { class: 'live', alt: `Live view of ${cam.name}`, src: streamUrl() });
        img.addEventListener('error', () => setTimeout(() => { if (img.isConnected) img.src = streamUrl(); }, 2000));
        img.addEventListener('click', () => zoom(`/api/stream/${enc(cam.id)}?z=${Date.now()}`));
        const title = h('b', {});
        const st = h('span', {});
        const foot = h('div', { class: 'foot' });
        const el = h('div', { class: 'cam' }, h('div', { class: 'head' }, title, st), img, foot);
        grid.append(el);
        c = { el, img, title, st, foot };
        cards.set(cam.id, c);
      }
      c.title.textContent = cam.name;
      c.st.replaceChildren(cam.rodent_alert ? h('span', { class: 'badge rodent' }, 'RODENT') : '', ' ', badge(cam.state));
      c.el.classList.toggle('rodent-alert', !!cam.rodent_alert);
      const names = (cam.names || []).join(', ');
      const rej = Object.entries(cam.rejected || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(', ');
      c.foot.replaceChildren(
        `FPS ${num(cam.fps)} · Det ${num(cam.det_fps)} · ${num(cam.infer_ms, 0)} ms · Faces ${cam.faces ?? 0}`,
        cam.saved != null ? h('div', {}, `Saved ${cam.saved}`, rej ? ` · Skipped: ${rej}` : '') : null,
        names ? h('div', { class: 'who' }, names) : '',
        cam.rodent_ms != null ? h('div', { class: cam.rodent_alert ? 'rodent' : '' },
          cam.rodent_alert ? `Rodent detected (${cam.rodents})` : `Rodents ${cam.rodents ?? 0}`,
          ` · ${num(cam.rodent_ms, 0)} ms · alerts ${cam.rodent_alerts ?? 0}`) : '');
    }
  });

  every(3000, async () => {
    let d;
    try { d = await api('GET', '/api/detections?limit=30'); } catch (e) { return; }
    if (!d.items.length) return;
    feed.replaceChildren(...d.items.map(r => h('div', { class: 'feed-item' },
      thumb(r.crop_url, r.annotated_url),
      h('div', {},
        h('b', {}, r.name || 'face', ' ', h('span', { class: 'muted small' }, num(r.confidence, 2))),
        h('div', { class: 'muted small' }, `${r.camera_id} · ${new Date(r.ts).toLocaleTimeString()}`),
        r.has_embedding ? h('button', { class: 'btn small', style: 'margin-top:4px', onclick: async () => {
          let persons;
          try { persons = await api('GET', '/api/persons'); } catch (e) { return fail(e); }
          enrollFromDetection(r, persons, () => {});
        } }, r.person_id ? 'Re-assign' : 'Assign person') : null))));
  });
}

// ------------------------------------------------------------------ Cameras

function pageCameras() {
  const tbody = h('tbody');
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Cameras'),
      h('button', { class: 'btn primary', onclick: () => cameraForm(null, load) }, '+ Add camera')),
    h('div', { class: 'table-wrap' }, h('table', { class: 'cards' },
      h('thead', {}, h('tr', {}, ['Camera', 'Source', 'Status', 'FPS', 'Frame skip', 'Enabled', ''].map(t => h('th', {}, t)))),
      tbody)));

  async function load() {
    let cams;
    try { cams = await api('GET', '/api/cameras'); } catch (e) { return fail(e); }
    if (!cams.length) {
      tbody.replaceChildren(h('tr', {}, h('td', { colspan: 7 }, h('div', { class: 'empty' }, 'No cameras yet. Click "Add camera".'))));
      return;
    }
    tbody.replaceChildren(...cams.map(c => h('tr', {},
      h('td', {}, h('b', {}, c.name), h('div', { class: 'muted small' }, c.id)),
      h('td', { class: 'mono' }, c.url),
      h('td', {}, badge(c.state)),
      h('td', {}, num(c.fps)),
      h('td', {}, c.frame_skip ?? h('span', { class: 'muted' }, 'default'),
        c.min_face_size != null ? h('div', { class: 'muted small' }, `min face ${c.min_face_size}px`) : null),
      h('td', {}, h('input', { type: 'checkbox', checked: c.enabled, 'aria-label': `Enable ${c.name}`,
        onchange: async e => {
          try { await api('PUT', `/api/cameras/${enc(c.id)}`, { enabled: e.target.checked }); toast(`${c.name} ${e.target.checked ? 'enabled' : 'disabled'}`, 'ok'); load(); }
          catch (err) { fail(err); e.target.checked = !e.target.checked; }
        } })),
      h('td', { class: 'actions' },
        h('button', { class: 'btn small', onclick: () => cameraForm(c, load) }, 'Edit'), ' ',
        h('button', { class: 'btn small danger', onclick: async () => {
          if (!await confirmBox(`Delete camera "${c.name}"? Its detection history is kept.`)) return;
          try { await api('DELETE', `/api/cameras/${enc(c.id)}`); toast('Camera deleted', 'ok'); load(); } catch (e) { fail(e); }
        } }, 'Delete')))));
    labelCells(tbody.closest('table'));
  }
  every(3000, load);
}

function cameraForm(cam, onSaved) {
  const editing = !!cam;
  const isDevice = u => /^device:\d+$/.test(u || '');
  let mode = cam && isDevice(cam.url) ? 'device' : 'rtsp';
  const id = h('input', { type: 'text', value: cam?.id || '', disabled: editing, placeholder: 'gate_1', pattern: '[A-Za-z0-9_-]+' });
  const name = h('input', { type: 'text', value: cam?.name || '', placeholder: 'Main gate' });
  const url = h('input', { type: 'text', class: 'mono', value: cam && !isDevice(cam.url) ? cam.url : '', placeholder: 'rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101' });
  const devSel = h('select', {}, h('option', { value: '' }, 'Looking for cameras…'));
  const skip = h('input', { type: 'number', min: 1, max: 100, value: cam?.frame_skip ?? '', placeholder: 'default' });
  const minSize = h('input', { type: 'number', min: 0, max: 1000, value: cam?.min_face_size ?? '', placeholder: 'global setting' });
  const minBlur = h('input', { type: 'number', min: 0, max: 5000, step: 1, value: cam?.min_blur_score ?? '', placeholder: 'global setting' });
  const enabled = h('input', { type: 'checkbox', checked: cam ? cam.enabled : true });
  const testOut = h('div', { class: 'test-result' });
  if (!editing) name.addEventListener('input', () => {
    if (!id.dataset.touched) id.value = name.value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 64);
  });
  id.addEventListener('input', () => { id.dataset.touched = '1'; });

  const segRtsp = h('button', { type: 'button', class: 'seg', onclick: () => setMode('rtsp') }, '🌐 IP / RTSP camera');
  const segDev = h('button', { type: 'button', class: 'seg', onclick: () => setMode('device') }, '💻 Device camera (webcam / USB)');
  const rtspField = field('RTSP URL', url, 'Format: rtsp://user:pass@ip:554/path. Encode special characters in the password (@ → %40). When editing, keep *** to leave the saved password unchanged.');
  const devField = field('Device', devSel, 'Cameras connected to the computer running this server (laptop webcam, USB camera).');

  function setMode(m2) {
    mode = m2;
    segRtsp.classList.toggle('on', mode === 'rtsp'); segRtsp.setAttribute('aria-pressed', String(mode === 'rtsp'));
    segDev.classList.toggle('on', mode === 'device'); segDev.setAttribute('aria-pressed', String(mode === 'device'));
    rtspField.hidden = mode !== 'rtsp';
    devField.hidden = mode !== 'device';
    testOut.textContent = '';
    if (!editing && mode === 'device' && !name.value.trim()) {
      name.value = 'Webcam'; name.dispatchEvent(new Event('input'));
    }
  }

  async function loadDevices() {
    try {
      const devs = await api('GET', '/api/devices');
      const taken = d => d.used_by && d.used_by_id !== cam?.id;
      devSel.replaceChildren(...(devs.length
        ? devs.map(d => h('option', { value: d.url, disabled: taken(d) },
            `${d.name} (${d.path})${taken(d) ? `: already added as "${d.used_by}"` : ''}`))
        : [h('option', { value: '' }, 'No camera found on the server computer')]));
      if (cam && isDevice(cam.url)) devSel.value = cam.url;
      else { const free = devs.find(d => !taken(d)); devSel.value = free ? free.url : ''; }
    } catch (e) { devSel.replaceChildren(h('option', { value: '' }, 'Could not list devices')); }
  }

  const currentUrl = () => (mode === 'device' ? devSel.value : url.value.trim());
  function validate() {
    if (!currentUrl()) {
      toast(mode === 'device' ? 'No free device camera to select' : 'Enter the RTSP URL first', 'error');
      (mode === 'device' ? devSel : url).focus();
      return false;
    }
    return true;
  }

  const testBtn = h('button', { class: 'btn left', onclick: () => {
    if (!validate()) return;
    busy(testBtn, async () => {
      testOut.replaceChildren(h('span', { class: 'muted' }, 'Connecting…'));
      try {
        const r = await api('POST', '/api/cameras/test', { url: currentUrl(), camera_id: cam?.id });
        testOut.replaceChildren(r.ok
          ? h('div', {},
              h('span', { class: 'ok-text' }, `✓ Connected: ${r.width}×${r.height}${r.fps ? ` @ ${r.fps} fps` : ''}${r.latency_ms ? ` (${r.latency_ms} ms)` : ''}`),
              r.note ? h('div', { class: 'help' }, r.note) : null,
              h('img', { src: r.snapshot, alt: 'Camera snapshot' }))
          : h('span', { class: 'err-text' }, `✗ ${r.error}`));
      } catch (e) { testOut.replaceChildren(h('span', { class: 'err-text' }, `✗ ${e.message}`)); }
    });
  } }, 'Test connection');

  const saveBtn = h('button', { class: 'btn primary', onclick: () => {
    if (!name.value.trim()) { toast('Enter a camera name', 'error'); name.focus(); return; }
    if (!validate()) return;
    busy(saveBtn, async () => {
      const body = { name: name.value.trim(), url: currentUrl(), enabled: enabled.checked };
      if (skip.value) body.frame_skip = Number(skip.value);
      if (minSize.value !== '') body.min_face_size = Number(minSize.value);
      if (minBlur.value !== '') body.min_blur_score = Number(minBlur.value);
      try {
        if (editing) {
          if (!skip.value) body.clear_frame_skip = true;
          body.clear_quality = [[minSize, 'min_face_size'], [minBlur, 'min_blur_score']].filter(([i]) => i.value === '').map(([, k]) => k);
          await api('PUT', `/api/cameras/${enc(cam.id)}`, body);
        } else {
          await api('POST', '/api/cameras', { id: id.value.trim(), ...body });
        }
        toast(editing ? 'Camera updated' : 'Camera added: stream is starting', 'ok');
        m.close(); onSaved();
      } catch (e) { fail(e); }
    });
  } }, 'Save');

  const m = modal(editing ? `Edit camera: ${cam.name}` : 'Add camera', h('div', {},
    h('div', { class: 'field' }, h('span', {}, 'Source'),
      h('div', { class: 'segmented', role: 'group', 'aria-label': 'Camera source' }, segRtsp, segDev)),
    field('Name', name),
    field('Camera ID', id, editing ? 'The ID cannot be changed (it is used in folder names).' : 'Letters, digits, _ and - only. Used for folder names.'),
    rtspField, devField,
    field('Frame skip', skip, 'Run detection on every N-th frame. Leave empty to use the default from Settings.'),
    h('details', { class: 'adv', open: !!(cam && (cam.min_face_size != null || cam.min_blur_score != null)) },
      h('summary', {}, 'Face quality for this camera'),
      h('div', { class: 'help', style: 'margin:6px 0 10px' },
        'For wide / far cameras whose faces are always small, lower the size so the best available crop of each person is still saved. Empty = global setting.'),
      field('Min face size (px)', minSize), field('Min sharpness (blur score)', minBlur)),
    h('label', { class: 'check' }, enabled, 'Enabled'),
    testOut),
  [testBtn, h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'), saveBtn]);
  setMode(mode);
  loadDevices();
}

// ------------------------------------------------------------------ Persons

function pagePersons() {
  const search = h('input', { type: 'search', placeholder: 'Search name or code…', style: 'max-width:260px' });
  const grid = h('div', { class: 'person-grid' });
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Persons'), search,
      h('button', { class: 'btn primary', onclick: () => personCreate(load) }, '+ Add person')),
    grid);

  async function load() {
    let list;
    try { list = await api('GET', '/api/persons' + (search.value.trim() ? `?q=${enc(search.value.trim())}` : '')); } catch (e) { return fail(e); }
    if (!list.length) {
      grid.replaceChildren(h('div', { class: 'empty', style: 'grid-column:1/-1' },
        search.value ? 'No match.' : 'No persons yet. Add a person with 2–5 clear face photos to enable recognition.'));
      return;
    }
    grid.replaceChildren(...list.map(p => h('button', { class: 'person', onclick: () => personDetail(p.id, load) },
      p.thumb_url ? h('img', { class: 'ph', src: p.thumb_url, alt: p.name, loading: 'lazy' }) : h('div', { class: 'ph' }, (p.name[0] || '?').toUpperCase()),
      h('div', { class: 'info' },
        h('b', {}, p.name),
        h('div', { class: 'muted small' }, [p.code, `${p.image_count} photo${p.image_count === 1 ? '' : 's'}`].filter(Boolean).join(' · ')),
        h('div', { class: 'muted small' }, p.last_seen ? `Seen ${ago(p.last_seen)} · ${p.last_camera}` : 'Not seen yet'),
        p.image_count === 0 ? h('div', { class: 'small err-text' }, 'No face photo: cannot be recognised') : null))));
  }
  let t;
  search.addEventListener('input', () => { clearTimeout(t); t = setTimeout(load, 250); });
  load();
}

function uploadResultsToast(res) {
  const ok = res.results.filter(r => r.ok).length;
  const bad = res.results.filter(r => !r.ok);
  if (ok) toast(`${ok} photo${ok === 1 ? '' : 's'} added`, 'ok');
  res.results.filter(r => r.warning).forEach(r => toast(`${r.file}: ${r.warning}`));
  bad.forEach(r => toast(`${r.file}: ${r.error}`, 'error', 7000));
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Face photo picker: files from disk + live capture from this device's camera (webcam / phone).
function photoPicker() {
  const pending = [];
  const strip = h('div', { class: 'pending' });

  async function add(file) {
    const item = { file, url: URL.createObjectURL(file), ok: null };
    const status = h('span', { class: 'badge' }, 'checking…');
    const el = h('div', { class: 'p' },
      h('img', { src: item.url, alt: 'Face photo', onclick: () => zoom(item.url) }),
      status,
      h('button', { class: 'x', type: 'button', title: 'Remove', 'aria-label': 'Remove photo', onclick: () => {
        pending.splice(pending.indexOf(item), 1); el.remove(); URL.revokeObjectURL(item.url);
      } }, '×'));
    pending.push(item); strip.append(el);
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await api('POST', '/api/faces/check', fd);
      item.ok = r.ok;
      status.textContent = r.ok ? `✓ ${r.message === 'Good' ? 'face' : r.message}` : r.message;
      status.className = `badge ${r.ok ? 'known' : 'failed'}`;
    } catch (e) { item.ok = false; status.textContent = e.message; status.className = 'badge failed'; }
    return item;
  }

  const fileInput = h('input', { type: 'file', accept: 'image/*', multiple: true, class: 'sr-only' });
  fileInput.addEventListener('change', () => { [...fileInput.files].forEach(add); fileInput.value = ''; });
  const pickBtn = h('label', { class: 'btn' }, '🖼 Choose files', fileInput);

  // Live camera needs a secure context (https or localhost). Elsewhere (e.g. http://<lan-ip> on a phone)
  // fall back to <input capture>, which opens the phone's camera app.
  const live = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.isSecureContext);
  const phoneInput = h('input', { type: 'file', accept: 'image/*', capture: 'user', class: 'sr-only' });
  phoneInput.addEventListener('change', () => { [...phoneInput.files].forEach(add); phoneInput.value = ''; });

  let stream = null, facing = 'user';
  const video = h('video', { autoplay: true, playsinline: true });
  video.muted = true; video.playsInline = true;
  const hint = h('div', { class: 'hint' }, 'Look at the camera');
  const wrap = h('div', { class: 'video-wrap' }, video, hint);
  const camStatus = h('div', { class: 'small muted', 'aria-live': 'polite' });
  const shootBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => shoot() }, 'Capture');
  const autoBtn = h('button', { class: 'btn', type: 'button', onclick: () => autoShoot() }, 'Auto 5 photos');
  const flipBtn = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => { facing = facing === 'user' ? 'environment' : 'user'; start(); } }, '🔄 Switch camera');
  const stopBtn = h('button', { class: 'btn', type: 'button', onclick: () => stop() }, 'Close camera');
  const panel = h('div', { class: 'capture', hidden: true }, wrap, h('div', { class: 'row' }, shootBtn, autoBtn, flipBtn, stopBtn), camStatus);
  const camBtn = live
    ? h('button', { class: 'btn', type: 'button', onclick: () => start() }, '📷 Use camera')
    : h('label', { class: 'btn' }, '📷 Take photo', phoneInput);

  function stopTracks() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } }
  async function start() {
    stopTracks();
    camStatus.textContent = 'Starting camera…';
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = stream;
      video.classList.toggle('mirror', facing === 'user');
      await video.play().catch(() => {});
      panel.hidden = false; camBtn.hidden = true;
      camStatus.textContent = 'Good light, face the camera, only one person in frame.';
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      flipBtn.hidden = cams.length < 2;
    } catch (e) {
      stop();
      toast(e.name === 'NotAllowedError' ? 'Camera permission denied: allow camera access in the browser settings.'
        : e.name === 'NotFoundError' ? 'No camera found on this device.' : `Camera error: ${e.message}`, 'error', 7000);
    }
  }
  function stop() { stopTracks(); video.srcObject = null; panel.hidden = true; camBtn.hidden = false; }
  async function shoot() {
    if (!stream || !video.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
    wrap.classList.remove('flash'); void wrap.offsetWidth; wrap.classList.add('flash');
    return add(new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' }));
  }
  async function autoShoot() {
    const prompts = ['Look straight at the camera', 'Turn your head slightly left', 'Turn your head slightly right',
                     'Tilt your head slightly up', 'Tilt your head slightly down'];
    shootBtn.disabled = autoBtn.disabled = true;
    try {
      for (const p of prompts) {
        if (!stream) break;
        hint.textContent = p;
        for (let s = 3; s > 0; s--) { camStatus.textContent = `${p}… ${s}`; await sleep(600); }
        if (!stream) break;
        camStatus.textContent = 'Capturing…';
        await shoot();
      }
      camStatus.textContent = 'Done. Remove any photo marked in red, then save.';
    } finally {
      shootBtn.disabled = autoBtn.disabled = false;
      hint.textContent = 'Look at the camera';
    }
  }

  const el = h('div', {},
    h('div', { class: 'row', style: 'margin-bottom:8px' }, camBtn, pickBtn),
    panel, strip,
    h('div', { class: 'help' }, 'Add 3–5 photos: front plus slight left/right turns, good light, one face per photo.',
      live ? '' : ' (Live preview needs https or localhost; here "Take photo" opens the phone camera.)'));
  return {
    el,
    stop,
    files: () => pending.filter(p => p.ok !== false).map(p => p.file),
    count: () => pending.filter(p => p.ok !== false).length,
    clear: () => { pending.splice(0).forEach(p => URL.revokeObjectURL(p.url)); strip.textContent = ''; },
  };
}

function personCreate(onSaved) {
  const name = h('input', { type: 'text', placeholder: 'Full name' });
  const code = h('input', { type: 'text', placeholder: 'e.g. EMP-0042 (optional)' });
  const notes = h('textarea', { placeholder: 'Department, phone, remarks… (optional)' });
  const picker = photoPicker();
  const save = h('button', { class: 'btn primary', onclick: () => busy(save, async () => {
    if (!name.value.trim()) return toast('Enter a name');
    try {
      const p = await api('POST', '/api/persons', { name: name.value.trim(), code: code.value.trim() || null, notes: notes.value });
      const files = picker.files();
      if (files.length) {
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        uploadResultsToast(await api('POST', `/api/persons/${p.id}/images`, fd));
      } else toast('Saved without photos: add photos so this person can be recognised');
      toast(`${p.name} added`, 'ok');
      m.close(); onSaved();
    } catch (e) { fail(e); }
  }) }, 'Save');
  const m = modal('Add person', h('div', {},
    field('Name', name), field('Code / Employee ID', code), field('Notes', notes),
    h('div', { class: 'small muted', style: 'margin-bottom:6px' }, 'FACE PHOTOS'), picker.el),
  [h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'), save], { wide: true, onClose: picker.stop });
}

async function personDetail(id, onChanged) {
  let p;
  try { p = await api('GET', `/api/persons/${id}`); } catch (e) { return fail(e); }
  const name = h('input', { type: 'text', value: p.name });
  const code = h('input', { type: 'text', value: p.code || '' });
  const notes = h('textarea', { value: p.notes || '' });
  const gallery = h('div', { class: 'gallery' });
  const picker = photoPicker();
  const title = h('div', { class: 'small muted' });

  function renderGallery() {
    const autos = p.images.filter(i => i.source === 'auto').length;
    title.textContent = `SAVED FACE PHOTOS (${p.images.length}${autos ? `, ${autos} auto-learned` : ''})`;
    gallery.replaceChildren(...(p.images.length ? p.images.map(img => h('div', { class: 'g' },
      h('img', { src: img.url, alt: `Photo of ${p.name}`, onclick: () => zoom(img.url) }),
      img.source === 'auto' ? h('span', { class: 'auto-tag', title: 'Learned automatically from a confident recognition. Remove it if it is not this person.' }, 'auto') : null,
      h('button', { class: 'x', title: 'Remove photo', 'aria-label': 'Remove photo', onclick: async () => {
        if (!await confirmBox('Remove this photo?', 'Remove')) return;
        try { await api('DELETE', `/api/persons/${p.id}/images/${img.id}`); p.images = p.images.filter(i => i.id !== img.id); renderGallery(); onChanged(); }
        catch (e) { fail(e); }
      } }, '×'))) : [h('div', { class: 'muted small' }, 'No photos yet.')]));
  }
  renderGallery();

  const uploadBtn = h('button', { class: 'btn primary', onclick: () => busy(uploadBtn, async () => {
    const files = picker.files();
    if (!files.length) return toast('Capture or choose photos first');
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    try {
      const res = await api('POST', `/api/persons/${p.id}/images`, fd);
      uploadResultsToast(res);
      p = res.person; picker.clear(); renderGallery(); onChanged();
    } catch (e) { fail(e); }
  }) }, 'Add photos to person');

  const save = h('button', { class: 'btn primary', onclick: () => busy(save, async () => {
    try {
      if (picker.count()) {
        const fd = new FormData();
        picker.files().forEach(f => fd.append('files', f));
        uploadResultsToast(await api('POST', `/api/persons/${p.id}/images`, fd));
        picker.clear();
      }
      await api('PUT', `/api/persons/${p.id}`, { name: name.value.trim(), code: code.value.trim(), notes: notes.value });
      toast('Saved', 'ok'); m.close(); onChanged();
    } catch (e) { fail(e); }
  }) }, 'Save');

  const del = h('button', { class: 'btn danger left', onclick: async () => {
    if (!await confirmBox(`Delete ${p.name} and all their photos? Detection history is kept but no longer linked.`)) return;
    try { await api('DELETE', `/api/persons/${p.id}`); toast('Person deleted', 'ok'); m.close(); onChanged(); } catch (e) { fail(e); }
  } }, 'Delete person');

  const m = modal(p.name, h('div', {},
    h('div', { class: 'muted small', style: 'margin-bottom:12px' },
      p.last_seen ? `Last seen ${fmtTime(p.last_seen)} on ${p.last_camera} · ` : 'Not seen yet · ',
      h('a', { href: `#/detections?person_id=${p.id}`, onclick: () => m.close() }, 'View detections')),
    field('Name', name), field('Code / Employee ID', code), field('Notes', notes),
    title, gallery,
    h('div', { class: 'small muted', style: 'margin-bottom:6px' }, 'ADD MORE PHOTOS'),
    picker.el,
    h('div', { class: 'row', style: 'margin-top:8px' }, uploadBtn)),
  [del, h('button', { class: 'btn', onclick: () => m.close() }, 'Close'), save], { wide: true, onClose: picker.stop });
}

// ------------------------------------------------------------------ Detections

async function pageDetections(params) {
  const camSel = h('select', {}, h('option', { value: '' }, 'All cameras'));
  const personSel = h('select', {}, h('option', { value: '' }, 'All persons'));
  const knownSel = h('select', {}, h('option', { value: '' }, 'Known + unknown'),
    h('option', { value: 'true' }, 'Known only'), h('option', { value: 'false' }, 'Unknown only'));
  const recSel = h('select', {}, h('option', { value: 'new' }, 'New (with embedding)'),
    h('option', { value: 'old' }, 'Old (no embedding)'), h('option', { value: 'all' }, 'All records'));
  const from = h('input', { type: 'date', 'aria-label': 'From date' });
  const to = h('input', { type: 'date', 'aria-label': 'To date' });
  const tbody = h('tbody');
  const pager = h('div', { class: 'row', style: 'margin-top:10px' });
  const csv = h('a', { class: 'btn', href: '/api/detections.csv' }, 'Export CSV');
  const limit = 50;
  let offset = 0;
  let persons = [];
  let pageIds = [];
  let pageRows = [];
  const selected = new Set();
  const selectAll = h('input', { type: 'checkbox', 'aria-label': 'Select all on this page', onchange: e => {
    pageIds.forEach(id => e.target.checked ? selected.add(id) : selected.delete(id));
    tbody.querySelectorAll('input.sel').forEach(cb => { cb.checked = e.target.checked; });
    syncSel();
  } });
  const delSel = h('button', { class: 'btn danger', disabled: true, onclick: async () => {
    const n = selected.size;
    if (!n || !await confirmBox(`Delete ${n} selected detection${n === 1 ? '' : 's'} and their saved images?`)) return;
    try {
      const r = await api('POST', '/api/detections/delete', { ids: [...selected] });
      toast(`Deleted ${r.deleted} detection${r.deleted === 1 ? '' : 's'}`, 'ok');
      selected.clear(); load();
    } catch (e) { fail(e); }
  } }, 'Delete selected');
  const sameSel = h('button', { class: 'btn', disabled: true,
    onclick: () => enrollFromDetection(pageRows.filter(r => selected.has(r.id)), persons, load) }, 'Same person');
  function syncSel() {
    delSel.disabled = selected.size === 0;
    sameSel.disabled = selected.size === 0;
    sameSel.textContent = selected.size ? `Same person (${selected.size})` : 'Same person';
    delSel.textContent = selected.size ? `Delete selected (${selected.size})` : 'Delete selected';
    selectAll.checked = pageIds.length > 0 && pageIds.every(id => selected.has(id));
    selectAll.indeterminate = !selectAll.checked && pageIds.some(id => selected.has(id));
  }

  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Detections'), sameSel, delSel, csv),
    h('div', { class: 'filters' },
      field('Camera', camSel), field('Person', personSel), field('Type', knownSel), field('Records', recSel), field('From', from), field('To', to),
      h('button', { class: 'btn primary', style: 'margin-bottom:12px', onclick: () => { offset = 0; load(); } }, 'Apply')),
    h('div', { class: 'table-wrap' }, h('table', { class: 'cards' },
      h('thead', {}, h('tr', {}, h('th', { class: 'sel-col' }, h('span', { class: 'sr-only' }, 'Select'), selectAll),
        ['Face', 'Time', 'Camera', 'Person', 'Similarity', 'Confidence', ''].map(t => h('th', {}, t)))),
      tbody)),
    pager);

  try {
    const [cams, ps] = await Promise.all([api('GET', '/api/cameras'), api('GET', '/api/persons')]);
    persons = ps;
    [...cams.map(c => [c.id, c.name]), ['images', 'images (CLI)'], ['upload', 'upload (web)']]
      .forEach(([v, t]) => camSel.append(h('option', { value: v }, t)));
    ps.forEach(p => personSel.append(h('option', { value: p.id }, p.name + (p.code ? ` (${p.code})` : ''))));
  } catch (e) { fail(e); }
  if (params.get('person_id')) personSel.value = params.get('person_id');
  if (params.get('camera_id')) camSel.value = params.get('camera_id');

  function query() {
    const q = new URLSearchParams();
    if (camSel.value) q.set('camera_id', camSel.value);
    if (personSel.value) q.set('person_id', personSel.value);
    if (knownSel.value) q.set('known', knownSel.value);
    if (recSel.value !== 'new') q.set('records', recSel.value);
    if (from.value) q.set('from', from.value);
    if (to.value) q.set('to', to.value);
    return q;
  }

  async function load() {
    const q = query();
    csv.href = '/api/detections.csv?' + q;
    q.set('limit', limit); q.set('offset', offset);
    let d;
    try { d = await api('GET', '/api/detections?' + q); } catch (e) { return fail(e); }
    pageRows = d.items;
    pageIds = d.items.map(r => r.id);
    selected.clear();
    if (!d.items.length) tbody.replaceChildren(h('tr', {}, h('td', { colspan: 8 }, h('div', { class: 'empty' }, 'No detections for these filters.'))));
    else tbody.replaceChildren(...d.items.map(r => h('tr', {},
      h('td', { class: 'sel-col' }, h('input', { type: 'checkbox', class: 'sel', 'aria-label': `Select detection ${r.id}`,
        onchange: e => { e.target.checked ? selected.add(r.id) : selected.delete(r.id); syncSel(); } })),
      h('td', {}, thumb(r.crop_url, r.annotated_url)),
      h('td', {}, fmtTime(r.ts), r.source ? h('div', { class: 'muted small' }, r.source.split('/').pop()) : null),
      h('td', {}, r.camera_id),
      h('td', {}, r.person_id ? h('span', { class: 'badge known' }, r.name) : h('span', { class: 'badge unknown' }, r.name || 'not recognised'),
        r.has_embedding ? null : h('div', { class: 'small err-text' }, 'old record: no embedding')),
      h('td', {}, num(r.similarity, 2)),
      h('td', {}, num(r.confidence, 2)),
      h('td', { class: 'actions' },
        h('button', { class: 'btn small', onclick: () => enrollFromDetection(r, persons, load) }, r.person_id ? 'Re-assign' : 'Assign person'), ' ',
        h('button', { class: 'btn small danger', 'aria-label': `Delete detection ${r.id}`, onclick: async () => {
          if (!await confirmBox('Delete this detection and its saved images?')) return;
          try { await api('DELETE', `/api/detections/${r.id}`); toast('Detection deleted', 'ok'); load(); } catch (e) { fail(e); }
        } }, 'Delete')))));
    syncSel();
    labelCells(tbody.closest('table'));
    pager.replaceChildren(
      h('span', { class: 'muted' }, d.total ? `${offset + 1}–${Math.min(offset + limit, d.total)} of ${d.total}` : '0 results'),
      h('button', { class: 'btn small', disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - limit); load(); } }, '‹ Newer'),
      h('button', { class: 'btn small', disabled: offset + limit >= d.total, onclick: () => { offset += limit; load(); } }, 'Older ›'));
  }
  load();
}

// dets: one detection or several selected ones that are the same person.
function enrollFromDetection(dets, persons, onDone) {
  dets = [].concat(dets);
  const usable = dets.filter(d => d.has_embedding);
  const sel = h('select', {}, h('option', { value: '' }, '— choose existing person —'),
    persons.map(p => h('option', { value: p.id }, p.name + (p.code ? ` (${p.code})` : ''))));
  const newName = h('input', { type: 'text', placeholder: 'or type a new person name' });
  const save = h('button', { class: 'btn primary', onclick: () => busy(save, async () => {
    try {
      let pid = sel.value;
      if (!pid) {
        if (!newName.value.trim()) return toast('Choose a person or enter a new name');
        const p = await api('POST', '/api/persons', { name: newName.value.trim() });
        persons.push(p); pid = p.id;
      }
      if (!usable.length) return toast('These are old records without an embedding and cannot be assigned');
      const r = await api('POST', `/api/persons/${pid}/from-detections`, { ids: usable.map(d => d.id) });
      toast(`${r.added} face${r.added === 1 ? '' : 's'} added to person. Future detections will be recognised.`, 'ok');
      if (r.skipped.length) toast(`${r.skipped.length} skipped (no embedding)`);
      m.close(); onDone();
    } catch (e) { fail(e); }
  }) }, 'Assign');
  const one = dets.length === 1;
  const m = modal(one ? 'Assign face to person' : `These ${dets.length} faces are the same person`, h('div', {},
    one ? h('div', { class: 'row', style: 'margin-bottom:12px' }, h('img', { src: dets[0].crop_url, alt: 'face', style: 'width:120px;border-radius:6px' }),
      h('div', { class: 'muted small' }, `${dets[0].camera_id} · ${fmtTime(dets[0].ts)}`, h('br'), 'The face embedding saved with this detection is added to the person (no re-detection).'))
    : h('div', { style: 'margin-bottom:12px' },
      h('div', { class: 'face-grid' }, dets.map(d => h('img', { src: d.crop_url, alt: 'face', class: d.has_embedding ? '' : 'no-emb',
        title: d.has_embedding ? `${d.camera_id} · ${fmtTime(d.ts)}` : 'old record: no embedding, will be skipped' }))),
      h('div', { class: 'muted small' }, 'All these faces are added to one person, so recognition learns several views of them.',
        usable.length < dets.length ? ` ${dets.length - usable.length} old record(s) without an embedding will be skipped.` : '')),
    field('Existing person', sel), field('New person', newName)),
  [h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'), save]);
}

// ------------------------------------------------------------------ Rodents

async function pageRodents() {
  const camSel = h('select', {}, h('option', { value: '' }, 'All cameras'));
  const from = h('input', { type: 'date', 'aria-label': 'From date' });
  const to = h('input', { type: 'date', 'aria-label': 'To date' });
  const tbody = h('tbody');
  const pager = h('div', { class: 'row', style: 'margin-top:10px' });
  const banner = h('div', { hidden: true });
  const limit = 50;
  let offset = 0;
  let pageIds = [];
  const selected = new Set();
  const selectAll = h('input', { type: 'checkbox', 'aria-label': 'Select all on this page', onchange: e => {
    pageIds.forEach(id => e.target.checked ? selected.add(id) : selected.delete(id));
    tbody.querySelectorAll('input.sel').forEach(cb => { cb.checked = e.target.checked; });
    syncSel();
  } });
  const delSel = h('button', { class: 'btn danger', disabled: true, onclick: async () => {
    const n = selected.size;
    if (!n || !await confirmBox(`Delete ${n} selected rodent alert${n === 1 ? '' : 's'} and their images?`)) return;
    try {
      const r = await api('POST', '/api/rodents/delete', { ids: [...selected] });
      toast(`Deleted ${r.deleted} alert${r.deleted === 1 ? '' : 's'}`, 'ok');
      load();
    } catch (e) { fail(e); }
  } }, 'Delete selected');
  function syncSel() {
    delSel.disabled = selected.size === 0;
    delSel.textContent = selected.size ? `Delete selected (${selected.size})` : 'Delete selected';
    selectAll.checked = pageIds.length > 0 && pageIds.every(id => selected.has(id));
    selectAll.indeterminate = !selectAll.checked && pageIds.some(id => selected.has(id));
  }

  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Rodents'),
      h('a', { class: 'btn', href: '#/rodent-frames' }, 'Training frames'), delSel),
    banner,
    h('div', { class: 'filters' },
      field('Camera', camSel), field('From', from), field('To', to),
      h('button', { class: 'btn primary', style: 'margin-bottom:12px', onclick: () => { offset = 0; load(); } }, 'Apply')),
    h('div', { class: 'table-wrap' }, h('table', { class: 'cards' },
      h('thead', {}, h('tr', {}, h('th', { class: 'sel-col' }, h('span', { class: 'sr-only' }, 'Select'), selectAll),
        ['Snapshot', 'Time', 'Camera', 'Rodents', 'Confidence', ''].map(t => h('th', {}, t)))),
      tbody)),
    pager);

  try {
    const [cams, sys] = await Promise.all([api('GET', '/api/cameras'), api('GET', '/api/system')]);
    cams.forEach(c => camSel.append(h('option', { value: c.id }, c.name)));
    if (!sys.rodent_model) {
      banner.className = 'banner bad'; banner.hidden = false;
      banner.textContent = 'Rodent model not loaded (models/rodent.onnx missing). Train it with scripts/rodent/train.py and restart.';
    } else if (!sys.rodent_enabled) {
      banner.className = 'banner'; banner.hidden = false;
      banner.replaceChildren('Rodent detection is switched off for the cameras. ', h('a', { href: '#/settings' }, 'Turn it on in Settings'), '.');
    }
  } catch (e) { fail(e); }

  async function load() {
    const q = new URLSearchParams();
    if (camSel.value) q.set('camera_id', camSel.value);
    if (from.value) q.set('from', from.value);
    if (to.value) q.set('to', to.value);
    q.set('limit', limit); q.set('offset', offset);
    let d;
    try { d = await api('GET', '/api/rodents?' + q); } catch (e) { return fail(e); }
    pageIds = d.items.map(r => r.id);
    selected.clear();
    if (!d.items.length) tbody.replaceChildren(h('tr', {}, h('td', { colspan: 7 }, h('div', { class: 'empty' }, 'No rodent alerts for these filters.'))));
    else tbody.replaceChildren(...d.items.map(r => h('tr', {},
      h('td', { class: 'sel-col' }, h('input', { type: 'checkbox', class: 'sel', 'aria-label': `Select alert ${r.id}`,
        onchange: e => { e.target.checked ? selected.add(r.id) : selected.delete(r.id); syncSel(); } })),
      h('td', {}, thumb(r.crop_url, r.annotated_url)),
      h('td', {}, fmtTime(r.ts)),
      h('td', {}, r.camera_id),
      h('td', {}, h('span', { class: 'badge rodent' }, r.count)),
      h('td', {}, num(r.confidence, 2)),
      h('td', { class: 'actions' },
        r.annotated_url ? h('button', { class: 'btn small', onclick: () => zoom(r.annotated_url) }, 'View frame') : null, ' ',
        h('button', { class: 'btn small danger', 'aria-label': `Delete alert ${r.id}`, onclick: async () => {
          if (!await confirmBox('Delete this rodent alert and its images?')) return;
          try { await api('POST', '/api/rodents/delete', { ids: [r.id] }); toast('Alert deleted', 'ok'); load(); } catch (e) { fail(e); }
        } }, 'Delete')))));
    syncSel();
    labelCells(tbody.closest('table'));
    pager.replaceChildren(
      h('span', { class: 'muted' }, d.total ? `${offset + 1}–${Math.min(offset + limit, d.total)} of ${d.total}` : '0 results'),
      h('button', { class: 'btn small', disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - limit); load(); } }, '‹ Newer'),
      h('button', { class: 'btn small', disabled: offset + limit >= d.total, onclick: () => { offset += limit; load(); } }, 'Older ›'));
  }
  load();
}

// Camera frames labelled here are added to the rodent training set (scripts/rodent/train.py).
async function pageRodentFrames() {
  const camSel = h('select', { 'aria-label': 'Camera' });
  const everySel = h('select', { 'aria-label': 'Capture interval' },
    [2, 3, 5, 10].map(s => h('option', { value: s, selected: s === 3 }, `every ${s} s`)));
  const forSel = h('select', { 'aria-label': 'Capture duration' },
    [1, 2, 5, 10].map(m => h('option', { value: m, selected: m === 2 }, `for ${m} min`)));
  const showSel = h('select', { 'aria-label': 'Show' }, h('option', { value: '' }, 'All frames'),
    h('option', { value: 'pending' }, 'Not labelled'), h('option', { value: 'rodent' }, 'With rodent'), h('option', { value: 'empty' }, 'No rodent'));
  const counts = h('div', { class: 'muted' });
  const grid = h('div', { class: 'frame-grid' });
  let items = [];
  let auto = null;

  const capture = async () => {
    if (!camSel.value) return toast('No camera running');
    try { await api('POST', '/api/rodent-frames', { camera_id: camSel.value }); load(); } catch (e) { fail(e); }
  };
  const autoBtn = h('button', { class: 'btn', onclick: () => {
    if (auto) return stopAuto();
    const end = Date.now() + Number(forSel.value) * 60000;
    auto = setInterval(() => {
      if (Date.now() > end) return stopAuto();
      autoBtn.textContent = `■ Stop (${Math.ceil((end - Date.now()) / 1000)} s)`;
      capture();
    }, Number(everySel.value) * 1000);
    autoBtn.className = 'btn danger';
    capture();
  } }, '▶ Auto capture');
  function stopAuto() {
    clearInterval(auto); auto = null;
    autoBtn.className = 'btn'; autoBtn.textContent = '▶ Auto capture';
  }
  pageCleanups.push(() => clearInterval(auto));

  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Rodent training frames'), h('a', { class: 'btn', href: '#/rodents' }, '‹ Rodent alerts')),
    h('p', { class: 'muted' }, 'Capture frames from your camera and draw a box around every rodent (or a toy rodent) in them. ',
      'Move it to different places between captures. Also mark some frames with nothing as "No rodent". ',
      'These frames are added to training (scripts/rodent/train.py), which teaches the model what rodents look like on this camera.'),
    h('div', { class: 'panel', style: 'margin-bottom:14px' },
      h('div', { class: 'row' }, camSel, h('button', { class: 'btn primary', onclick: capture }, 'Capture frame'),
        h('span', { class: 'muted' }, 'or'), everySel, forSel, autoBtn)),
    h('div', { class: 'row', style: 'margin-bottom:10px' }, counts, h('span', { style: 'margin-left:auto' }), showSel),
    grid);
  showSel.addEventListener('change', render);

  try {
    const cams = await api('GET', '/api/status');
    cams.forEach(c => camSel.append(h('option', { value: c.id }, c.name)));
  } catch (e) { fail(e); }

  function render() {
    const shown = items.filter(i => !showSel.value || i.status === showSel.value);
    if (!shown.length) return grid.replaceChildren(h('div', { class: 'empty' }, items.length ? 'No frames for this filter.' : 'No frames yet. Capture some from the camera.'));
    grid.replaceChildren(...shown.map(it => h('div', { class: 'frame-tile', onclick: () => frameLabeler(items, items.indexOf(it), load) },
      h('img', { src: it.url, alt: it.name, loading: 'lazy' }),
      h('div', { class: 'meta' }, h('span', { class: 'muted' }, new Date(it.mtime * 1000).toLocaleTimeString()),
        h('span', { class: `badge ${it.status === 'rodent' ? 'rodent' : it.status}` },
          it.status === 'pending' ? 'not labelled' : it.status === 'empty' ? 'no rodent' : `${it.boxes.length} rodent`)))));
  }

  async function load() {
    let d;
    try { d = await api('GET', '/api/rodent-frames'); } catch (e) { return fail(e); }
    items = d.items;
    counts.textContent = `${items.length} frames · ${d.counts.pending} not labelled · ${d.counts.rodent} with rodent · ${d.counts.empty} no rodent`;
    render();
  }
  load();
}

// Draw boxes on a frame: drag to add, Undo / Clear to fix, then save. Opens the next unlabelled frame after saving.
function frameLabeler(items, index, onChange) {
  let i = index, boxes = [], drag = null, zoomLevel = 0;
  const zooms = ['fit', 1, 2];
  const img = new Image();
  const canvas = h('canvas', {});
  const ctx2d = canvas.getContext('2d');
  const wrap = h('div', { class: 'lbl-wrap' }, canvas);
  const info = h('div', { class: 'muted small', style: 'margin-top:8px' });
  const zoomBtn = h('button', { class: 'btn', onclick: () => { zoomLevel = (zoomLevel + 1) % zooms.length; layout(); } }, 'Zoom: fit');

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx2d.drawImage(img, 0, 0, W, H);
    ctx2d.lineWidth = 2;
    ctx2d.strokeStyle = '#ff3b3b';
    boxes.forEach(b => ctx2d.strokeRect(b[0] * W, b[1] * H, (b[2] - b[0]) * W, (b[3] - b[1]) * H));
    if (drag && drag.cur) {
      ctx2d.setLineDash([5, 4]);
      ctx2d.strokeRect(drag.x * W, drag.y * H, (drag.cur.x - drag.x) * W, (drag.cur.y - drag.y) * H);
      ctx2d.setLineDash([]);
    }
    info.textContent = `${i + 1} / ${items.length} · ${items[i].name} · ${boxes.length} box(es) · drag on the image to draw a box around each rodent`;
  }
  function layout() {
    const z = zooms[zoomLevel];
    const s = z === 'fit' ? Math.min(1, (wrap.clientWidth || 900) / img.naturalWidth) : z;
    canvas.width = Math.round(img.naturalWidth * s);
    canvas.height = Math.round(img.naturalHeight * s);
    zoomBtn.textContent = `Zoom: ${z === 'fit' ? 'fit' : z + '×'}`;
    draw();
  }
  function load(n) {
    i = n;
    boxes = (items[i].boxes || []).map(b => [...b]);
    img.onload = layout;
    img.src = items[i].url;
  }
  const pos = e => {
    const r = canvas.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); drag = pos(e); });
  canvas.addEventListener('pointermove', e => { if (drag) { drag.cur = pos(e); draw(); } });
  canvas.addEventListener('pointerup', e => {
    if (!drag) return;
    const p = pos(e);
    const b = [Math.min(drag.x, p.x), Math.min(drag.y, p.y), Math.max(drag.x, p.x), Math.max(drag.y, p.y)];
    drag = null;
    if ((b[2] - b[0]) * canvas.width > 4 && (b[3] - b[1]) * canvas.height > 4) boxes.push(b);
    draw();
  });

  async function save(list) {
    try {
      const r = await api('PUT', `/api/rodent-frames/${enc(items[i].name)}`, { boxes: list });
      items[i] = r;
      onChange();
      const next = items.findIndex((it, k) => k > i && it.status === 'pending');
      if (next >= 0) load(next);
      else { toast('Saved. No more unlabelled frames.', 'ok'); m.close(); }
    } catch (e) { fail(e); }
  }

  const m = modal('Label rodents', h('div', {}, wrap, info), [
    h('button', { class: 'btn left', onclick: () => i > 0 && load(i - 1) }, '‹ Prev'),
    h('button', { class: 'btn', onclick: () => i < items.length - 1 && load(i + 1) }, 'Next ›'),
    zoomBtn,
    h('button', { class: 'btn', onclick: () => { boxes.pop(); draw(); } }, 'Undo box'),
    h('button', { class: 'btn danger', onclick: async () => {
      if (!await confirmBox('Delete this frame?')) return;
      try {
        await api('DELETE', `/api/rodent-frames/${enc(items[i].name)}`);
        items.splice(i, 1); onChange();
        if (!items.length) return m.close();
        load(Math.min(i, items.length - 1));
      } catch (e) { fail(e); }
    } }, 'Delete frame'),
    h('button', { class: 'btn', onclick: () => save([]) }, 'No rodent'),
    h('button', { class: 'btn primary', onclick: () => boxes.length ? save(boxes) : toast('Draw a box first, or use "No rodent"') }, 'Save boxes'),
  ], { wide: true });
  m.el.querySelector('.modal').classList.add('xwide');
  load(i);
}

// ------------------------------------------------------------------ Image test

function pageImages() {
  const input = h('input', { type: 'file', accept: 'image/*', multiple: true });
  const results = h('div', { class: 'results' });
  const drop = h('div', { class: 'drop' }, 'Drop images here or ', input);
  const run = h('button', { class: 'btn primary', onclick: () => detect(input.files) }, 'Detect faces');
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Image test'), run),
    h('p', { class: 'muted' }, 'Upload photos to run face detection (and recognition, if enabled) and rodent detection. Faces are saved to Detections under camera "upload".'),
    drop, results);

  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); detect(e.dataTransfer.files); });

  async function detect(fileList) {
    if (!fileList || !fileList.length) return toast('Choose images first');
    const fd = new FormData();
    [...fileList].forEach(f => fd.append('files', f));
    await busy(run, async () => {
      try {
        const res = await api('POST', '/api/images/detect', fd);
        results.replaceChildren(...res.results.map(r => h('div', { class: 'panel result' },
          h('div', { class: 'row', style: 'margin-bottom:8px' }, h('b', {}, r.file),
            r.ok ? h('span', { class: 'muted small' }, `${r.faces.length} face(s) · ${r.infer_ms} ms · ${r.width}×${r.height}`) : h('span', { class: 'err-text' }, r.error),
            r.ok && r.rodents ? (r.rodents.length
              ? h('span', { class: 'badge rodent' }, `${r.rodents.length} rodent(s) · ${r.rodents.map(x => num(x.confidence, 2)).join(', ')}`)
              : h('span', { class: 'muted small' }, 'no rodent')) : null),
          r.ok ? h('img', { src: r.annotated, alt: `Result for ${r.file}`, onclick: () => zoom(r.annotated) }) : null,
          r.ok && r.faces.length ? h('div', { class: 'small', style: 'margin-top:8px' },
            r.faces.map((f, i) => h('div', { style: 'margin-bottom:4px' }, `#${i + 1} `, h('b', {}, f.name || 'face'),
              ` · conf ${num(f.confidence, 2)}`, f.similarity != null ? ` · sim ${num(f.similarity, 2)}` : '',
              f.quality ? h('div', { class: 'muted' },
                `quality ${num(f.quality.score, 2)} · ${f.quality.size}px · blur ${f.quality.blur} · yaw ${f.quality.yaw}° · pitch ${f.quality.pitch}°`,
                f.reject ? h('span', { class: 'err-text' }, ` · would be skipped: ${f.reject}`) : h('span', { class: 'ok-text' }, ' · good')) : null))) : null)));
      } catch (e) { fail(e); }
    });
  }
}

// ------------------------------------------------------------------ Settings

async function pageSettings() {
  view.append(h('div', { class: 'page-head' }, h('h1', {}, 'Settings')));
  let data, sys;
  try { [data, sys] = await Promise.all([api('GET', '/api/settings'), api('GET', '/api/system')]); } catch (e) { return fail(e); }
  const inputs = {};
  const groups = new Map();
  for (const s of data.schema) {
    if (!groups.has(s.group)) groups.set(s.group, []);
    const v = data.values[s.key];
    let input;
    if (s.type === 'bool') input = h('input', { type: 'checkbox', checked: !!v });
    else if (s.type === 'secret') input = h('input', { type: 'password', autocomplete: 'new-password', placeholder: v ? '•••••••• (saved)' : 'not set' });
    else if (s.type === 'str') input = h('input', { type: 'text', value: v ?? '' });
    else input = h('input', { type: 'number', value: v ?? '', min: s.min, max: s.max, step: s.type === 'int' ? 1 : 0.01 });
    inputs[s.key] = { s, input };
    groups.get(s.group).push(s.type === 'bool'
      ? h('div', { style: 'margin-bottom:12px' }, h('label', { class: 'check' }, input, s.label), s.help ? h('div', { class: 'help' }, s.help) : null)
      : field(s.label, input, s.help));
  }
  const save = h('button', { class: 'btn primary', onclick: () => busy(save, async () => {
    const body = {};
    for (const [k, { s, input }] of Object.entries(inputs)) {
      if (s.type === 'bool') body[k] = input.checked;
      else if (s.type === 'secret') { if (input.value) body[k] = input.value; }
      else if (s.type === 'str') body[k] = input.value.trim();
      else body[k] = Number(input.value);
    }
    try { await api('PUT', '/api/settings', body); toast('Settings saved and applied', 'ok'); route(); } catch (e) { fail(e); }
  }) }, 'Save settings');

  view.firstElementChild.append(save);
  view.append(h('div', { class: 'settings-grid' },
    [...groups].map(([g, els]) => h('section', { class: 'panel' }, h('h2', {}, g), els)),
    h('section', { class: 'panel' }, h('h2', {}, 'System (config.yaml, restart to change)'),
      h('table', {}, h('tbody', {}, [
        ['Mode', sys.mode], ['Device', sys.device.toUpperCase()], ['Model', sys.model],
        ['Detector input', sys.det_size.join('×')], ['Uptime', `${Math.round(sys.uptime_sec / 60)} min`],
        ['API docs', h('a', { href: '/api/docs', target: '_blank' }, '/api/docs')],
      ].map(([k, v]) => h('tr', {}, h('td', { class: 'muted' }, k), h('td', {}, v))))))));
}

// ------------------------------------------------------------------ boot
refreshSys();
route();
