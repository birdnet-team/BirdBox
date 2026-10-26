// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * BirdBox browser demo (the Demo tab of the docs): bird vocalization boxes
 * with the self-contained Just-Bird ONNX graph (PCEN, YOLO, NMS inside).
 *
 * The graph runs once per recording at the lowest confidence and returns its
 * per-window boxes. fusion.js merges them exactly like the graph's window
 * fusion for any (confidence, phrase gap) setting, so both sliders update
 * instantly.
 *
 * On the spectrogram, the selected box can be moved and resized and new
 * boxes can be drawn. Exports use the BirdBox output formats.
 */
import { AudioSource, MAX_SECONDS } from './audio-source.js';
import { SAMPLE_RATE as SR } from './audio.js';
import { detector, modelEntry, RAW_CONF } from './detector.js';
import { fuseBoxes } from './fusion.js';
import { player } from './player.js';
import { SpecView } from './spec-view.js';
import { csvField, downloadFile, fill, fmtBytes, fmtNumber, fmtTime, icon, statLine, toast } from './util.js';

const MS_PER_CLIP_KEY = 'birdbox-demo-ms-per-clip';
const CLIP = 3, HOP = 1.5;       // the graph's window grid (onnx_export.py)
const NMS_IOU = 0.7;             // baked into the graph (DEFAULT_NMS_IOU)
const FREQ_MAX = 15000;          // top of the spectrogram and of any box
const MIN_DUR = 0.05;            // smallest box: seconds
const MIN_BAND = 100;            // smallest box: Hz
const COLORS = { detected: '#22d3ee', verified: '#4ade80', drawn: '#fbbf24', removed: 'rgba(148,163,184,0.7)' };

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Detector boxes keep their identity through slider changes by their geometry. */
const boxKey = b => `${b.t0.toFixed(4)}|${b.t1.toFixed(4)}|${b.flo}|${b.fhi}`;
const geomOf = g => ({ t0: g.t0, t1: g.t1, flo: g.flo, fhi: g.fhi });
const round = (v, digits) => Number(v.toFixed(digits));

class Demo {
  constructor() {
    this.audio = null;
    this.raw = null;               // { data, rows, ms }: raw_all of the current recording
    this.songs = [];               // detector boxes at the current sliders
    this.userBoxes = [];           // boxes drawn by the user
    this.boxes = [];               // both, ordered by start time
    this.edits = new Map();        // box id -> { include?, verified?, geom? }
    this.selectedId = null;
    this.hoverId = null;
    this.drawMode = false;
    this.gesture = null;           // move, resize or draw in progress
    this.live = null;              // { id, g }: geometry while dragging
    this.draft = null;             // box being drawn
    this.userSeq = 0;
    this.runToken = 0;
  }

  init() {
    this.source = new AudioSource($('bbd-source'));
    this.spec = new SpecView($('bbd-spec'), {
      height: this._specHeight(), kind: 'stft', kinds: { stft: 'STFT', pcen: 'PCEN' },
      axis: { type: 'linear', fMin: 0, fMax: FREQ_MAX },
      toolbar: document.querySelector('#bbd-viewer .spec-tools-host'),
    });
    window.addEventListener('resize', () => this.spec.setHeight(this._specHeight()));
    this.spec.overlay = (ctx, h) => this._drawBoxes(ctx, h);
    this.spec.minimapOverlay = (ctx, h) => {
      for (const b of this.boxes) {
        const g = this._geom(b);
        ctx.fillStyle = this._color(b);
        ctx.fillRect(h.timeToX(g.t0), h.height - 4, Math.max(1, h.timeToX(g.t1) - h.timeToX(g.t0)), 4);
      }
    };
    this.spec.editor = {
      cursor: p => this._editCursor(p),
      down: p => this._editDown(p),
      move: p => this._editMove(p),
      up: (p, commit) => this._editUp(commit),
    };
    this.spec.tools?.insertAdjacentHTML('afterbegin',
      '<button type="button" class="spec-tool" data-role="help" aria-expanded="false" aria-controls="bbd-help" title="How to edit boxes" aria-label="How to edit boxes">' + icon('help') + '</button>'
      + '<button type="button" class="spec-tool" data-role="draw" aria-pressed="false" title="Draw a box (B)" aria-label="Draw a box (B)">' + icon('draw') + '</button>');
    this.drawBtn = this.spec.tools?.querySelector('[data-role="draw"]');
    this.drawBtn?.addEventListener('click', e => { e.stopPropagation(); this._setDrawMode(!this.drawMode); });
    this.helpBtn = this.spec.tools?.querySelector('[data-role="help"]');
    this.helpBtn?.addEventListener('click', e => { e.stopPropagation(); this._showHelp($('bbd-help').hidden); });
    $('bbd-help').querySelector('[data-role="close"]').addEventListener('click', () => this._showHelp(false));
    this._showHelp(false);   // closed until the ? button opens it
    $('bbd-conf').min = RAW_CONF;
    this._bind();
    modelEntry().then(m => {
      this.model = m;
      if (m.phrase_gap != null) { $('bbd-gap').value = m.phrase_gap; this._syncSliders(); }
      $('bbd-model-name').textContent = m.label || m.file;
    }).catch(err => this._error(err.message));
    this._refresh();
  }

  /** The spectrogram takes what the page leaves it, within limits. */
  _specHeight() {
    return Math.round(clamp(window.innerHeight * 0.42, 240, 560));
  }

  _bind() {
    this.source.addEventListener('audio', e => this._onAudio(e.detail));
    for (const id of ['bbd-conf', 'bbd-gap']) $(id).addEventListener('input', () => { this._syncSliders(); this._update(); });
    this._syncSliders();
    $('bbd-export').addEventListener('click', e => {
      const btn = e.target.closest('[data-export]');
      if (btn) this._export(btn.dataset.export);
    });
    $('bbd-rerun').addEventListener('click', () => this.run());
    $('bbd-stop').addEventListener('click', () => {
      this.runToken++;
      detector.abort();
      this._setProgress(null);
      $('bbd-rerun').hidden = false;
      this._renderHint('Detection stopped. Run it again with ↻.');
    });

    player.addEventListener('time', e => {
      this.spec.setCursor(e.detail);
      if (player.playing) this.spec.follow(e.detail);
    });
    this.spec.addEventListener('toggleplay', () => player.toggle());
    this.spec.addEventListener('kind', () => { if (this.audio) this._streamSpectrogram({ keepView: true }); });
    this.spec.addEventListener('pick', e => {
      const { time, freq } = e.detail;
      const hit = freq != null ? this._boxAt(time, freq) : null;
      // No scrolling here: the page stays where the person is looking
      if (hit) { this._select(hit.id, { play: true }); return; }
      if (this.selectedId != null) this._select(null);
      player.play(time);
    });
    this.spec.addEventListener('hover', e => {
      const d = e.detail;
      const b = d && d.freq != null && !this.gesture && !this.drawMode ? this._boxAt(d.time, d.freq) : null;
      if ((b?.id ?? null) !== this.hoverId) {
        this.hoverId = b?.id ?? null;
        this._markRow('is-hover', this.hoverId);
        this.spec.render();
      }
      if (!b) { this.spec.showTooltip(null); return; }
      const g = this._geom(b), n = this.boxes.indexOf(b) + 1, edit = this._edit(b);
      const origin = b.manual ? 'Drawn by you'
        : `Confidence <b>${fmtNumber(b.avg, 2)}</b> · max ${fmtNumber(b.max, 2)} · ${b.n} window boxes${edit.verified ? ' · verified' : ''}`;
      this.spec.showTooltip(`<div class="tt-time">#${n} · ${fmtTime(g.t0, 2)} – ${fmtTime(g.t1, 2)}</div><div>${fmtNumber(g.flo)}–${fmtNumber(g.fhi)} Hz</div><div>${origin}</div>`, d.x, d.y);
    });
    // Editing keys work anywhere on the page except while typing in a field.
    // Caught on the way down and stopped there: Material's own shortcuts would
    // otherwise also see them (N and P open the next and previous docs page).
    window.addEventListener('keydown', e => {
      if (!this.audio || e.ctrlKey || e.metaKey || e.altKey) return;
      // The volume fader is no text field: M and Space work there too
      if (e.target.closest?.('input:not([type="range"]), select, textarea, [contenteditable], .md-search')) return;
      const selected = this.boxes.find(b => b.id === this.selectedId);
      if (e.key === 'Escape') {
        if (this.drawMode || this.gesture) this._setDrawMode(false);
        else if (selected) this._select(null);
        else return;
      } else if (e.key === 'b' || e.key === 'B') { if (this.raw) this._setDrawMode(!this.drawMode); }
      else if ((e.key === 'v' || e.key === 'V') && selected) this._toggleVerified(selected);
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selected) this._removeBox(selected);
      // Taken even without boxes, so a stray N or P never leaves a loaded recording
      else if (e.key === 'n' || e.key === 'p') { if (this.boxes.length) this._step(e.key === 'n' ? 1 : -1); }
      // Space plays and pauses, also on the mute button (which it would otherwise toggle again)
      else if (e.key === ' ' && !e.target.closest?.('button:not([data-role="mute"]), a, .spec-main')) player.toggle();
      else if (e.key === 'm' || e.key === 'M') player.toggleMute();
      else return;
      e.preventDefault();
      e.stopPropagation();
    }, true);

    const table = $('bbd-table');
    table.addEventListener('click', e => {
      const tr = e.target.closest('tr[data-i]');
      const b = tr ? this.boxes[+tr.dataset.i] : null;
      if (!b) return;
      if (e.target.closest('[data-act="play"]')) this._select(b.id, { play: true });
      else if (e.target.closest('[data-act="toggle"]')) this._removeBox(b);
      else if (e.target.closest('[data-act="verify"]')) this._toggleVerified(b);
      else if (e.target.closest('[data-act="reset-geom"]')) { this._override(b, { geom: undefined }); this._refresh(); }
      else this._select(b.id);
    });
    table.addEventListener('mouseover', e => {
      const id = this.boxes[+e.target.closest('tr[data-i]')?.dataset.i]?.id ?? null;
      if (id !== this.hoverId) { this.hoverId = id; this.spec.render(); }
    });
    table.addEventListener('mouseleave', () => { this.hoverId = null; this.spec.render(); });
  }

  _syncSliders() {
    const conf = +$('bbd-conf').value, gap = +$('bbd-gap').value;
    $('bbd-conf-val').textContent = fmtNumber(conf, 2);
    $('bbd-gap-val').textContent = gap > 0 ? `${fmtNumber(gap, 2)} s` : 'off';
  }

  _showHelp(show) {
    $('bbd-help').hidden = !show;
    this.helpBtn?.classList.toggle('is-active', show);
    this.helpBtn?.setAttribute('aria-expanded', String(show));
  }

  /** What the pointer does right now, under the spectrogram. */
  _renderHint(text) {
    if (!text) {
      if (!this.raw) text = 'Boxes appear here when detection is done.';
      else if (this.drawMode) text = 'Drag across a sound to draw a box · Esc cancels';
      else if (this.selectedId != null) text = 'Drag the box to move it, or its edges to resize · V verifies · Delete removes · N / P next and previous · Esc deselects';
      else text = 'Click a box to hear and select it · B draws a new one · Ctrl + wheel zooms';
    }
    $('bbd-edit-hint').textContent = text;
  }

  _setProgress(fraction, text) {
    const wrap = $('bbd-progress');
    if (fraction == null) { wrap.hidden = true; return; }
    wrap.hidden = false;
    wrap.querySelector('.bbd-progress__bar').style.width = `${Math.round(clamp(fraction, 0.02, 1) * 100)}%`;
    $('bbd-progress-text').textContent = text || '';
  }

  _error(message) {
    $('bbd-error').hidden = !message;
    $('bbd-error').textContent = message || '';
  }

  _onAudio(audio) {
    this.runToken++;
    detector.abort();
    this.audio = audio;
    this.raw = null;
    this.songs = [];
    this.userBoxes = [];
    this.edits.clear();
    this.selectedId = null;
    this.hoverId = null;
    this._setDrawMode(false);
    $('bbd').classList.add('has-audio');
    $('bbd-rerun').hidden = true;
    const notes = [];
    if (audio.truncatedFrom) notes.push(`The demo analyzes the first ${fmtTime(MAX_SECONDS)} of this ${fmtTime(audio.truncatedFrom)} recording. Use the CLI for longer files.`);
    if (audio.pcm.length < CLIP * SR) this._error('BirdBox needs at least 3 seconds of audio.');
    else this._error(null);
    $('bbd-note').hidden = !notes.length;
    $('bbd-note').textContent = notes.join(' ');
    this.spec.clear();
    this._streamSpectrogram();
    this.spec.setCursor(0);
    this._update();
    this.run();
  }

  /** STFT for reading the sounds, or PCEN, the mel spectrogram the model detects on. */
  _streamSpectrogram({ keepView = false } = {}) {
    this.specWorker?.terminate();
    const worker = this.specWorker = new Worker(new URL('./spec-worker.js', import.meta.url));
    const duration = this.audio.duration, view = { ...this.spec.view };
    worker.onmessage = ({ data }) => {
      if (data.type === 'meta') {
        this.spec.setSource({ ...data, duration, floor: data.floor || 0 });
        if (keepView) this.spec.setView(view.t0, view.t1, { silent: true });
      }
      else if (data.type === 'slice') this.spec.addSlice(data.start, data.data);
      else if (data.type === 'error') console.error('spectrogram:', data.error);
      if (data.type === 'done' || data.type === 'error') {
        worker.terminate();
        if (this.specWorker === worker) this.specWorker = null;
      }
    };
    worker.postMessage({ type: this.spec.kind, id: 1, pcm: this.audio.pcm, sampleRate: SR, fMax: FREQ_MAX });
  }

  async run() {
    if (!this.audio || this.audio.pcm.length < CLIP * SR) return;
    const token = ++this.runToken;
    this.raw = null;
    $('bbd-rerun').hidden = true;
    this._update();
    let ticker = null;
    try {
      this._setProgress(0.01, 'Preparing BirdBox…');
      await detector.ensure(p => {
        if (token !== this.runToken) return;
        if (p.starting) this._setProgress(0.3, 'Starting ONNX Runtime…');
        else if (p.total) this._setProgress(p.loaded / p.total * 0.3, fill('Downloading the model: {done} of {total}', { done: fmtBytes(p.loaded), total: fmtBytes(p.total) }));
        else this._setProgress(0.15, fill('Downloading the model: {done}', { done: fmtBytes(p.loaded) }));
      });
      this._renderModel();
      if (token !== this.runToken) return;
      const clips = Math.max(1, Math.ceil((this.audio.duration - CLIP) / HOP) + 1);
      let msPerClip = 280;
      try { msPerClip = +localStorage.getItem(MS_PER_CLIP_KEY) || msPerClip; } catch { /* storage blocked */ }
      const t0 = performance.now();
      // ONNX Runtime reports no progress inside one run; pace a counter on the clip grid.
      ticker = setInterval(() => {
        const elapsed = performance.now() - t0, frac = Math.min(0.95, elapsed / (clips * msPerClip));
        const left = (clips * msPerClip - elapsed) / 1000;
        let text = fill('Detecting vocalizations · window {n} of {total}', { n: Math.max(1, Math.min(clips, Math.round(frac * clips))), total: clips });
        if (left > 20) text += ` · about ${fmtTime(Math.ceil(left / 10) * 10)} left`;
        this._setProgress(0.3 + 0.7 * frac, text);
      }, 100);
      const res = await detector.detect(this.audio.pcm.slice());
      const ms = performance.now() - t0;
      try { localStorage.setItem(MS_PER_CLIP_KEY, String(Math.round(ms / clips))); } catch { /* not persisted */ }
      if (token !== this.runToken) return;
      this.raw = { data: res.data, rows: res.rows, ms };
    } catch (err) {
      if (token !== this.runToken || err.cancelled) return;
      console.error(err);
      this._error(fill('BirdBox failed: {message}', { message: err.message }));
      $('bbd-rerun').hidden = false;
    } finally {
      clearInterval(ticker);
      if (token === this.runToken) {
        this._setProgress(null);
        this._renderModel();
        this._update();
      }
    }
  }

  _renderModel() {
    const chip = $('bbd-model-state');
    chip.textContent = detector.loaded ? 'loaded' : '';
    chip.hidden = !detector.loaded;
  }

  /* ---------------- boxes + edits ---------------- */

  _update() {
    const conf = +$('bbd-conf').value, gap = +$('bbd-gap').value;
    this.songs = this.raw ? fuseBoxes(this.raw.data, this.raw.rows, conf, gap) : [];
    this.songs.forEach(b => { b.id = boxKey(b); });
    this._refresh();
  }

  /** Rebuild the combined box list and everything drawn from it. */
  _refresh() {
    this.boxes = [...this.songs, ...this.userBoxes].sort((a, b) => this._geom(a).t0 - this._geom(b).t0);
    if (this.selectedId != null && !this.boxes.some(b => b.id === this.selectedId)) this.selectedId = null;
    this.spec.main.classList.toggle('is-editing', this.drawMode || this.selectedId != null);
    this._renderSummary();
    this._renderTable();
    this._renderHint();
    this.spec.render();
  }

  /** Box geometry with the user's changes (and the live one while dragging). */
  _geom(b) {
    if (this.live?.id === b.id) return this.live.g;
    return this.edits.get(b.id)?.geom || b;
  }

  _edit(b) {
    const o = this.edits.get(b.id) || {};
    return {
      include: o.include ?? true,
      verified: !!b.manual || !!o.verified,   // drawn boxes are the person's own
      adjusted: !b.manual && !!o.geom,
    };
  }

  _color(b) {
    const edit = this._edit(b);
    return !edit.include ? COLORS.removed : b.manual ? COLORS.drawn : edit.verified ? COLORS.verified : COLORS.detected;
  }

  /** Verify a detected box (or take it back); drawn boxes always count as verified. */
  _toggleVerified(b) {
    if (!b || b.manual) return;
    this._override(b, { verified: !this._edit(b).verified || undefined, include: undefined });
    this._refresh();
  }

  _override(b, patch) {
    const next = { ...(this.edits.get(b.id) || {}), ...patch };
    Object.keys(next).forEach(k => { if (next[k] === undefined) delete next[k]; });
    this.edits.set(b.id, next);
  }

  /** Drawn boxes are deleted (with undo); detected ones are toggled off and on. */
  _removeBox(b) {
    if (!b) return;
    if (!b.manual) {
      this._override(b, { include: !this._edit(b).include });
      this._refresh();
      return;
    }
    const n = this.boxes.indexOf(b) + 1;
    this.userBoxes = this.userBoxes.filter(x => x !== b);
    this._refresh();
    toast(`Box ${n} deleted.`, { label: 'Undo', onClick: () => { this.userBoxes.push(b); this._refresh(); } });
  }

  /** Select the next or previous box and play it. */
  _step(dir) {
    const i = this.boxes.findIndex(b => b.id === this.selectedId);
    const next = this.boxes[i < 0 ? (dir > 0 ? 0 : this.boxes.length - 1) : clamp(i + dir, 0, this.boxes.length - 1)];
    this._select(next.id, { play: true, scroll: true });
  }

  /** Smallest box under a point; the selected one wins when it is there. */
  _boxAt(time, freq) {
    let best = null, bestArea = Infinity;
    for (const b of this.boxes) {
      const g = this._geom(b);
      if (time < g.t0 || time > g.t1 || freq < g.flo * 0.97 || freq > g.fhi * 1.03) continue;
      if (b.id === this.selectedId) return b;
      const area = (g.t1 - g.t0) * (g.fhi - g.flo);
      if (area < bestArea) { best = b; bestArea = area; }
    }
    return best;
  }

  /* ---------------- editing on the spectrogram ---------------- */

  _setDrawMode(on) {
    this.drawMode = on;
    this.gesture = null;
    this.draft = null;
    this.live = null;
    this.drawBtn?.classList.toggle('is-active', on);
    this.drawBtn?.setAttribute('aria-pressed', String(on));
    this.spec.main.classList.toggle('is-editing', on || this.selectedId != null);
    if (on) { this.hoverId = null; this.spec.showTooltip(null); }
    this._renderHint();
    this.spec.render();
  }

  /** Which part of the selected box is under the pointer: an edge, a corner or its body. */
  _handleAt(p) {
    const b = this.selectedId != null && this.boxes.find(x => x.id === this.selectedId);
    if (!b) return null;
    const g = this._geom(b), s = this.spec;
    const x0 = s.timeToX(g.t0), x1 = s.timeToX(g.t1), y0 = s.freqToY(g.fhi), y1 = s.freqToY(g.flo);
    const out = p.pointerType === 'mouse' ? 6 : 14;          // grab distance outside the box
    const inX = Math.min(out, Math.max(3, (x1 - x0) / 3));   // and inside, so small boxes stay movable
    const inY = Math.min(out, Math.max(3, (y1 - y0) / 3));
    if (p.x < x0 - out || p.x > x1 + out || p.y < y0 - out || p.y > y1 + out) return null;
    const edges = {
      l: p.x <= x0 + inX && p.x >= x0 - out,
      r: p.x >= x1 - inX && p.x <= x1 + out,
      t: p.y <= y0 + inY && p.y >= y0 - out,
      b: p.y >= y1 - inY && p.y <= y1 + out,
    };
    if (edges.l && edges.r) edges[p.x - x0 < x1 - p.x ? 'r' : 'l'] = false;
    if (edges.t && edges.b) edges[p.y - y0 < y1 - p.y ? 'b' : 't'] = false;
    const onEdge = edges.l || edges.r || edges.t || edges.b;
    const inside = p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
    if (!onEdge && !inside) return null;
    return { b, g, edges, move: !onEdge };
  }

  _editCursor(p) {
    if (this.drawMode) return 'crosshair';
    const h = this._handleAt(p);
    if (h) {
      const { l, r, t: top, b: bottom } = h.edges;
      if ((l && top) || (r && bottom)) return 'nwse-resize';
      if ((r && top) || (l && bottom)) return 'nesw-resize';
      if (l || r) return 'ew-resize';
      if (top || bottom) return 'ns-resize';
      return 'move';
    }
    return this._boxAt(p.time, p.freq) ? 'pointer' : '';
  }

  _editDown(p) {
    if (!this.raw) return false;
    if (this.drawMode) {
      this.gesture = { type: 'draw', start: p };
      return true;
    }
    const h = this._handleAt(p);
    if (!h) return false;
    this.gesture = { type: h.move ? 'move' : 'resize', id: h.b.id, start: p, from: geomOf(h.g), edges: h.edges };
    this.hoverId = null;
    this.spec.showTooltip(null);
    return true;
  }

  _editMove(p) {
    const gs = this.gesture;
    if (!gs) return;
    const dur = this.audio.duration, time = clamp(p.time, 0, dur), freq = clamp(p.freq, 0, FREQ_MAX);
    if (gs.type === 'draw') {
      const a = { time: clamp(gs.start.time, 0, dur), freq: clamp(gs.start.freq, 0, FREQ_MAX) };
      this.draft = { t0: Math.min(a.time, time), t1: Math.max(a.time, time), flo: Math.min(a.freq, freq), fhi: Math.max(a.freq, freq) };
    } else {
      let { t0, t1, flo, fhi } = gs.from;
      if (gs.type === 'move') {
        const dt = clamp(p.time - gs.start.time, -t0, dur - t1);
        const df = clamp(p.freq - gs.start.freq, -flo, FREQ_MAX - fhi);
        t0 += dt; t1 += dt; flo += df; fhi += df;
      } else {
        if (gs.edges.l) t0 = Math.min(time, t1 - MIN_DUR);
        if (gs.edges.r) t1 = Math.max(time, t0 + MIN_DUR);
        if (gs.edges.t) fhi = Math.max(freq, flo + MIN_BAND);
        if (gs.edges.b) flo = Math.min(freq, fhi - MIN_BAND);
      }
      this.live = { id: gs.id, g: { t0, t1, flo, fhi } };
    }
    this.spec.render();
  }

  _editUp(commit) {
    const gs = this.gesture, draft = this.draft, live = this.live;
    this.gesture = null;
    this.draft = null;
    this.live = null;
    if (!gs) return;
    if (gs.type === 'draw') {
      if (commit && draft && draft.t1 - draft.t0 >= MIN_DUR && draft.fhi - draft.flo >= MIN_BAND) {
        const box = { id: `user-${++this.userSeq}`, manual: true, ...draft };
        this.userBoxes.push(box);
        this.selectedId = box.id;
        this._setDrawMode(false);
        this._refresh();
      } else this.spec.render();
      return;
    }
    const b = this.boxes.find(x => x.id === gs.id);
    if (commit && live && b) {
      if (b.manual) Object.assign(b, live.g);
      else this._override(b, { geom: live.g, verified: true });   // adjusting a box is a human judgment
    }
    this._refresh();
  }

  /* ---------------- rendering ---------------- */

  _counts() {
    const c = { all: this.boxes.length, included: 0, verified: 0, adjusted: 0, drawn: 0, removed: 0 };
    for (const b of this.boxes) {
      const edit = this._edit(b);
      if (!edit.include) { c.removed++; continue; }
      c.included++;
      if (b.manual) c.drawn++;
      else if (edit.verified) c.verified++;
      if (edit.adjusted) c.adjusted++;
    }
    return c;
  }

  _renderSummary() {
    const el = $('bbd-summary'), ok = !!this.raw;
    $('bbd-export').querySelectorAll('button').forEach(btn => { btn.disabled = !ok || !this._included().length; });
    if (!ok) { el.innerHTML = ''; return; }
    const c = this._counts();
    const stats = [[c.included, c.included === 1 ? 'box' : 'boxes']];
    if (c.verified) stats.push([c.verified, 'verified']);
    if (c.drawn) stats.push([c.drawn, 'drawn']);
    if (c.removed) stats.push([c.removed, 'removed']);
    stats.push([`${fmtNumber(this.raw.ms / 1000, 1)} s`, 'detection']);
    el.innerHTML = statLine(stats);
  }

  _renderTable() {
    const tbody = $('bbd-table').querySelector('tbody');
    $('bbd-results').classList.toggle('has-results', !!this.raw);
    if (!this.raw) {
      tbody.innerHTML = `<tr class="bbd-table__empty"><td colspan="5">${this.audio ? 'Detection is running…' : 'Load a recording to see its boxes here.'}</td></tr>`;
      return;
    }
    if (!this.boxes.length) {
      tbody.innerHTML = '<tr class="bbd-table__empty"><td colspan="5">No bird vocalizations above this confidence. Try a lower threshold.</td></tr>';
      return;
    }
    tbody.innerHTML = this.boxes.map((b, i) => {
      const g = this._geom(b), edit = this._edit(b);
      const mark = b.manual
        ? `<span class="bbd-mark" title="Drawn by you">${icon('pencil')}</span>`
        : edit.adjusted ? `<button type="button" class="bbd-mark bbd-iconbtn" data-act="reset-geom" title="Restore the detected box" aria-label="Restore the detected box">${icon('reset')}</button>` : '';
      const removeLabel = b.manual ? 'Delete box (Delete)' : edit.include ? 'Remove box (Delete)' : 'Restore box';
      const removeIcon = b.manual ? 'trash' : edit.include ? 'close' : 'reset';
      const verifyLabel = edit.verified ? 'Verified · click to undo (V)' : 'Verify box (V)';
      const verify = b.manual
        ? `<span class="bbd-verify is-on is-drawn" title="Drawn by you">${icon('check-on')}</span>`
        : `<button type="button" class="bbd-verify bbd-iconbtn${edit.verified ? ' is-on' : ''}" data-act="verify" aria-pressed="${edit.verified}" title="${verifyLabel}" aria-label="${verifyLabel}">${icon(edit.verified ? 'check-on' : 'check')}</button>`;
      const conf = b.manual ? '<td class="bbd-c-conf" title="Drawn by you">–</td>'
        : `<td class="bbd-c-conf" title="max ${fmtNumber(b.max, 3)} · ${b.n} window boxes">${fmtNumber(b.avg, 2)}<span>max ${fmtNumber(b.max, 2)}</span></td>`;
      return `<tr data-i="${i}" class="${b.id === this.selectedId ? 'is-selected' : ''}${edit.include ? '' : ' is-excluded'}" style="--box:${this._color(b)}">
        <td class="bbd-c-play"><button type="button" class="bbd-play" data-act="play" aria-label="Play box ${i + 1}">${icon('play')}${i + 1}</button></td>
        <td class="bbd-c-time">${fmtTime(g.t0, 2)} – ${fmtTime(g.t1, 2)}<span>${fmtNumber(g.t1 - g.t0, 2)} s</span></td>
        <td class="bbd-c-freq">${fmtNumber(g.flo / 1000, 1)} – ${fmtNumber(g.fhi / 1000, 1)} kHz${mark}</td>
        ${conf}
        <td class="bbd-c-act">${verify}<button type="button" class="bbd-iconbtn" data-act="toggle" title="${removeLabel}" aria-label="${removeLabel}">${icon(removeIcon)}</button></td>
      </tr>`;
    }).join('');
  }

  _markRow(cls, id) {
    const i = this.boxes.findIndex(b => b.id === id);
    document.querySelectorAll('#bbd-table tr[data-i]').forEach(tr => tr.classList.toggle(cls, +tr.dataset.i === i));
  }

  /**
   * Select a box (or none): highlight its row, bring the box into the
   * spectrogram view, optionally play it. scroll moves the table's own list to
   * the row (for N / P); the page itself never scrolls.
   */
  _select(id, { play = false, scroll = false } = {}) {
    this.selectedId = id;
    const i = this.boxes.findIndex(b => b.id === id), b = this.boxes[i];
    this._markRow('is-selected', id);
    this.spec.main.classList.toggle('is-editing', this.drawMode || !!b);
    if (b) {
      if (scroll) this._revealRow(i);
      const g = this._geom(b), span = this.spec.view.t1 - this.spec.view.t0;
      if (g.t0 < this.spec.view.t0 || g.t1 > this.spec.view.t1) this.spec.setView(g.t0 - span * 0.3, g.t0 - span * 0.3 + span);
      if (play) player.play(Math.max(0, g.t0 - 0.15), g.t1 + 0.15);
    }
    this._renderHint();
    this.spec.render();
  }

  /** Scroll the table's list (not the page) just enough to show row i below the sticky header. */
  _revealRow(i) {
    const list = $('bbd-results'), row = list.querySelector(`tr[data-i="${i}"]`);
    if (!row) return;
    const r = row.getBoundingClientRect(), c = list.getBoundingClientRect();
    const top = c.top + list.querySelector('thead').offsetHeight;
    if (r.top < top) list.scrollBy({ top: r.top - top, behavior: 'smooth' });
    else if (r.bottom > c.bottom) list.scrollBy({ top: r.bottom - c.bottom, behavior: 'smooth' });
  }

  _drawBoxes(ctx, h) {
    ctx.font = '600 11px Roboto, system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    this.boxes.forEach((b, i) => {
      const g = this._geom(b), x0 = h.timeToX(g.t0), x1 = h.timeToX(g.t1);
      if (x1 < 0 || x0 > h.width) return;
      const y0 = h.freqToY(g.fhi), y1 = h.freqToY(g.flo), edit = this._edit(b);
      const selected = b.id === this.selectedId, active = selected || b.id === this.hoverId;
      const base = this._color(b), color = active && edit.include ? '#ffffff' : base;
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.setLineDash(edit.include ? [] : [4, 3]);
      ctx.strokeRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0));
      ctx.setLineDash([]);
      if (active) {
        ctx.fillStyle = base;
        ctx.globalAlpha = 0.08;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.globalAlpha = 1;
      }
      const num = `${i + 1}${edit.verified && !b.manual ? ' ✓' : ''}`;
      let label = b.manual || edit.verified ? num : `${num} · ${fmtNumber(b.avg, 2)}`;
      // Keep crowded views readable: numbers only unless the label fits or the box is active
      if (!active && ctx.measureText(label).width > Math.max(x1 - x0, 0) + 24) label = num;
      const tw = ctx.measureText(label).width + 8, ly = Math.max(14, y0 - 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x0, ly - 13, tw, 15);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#05040c';
      ctx.fillText(label, x0 + 4, ly - 2);
      // Handles on the selected box show that it can be moved and resized
      if (selected && !this.drawMode && edit.include) {
        ctx.fillStyle = '#ffffff';
        const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
        for (const [hx, hy] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1], [xm, y0], [xm, y1], [x0, ym], [x1, ym]]) ctx.fillRect(hx - 3, hy - 3, 6, 6);
      }
    });
    if (this.draft) {
      const d = this.draft, x0 = h.timeToX(d.t0), x1 = h.timeToX(d.t1), y0 = h.freqToY(d.fhi), y1 = h.freqToY(d.flo);
      ctx.fillStyle = 'rgba(251,191,36,0.14)';
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = COLORS.drawn;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
  }

  /* ---------------- export ---------------- */

  /** Included boxes as BirdBox detections, numbered as in the table. */
  _included() {
    return this.boxes.map((b, i) => ({ b, i, g: this._geom(b), edit: this._edit(b) })).filter(r => r.edit.include);
  }

  /** How a box came about: detected, verified, adjusted (and verified) or drawn. */
  _review(r) {
    return r.b.manual ? 'drawn' : r.edit.adjusted ? 'adjusted' : r.edit.verified ? 'verified' : 'detected';
  }

  _baseName() { return (this.audio?.name || 'audio').replace(/\.[^.]+$/, ''); }

  _detections() {
    return this._included().map(r => ({
      species: 'bird',
      species_id: 0,
      time_start: round(r.g.t0, 4),
      time_end: round(r.g.t1, 4),
      freq_low_hz: round(r.g.flo, 1),
      freq_high_hz: round(r.g.fhi, 1),
      // Drawn boxes have no detector score
      avg_confidence: r.b.manual ? null : round(r.b.avg, 6),
      max_confidence: r.b.manual ? null : round(r.b.max, 6),
      detections_merged: r.b.manual ? 0 : Math.round(r.b.n),
      review: this._review(r),
    }));
  }

  _export(kind) {
    const base = this._baseName(), name = this.audio?.name || 'audio';
    const dets = this._detections();
    if (!dets.length) return;
    const conf = +$('bbd-conf').value, gap = +$('bbd-gap').value;
    if (kind === 'json') {
      // Layout of detect_birds.py --output-format json-with-algorithm-metadata
      const payload = {
        audio_file: name,
        model_config: {
          model: this.model?.file || 'Just-Bird_fp16.onnx',
          confidence_threshold: conf,
          nms_iou_threshold: NMS_IOU,
          species_mapping: 'Just-Bird',
          merge_method: 'fusion',
          phrase_gap: gap,
        },
        demo: {
          source: 'BirdBox browser demo',
          review: 'detected: unchanged detector box; verified: checked by the user; adjusted: moved or resized by the user; drawn: added by the user',
          removed_by_user: this._counts().removed,
        },
        detection_count: dets.length,
        detections: dets,
      };
      downloadFile(`${base}.with_algorithm_metadata.json`, `${JSON.stringify(payload, null, 2)}\n`, 'application/json');
    } else if (kind === 'csv') {
      // Layout of --output-format simplified-csv (times to 0.1 s, as the CLI writes them)
      const lines = ['Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence'];
      for (const d of dets) {
        lines.push([csvField(name), d.time_start.toFixed(1), d.time_end.toFixed(1), Math.round(d.freq_low_hz), Math.round(d.freq_high_hz), d.species,
          d.avg_confidence == null ? '' : d.avg_confidence.toFixed(3)].join(','));
      }
      downloadFile(`${base}.simplified.csv`, `${lines.join('\n')}\n`, 'text/csv;charset=utf-8');
    } else if (kind === 'raven') {
      // Layout of --output-format raven-selection-table
      const lines = [['Selection', 'View', 'Channel', 'Begin Time (S)', 'End Time (S)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Annotation'].join('\t')];
      dets.forEach((d, i) => lines.push([i + 1, 'Spectrogram 1', 1, d.time_start.toFixed(3), d.time_end.toFixed(3), Math.round(d.freq_low_hz), Math.round(d.freq_high_hz), d.species].join('\t')));
      downloadFile(`${base}.raven_selection_table.txt`, `${lines.join('\n')}\n`);
    } else if (kind === 'table') {
      // Everything in the table at full precision, like the earlier ONNX demo's CSV
      const header = ['box', 'time_start', 'time_end', 'freq_low_hz', 'freq_high_hz', 'avg_confidence', 'max_confidence', 'detections_merged', 'class_id', 'species', 'review'];
      const lines = [header.join(',')];
      this._included().forEach((r, k) => {
        const d = dets[k];
        lines.push([r.i + 1, r.g.t0.toFixed(6), r.g.t1.toFixed(6), r.g.flo.toFixed(3), r.g.fhi.toFixed(3),
          r.b.manual ? '' : r.b.avg.toFixed(6), r.b.manual ? '' : r.b.max.toFixed(6), d.detections_merged, 0, 'bird', d.review].join(','));
      });
      downloadFile(`${base}.BirdBox.csv`, `${lines.join('\n')}\n`, 'text/csv;charset=utf-8');
    }
  }
}

new Demo().init();
