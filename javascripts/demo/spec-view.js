// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Zoomable spectrogram viewer with a minimap, playback cursor and overlay
 * hooks. Data arrives as Uint8 levels (frame-major, row 0 = lowest band)
 * from spec-worker.js and is drawn through cached tiles. Zoomed-out views use
 * max-pooled levels so short calls stay visible.
 */
import { COLORMAPS } from './colormaps.js';
import { esc, fmtTime, icon } from './util.js';

const DISPLAY_KEY = 'birdbox-demo-display';
const CMAP_ORDER = ['inferno', 'magma', 'viridis', 'gray', 'paper'];

/** Display preferences: colormap and frequency scale, brightness/contrast per spectrogram kind. */
function loadDisplay() {
  try { return JSON.parse(localStorage.getItem(DISPLAY_KEY)) || {}; } catch { return {}; }
}

function saveDisplay(prefs) {
  try { localStorage.setItem(DISPLAY_KEY, JSON.stringify(prefs)); } catch { /* not persisted */ }
}

const TILE = 1024;
const MINI_H = 34;
const MAX_TILES = 48;
const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
const melToHz = mel => 700 * (Math.pow(10, mel / 2595) - 1);

function niceStep(span, target) {
  const raw = span / target, pow = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return m * pow;
  return 10 * pow;
}

export class SpecView extends EventTarget {
  /**
   * @param {HTMLElement} root
   * @param {{height?:number, axis:{type:'linear'|'mel', fMin:number, fMax:number}, minimap?:boolean, kind?:string, kinds?:Object<string,string>}} opts
   *   kinds: spectrogram kinds to offer in the display settings, { kind: label }; a change fires 'kind'
   */
  constructor(root, opts) {
    super();
    this.root = root;
    this.opts = { height: 220, minimap: true, ...opts };
    this.axis = this.opts.axis;
    this.data = null;
    this.frames = 0;
    this.rows = 0;
    this.fps = 100;
    this.loadedFrames = 0;
    this.duration = 0;
    this.view = { t0: 0, t1: 1 };
    this.cursor = null;
    this.levels = [];       // max-pooled pyramids, levels[k] = { cols, data, built }
    this.tiles = new Map(); // `${level}:${index}` -> { canvas, version }
    this.version = 0;
    this.overlay = null;    // (ctx, helpers) => void
    this.minimapOverlay = null;
    // Optional editing hook: { cursor(p), down(p) -> claims the gesture, move(p), up(p, commit) },
    // where p = { x, y, time, freq, pointerType }. Unclaimed drags pan the view.
    this.editor = null;
    this.pointers = new Map();
    this.kinds = this.opts.kinds || null;
    const saved = loadDisplay().kind;
    this.kind = this.kinds?.[saved] ? saved : this.opts.kind || 'stft';
    this.source = { rowScale: 'linear' };
    this.defaultFloor = 0;
    this._loadDisplay();
    this._build();
    window.addEventListener('demo:display', () => { this._loadDisplay(); this._syncControls(); this._applyDisplay(); });
  }

  /* ---------------- display settings ---------------- */

  _loadDisplay() {
    const prefs = loadDisplay(), own = prefs[this.kind] || {};
    this.display = {
      cmap: COLORMAPS[prefs.cmap] ? prefs.cmap : 'inferno',
      scale: prefs.scale === 'mel' ? 'mel' : 'linear',
      brightness: Number.isFinite(own.brightness) ? own.brightness : 0,
      contrast: Number.isFinite(own.contrast) ? own.contrast : 0,
    };
    this._setAxis();
    this._buildLut();
  }

  /**
   * Linear data rows follow the chosen scale (a mel view remaps them when
   * tiles are drawn); mel rows (PCEN) are shown on their own mel axis.
   */
  _setAxis() {
    const { rowScale, fMin, fMax } = this.source;
    this.axis = rowScale === 'mel' ? { type: 'mel', fMin, fMax } : { ...this.opts.axis, type: this.display.scale };
    this.rowMap = null;
  }

  /** Switch the spectrogram kind (remembered); the owner recomputes the data on 'kind'. */
  setKind(kind) {
    if (kind === this.kind || !this.kinds?.[kind]) return;
    const prefs = loadDisplay();
    prefs.kind = kind;
    saveDisplay(prefs);
    this.kind = kind;
    this._loadDisplay();
    this._syncControls();
    this.dispatchEvent(new Event('kind'));
  }

  _saveDisplay() {
    const prefs = loadDisplay();
    prefs.cmap = this.display.cmap;
    prefs.scale = this.display.scale;
    prefs[this.kind] = { brightness: this.display.brightness, contrast: this.display.contrast };
    saveDisplay(prefs);
    window.dispatchEvent(new Event('demo:display'));
  }

  /** Fold brightness, contrast and the colormap into one 256-entry RGB table. */
  _buildLut() {
    const cmap = COLORMAPS[this.display.cmap];
    const floor = this.defaultFloor - this.display.brightness * 0.25;
    const gamma = Math.pow(2, this.display.contrast * 1.2);
    const lut = new Uint8Array(768);
    for (let i = 0; i < 256; i++) {
      let x = (i / 255 - floor) / Math.max(0.05, 1 - floor);
      x = x <= 0 ? 0 : x >= 1 ? 1 : Math.pow(x, gamma);
      const j = Math.round(x * 255) * 3;
      lut[i * 3] = cmap.lut[j]; lut[i * 3 + 1] = cmap.lut[j + 1]; lut[i * 3 + 2] = cmap.lut[j + 2];
    }
    this.lut = lut;
    this.light = !!cmap.light;
    this.root.classList?.toggle('is-light', this.light);
  }

  _applyDisplay() {
    this._buildLut();
    this.tiles.clear();
    this.render();
  }

  _build() {
    this.root.classList.add('spec-view');
    this.root.innerHTML = `
      <div class="spec-main" tabindex="0" role="img" aria-label="Spectrogram">
        <canvas class="spec-canvas"></canvas>
        <div class="spec-tooltip" hidden></div>
      </div>
      ${this.opts.minimap ? '<div class="spec-minimap"><canvas></canvas><div class="spec-window"></div></div>' : ''}`;
    this.main = this.root.querySelector('.spec-main');
    this.canvas = this.root.querySelector('.spec-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.tooltip = this.root.querySelector('.spec-tooltip');
    this.main.style.height = `${this.opts.height}px`;
    if (this.opts.minimap) {
      this.mini = this.root.querySelector('.spec-minimap');
      this.miniCanvas = this.mini.querySelector('canvas');
      this.miniCtx = this.miniCanvas.getContext('2d');
      this.miniWindow = this.mini.querySelector('.spec-window');
      this._bindMinimap();
    }
    this._buildToolbar();
    this.zoomBtns = {};
    this.tools?.querySelectorAll('[data-zoom]').forEach(btn => {
      this.zoomBtns[btn.dataset.zoom] = btn;
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const z = btn.dataset.zoom;
        if (z === 'fit') this.fit();
        else this.zoomBy(z === 'in' ? 0.5 : 2, (this.view.t0 + this.view.t1) / 2);
      });
    });
    this._bindMain();
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(this.root);
    this._resize();
  }

  _buildToolbar() {
    const host = this.opts.toolbar;
    if (!host) return;
    const zoom = (z, name, label) => `<button type="button" class="spec-tool" data-zoom="${z}" title="${label}" aria-label="${label}">${icon(name)}</button>`;
    const swatch = name => {
      const lut = COLORMAPS[name].lut, stops = [0, 64, 128, 192, 255].map(i => `rgb(${lut[i * 3]},${lut[i * 3 + 1]},${lut[i * 3 + 2]})`);
      return `<button type="button" class="spec-swatch" data-cmap="${name}" title="${esc(name)}" aria-label="${esc(name)}" style="background:linear-gradient(90deg,${stops.join(',')})"></button>`;
    };
    host.innerHTML = `
      <div class="spec-tools">
        ${zoom('out', 'zoom-out', 'Zoom out (−)')}${zoom('in', 'zoom-in', 'Zoom in (+)')}${zoom('fit', 'fit', 'Show all')}
        <div class="spec-display">
          <button type="button" class="spec-tool" data-role="display" aria-expanded="false" title="Display" aria-label="Display">${icon('display')}</button>
          <div class="spec-popover" hidden>
            ${this.kinds ? `<div class="spec-popover-label">Spectrogram</div>
            <div class="spec-seg">${Object.entries(this.kinds).map(([k, label]) => `<button type="button" data-kind="${k}">${esc(label)}</button>`).join('')}</div>` : ''}
            <div class="spec-popover-label">Colors</div>
            <div class="spec-swatches">${CMAP_ORDER.map(swatch).join('')}</div>
            <div class="spec-popover-label">Frequency scale</div>
            <div class="spec-seg">
              <button type="button" data-scale="linear">Linear</button>
              <button type="button" data-scale="mel">Mel</button>
            </div>
            <label class="spec-popover-label" for="${this.root.id}-brightness">Brightness</label>
            <input type="range" id="${this.root.id}-brightness" data-set="brightness" min="-1" max="1" step="0.05">
            <label class="spec-popover-label" for="${this.root.id}-contrast">Contrast</label>
            <input type="range" id="${this.root.id}-contrast" data-set="contrast" min="-1" max="1" step="0.05">
            <button type="button" class="spec-link" data-role="reset">Reset</button>
          </div>
        </div>
      </div>`;
    this.tools = host.querySelector('.spec-tools');
    const btn = host.querySelector('[data-role="display"]'), pop = host.querySelector('.spec-popover');
    const toggle = open => { pop.hidden = !open; btn.setAttribute('aria-expanded', String(open)); };
    btn.addEventListener('click', e => { e.stopPropagation(); toggle(pop.hidden); });
    document.addEventListener('click', e => { if (!pop.hidden && !pop.contains(e.target)) toggle(false); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') toggle(false); });
    pop.querySelectorAll('[data-cmap]').forEach(sw => sw.addEventListener('click', () => { this.display.cmap = sw.dataset.cmap; this._saveDisplay(); }));
    pop.querySelectorAll('[data-scale]').forEach(btn => btn.addEventListener('click', () => { this.display.scale = btn.dataset.scale; this._saveDisplay(); }));
    pop.querySelectorAll('[data-kind]').forEach(btn => btn.addEventListener('click', () => this.setKind(btn.dataset.kind)));
    pop.querySelectorAll('[data-set]').forEach(input => input.addEventListener('input', () => {
      this.display[input.dataset.set] = +input.value;
      this._applyDisplay();
      clearTimeout(this._saveTimer);
      this._saveTimer = setTimeout(() => this._saveDisplay(), 250);
    }));
    pop.querySelector('[data-role="reset"]').addEventListener('click', () => {
      Object.assign(this.display, { cmap: 'inferno', scale: 'linear', brightness: 0, contrast: 0 });
      this._saveDisplay();
    });
    this._syncControls();
  }

  _syncControls() {
    if (!this.tools) return;
    this.tools.querySelectorAll('[data-cmap]').forEach(sw => sw.classList.toggle('active', sw.dataset.cmap === this.display.cmap));
    const melRows = this.source.rowScale === 'mel';
    this.tools.querySelectorAll('[data-scale]').forEach(btn => {
      const on = melRows ? btn.dataset.scale === 'mel' : btn.dataset.scale === this.display.scale;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', String(on));
      btn.disabled = melRows;
    });
    this.tools.querySelectorAll('[data-kind]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.kind === this.kind);
      btn.setAttribute('aria-pressed', String(btn.dataset.kind === this.kind));
    });
    this.tools.querySelectorAll('[data-set]').forEach(input => { input.value = this.display[input.dataset.set]; });
  }

  /* ---------------- data ---------------- */

  /**
   * Start a new spectrogram; slices follow via addSlice(). rowScale 'mel'
   * means the rows are mel bands from fMin to fMax.
   */
  setSource({ frames, rows, fps, duration, floor = 0, rowScale = 'linear', fMin, fMax }) {
    this.source = { rowScale, fMin, fMax };
    this._setAxis();
    this._syncControls();
    this.defaultFloor = floor;
    this._buildLut();
    this.frames = frames;
    this.rows = rows;
    this.fps = fps;
    this.duration = duration || frames / fps;
    this.data = new Uint8Array(frames * rows);
    this.loadedFrames = 0;
    this.levels = [{ cols: frames, data: this.data, built: 0 }];
    this.tiles.clear();
    this.version++;
    this.view = { t0: 0, t1: Math.max(0.5, this.duration) };
    this.render();
  }

  addSlice(start, slice) {
    if (!this.data) return;
    const end = start + slice.length / this.rows;
    this.data.set(slice, start * this.rows);
    this.loadedFrames = Math.max(this.loadedFrames, end);
    this.levels[0].built = this.loadedFrames;
    this._invalidate(0, start, end);
    this.render();
  }

  _invalidate(k, c0, c1) {
    for (let i = Math.floor(c0 / TILE); i <= Math.floor((c1 - 1) / TILE); i++) this.tiles.delete(`${k}:${i}`);
  }

  clear() {
    this.data = null;
    this.frames = 0;
    this.duration = 0;
    this.tiles.clear();
    this.levels = [];
    this.cursor = null;
    this.render();
  }

  /** Max-pooled level k, extended incrementally as frames stream in. */
  _level(k) {
    if (k === 0) return this.levels[0];
    const src = this._level(k - 1), rows = this.rows;
    let level = this.levels[k];
    if (!level) {
      const cols = Math.ceil(src.cols / 2);
      level = this.levels[k] = { cols, data: new Uint8Array(cols * rows), built: 0 };
    }
    const target = Math.ceil(src.built / 2);
    if (level.built < target) {
      const from = Math.max(0, level.built - 1); // the last column may have been half-pooled
      for (let c = from; c < target; c++) {
        const a = 2 * c * rows, b = Math.min(2 * c + 1, src.built - 1) * rows, o = c * rows;
        for (let r = 0; r < rows; r++) level.data[o + r] = Math.max(src.data[a + r], src.data[b + r]);
      }
      this._invalidate(k, from, target);
      level.built = target;
    }
    return level;
  }

  /** Mel view: the linear rows each display row covers (max-pooled, so narrow high calls stay visible). */
  _rowRanges() {
    if (this.axis.type !== 'mel' || !this.rows || this.source.rowScale === 'mel') return null;
    if (this.rowMap?.rows === this.rows) return this.rowMap;
    const n = this.rows, { fMin, fMax } = this.axis, binHz = (fMax - fMin) / n;
    const m0 = hzToMel(fMin), m1 = hzToMel(fMax), lo = new Int32Array(n), hi = new Int32Array(n);
    for (let o = 0; o < n; o++) {
      const a = (melToHz(m0 + (m1 - m0) * o / n) - fMin) / binHz, b = (melToHz(m0 + (m1 - m0) * (o + 1) / n) - fMin) / binHz;
      lo[o] = Math.min(n - 1, Math.floor(a));
      hi[o] = Math.max(lo[o], Math.min(n - 1, Math.ceil(b) - 1));
    }
    return (this.rowMap = { rows: n, lo, hi });
  }

  _tile(k, index) {
    const key = `${k}:${index}`;
    let tile = this.tiles.get(key);
    if (tile && tile.version === this.version) {
      this.tiles.delete(key);
      this.tiles.set(key, tile); // LRU refresh
      return tile.canvas;
    }
    const level = this._level(k), rows = this.rows;
    const c0 = index * TILE, width = Math.min(TILE, level.built - c0);
    if (width <= 0) return null;
    const canvas = tile?.canvas || document.createElement('canvas');
    canvas.width = width;
    canvas.height = rows;
    const tctx = canvas.getContext('2d');
    const img = tctx.createImageData(width, rows), px = img.data, src = level.data, lut = this.lut, map = this._rowRanges();
    for (let c = 0; c < width; c++) {
      const so = (c0 + c) * rows;
      for (let r = 0; r < rows; r++) {
        let level8 = src[so + r];
        if (map) { level8 = 0; for (let q = so + map.lo[r], end = so + map.hi[r]; q <= end; q++) if (src[q] > level8) level8 = src[q]; }
        const v = level8 * 3, o = ((rows - 1 - r) * width + c) * 4;
        px[o] = lut[v]; px[o + 1] = lut[v + 1]; px[o + 2] = lut[v + 2]; px[o + 3] = 255;
      }
    }
    tctx.putImageData(img, 0, 0);
    this.tiles.delete(key);
    this.tiles.set(key, { canvas, version: this.version });
    while (this.tiles.size > MAX_TILES) this.tiles.delete(this.tiles.keys().next().value);
    return canvas;
  }

  /* ---------------- geometry ---------------- */

  get width() { return this.cssWidth || 1; }
  get height() { return this.opts.height; }

  /** Change the viewer height (CSS pixels). */
  setHeight(px) {
    if (px === this.opts.height) return;
    this.opts.height = px;
    this.main.style.height = `${px}px`;
    this._resize();
  }
  timeToX(time) { return (time - this.view.t0) / (this.view.t1 - this.view.t0) * this.width; }
  xToTime(x) { return this.view.t0 + x / this.width * (this.view.t1 - this.view.t0); }

  /** Map a frequency (Hz) to a y pixel; 0 = top. */
  freqToY(hz) {
    const { type, fMin, fMax } = this.axis;
    const n = type === 'mel'
      ? (hzToMel(hz) - hzToMel(fMin)) / (hzToMel(fMax) - hzToMel(fMin))
      : (hz - fMin) / (fMax - fMin);
    return (1 - Math.max(0, Math.min(1, n))) * this.height;
  }

  yToFreq(y) {
    const { type, fMin, fMax } = this.axis, n = 1 - y / this.height;
    return type === 'mel' ? melToHz(hzToMel(fMin) + n * (hzToMel(fMax) - hzToMel(fMin))) : fMin + n * (fMax - fMin);
  }

  setView(t0, t1, { silent } = {}) {
    const dur = Math.max(0.5, this.duration || 1), minSpan = Math.min(dur, 1.5);
    let span = Math.max(minSpan, Math.min(dur, t1 - t0));
    t0 = Math.max(0, Math.min(dur - span, t0));
    this.view = { t0, t1: t0 + span };
    this.render();
    if (!silent) this.dispatchEvent(new CustomEvent('view', { detail: this.view }));
  }

  zoomBy(factor, center) {
    const { t0, t1 } = this.view, span = (t1 - t0) * factor;
    const ratio = (center - t0) / (t1 - t0);
    this.setView(center - span * ratio, center - span * ratio + span);
  }

  fit() { this.setView(0, this.duration); }

  /** Keep a time visible, paging the view like a scrolling score. */
  follow(time) {
    const { t0, t1 } = this.view, span = t1 - t0;
    if (time < t0 || time > t1 - span * 0.05) this.setView(time - span * 0.1, time - span * 0.1 + span);
  }

  setCursor(time) {
    this.cursor = time;
    this.render();
  }

  /* ---------------- rendering ---------------- */

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    this.cssWidth = this.main.clientWidth;
    this.canvas.width = Math.round(this.cssWidth * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.canvas.style.height = `${this.height}px`;
    if (this.miniCanvas) {
      this.miniCanvas.width = Math.round(this.mini.clientWidth * dpr);
      this.miniCanvas.height = Math.round(MINI_H * dpr);
    }
    this.render();
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this._draw();
    });
  }

  _drawSpectrum(ctx, width, height, t0, t1) {
    if (!this.data || !this.frames) return;
    const framesPerPx = (t1 - t0) * this.fps / width;
    let k = 0;
    while (framesPerPx / Math.pow(2, k) > 1.6 && Math.pow(2, k + 1) < this.frames) k++;
    const level = this._level(k), scale = Math.pow(2, k);
    const c0 = t0 * this.fps / scale, c1 = t1 * this.fps / scale;
    const first = Math.max(0, Math.floor(c0 / TILE)), last = Math.min(Math.ceil(level.cols / TILE) - 1, Math.floor(c1 / TILE));
    const limit = level.built;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    for (let i = first; i <= last; i++) {
      const tile = this._tile(k, i);
      if (!tile) continue;
      const tc0 = i * TILE, tc1 = Math.min(tc0 + tile.width, limit);
      const s0 = Math.max(c0, tc0), s1 = Math.min(c1, tc1);
      if (s1 <= s0) continue;
      const x0 = (s0 - c0) / (c1 - c0) * width, x1 = (s1 - c0) / (c1 - c0) * width;
      ctx.drawImage(tile, s0 - tc0, 0, s1 - s0, tile.height, x0, 0, x1 - x0, height);
    }
  }

  _draw() {
    const ctx = this.ctx, dpr = window.devicePixelRatio || 1, W = this.width, H = this.height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.light ? '#ffffff' : '#05040c';
    ctx.fillRect(0, 0, W, H);
    const { t0, t1 } = this.view;
    this._drawSpectrum(ctx, W, H, t0, t1);
    if (this.overlay) {
      ctx.save();
      this.overlay(ctx, { width: W, height: H, timeToX: x => this.timeToX(x), freqToY: f => this.freqToY(f), view: this.view });
      ctx.restore();
    }
    this._drawAxes(ctx, W, H);
    if (this.cursor != null && this.cursor >= t0 && this.cursor <= t1) {
      const x = Math.round(this.timeToX(this.cursor)) + 0.5;
      ctx.strokeStyle = this.light ? 'rgba(15,23,42,0.9)' : 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    this._drawMinimap();
    this._syncZoom();
  }

  /** Gray out zoom buttons that would not change the view (same limits as setView). */
  _syncZoom() {
    const { in: zoomIn, out, fit } = this.zoomBtns || {};
    if (!fit) return;
    const dur = Math.max(0.5, this.duration || 1), span = this.view.t1 - this.view.t0;
    const full = !this.duration || span >= dur - 1e-3;
    out.disabled = fit.disabled = full;
    zoomIn.disabled = !this.duration || span <= Math.min(dur, 1.5) + 1e-3;
  }

  _drawAxes(ctx, W, H) {
    ctx.font = '10px Roboto, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = this.light ? 'rgba(15,23,42,0.75)' : 'rgba(255,255,255,0.75)';
    ctx.strokeStyle = this.light ? 'rgba(15,23,42,0.25)' : 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    const ticks = this.axis.type === 'mel' ? [500, 1000, 2000, 4000, 8000, 12000] : Array.from({ length: Math.floor(this.axis.fMax / 2000) }, (_, i) => (i + 1) * 2000);
    for (const hz of ticks) {
      if (hz <= this.axis.fMin || hz >= this.axis.fMax) continue;
      const y = Math.round(this.freqToY(hz)) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(6, y); ctx.stroke();
      ctx.fillText(`${hz >= 1000 ? hz / 1000 : hz}${hz >= 1000 ? ' kHz' : ' Hz'}`, 9, y);
    }
    if (!this.duration) return;
    const { t0, t1 } = this.view, step = niceStep(t1 - t0, Math.max(3, W / 110));
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'center';
    for (let s = Math.ceil(t0 / step) * step; s <= t1; s += step) {
      const x = Math.round(this.timeToX(s)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, H); ctx.lineTo(x, H - 5); ctx.stroke();
      if (x > 24 && x < W - 24) ctx.fillText(fmtTime(s, step < 1 ? 1 : 0), x, H - 6);
    }
    ctx.textAlign = 'left';
  }

  _drawMinimap() {
    if (!this.miniCanvas) return;
    const ctx = this.miniCtx, dpr = window.devicePixelRatio || 1;
    const W = this.mini.clientWidth, H = MINI_H;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.light ? '#ffffff' : '#05040c';
    ctx.fillRect(0, 0, W, H);
    if (!this.duration) { this.miniWindow.style.display = 'none'; return; }
    this._drawSpectrum(ctx, W, H, 0, this.duration);
    if (this.minimapOverlay) {
      ctx.save();
      this.minimapOverlay(ctx, { width: W, height: H, timeToX: x => x / this.duration * W });
      ctx.restore();
    }
    if (this.cursor != null) {
      const x = Math.round(this.cursor / this.duration * W) + 0.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    // The overview only earns its space once the main view is zoomed in
    const full = this.view.t1 - this.view.t0 >= this.duration - 1e-3;
    this.mini.classList.toggle('is-collapsed', full);
    this.miniWindow.style.display = full ? 'none' : 'block';
    this.miniWindow.style.left = `${this.view.t0 / this.duration * 100}%`;
    this.miniWindow.style.width = `${(this.view.t1 - this.view.t0) / this.duration * 100}%`;
  }

  /* ---------------- interaction ---------------- */

  _localX(e) { return e.clientX - this.main.getBoundingClientRect().left; }
  _localY(e) { return e.clientY - this.main.getBoundingClientRect().top; }
  _point(e) {
    const x = this._localX(e), y = this._localY(e);
    return { x, y, time: this.xToTime(x), freq: this.yToFreq(y), pointerType: e.pointerType };
  }

  _bindMain() {
    const el = this.main;
    let drag = null;
    el.addEventListener('pointerdown', e => {
      if (!this.duration) return;
      el.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, this._localX(e));
      if (this.pointers.size === 2) {
        if (drag?.edit) this.editor.up(this._point(e), false);
        const [a, b] = [...this.pointers.values()];
        drag = { pinch: true, dist: Math.abs(a - b) || 1, view: { ...this.view }, center: this.xToTime((a + b) / 2) };
      } else if (this.editor?.down(this._point(e))) {
        drag = { edit: true, x: this._localX(e), y: this._localY(e), moved: false };
      } else {
        drag = { x: this._localX(e), y: this._localY(e), view: { ...this.view }, moved: false };
      }
    });
    el.addEventListener('pointermove', e => {
      const x = this._localX(e), y = this._localY(e);
      if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, x);
      if (drag?.pinch && this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        const span = (drag.view.t1 - drag.view.t0) * drag.dist / (Math.abs(a - b) || 1);
        const ratio = (drag.center - drag.view.t0) / (drag.view.t1 - drag.view.t0);
        this.setView(drag.center - span * ratio, drag.center - span * ratio + span);
        return;
      }
      if (drag?.edit) {
        if (!drag.moved && Math.hypot(x - drag.x, y - drag.y) > 4) drag.moved = true;
        if (drag.moved) this.editor.move(this._point(e));
        return;
      }
      if (drag && !drag.pinch) {
        const dx = x - drag.x;
        if (Math.abs(dx) > 4) drag.moved = true;
        if (drag.moved) {
          const dt = dx / this.width * (drag.view.t1 - drag.view.t0);
          this.setView(drag.view.t0 - dt, drag.view.t1 - dt);
          el.classList.add('is-panning');
          el.style.cursor = '';
        }
        return;
      }
      if (this.editor) el.style.cursor = this.editor.cursor(this._point(e)) || '';
      this.dispatchEvent(new CustomEvent('hover', { detail: { time: this.xToTime(x), freq: this.yToFreq(y), x, y } }));
    });
    const end = e => {
      this.pointers.delete(e.pointerId);
      el.classList.remove('is-panning');
      if (drag?.edit) this.editor.up(this._point(e), drag.moved && e.type === 'pointerup');
      if (drag && !drag.pinch && !drag.moved && e.type === 'pointerup') {
        const x = this._localX(e), y = this._localY(e);
        this.dispatchEvent(new CustomEvent('pick', { detail: { time: this.xToTime(x), freq: this.yToFreq(y), x, y } }));
      }
      if (!this.pointers.size) drag = null;
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => this.dispatchEvent(new CustomEvent('hover', { detail: null })));
    el.addEventListener('wheel', e => {
      if (!this.duration) return;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        this.zoomBy(Math.exp(e.deltaY * 0.0025), this.xToTime(this._localX(e)));
      } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
        e.preventDefault();
        const d = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / this.width * (this.view.t1 - this.view.t0);
        this.setView(this.view.t0 + d, this.view.t1 + d);
      }
    }, { passive: false });
    el.addEventListener('keydown', e => {
      if (!this.duration) return;
      const span = this.view.t1 - this.view.t0, c = (this.view.t0 + this.view.t1) / 2;
      if (e.key === '+' || e.key === '=') this.zoomBy(0.5, c);
      else if (e.key === '-') this.zoomBy(2, c);
      else if (e.key === 'ArrowLeft') this.setView(this.view.t0 - span * 0.2, this.view.t1 - span * 0.2);
      else if (e.key === 'ArrowRight') this.setView(this.view.t0 + span * 0.2, this.view.t1 + span * 0.2);
      else if (e.key === ' ') this.dispatchEvent(new Event('toggleplay'));
      else return;
      e.preventDefault();
    });
  }

  _bindMinimap() {
    let dragging = false;
    const move = e => {
      const r = this.mini.getBoundingClientRect();
      const time = (e.clientX - r.left) / r.width * this.duration, span = this.view.t1 - this.view.t0;
      this.setView(time - span / 2, time + span / 2);
    };
    this.mini.addEventListener('pointerdown', e => {
      if (!this.duration) return;
      dragging = true;
      this.mini.setPointerCapture(e.pointerId);
      if (this.view.t1 - this.view.t0 >= this.duration - 1e-3) {
        // Fully zoomed out: a click on the overview seeks instead
        const r = this.mini.getBoundingClientRect();
        this.dispatchEvent(new CustomEvent('pick', { detail: { time: (e.clientX - r.left) / r.width * this.duration, freq: null, minimap: true } }));
        dragging = false;
        return;
      }
      move(e);
    });
    this.mini.addEventListener('pointermove', e => { if (dragging) move(e); });
    this.mini.addEventListener('pointerup', () => { dragging = false; });
    this.mini.addEventListener('pointercancel', () => { dragging = false; });
  }

  showTooltip(html, x, y) {
    if (!html) { this.tooltip.hidden = true; return; }
    this.tooltip.innerHTML = html;
    this.tooltip.hidden = false;
    const w = this.tooltip.offsetWidth, h = this.tooltip.offsetHeight;
    this.tooltip.style.left = `${Math.min(this.width - w - 4, Math.max(4, x + 12))}px`;
    this.tooltip.style.top = `${Math.max(4, Math.min(this.height - h - 4, y - h - 8))}px`;
  }
}
