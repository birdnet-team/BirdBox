// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * BirdBox browser demo (the Demo tab of the docs): bird vocalization boxes
 * with a self-contained BirdBox ONNX graph (log-floor spectrogram, YOLO, NMS inside), Just-Bird
 * by default; the models are listed in docs/models/manifest.json.
 *
 * The graph runs once per recording at the lowest confidence and reports its
 * per-window boxes a few seconds of audio at a time, so boxes appear while it
 * runs. fusion.js (window fusion, Just-Bird) or songs.js (song reconstruction,
 * the species models) merges them exactly like the graph for any (confidence,
 * gap) setting, so both sliders update instantly.
 *
 * While recording, the spectrogram flies in from the right, and each window is
 * detected as soon as it is complete. Its noise floor comes from the audio before its
 * end (the rolling floor), so its boxes are final: when the recording ends, only the
 * last windows are left, and the result is the same as for the saved file.
 *
 * On the spectrogram, the selected box can be moved and resized and new
 * boxes can be drawn. Exports use the BirdBox output formats.
 */
import { AudioSource, MAX_SECONDS } from './audio-source.js';
import { SAMPLE_RATE as SR } from './audio.js';
import { fileEnd, readyWindows } from './chunked-graph.js';
import { detector, models, RAW_CONF } from './detector.js';
import { fuseBoxes } from './fusion.js';
import { player } from './player.js';
import { reconstructSongs } from './songs.js';
import { SpecView } from './spec-view.js';
import { csvField, downloadFile, esc, fill, fmtBytes, fmtNumber, fmtTime, icon, statLine, toast } from './util.js';

const UPDATE_MS = 100;            // redraw at most this often while detection runs
const LIVE_MAX_CLIPS = 8;         // windows per live detection when it has fallen behind
const LIVE_BEHIND = 6;            // seconds behind at which a note says so
const LIVE_OWNED = 2.25;   // a window owns its boxes up to here (fusion.js)
const LIVE_EDGE = 0.13;           // seconds the right edge stays behind the microphone, for the last frames
const POP_MS = 420;               // a new box pops in over this time
const CLIP = 3, HOP = 1.5;       // the graph's window grid (onnx_export.py)
const NMS_IOU = 0.7;             // baked into the graph (DEFAULT_NMS_IOU)
const FREQ_MAX = 15000;          // top of the spectrogram and of any box
const MIN_DUR = 0.05;            // smallest box: seconds
const MIN_BAND = 100;            // smallest box: Hz
const MERGE = { fusion: fuseBoxes, reconstruct: reconstructSongs };
const GAP_TEXT = {
  fusion: ['Phrase gap', 'Calls in the same frequency band closer than this are joined into one phrase box. 0 keeps single calls.'],
  reconstruct: ['Song gap', 'Boxes of the same species closer than this are merged into one song box.'],
};
const COLORS = { detected: '#22d3ee', verified: '#4ade80', drawn: '#fbbf24', removed: 'rgba(148,163,184,0.7)' };

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Detector boxes keep their identity through slider changes by their geometry and class. */
const boxKey = b => `${b.t0.toFixed(4)}|${b.t1.toFixed(4)}|${b.flo}|${b.fhi}|${b.cls}`;
const geomOf = g => ({ t0: g.t0, t1: g.t1, flo: g.flo, fhi: g.fhi });
const round = (v, digits) => Number(v.toFixed(digits));
const concat = (a, b) => {
  const out = new Float32Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
};

class Demo {
  constructor() {
    this.audio = null;
    this.model = null;             // manifest entry of the selected model
    this.raw = null;               // { data, rows, ms, next, done, stopped }: raw_all of the current recording, so far
    this.analyzedTo = null;        // seconds analyzed while detection runs, else null
    this.tableHeld = false;        // a press on the list: redrawing it now would lose the click
    this.rec = null;               // the recording in progress
    this.liveEnded = null;         // { boxes, rec }: it ended, and its file is on the way
    this.liveIdle = null;          // settles when no live detection is using the worker any more
    this.ghosts = [];              // boxes seen while recording that the file run has not reached yet
    this.born = new Map();         // box id -> when it appeared while recording
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
      height: this._specHeight(), kind: 'stft', kinds: { stft: 'STFT', logfloor: 'Log-floor' },
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
    this.modelsReady = models().then(list => {
      this.models = list;
      $('bbd-model').replaceChildren(...list.map((m, i) => new Option(m.label || m.file, String(i))));
      $('bbd-model').disabled = list.length < 2;
      this._setModel(list[0]);
    }).catch(err => this._error(err.message));
    this._refresh();
  }

  /** Use a manifest model: sliders to its suggested values, gap slider and table to its merge and classes. */
  _setModel(m) {
    this.model = m;
    $('bbd-model').value = String(this.models.indexOf(m));
    const fusion = m.merge_method === 'fusion';
    if (m.conf != null) $('bbd-conf').value = m.conf;
    const gap = fusion ? m.phrase_gap : m.song_gap;
    if (gap != null) $('bbd-gap').value = gap;
    const [label, help] = GAP_TEXT[m.merge_method];
    $('bbd-gap-label').textContent = label;
    $('bbd-gap-help').textContent = help;
    $('bbd-table').querySelector('.bbd-c-species').hidden = !this._multiClass();
    this._syncSliders();
    this._renderModel();
    this._update();
  }

  /** Switch the model: boxes of the old one and changes to them go, drawn boxes stay. */
  _changeModel(m) {
    if (m === this.model) return;
    this.runToken++;
    detector.abort();
    this._setProgress(null);
    const edited = [...this.edits.values()].some(e => Object.keys(e).length);
    this.raw = null;
    this.analyzedTo = null;
    this.edits.clear();
    this._setModel(m);
    if (edited) toast('Your changes to detected boxes were reset for the new model.');
    this.run();
  }

  /** Several classes (species models): the table and the tooltip name them. */
  _multiClass() {
    return Object.keys(this.model?.names || {}).length > 1;
  }

  /** The spectrogram takes what the page leaves it, within limits. */
  _specHeight() {
    return Math.round(clamp(window.innerHeight * 0.42, 240, 560));
  }

  _bind() {
    this.source.addEventListener('audio', e => this._onAudio(e.detail));
    $('bbd-model').addEventListener('change', e => { const m = this.models?.[+e.target.value]; if (m) this._changeModel(m); });
    for (const id of ['bbd-conf', 'bbd-gap']) $(id).addEventListener('input', () => { this._syncSliders(); this._update(); });
    this._syncSliders();
    $('bbd-export').addEventListener('click', e => {
      const btn = e.target.closest('[data-export]');
      if (btn) this._export(btn.dataset.export);
    });
    // Leaving the page would lose a recording in progress
    window.addEventListener('beforeunload', e => { if (this.rec) e.preventDefault(); });
    this.source.addEventListener('livestart', e => this._liveStart(e.detail));
    this.source.addEventListener('liveaudio', e => this._liveAudio(e.detail.pcm));
    this.source.addEventListener('liveend', () => this._liveEnd());
    this.source.addEventListener('livecancel', () => this._liveCancel());
    $('bbd-rerun').addEventListener('click', () => this.run());
    $('bbd-stop').addEventListener('click', () => {
      this.runToken++;
      detector.abort();
      this._showStopped();
    });
    $('bbd-continue').addEventListener('click', () => this.run(true));

    player.addEventListener('time', e => {
      this.spec.setCursor(e.detail);
      if (player.playing) this.spec.follow(e.detail);
    });
    this.spec.addEventListener('toggleplay', () => { if (!this.rec) player.toggle(); });
    this.spec.addEventListener('kind', () => {
      if (this.rec) this._liveSpectrogram(this.rec);
      else if (this.audio) this._streamSpectrogram({ keepView: true });
    });
    this.spec.addEventListener('pick', e => {
      if (this.rec) return;
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
      const species = this._multiClass() && !b.manual ? `<div><b>${esc(this._speciesName(b))}</b></div>` : '';
      this.spec.showTooltip(`<div class="tt-time">#${n} · ${fmtTime(g.t0, 2)} – ${fmtTime(g.t1, 2)}</div>${species}<div>${fmtNumber(g.flo)}–${fmtNumber(g.fhi)} Hz</div><div>${origin}</div>`, d.x, d.y);
    });
    // Editing keys work anywhere on the page except while typing in a field.
    // Caught on the way down and stopped there: Material's own shortcuts would
    // otherwise also see them (N and P open the next and previous docs page).
    window.addEventListener('keydown', e => {
      if (!(this.audio || this.rec) || e.ctrlKey || e.metaKey || e.altKey) return;
      // The volume fader is no text field: M and Space work there too
      if (e.target.closest?.('input:not([type="range"]), select, textarea, [contenteditable], .md-search')) return;
      // Nothing to edit while recording, and N or P must not leave the page and lose the recording
      if (this.rec) {
        if (e.key === 'n' || e.key === 'p') { e.preventDefault(); e.stopPropagation(); }
        return;
      }
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
    table.addEventListener('pointerdown', () => { this.tableHeld = true; });
    for (const type of ['pointerup', 'pointercancel']) window.addEventListener(type, () => { this.tableHeld = false; });
    table.addEventListener('click', e => {
      if (this.rec) return;
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
    // Phrase gap 0 turns the joining off; song gap 0 still merges overlapping boxes
    $('bbd-gap-val').textContent = gap > 0 || this.model?.merge_method !== 'fusion' ? `${fmtNumber(gap, 2)} s` : 'off';
  }

  _showHelp(show) {
    $('bbd-help').hidden = !show;
    this.helpBtn?.classList.toggle('is-active', show);
    this.helpBtn?.setAttribute('aria-expanded', String(show));
  }

  /** What the pointer does right now, under the spectrogram. */
  _renderHint(text) {
    if (!text) {
      if (this.rec) {
        const behind = Math.floor(this.rec.n / SR - (this.analyzedTo ?? 0) - CLIP);
        text = this.raw && behind >= LIVE_BEHIND
          ? fill('This device is {n} s behind the sound. Nothing is lost: the final boxes cover it all when you stop.', { n: behind })
          : !detector.loaded ? 'Boxes appear a few seconds after a sound.'
          : 'Boxes appear a few seconds after a sound, and they are final: the saved file gets the same ones.';
      }
      else if (this.liveEnded) text = 'Finishing the recording…';
      else if (this.ghosts.length) text = 'Detecting the last windows: dashed boxes turn solid when it reaches them.';
      else if (!this.raw) text = 'Boxes appear here while detection runs.';
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
    $('bbd-stop').hidden = !!this.rec;
    $('bbd-continue').hidden = true;
    wrap.querySelector('.bbd-progress__bar').style.width = `${Math.round(clamp(fraction, 0.02, 1) * 100)}%`;
    $('bbd-progress-text').textContent = text || '';
  }

  _error(message) {
    $('bbd-error').hidden = !message;
    $('bbd-error').textContent = message || '';
  }

  /** The boxes found so far stay and can be exported, and detection can continue from there. */
  _showStopped() {
    if (this.raw) this.raw.stopped = true;
    else this.analyzedTo = null;
    this.ghosts = [];
    $('bbd-rerun').hidden = false;
    $('bbd-progress').hidden = false;
    $('bbd-stop').hidden = true;
    $('bbd-continue').hidden = false;
    const text = this.raw ? fill('Stopped at window {n} of {total}', { n: this.raw.next.clips, total: this._windows() }) : 'Detection stopped';
    $('bbd-progress-text').textContent = text;
    this._update();
    this._renderHint('Detection stopped. Continue it below the spectrogram, or run it again from the start.');
  }

  /* ---------------- recording in progress ---------------- */

  /**
   * The microphone is on. The current recording, if any, steps aside: it comes
   * back as it was if this recording is cancelled.
   */
  _liveStart({ maxSeconds }) {
    this.runToken++;
    detector.abort();
    this.specWorker?.terminate();
    this.specWorker = null;
    const prev = this.audio ? { audio: this.audio, raw: this.raw, analyzedTo: this.analyzedTo, userBoxes: this.userBoxes, edits: this.edits, selectedId: this.selectedId } : null;
    const rec = this.rec = {
      maxSeconds, prev, pcm: new Float32Array((maxSeconds + 1) * SR), n: 0, arrivedAt: 0,
      final: { data: new Float32Array(0), rows: 0, ms: 0, next: { segment: 0, clip: 0, clips: 0 } },   // rows that will not change
      finalTo: 0,   // a box ending before this has only final rows
      t0: performance.now(), ended: false, wake: null, specId: 0, specWorker: null, loop: null,
    };
    Object.assign(this, { audio: null, raw: null, analyzedTo: 0, songs: [], userBoxes: [], edits: new Map(), selectedId: null, hoverId: null, ghosts: [], liveEnded: null });
    this.born.clear();
    this._setDrawMode(false);
    this._setProgress(null);
    this._error(null);
    $('bbd-rerun').hidden = true;
    $('bbd-note').hidden = true;
    $('bbd').classList.add('has-audio', 'is-recording');
    $('bbd-model').disabled = true;
    this.spec.clear();
    this._liveSpectrogram(rec);
    this._update();
    rec.loop = this._liveLoop(rec);
  }

  /** The spectrogram of the recording in progress; a change of kind starts over on the audio so far. */
  _liveSpectrogram(rec) {
    rec.specWorker?.terminate();
    const worker = rec.specWorker = new Worker(new URL('./spec-worker.js', import.meta.url)), id = ++rec.specId;
    worker.onmessage = ({ data }) => {
      if (this.rec !== rec || data.id !== id) return;
      if (data.type === 'meta') this.spec.startLive({ ...data, capacity: rec.maxSeconds + 1, clock: () => this._liveClock(rec) });
      else if (data.type === 'live-frames') this.spec.addFrames(data.start, data.data);
      else if (data.type === 'error') console.error('spectrogram:', data.error);
    };
    worker.postMessage({ type: 'live-start', id, kind: this.spec.kind, sampleRate: SR, fMax: FREQ_MAX });
    if (rec.n) worker.postMessage({ type: 'live-audio', id, pcm: rec.pcm.slice(0, rec.n) });
  }

  /** Seconds to show at the right edge: smooth between the pieces of audio, which arrive in bursts. */
  _liveClock(rec) {
    if (!rec.n) return 0;
    return Math.max(0, rec.n / SR + Math.min(performance.now() - rec.arrivedAt, 120) / 1000 - LIVE_EDGE);
  }

  _liveAudio(pcm) {
    const rec = this.rec;
    if (!rec || rec.ended) return;
    rec.pcm.set(pcm.subarray(0, rec.pcm.length - rec.n), rec.n);
    rec.n = Math.min(rec.pcm.length, rec.n + pcm.length);
    rec.arrivedAt = performance.now();
    rec.specWorker?.postMessage({ type: 'live-audio', id: rec.specId, pcm }, [pcm.buffer]);
    rec.wake?.();
  }

  /**
   * Detect on each new window as soon as it is complete. The worker keeps the recording's
   * spectrogram and adds frames for new audio only. Each window is run once, as soon as the
   * audio reaches its end (readyWindows), and its rows are the final ones, the rows of the
   * saved file. A run takes every window that has become ready meanwhile (up to
   * LIVE_MAX_CLIPS), so a slower device falls behind by a run, not more. At Stop the file run
   * continues from the last window done (_finalSoFar).
   */
  async _liveLoop(rec) {
    try {
      await this.modelsReady;
      if (rec.ended || !this.model) return;
      await detector.ensure(this.model, p => {
        if (this.rec !== rec) return;
        if (p.starting) this._setProgress(0.3, 'Starting ONNX Runtime…');
        else if (p.total) this._setProgress(p.loaded / p.total * 0.3, fill('Downloading the model: {done} of {total}', { done: fmtBytes(p.loaded), total: fmtBytes(p.total) }));
        else this._setProgress(0.15, fill('Downloading the model: {done}', { done: fmtBytes(p.loaded) }));
      });
      if (this.rec === rec) {
        this._setProgress(null);
        this._renderModel();
      }
      const final = rec.final;
      detector.liveStart();
      let sent = 0;   // samples the worker has: it turns only new ones into spectrogram frames
      while (!rec.ended) {
        if (rec.n > sent) {
          detector.liveAudio(rec.pcm.subarray(sent, rec.n));
          sent = rec.n;
        }
        const ready = readyWindows(sent), { next } = final;
        if (ready <= next.clips) {
          await new Promise(resolve => { rec.wake = resolve; });
          continue;
        }
        const count = Math.min(ready - next.clips, LIVE_MAX_CLIPS);
        const res = await detector.liveWindows(next, count);
        if (rec.ended) return;
        final.data = concat(final.data, res.data);
        final.rows += res.rows;
        final.ms += res.ms;
        final.next = res.next;
        rec.finalTo = this.analyzedTo = (res.next.clips - 1) * HOP + LIVE_OWNED;
        this._liveShow(rec);
      }
    } catch (err) {
      if (err.cancelled || rec.ended) return;
      console.error(err);
      this._error(fill('Boxes stopped appearing: {message}. The recording goes on, and the final boxes still come when you stop.', { message: err.message }));
    }
  }

  _liveShow(rec) {
    this.raw = { data: rec.final.data, rows: rec.final.rows, ms: performance.now() - rec.t0, done: false };
    this._liveUpdate();
  }

  /** New boxes, marked so they pop in. A box that only grew does not. The list stays at its end. */
  _liveUpdate() {
    if (this.gesture || this.tableHeld) return;
    const list = $('bbd-results'), atEnd = list.scrollHeight - list.scrollTop - list.clientHeight < 8;
    const before = this.songs, now = performance.now();
    this._update();
    for (const b of this.songs) {
      if (before.some(o => o.t0 < b.t1 && b.t0 < o.t1 && o.flo < b.fhi && b.flo < o.fhi)) continue;
      this.born.set(b.id, now);
    }
    if (atEnd) list.scrollTop = list.scrollHeight;
  }

  /** Stop was pressed: the file is on its way. What is on screen stays until then. */
  _liveEnd() {
    const rec = this.rec;
    if (!rec) return;
    this._liveClose(rec);
    this.liveEnded = { boxes: this.songs.slice(), rec };
    this._renderHint();
  }

  _liveClose(rec) {
    rec.ended = true;
    rec.wake?.();
    rec.specWorker?.terminate();
    detector.liveEnd();
    this.liveIdle = rec.loop;
    this.rec = null;
    this.spec.stopLive();
    $('bbd').classList.remove('is-recording');
    $('bbd-model').disabled = (this.models?.length ?? 0) < 2;
    this._setProgress(null);
  }

  /** Cancelled: back to the recording that was there, as it was. */
  _liveCancel() {
    const rec = this.rec;
    if (!rec) return;
    this._liveClose(rec);
    this.spec.clear();
    const { prev } = rec;
    Object.assign(this, { songs: [], ghosts: [], selectedId: prev?.selectedId ?? null, audio: prev?.audio ?? null, raw: prev?.raw ?? null, analyzedTo: prev?.analyzedTo ?? null, userBoxes: prev?.userBoxes ?? [], edits: prev?.edits ?? new Map() });
    this.born.clear();
    this._error(null);
    if (!prev) {
      $('bbd').classList.remove('has-audio');
      this._update();
      return;
    }
    this._streamSpectrogram();
    this.spec.setCursor(0);
    if (prev.raw?.stopped) this._showStopped();
    else this._update();
    if (!prev.raw) this.run();
    else if (!prev.raw.done && !prev.raw.stopped) this.run(true);
  }

  /**
   * The final rows found while recording, as a run stopped at their end, if the file holds
   * the samples they were found on. It does by construction (Recorder): this makes sure.
   */
  _finalSoFar({ rec }, audio) {
    const { final } = rec, { pcm } = audio;
    if (!final.next.clips || pcm.length !== rec.n) return null;
    for (let i = 0; i < pcm.length; i++) if (pcm[i] !== rec.pcm[i]) return null;
    const end = fileEnd(pcm.length);
    for (let r = 0; r < final.rows; r++) final.data[r * 9 + 8] = end;
    return { data: final.data, rows: final.rows, ms: final.ms, next: final.next, done: false };
  }

  _onAudio(audio) {
    this.runToken++;
    const fromLive = this.liveEnded;
    this.liveEnded = null;
    // A live detection still running ends by itself in a moment, and the worker stays warm
    if (!fromLive) detector.abort();
    this.audio = audio;
    this.raw = fromLive ? this._finalSoFar(fromLive, audio) : null;
    this.analyzedTo = this.raw ? (this.raw.next.clips - 1) * HOP + LIVE_OWNED : null;
    this._setProgress(null);
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
    this.born.clear();
    this.ghosts = fromLive && audio.pcm.length >= CLIP * SR ? fromLive.boxes : [];
    if (fromLive) {
      // The live picture stays until the finished spectrogram replaces it, and the view opens up to the whole recording
      this.spec.duration = audio.duration;
      this._streamSpectrogram({ swap: true });
      this.spec.animateView(0, audio.duration);
    } else {
      this.spec.clear();
      this._streamSpectrogram();
    }
    this.spec.setCursor(0);
    this._update();
    this.run(!!this.raw);
  }

  /** STFT for reading the sounds, or log-floor, the mel spectrogram the model detects on. */
  _streamSpectrogram({ keepView = false, swap = false } = {}) {
    this.specWorker?.terminate();
    const worker = this.specWorker = new Worker(new URL('./spec-worker.js', import.meta.url));
    const duration = this.audio.duration, view = { ...this.spec.view };
    let held = null;   // swap: everything is gathered first, so nothing goes dark meanwhile
    worker.onmessage = ({ data }) => {
      if (data.type === 'meta') {
        if (swap) held = { meta: data, levels: new Uint8Array(data.frames * data.rows) };
        else {
          this.spec.setSource({ ...data, duration, floor: data.floor || 0 });
          if (keepView) this.spec.setView(view.t0, view.t1, { silent: true });
        }
      }
      else if (data.type === 'slice') {
        if (held) held.levels.set(data.data, data.start * held.meta.rows);
        else this.spec.addSlice(data.start, data.data);
      }
      else if (data.type === 'error') console.error('spectrogram:', data.error);
      if (data.type === 'done' && held) {
        const current = { ...this.spec.view };
        this.spec.setSource({ ...held.meta, duration, floor: held.meta.floor || 0 });
        this.spec.addSlice(0, held.levels);
        this.spec.setView(current.t0, current.t1, { silent: true });
      }
      if (data.type === 'done' || data.type === 'error') {
        worker.terminate();
        if (this.specWorker === worker) this.specWorker = null;
      }
    };
    worker.postMessage({ type: this.spec.kind, id: 1, pcm: this.audio.pcm, sampleRate: SR, fMax: FREQ_MAX });
  }

  /** The windows of the graph's 1.5 s grid in this recording. */
  _windows() {
    return Math.max(1, Math.ceil((this.audio.duration - CLIP) / HOP) + 1);
  }

  /** Detect over the recording, or continue a stopped run where it ended (keeping its boxes). */
  async run(resume = false) {
    if (!this.audio || this.audio.pcm.length < CLIP * SR) return;
    const token = ++this.runToken;
    const clips = this._windows();
    const from = resume ? this.raw?.next : undefined;   // nothing to continue if no box arrived yet
    const resumed = from ? 0.3 + 0.7 * from.clips / clips : null;
    if (from) {
      this.raw.stopped = false;
    } else {
      this.raw = null;
      this.analyzedTo = 0;
    }
    $('bbd-rerun').hidden = true;
    this._update();
    try {
      this._setProgress(resumed ?? 0.01, resumed ? 'Continuing detection…' : 'Preparing BirdBox…');
      await this.liveIdle;   // a live detection that was still running needs the worker first
      if (token !== this.runToken) return;
      await this.modelsReady;
      if (!this.model || token !== this.runToken) return;
      await detector.ensure(this.model, p => {
        if (token !== this.runToken) return;
        if (resumed) return;
        if (p.starting) this._setProgress(0.3, 'Starting ONNX Runtime…');
        else if (p.total) this._setProgress(p.loaded / p.total * 0.3, fill('Downloading the model: {done} of {total}', { done: fmtBytes(p.loaded), total: fmtBytes(p.total) }));
        else this._setProgress(0.15, fill('Downloading the model: {done}', { done: fmtBytes(p.loaded) }));
      });
      this._renderModel();
      if (token !== this.runToken) return;
      // Detection time of the whole recording: a continued run adds to the time so far
      const t0 = performance.now() - (from ? this.raw.ms : 0);
      let data = from ? this.raw.data : new Float32Array(0), rows = from ? this.raw.rows : 0, lastUpdate = 0;
      this._setProgress(resumed ?? 0.3, fill('Detecting vocalizations · window {n} of {total}', { n: from ? from.clips : 1, total: clips }));
      await detector.detect(this.audio.pcm.slice(), chunk => {
        if (token !== this.runToken) return;
        const grown = new Float32Array(data.length + chunk.data.length), now = performance.now();
        grown.set(data);
        grown.set(chunk.data, data.length);
        data = grown;
        rows += chunk.rows;
        this.analyzedTo = chunk.analyzedTo;
        this.raw = { data, rows, ms: now - t0, next: chunk.next, done: false };
        const done = chunk.next.clips, left = (clips - done) * (now - t0) / done / 1000;
        let text = fill('Detecting vocalizations · window {n} of {total}', { n: done, total: clips });
        if (left > 20) text += ` · about ${fmtTime(Math.ceil(left / 10) * 10)} left`;
        this._setProgress(0.3 + 0.7 * done / clips, text);
        // Not in the middle of a drag or a click, which a redraw would disturb
        if (now - lastUpdate < UPDATE_MS || this.gesture || this.tableHeld) return;
        lastUpdate = now;
        this._update();
      }, from);
      if (token !== this.runToken) return;
      this.analyzedTo = null;
      this.ghosts = [];
      this.raw = { data, rows, ms: performance.now() - t0, done: true };
    } catch (err) {
      if (token !== this.runToken || err.cancelled) return;
      console.error(err);
      this._error(fill('BirdBox failed: {message}', { message: err.message }));
      $('bbd-rerun').hidden = false;
    } finally {
      if (token === this.runToken) {
        this._setProgress(null);
        this._renderModel();
        this._update();
      }
    }
  }

  _renderModel() {
    const chip = $('bbd-model-state'), loaded = detector.loaded && detector.entry === this.model;
    chip.textContent = loaded ? 'loaded' : '';
    chip.hidden = !loaded;
  }

  /* ---------------- boxes + edits ---------------- */

  _update() {
    const conf = +$('bbd-conf').value, gap = +$('bbd-gap').value;
    this.songs = this.raw ? MERGE[this.model.merge_method](this.raw.data, this.raw.rows, conf, gap) : [];
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
    if (this.rec) return '';
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
    if (!this.raw || this.rec) return false;
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
    const el = $('bbd-summary');
    // Exports wait for the whole recording, unless detection was stopped
    const finished = this.raw?.done || this.raw?.stopped;
    $('bbd-export').querySelectorAll('button').forEach(btn => { btn.disabled = !finished || !this._included().length; });
    const note = $('bbd-export-note');
    note.hidden = !this.raw?.stopped;
    if (this.raw?.stopped) note.textContent = fill('Detection was stopped at {done} of {total}. Exports only cover this part, not the whole file.', { done: fmtTime(this.analyzedTo ?? 0), total: fmtTime(this.audio.duration) });
    if (!this.raw) { el.innerHTML = ''; return; }
    const c = this._counts();
    const stats = [[c.included, c.included === 1 ? 'box' : 'boxes']];
    if (c.verified) stats.push([c.verified, 'verified']);
    if (c.drawn) stats.push([c.drawn, 'drawn']);
    if (c.removed) stats.push([c.removed, 'removed']);
    stats.push(this.raw.done ? [`${fmtNumber(this.raw.ms / 1000, 1)} s`, 'detection'] : [fmtTime(this.analyzedTo ?? 0), 'audio']);
    el.innerHTML = statLine(stats);
  }

  _renderTable() {
    const tbody = $('bbd-table').querySelector('tbody'), multi = this._multiClass(), cols = multi ? 6 : 5;
    $('bbd-results').classList.toggle('has-results', !!this.raw);
    if (!this.raw) {
      tbody.innerHTML = `<tr class="bbd-table__empty"><td colspan="${cols}">${this.rec ? 'Listening. Boxes appear here as they are found.' : this.audio ? ($('bbd-rerun').hidden ? 'Detection is running…' : 'Detection is not running.') : 'Load a recording to see its boxes here.'}</td></tr>`;
      return;
    }
    if (!this.boxes.length) {
      const note = this.raw.done || this.raw.stopped ? 'No bird vocalizations above this confidence. Try a lower threshold.'
        : this.rec ? 'Listening. Boxes appear here as they are found.' : 'Detection is running…';
      tbody.innerHTML = `<tr class="bbd-table__empty"><td colspan="${cols}">${note}</td></tr>`;
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
      const species = !multi ? '' : b.manual ? '<td class="bbd-c-species">–</td>'
        : `<td class="bbd-c-species">${esc(this._speciesName(b))}<span>${esc(this._class(b)[1])}</span></td>`;
      const conf = b.manual ? '<td class="bbd-c-conf" title="Drawn by you">–</td>'
        : `<td class="bbd-c-conf" title="max ${fmtNumber(b.max, 3)} · ${b.n} window boxes">${fmtNumber(b.avg, 2)}<span>max ${fmtNumber(b.max, 2)}</span></td>`;
      return `<tr data-i="${i}" class="${b.id === this.selectedId ? 'is-selected' : ''}${edit.include ? '' : ' is-excluded'}" style="--box:${this._color(b)}">
        <td class="bbd-c-play"><button type="button" class="bbd-play" data-act="play" aria-label="Play box ${i + 1}">${icon('play')}${i + 1}</button></td>
        <td class="bbd-c-time">${fmtTime(g.t0, 2)} – ${fmtTime(g.t1, 2)}<span>${fmtNumber(g.t1 - g.t0, 2)} s</span></td>
        <td class="bbd-c-freq">${fmtNumber(g.flo / 1000, 1)} – ${fmtNumber(g.fhi / 1000, 1)} kHz${mark}</td>
        ${species}${conf}
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

  /**
   * Scroll the table's list (not the page) just enough to show row i in the
   * part of the list that is on screen: below its sticky header and the page
   * header, above the bottom of the window.
   */
  _revealRow(i) {
    const list = $('bbd-results'), row = list.querySelector(`tr[data-i="${i}"]`);
    if (!row) return;
    const r = row.getBoundingClientRect(), c = list.getBoundingClientRect();
    const header = document.querySelector('.md-header')?.getBoundingClientRect().bottom || 0;
    const top = Math.max(c.top, header) + list.querySelector('thead').offsetHeight;
    const bottom = Math.min(c.bottom, window.innerHeight);
    // Instant: smooth scrolls cut each other short when N or P is held down
    if (r.top < top) list.scrollTop -= top - r.top;
    else if (r.bottom > bottom && bottom - top >= r.height) list.scrollTop += r.bottom - bottom;
  }

  /** While recording, a soft edge marks how far detection has come; the newest sound stays bright. */
  _drawFrontier(ctx, h) {
    const x = h.timeToX(this.analyzedTo);
    if (x < 0 || x > h.width) return;
    const tint = this.spec.light ? '15,23,42' : '34,211,238', band = Math.min(h.width - x, 90);
    if (band > 0) {
      const fade = ctx.createLinearGradient(x, 0, x + band, 0);
      fade.addColorStop(0, `rgba(${tint},0.14)`);
      fade.addColorStop(1, `rgba(${tint},0)`);
      ctx.fillStyle = fade;
      ctx.fillRect(x, 0, band, h.height);
    }
    ctx.fillStyle = `rgba(${tint},0.55)`;
    ctx.fillRect(Math.round(x), 0, 1, h.height);
  }

  /** Boxes seen while recording stay visible, dashed, until the file run reaches them. */
  _drawGhosts(ctx, h) {
    const from = this.analyzedTo ?? Infinity;
    ctx.strokeStyle = COLORS.detected;
    ctx.globalAlpha = 0.65;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    for (const b of this.ghosts) {
      if (b.t1 < from) continue;
      const x0 = h.timeToX(b.t0), x1 = h.timeToX(b.t1);
      if (x1 < 0 || x0 > h.width) continue;
      const y0 = h.freqToY(b.fhi), y1 = h.freqToY(b.flo);
      ctx.strokeRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0));
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  /** How far along the box's pop-in is: 1 when done. */
  _popOf(b, now) {
    if (!this.rec) return 1;
    const born = this.born.get(b.id);
    return born == null ? 1 : Math.min(1, (now - born) / POP_MS);
  }

  _drawBoxes(ctx, h) {
    const now = performance.now(), calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (this.rec) {
      if (this.analyzedTo != null) this._drawFrontier(ctx, h);
    }
    // Dim what is not analyzed yet
    else if (this.analyzedTo != null) {
      const x0 = Math.max(0, h.timeToX(this.analyzedTo));
      ctx.fillStyle = this.spec.light ? 'rgba(255,255,255,0.6)' : 'rgba(5,4,12,0.55)';
      if (x0 < h.width) ctx.fillRect(x0, 0, h.width - x0, h.height);
    }
    if (this.ghosts.length) this._drawGhosts(ctx, h);
    ctx.font = '600 11px Roboto, system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    this.boxes.forEach((b, i) => {
      const g = this._geom(b), x0 = h.timeToX(g.t0), x1 = h.timeToX(g.t1);
      if (x1 < 0 || x0 > h.width) return;
      const y0 = h.freqToY(g.fhi), y1 = h.freqToY(g.flo), edit = this._edit(b);
      const selected = b.id === this.selectedId, active = selected || b.id === this.hoverId;
      const base = this._color(b), color = active && edit.include ? '#ffffff' : base;
      const pop = calm ? 1 : this._popOf(b, now);
      if (pop < 1) {
        // A ring that settles on the box as it appears
        const grow = (1 - pop) * 10, eased = pop * pop;
        ctx.strokeStyle = base;
        ctx.globalAlpha = (1 - eased) * 0.8;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0 - grow, y0 - grow, Math.max(2, x1 - x0) + 2 * grow, Math.max(2, y1 - y0) + 2 * grow);
        ctx.fillStyle = base;
        ctx.globalAlpha = (1 - eased) * 0.22;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      const provisional = this.rec && g.t1 > this.rec.finalTo;
      ctx.setLineDash(!edit.include ? [4, 3] : provisional ? [6, 3] : []);
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

  /** Class of a box as [id, name] from the manifest's names; drawn boxes take the model's only class, or none. */
  _class(b) {
    const names = this.model?.names || { 0: 'bird' }, ids = Object.keys(names);
    if (b.manual) return ids.length === 1 ? [+ids[0], names[ids[0]]] : [null, ''];
    const id = Math.round(b.cls);
    return [id, names[id] ?? `class_${id}`];
  }

  /** Common name of a detected box's class, else its eBird code. */
  _speciesName(b) {
    const [id, code] = this._class(b);
    return this.model?.common_names?.[id] || code;
  }

  _detections() {
    return this._included().map(r => ({
      species: this._class(r.b)[1],
      species_id: this._class(r.b)[0],
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
    // A stopped run does not cover the whole file: the file name says so in every format
    const stopped = !!this.raw?.stopped, base = `${this._baseName()}${stopped ? '_partial' : ''}`, name = this.audio?.name || 'audio';
    const dets = this._detections();
    if (!dets.length) return;
    const conf = +$('bbd-conf').value, gap = +$('bbd-gap').value;
    if (kind === 'json') {
      // Layout of detect_birds.py --output-format json-with-algorithm-metadata
      const payload = {
        audio_file: name,
        model_config: {
          model: this.model.file,
          confidence_threshold: conf,
          nms_iou_threshold: NMS_IOU,
          species_mapping: this.model.species_mapping ?? null,
          merge_method: this.model.merge_method,
          ...(this.model.merge_method === 'fusion' ? { phrase_gap: gap } : { song_gap_threshold: gap }),
        },
        demo: {
          source: 'BirdBox browser demo',
          review: 'detected: unchanged detector box; verified: checked by the user; adjusted: moved or resized by the user; drawn: added by the user',
          removed_by_user: this._counts().removed,
          ...(stopped ? { partial: true, analyzed_until_s: round(this.analyzedTo ?? 0, 3), audio_duration_s: round(this.audio.duration, 3) } : {}),
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
      // Everything in the table at full precision
      const header = ['box', 'time_start', 'time_end', 'freq_low_hz', 'freq_high_hz', 'avg_confidence', 'max_confidence', 'detections_merged', 'class_id', 'species', 'review'];
      const lines = [header.join(',')];
      this._included().forEach((r, k) => {
        const d = dets[k];
        lines.push([r.i + 1, r.g.t0.toFixed(6), r.g.t1.toFixed(6), r.g.flo.toFixed(3), r.g.fhi.toFixed(3),
          r.b.manual ? '' : r.b.avg.toFixed(6), r.b.manual ? '' : r.b.max.toFixed(6), d.detections_merged, d.species_id ?? '', csvField(d.species), d.review].join(','));
      });
      downloadFile(`${base}.BirdBox.csv`, `${lines.join('\n')}\n`, 'text/csv;charset=utf-8');
    }
  }
}

new Demo().init();
