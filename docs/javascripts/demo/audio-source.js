// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Audio input for the demo: file, drag & drop, the sample, the microphone or
 * a Xeno-canto number. Emits 'audio' with the decoded recording:
 * { name, size, url, isSample, xc, pcm, duration, sourceRate, channels, truncatedFrom }.
 */
import { decodeAudio, Recorder, SAMPLE_RATE } from './audio.js';
import { esc, fill, fmtBytes, fmtPct, fmtTime, icon, toast } from './util.js';
import { player } from './player.js';

// Same cap as the Streamlit demo (config.MAX_DURATION_SECONDS). The graph
// itself has no length limit; browser memory is the practical one.
export const MAX_SECONDS = 600;
const MAX_RECORD_SECONDS = 300;
// The sample of the BirdNET website demo: BirdNET-Analyzer's example soundscape at 32 kHz
const SAMPLE = { url: new URL('../../audio/BirdNET_soundscape_32kHz.flac', import.meta.url), name: 'BirdNET_soundscape_32kHz.flac', type: 'audio/flac' };
const XC_EXT = { mpeg: 'mp3', mp3: 'mp3', wav: 'wav', 'x-wav': 'wav', wave: 'wav', flac: 'flac', 'x-flac': 'flac', ogg: 'ogg' };

/** Xeno-canto number from "XC1059662", "1059662" or a recording link; null if none. */
export function parseXcNumber(value) {
  const s = String(value ?? '').trim();
  const m = s.match(/^XC\s*(\d+)$/i) || s.match(/^(\d+)$/) || s.match(/xeno-canto\.org\/(\d+)(?:[/?#]|$)/i);
  return m ? m[1] : null;
}

class XcError extends Error {
  constructor(message, nr) { super(message); this.nr = nr; }
}

export class AudioSource extends EventTarget {
  /** @param {HTMLElement} root  the .bbd-source element of the page */
  constructor(root) {
    super();
    this.root = root;
    this.$ = role => root.querySelector(`[data-role="${role}"]`);
    this.audio = null;
    this.recorder = null;
    this.loadToken = 0;
    this._bind();
    player.addEventListener('state', () => {
      this._renderPlay();
      if (player.playing && player.muted) this._nudgeVolume();
    });
    player.addEventListener('volume', () => this._renderVolume());
    player.addEventListener('time', e => {
      if (this.audio) this.$('time').textContent = `${fmtTime(e.detail)} / ${fmtTime(this.audio.duration)}`;
    });
    this.render();
  }

  _bind() {
    const input = this.$('input');
    this.$('choose').addEventListener('click', () => input.click());
    input.addEventListener('change', () => { if (input.files[0]) this._loadFile(input.files[0]); input.value = ''; });
    this.$('sample').addEventListener('click', () => this._run(async () => {
      this._status('Loading the sample…');
      const res = await fetch(SAMPLE.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return this.load(new File([await res.blob()], SAMPLE.name, { type: SAMPLE.type }), { isSample: true });
    }));
    this.$('record').addEventListener('click', () => this._startRecording());
    this.$('stop').addEventListener('click', () => this._stopRecording());
    this.$('cancel').addEventListener('click', () => this._cancelRecording());
    this.$('replace').addEventListener('click', () => this._showDrop(true));
    this.$('back').addEventListener('click', () => this._showDrop(false));
    this.$('play').addEventListener('click', () => player.toggle());
    this.$('mute').addEventListener('click', () => player.toggleMute());
    this.$('volume').addEventListener('input', e => player.setVolume(+e.target.value));
    this.$('xc-form').addEventListener('submit', e => {
      e.preventDefault();
      const field = this.$('xc'), nr = parseXcNumber(field.value);
      field.classList.toggle('is-invalid', !nr);
      if (!nr) {
        this._message('Enter a Xeno-canto number such as XC1059662, or paste the recording link.');
        field.focus();
        return;
      }
      this._run(() => this._loadXenoCanto(nr).then(audio => { if (audio) field.value = ''; return audio; }));
    });
    this.$('xc').addEventListener('input', e => e.target.classList.remove('is-invalid'));
    // Files can be dropped anywhere on the demo
    const zone = this.root.closest('.bbd') || this.root, drop = this.$('drop');
    ['dragenter', 'dragover'].forEach(type => zone.addEventListener(type, e => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      zone.classList.add('is-dragover');
      drop.classList.add('is-over');
    }));
    ['dragleave', 'drop'].forEach(type => zone.addEventListener(type, e => {
      if (type === 'dragleave' && zone.contains(e.relatedTarget)) return;
      e.preventDefault();
      zone.classList.remove('is-dragover');
      drop.classList.remove('is-over');
    }));
    zone.addEventListener('drop', e => {
      const file = e.dataTransfer?.files?.[0];
      if (file) this._loadFile(file);
    });
  }

  _status(text) {
    this.$('status').textContent = text || '';
    this.root.classList.toggle('is-busy', !!text);
  }

  _message(text, xc) {
    const link = xc ? ` <a href="https://xeno-canto.org/${xc}" target="_blank" rel="noopener">xeno-canto.org/${xc}</a>` : '';
    this.$('status').innerHTML = `<span class="bbd-warn">${icon('alert')}${esc(text)}</span>${link}`;
    this.root.classList.remove('is-busy');
  }

  async _run(fn) {
    try {
      await fn();
      this._status(null);
    } catch (err) {
      console.error(err);
      if (err instanceof XcError) this._message(err.message, err.nr);
      else this._message('Could not decode this audio file. Try WAV, FLAC or MP3.');
    }
  }

  _loadFile(file) {
    if (!/^audio\/|^video\/(webm|mp4|ogg)/.test(file.type) && !/\.(wav|flac|mp3|ogg|oga|opus|m4a|aac|webm|aiff?)$/i.test(file.name)) {
      this._message('This does not look like an audio file.');
      return;
    }
    this._run(() => this.load(file));
  }

  /** Decode a File and make it the current recording. */
  async load(file, { isSample = false, xc = null } = {}) {
    const token = ++this.loadToken;
    this._status('Decoding audio…');
    const decoded = await decodeAudio(await file.arrayBuffer());
    if (token !== this.loadToken) return null;
    let truncatedFrom = null;
    if (decoded.pcm.length > MAX_SECONDS * SAMPLE_RATE) {
      truncatedFrom = decoded.duration;
      decoded.pcm = decoded.pcm.slice(0, MAX_SECONDS * SAMPLE_RATE);
      decoded.duration = MAX_SECONDS;
    }
    if (!xc) xc = file.name.match(/(?:^|[^A-Za-z0-9])XC[_-]?(\d+)/i)?.[1] || null;
    if (this.audio?.url) URL.revokeObjectURL(this.audio.url);
    const url = URL.createObjectURL(file);
    this.audio = { name: file.name, size: file.size, url, isSample, xc, truncatedFrom, ...decoded };
    player.setSource(url);
    this.render();
    this.dispatchEvent(new CustomEvent('audio', { detail: this.audio }));
    return this.audio;
  }

  /** Download a recording from Xeno-canto (served with open CORS) and load it. */
  async _loadXenoCanto(nr) {
    const loading = fill('Loading XC{nr} from Xeno-canto…', { nr });
    this._status(loading);
    // Offline, or Xeno-canto answered with a bot check instead of the file (no CORS header)
    const res = await fetch(`https://xeno-canto.org/${nr}/download`).catch(() => {
      throw new XcError(fill('Could not load XC{nr} from Xeno-canto. You can download it there and open the file here:', { nr }), nr);
    });
    const type = (res.headers.get('Content-Type') || '').split(';')[0].trim();
    if (res.status === 404 || res.status === 410 || type.startsWith('text/')) {
      throw new XcError(fill('Xeno-canto has no downloadable recording XC{nr}.', { nr }));
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const total = +res.headers.get('Content-Length') || 0, chunks = [];
    let loaded = 0;
    if (res.body?.getReader) {
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        this._status(`${loading} ${total ? fmtPct(loaded / total) : fmtBytes(loaded)}`);
      }
    } else chunks.push(new Uint8Array(await res.arrayBuffer()));
    const ext = XC_EXT[type.split('/')[1]] || 'mp3';
    return this.load(new File(chunks, `XC${nr}.${ext}`, { type: type || 'audio/mpeg' }), { xc: nr });
  }

  _showDrop(show) {
    this.$('drop').hidden = !show;
    this.$('bar').hidden = show || !this.audio;
    this.$('back').hidden = !this.audio;
    this.root.classList.toggle('has-audio', !show && !!this.audio);
  }

  render() {
    const a = this.audio;
    this._showDrop(!a);
    if (!a) return;
    this.$('name').textContent = a.xc ? `XC${a.xc}` : a.name;
    const link = this.$('xc-link');
    link.hidden = !a.xc;
    if (a.xc) {
      link.href = `https://xeno-canto.org/${a.xc}`;
      link.title = `Open XC${a.xc} on Xeno-canto`;
      link.setAttribute('aria-label', link.title);
    }
    const meta = [fmtTime(a.duration, a.duration < 60 ? 1 : 0)];
    if (a.size) meta.push(fmtBytes(a.size));
    if (a.sourceRate === SAMPLE_RATE) meta.push('32 kHz');
    else if (a.sourceRate) meta.push(`${(a.sourceRate / 1000).toLocaleString('en-US')} kHz → 32 kHz`);
    if (a.channels > 1) meta.push(`${a.channels} channels → mono`);
    if (a.truncatedFrom) meta.push(`first ${fmtTime(MAX_SECONDS)} of ${fmtTime(a.truncatedFrom)}`);
    this.$('meta').textContent = meta.join(' · ');
    this.$('kind').innerHTML = icon(a.isSample ? 'sample' : a.xc ? 'xc' : a.name.startsWith('recording_') ? 'mic' : 'file');
    this.$('time').textContent = `${fmtTime(player.time)} / ${fmtTime(a.duration)}`;
    this._renderPlay();
    this._renderVolume();
  }

  _renderVolume() {
    const v = player.volume, btn = this.$('mute');
    btn.innerHTML = icon(v === 0 ? 'volume-off' : v < 0.34 ? 'volume-low' : v < 0.67 ? 'volume-medium' : 'volume-high');
    btn.title = v === 0 ? 'Sound is off · click to turn it on (M)' : 'Mute (M)';
    btn.setAttribute('aria-label', v === 0 ? 'Turn sound on' : 'Mute');
    btn.setAttribute('aria-pressed', String(v === 0));
    const fader = this.$('volume');
    fader.value = v;
    fader.setAttribute('aria-valuetext', v === 0 ? 'off' : `${Math.round(v * 100)}%`);
    this.$('volume-group').classList.toggle('is-muted', v === 0);
  }

  /** Playing without sound: point at the fader, and say so once per visit. */
  _nudgeVolume() {
    const group = this.$('volume-group');
    group.classList.remove('is-nudged');
    void group.offsetWidth;   // restart the animation
    group.classList.add('is-nudged');
    if (this.nudged) return;
    this.nudged = true;
    toast('Sound is off. Turn it up with the volume fader or press M.', { label: 'Sound on', onClick: () => player.toggleMute() });
  }

  _renderPlay() {
    const btn = this.$('play');
    btn.innerHTML = icon(player.playing ? 'pause' : 'play');
    btn.title = player.playing ? 'Pause (Space)' : 'Play (Space)';
    btn.setAttribute('aria-label', btn.title);
  }

  async _startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      this._message('This browser cannot record audio.');
      return;
    }
    this.recorder = new Recorder({
      maxSeconds: MAX_RECORD_SECONDS,
      onTick: s => { this.$('timer').textContent = fmtTime(s); },
    });
    try {
      await this.recorder.start();
    } catch (err) {
      this._message(err.name === 'NotAllowedError'
        ? 'Microphone access was denied. Allow it in the browser settings to record.'
        : err.name === 'NotFoundError' ? 'No microphone found.' : 'Could not access the microphone.');
      this.recorder = null;
      return;
    }
    player.pause();
    this.$('timer').textContent = '0:00';
    this.$('idle').hidden = true;
    this.$('recording').hidden = false;
  }

  async _stopRecording() {
    const file = await this.recorder?.stop();
    this.recorder = null;
    this.$('idle').hidden = false;
    this.$('recording').hidden = true;
    if (file) this._run(() => this.load(file));
  }

  async _cancelRecording() {
    await this.recorder?.cancel();
    this.recorder = null;
    this.$('idle').hidden = false;
    this.$('recording').hidden = true;
  }
}
