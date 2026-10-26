// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * One audio element for the demo. Emits 'time' while playing, 'state' on
 * play/pause and 'volume' on volume changes; play(from, to) plays a segment.
 *
 * Sound is off until the person turns it up: clicking a box plays it, which
 * should not surprise anyone with sound. The level is remembered.
 *
 * Playback runs through a Web Audio gain node once something plays, so quiet
 * recordings can be played louder than the file (setSource's gain) and the
 * fader also works on iOS, which ignores the element volume. Detection
 * always uses the decoded audio, not this.
 */
const VOLUME_KEY = 'birdbox-demo-volume';
const UNMUTE_LEVEL = 1;     // the mute button turns sound on at full volume unless a level was set

class Player extends EventTarget {
  constructor() {
    super();
    this.audio = new Audio();
    this.audio.preload = 'auto';
    let saved = NaN;
    try { saved = parseFloat(localStorage.getItem(VOLUME_KEY)); } catch { /* storage blocked */ }
    this.lastLevel = UNMUTE_LEVEL;
    this.gain = 1;
    this.graph = undefined;   // { ctx, node } once routed, null if Web Audio is unavailable
    this._applyVolume(Number.isFinite(saved) ? Math.min(1, Math.max(0, saved)) : 0);
    this.stopAt = null;
    this.raf = 0;
    const tick = () => {
      if (this.stopAt != null && this.audio.currentTime >= this.stopAt) {
        this.audio.pause();
        this.audio.currentTime = this.stopAt;
        this.stopAt = null;
      }
      this.dispatchEvent(new CustomEvent('time', { detail: this.audio.currentTime }));
      if (!this.audio.paused) this.raf = requestAnimationFrame(tick);
    };
    this.audio.addEventListener('play', () => {
      cancelAnimationFrame(this.raf);
      this.raf = requestAnimationFrame(tick);
      this.dispatchEvent(new Event('state'));
    });
    this.audio.addEventListener('pause', () => { this.dispatchEvent(new Event('state')); tick(); });
    this.audio.addEventListener('ended', () => { this.stopAt = null; this.dispatchEvent(new Event('state')); });
    this.audio.addEventListener('seeked', () => this.dispatchEvent(new CustomEvent('time', { detail: this.audio.currentTime })));
  }

  get time() { return this.audio.currentTime || 0; }
  get playing() { return !this.audio.paused; }
  get duration() { return this.audio.duration || 0; }
  get volume() { return this.level; }
  get muted() { return this.level === 0; }

  /**
   * Fader position 0-1, applied linearly: field recordings are often quiet,
   * so a curve that lowers the middle of the fader makes them inaudible.
   */
  _applyVolume(level) {
    this.level = level;
    if (level > 0) this.lastLevel = level;
    if (this.graph) {
      this.audio.volume = 1;
      this.audio.muted = false;
      this.graph.node.gain.value = level * this.gain;
    } else {
      this.audio.volume = level;
      // iOS ignores volume (the hardware buttons set it) but honors muted
      this.audio.muted = level === 0;
    }
  }

  /** Route the element through a gain node; needs a user gesture, so it happens on the first play. */
  _route() {
    if (this.graph !== undefined) {
      if (this.graph?.ctx.state === 'suspended') this.graph.ctx.resume();
      return;
    }
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const node = ctx.createGain();
      ctx.createMediaElementSource(this.audio).connect(node).connect(ctx.destination);
      this.graph = { ctx, node };
    } catch (err) {
      console.warn('Playback without Web Audio, so no gain for quiet recordings:', err);
      this.graph = null;
    }
    this._applyVolume(this.level);
  }

  setVolume(level) {
    level = Math.min(1, Math.max(0, level));
    if (level === this.level) return;
    this._applyVolume(level);
    try { localStorage.setItem(VOLUME_KEY, String(level)); } catch { /* not persisted */ }
    this.dispatchEvent(new Event('volume'));
  }

  toggleMute() { this.setVolume(this.muted ? this.lastLevel : 0); }

  /** gain: playback amplification for this recording (1 = as recorded). */
  setSource(url, { gain = 1 } = {}) {
    this.pause();
    this.stopAt = null;
    this.gain = gain;
    this._applyVolume(this.level);
    this.audio.src = url || '';
    if (url) this.audio.load();
  }

  async play(from, to) {
    if (!this.audio.src) return;
    if (from != null) this.audio.currentTime = Math.max(0, from);
    this.stopAt = to != null ? to : null;
    this._route();
    try { await this.audio.play(); } catch { /* autoplay blocked until a user gesture */ }
  }

  pause() { this.audio.pause(); }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(time) {
    if (!this.audio.src) return;
    this.stopAt = null;
    this.audio.currentTime = Math.max(0, time);
  }
}

export const player = new Player();
