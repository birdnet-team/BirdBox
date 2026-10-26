// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * One audio element for the demo. Emits 'time' while playing, 'state' on
 * play/pause and 'volume' on volume changes; play(from, to) plays a segment.
 *
 * Sound is off until the person turns it up: clicking a box plays it, which
 * should not surprise anyone with sound. The level is remembered.
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
   * Fader position 0-1, used as the element volume as it is: field recordings
   * are often quiet (the sample peaks at -32 dBFS), so a curve that lowers the
   * middle of the fader makes them inaudible.
   */
  _applyVolume(level) {
    this.level = level;
    if (level > 0) this.lastLevel = level;
    this.audio.volume = level;
    // iOS ignores volume (the hardware buttons set it) but honors muted
    this.audio.muted = level === 0;
  }

  setVolume(level) {
    level = Math.min(1, Math.max(0, level));
    if (level === this.level) return;
    this._applyVolume(level);
    try { localStorage.setItem(VOLUME_KEY, String(level)); } catch { /* not persisted */ }
    this.dispatchEvent(new Event('volume'));
  }

  toggleMute() { this.setVolume(this.muted ? this.lastLevel : 0); }

  setSource(url) {
    this.pause();
    this.stopAt = null;
    this.audio.src = url || '';
    if (url) this.audio.load();
  }

  async play(from, to) {
    if (!this.audio.src) return;
    if (from != null) this.audio.currentTime = Math.max(0, from);
    this.stopAt = to != null ? to : null;
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
