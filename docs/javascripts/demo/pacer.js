// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Shows the windows of a file evenly spaced while detection runs.
 *
 * The worker finishes windows unevenly: now and then a few take longer, and with a
 * per-minute graph (no rolling floor) the first window of each minute also computes
 * that minute's spectrogram, which takes about as long as ten others. Shown as they
 * come, the line would stall and jump. So a few finished windows are held in hand
 * (about HAND_MS of them, at least two), and the windows show one interval apart:
 *
 *   - the interval is the worker's mean time per window, a mean that follows it
 *     slowly (FOLLOW). For a per-minute graph, while another minute is still to
 *     start, the extra time of a minute's first window is spread over the minute's
 *     windows: the worker then gets ahead by about that time over a minute, and the
 *     windows in hand cover the next minute's start;
 *   - with fewer windows in hand than that the pace slows a little, with more it
 *     quickens a little (CORRECT per window); with none, the next window shows as
 *     soon as it is done;
 *   - the interval changes by at most GLIDE per step, so the pace glides, also in
 *     the last minute, where the windows held for a minute's start are let go;
 *   - the first window waits for a full hand, which the gentle pace would take long
 *     to fill.
 *
 * When a redraw costs more than its share of the time (SHARE: a long recording with
 * many boxes) or windows come faster than MIN_STEP_MS, each step shows the same number
 * of windows. tests/js/demo_pacer.mjs checks the spacing on simulated workers.
 */

export const HAND_MS = 600;   // windows held back: about this much of the worker's time
const FOLLOW = 0.05;          // how fast the mean follows the worker's time per window
const CORRECT = 0.03;         // pace change per window in hand too few or too many
const GLIDE = 0.03;           // the interval changes by at most this share per step
const MIN_STEP_MS = 50;       // redraw at most this often,
const SHARE = 0.25;           // and spend at most this share of the time on it

export class Pacer {
  /**
   * @param {(item: object) => void} show  draws the last item of a step
   * @param {object} [options]
   * @param {number} [options.period]  windows per minute whose first window also computes
   *   the minute's spectrogram: 40 for a per-minute graph, 0 with a rolling floor
   * @param {number} [options.total]  windows in the file
   * @param {() => number} [options.now]  clock in ms (tests)
   * @param {(fn: () => void, ms: number) => any} [options.later]  setTimeout (tests), and cancel: clearTimeout
   * @param {{handMs?: number, follow?: number, correct?: number}} [options.tune]  other settings (tests)
   */
  constructor(show, { period = 0, total = Infinity, now = () => performance.now(), later = (fn, ms) => setTimeout(fn, ms), cancel = t => clearTimeout(t), tune = {} } = {}) {
    Object.assign(this, { show, period, total, now, later, cancel });
    this.handMs = tune.handMs ?? HAND_MS;
    this.follow = tune.follow ?? FOLLOW;
    this.correct = tune.correct ?? CORRECT;
    this.queue = [];
    this.mean = 0;            // ms per window, without a minute's start
    this.extra = 0;           // ms a minute's first window takes on top of the mean
    this.lastIn = this.now(); // the run starts now: the first window's time counts from here
    this.lastGap = 0;
    this.lastIndex = -1;      // the last window the worker finished
    this.count = 0;           // windows pushed
    this.per = 0;             // the interval in use, ms per window
    this.minStep = MIN_STEP_MS;
    this.timer = null;
    this.idle = true;
    this.started = false;
    this.ending = false;      // the worker is done: the rest shows, also in a smaller step
    this.finished = null;
  }

  /** Is a minute still to start for the worker? Its extra time is then spread over the windows before it. */
  get minuteAhead() {
    return !!this.period && (Math.floor(this.lastIndex / this.period) + 1) * this.period < this.total;
  }

  /** Mean time per window, with a minute's start while one is ahead. */
  get interval() { return this.mean + (this.minuteAhead ? this.extra / this.period : 0); }

  /** Windows held back. Before the mean is known: what the first windows took, at least two. */
  get inHand() { return Math.max(2, Math.round(this.handMs / (this.mean || this.lastGap || this.handMs))); }

  /** Windows in hand above which the pace quickens: with a minute ahead, enough to cover its start. */
  get most() { return this.inHand + (this.minuteAhead && this.mean ? Math.ceil(this.extra / this.mean) : 0); }

  /** Windows per step: a redraw at most every minStep ms. */
  get size() { return Math.max(1, Math.ceil(this.minStep / (this.per || this.interval || this.minStep))); }

  /**
   * Window `index` (counted from the file start) is done; `item` is what show() gets.
   * The first window of a run does not count towards the mean: it also waits for the first frames.
   */
  push(item, index) {
    const now = this.now(), gap = now - this.lastIn, first = !this.count++;
    this.lastIn = now;
    this.lastGap = gap;
    this.lastIndex = index;
    if (this.period && index % this.period === 0) {
      // A minute's start: its extra time over the mean (the first run's mean comes from the windows after it)
      this.startGap = gap;
    } else if (!first) {
      this.mean = this.mean ? this.mean + this.follow * (gap - this.mean) : gap;
    }
    if (this.startGap != null && this.mean) {
      const extra = Math.max(0, this.startGap - this.mean);
      this.extra = this.extra ? this.extra + 0.5 * (extra - this.extra) : extra;
      this.startGap = null;
    }
    this.queue.push(item);
    // A step waits for its windows, and the first one for a full hand besides
    if (this.idle && this.queue.length >= this.size + (this.started ? 0 : this.inHand)) {
      this.started = true;
      this._step();
    }
  }

  /** Resolves once the last window is shown. */
  finish() {
    this.ending = true;
    if (this.idle && this.queue.length) {   // the rest: fewer windows than a hand or a step
      this.started = true;
      this._step();
    }
    if (this.idle) return Promise.resolve();
    return new Promise(resolve => { this.finished = resolve; });
  }

  /** Drop what is in hand: the run was stopped or replaced. */
  stop() {
    this.cancel(this.timer);
    this.queue = [];
    this.idle = true;
    this.finished = null;
  }

  _step() {
    this.timer = null;
    const size = this.size;
    if (!this.queue.length || (this.queue.length < size && !this.ending)) {
      // Nothing to show, or not a whole step: the next window shows (or completes the step) as soon as it is done
      this.idle = true;
      if (!this.queue.length) {
        this.finished?.();
        this.finished = null;
      }
      return;
    }
    this.idle = false;
    const t0 = this.now(), most = this.most;
    // Far behind (a hidden page, whose timers are slowed): catch up in larger steps
    const n = Math.min(this.queue.length, Math.max(size, Math.ceil((this.queue.length - most - 2) / 3)));
    const items = this.queue.splice(0, n);
    this.show(items[items.length - 1]);
    const now = this.now();
    this.minStep = Math.max(MIN_STEP_MS, (now - t0) / SHARE);
    if (this.finished && !this.queue.length) {
      this.idle = true;
      this.finished();
      this.finished = null;
      return;
    }
    const q = this.queue.length, inHand = this.inHand;
    const pace = Math.min(1.2, Math.max(0.6, q < inHand ? 1 + this.correct * (inHand - q) : q > most ? 1 - this.correct * (q - most) : 1));
    const target = (this.interval || this.minStep) * pace;
    this.per = this.per ? Math.min(this.per * (1 + GLIDE), Math.max(this.per * (1 - GLIDE), target)) : target;
    // The next step comes after the time of its own windows (after the first redraw, steps may grow)
    this.timer = this.later(() => this._step(), Math.max(0, t0 + this.size * this.per - now));
  }
}
