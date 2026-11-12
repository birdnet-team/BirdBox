// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The Demo tab shows the windows of a file evenly spaced (docs/javascripts/demo/pacer.js).
 * This replays simulated workers on a simulated clock and checks the spacing:
 *
 *   - Just-Bird: about 150 ms a window (172 ms on one wasm thread for the sample), slowly
 *     rising over the first minute, with now and then three windows at twice the time;
 *   - Birds-With-Labels: about 490 ms a window, now and then two at 1.5 times the time;
 *   - a redraw that costs more than its share of the time: each step shows as many windows;
 *   - fewer windows than a full hand, and a run stopped halfway.
 *
 * Shown as they come, these windows are up to twice as far apart as the median. Paced,
 * every step must be within 30% of the median and the spread (sd over mean) under 10%,
 * every window must be shown in order, and the last one at most the hand's time and
 * three steps after the worker is done.
 *
 * Usage: node demo_pacer.mjs
 */
const { Pacer, HAND_MS } = await import(new URL('../../docs/javascripts/demo/pacer.js', import.meta.url));

/** Seeded random numbers in [0, 1). */
const random = seed => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

/** Absolute times at which the worker finishes each window. */
function worker(windows, timeOf) {
  const at = [];
  let t = 0;
  for (let w = 0; w < windows; w++) at.push(t += timeOf(w));
  return at;
}

/** Replay a worker through a Pacer on a simulated clock. */
async function replay(arrivals, { cost = 0, stopAfter = Infinity } = {}) {
  let t = 0, seq = 0, doneAt = null, stopped = false;
  const events = [], shows = [];
  const schedule = (at, fn) => { const e = { at, fn, seq: seq++, live: true }; events.push(e); return e; };
  const pacer = new Pacer(item => {
    shows.push({ at: t, index: item.index, stopped });
    t += cost;   // the page is busy while it redraws
  }, { now: () => t, later: (fn, ms) => schedule(t + ms, fn), cancel: e => { if (e) e.live = false; } });
  arrivals.forEach((at, i) => schedule(at, () => {
    if (stopped) return;
    pacer.push({ index: i });
    if (i + 1 === stopAfter) { pacer.stop(); stopped = true; }
    else if (i + 1 === arrivals.length) pacer.finish().then(() => { doneAt = t; });
  }));
  for (;;) {
    const live = events.filter(e => e.live);
    if (!live.length) break;
    const e = live.reduce((a, b) => (a.at < b.at || (a.at === b.at && a.seq < b.seq) ? a : b));
    e.live = false;
    t = Math.max(t, e.at);   // a busy page delivers late
    e.fn();
    await new Promise(resolve => setImmediate(resolve));   // promise callbacks, as between tasks
  }
  return { shows, doneAt };
}

/** Spacing of steps per window: median, spread, the longest against the median. */
function spacing(times, indexes) {
  const per = [];
  for (let k = 1; k < times.length; k++) per.push((times[k] - times[k - 1]) / (indexes[k] - indexes[k - 1]));
  const sorted = [...per].sort((a, b) => a - b), median = sorted[sorted.length >> 1];
  const mean = per.reduce((a, b) => a + b, 0) / per.length;
  const sd = Math.sqrt(per.reduce((a, b) => a + (b - mean) ** 2, 0) / per.length);
  const worst = Math.max(...per.map(v => Math.abs(v - median) / median));
  return { median, spread: sd / mean, worst };
}

let failed = false;
const check = (ok, what) => { console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) failed = true; };

async function scenario(name, arrivals, options = {}) {
  console.log(name);
  const raw = spacing(arrivals, arrivals.map((_, i) => i));
  const { shows, doneAt } = await replay(arrivals, options);
  const steps = shows.slice(1).map((x, k) => x.index - shows[k].index);
  // The last step shows what is left, which may be fewer windows than the others
  const timed = steps.length > 1 && steps.at(-1) < steps.at(-2) ? shows.slice(0, -1) : shows;
  const s = spacing(timed.map(x => x.at), timed.map(x => x.index));
  const last = arrivals.at(-1), lag = doneAt - last;
  console.log(`  as they come: median ${raw.median.toFixed(0)} ms, spread ${(raw.spread * 100).toFixed(0)}%, longest ${(1 + raw.worst).toFixed(1)}x the median`);
  console.log(`  paced: median ${s.median.toFixed(0)} ms, spread ${(s.spread * 100).toFixed(1)}%, furthest ${(s.worst * 100).toFixed(0)}% from the median; windows per step ${[...new Set(steps)].join(', ')}; first shown ${(shows[0].at - arrivals[0]).toFixed(0)} ms after the worker, last ${lag.toFixed(0)} ms after`);
  check(shows.map(x => x.index).every((v, k, a) => !k || v > a[k - 1]) && shows.at(-1).index === arrivals.length - 1, 'every window shown, in order');
  check(s.spread < 0.1 && s.worst < 0.3, 'evenly spaced: spread under 10%, every step within 30% of the median');
  // What is in hand at the end: the hand, and a little more while the pace corrects
  const allowed = HAND_MS + 3 * s.median * Math.max(...steps);
  check(doneAt != null && lag <= allowed, `done soon after the worker (${lag.toFixed(0)} ms, at most ${allowed.toFixed(0)})`);
  return { shows, steps };
}

// Just-Bird: 130 ms rising to 150 ms over the first minute, now and then three windows at twice the time
for (const seed of [7, 8, 9]) {
  const r = random(seed), slow = new Set();
  for (let w = 8; w < 200; w += 15 + Math.floor(r() * 20)) for (let k = 0; k < 3; k++) slow.add(w + k);
  const time = w => (130 + 20 * Math.min(1, w / 40)) * (slow.has(w) ? 2 : 1) * (0.95 + 0.1 * r());
  await scenario(`Just-Bird, 200 windows (5 minutes), seed ${seed}`, worker(200, time));
}
// Birds-With-Labels: 490 ms, now and then two windows at 1.5 times the time
{
  const r = random(11), slow = new Set();
  for (let w = 8; w < 200; w += 15 + Math.floor(r() * 20)) for (let k = 0; k < 2; k++) slow.add(w + k);
  const time = w => 490 * (slow.has(w) ? 1.5 : 1) * (0.98 + 0.04 * r());
  await scenario('Birds-With-Labels, 200 windows (5 minutes)', worker(200, time));
}
// A redraw of 60 ms with windows every 100 ms: at most a quarter of the time on redraws
{
  const r = random(3);
  const { steps } = await scenario('Costly redraws: 60 ms each, a window every 100 ms', worker(120, () => 100 * (0.95 + 0.1 * r())), { cost: 60 });
  check(new Set(steps.slice(0, -1)).size === 1 && steps[0] > 1, `each step shows as many windows (${steps[0]})`);
}
// Fewer windows than a full hand
{
  console.log('Two windows');
  const { shows, doneAt } = await replay([300, 450]);
  check(shows.length >= 1 && shows.at(-1).index === 1 && doneAt != null, 'both shown, and the run is done');
}
// Stopped halfway: what is in hand is dropped
{
  console.log('Stopped after 30 of 80 windows');
  const { shows } = await replay(worker(80, () => 140), { stopAfter: 30 });
  check(!shows.some(x => x.stopped) && shows.at(-1).index < 30, `nothing shown after the stop (last shown ${shows.at(-1).index})`);
}

console.log(failed ? 'Some checks FAILED.' : 'All checks passed.');
process.exit(failed ? 1 : 0);
