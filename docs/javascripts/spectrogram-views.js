// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The spectrogram figures of how-it-works/demo-spectrogram.md: axes, boxes, the window and a
 * linked cross over the images, the level chart and the frequency axes. Data from
 * spectrogram_views.json (scripts/make_spectrogram_views.mjs). Runs only on pages
 * with a #sv-views figure.
 */
(() => {
  const figure = document.getElementById('sv-views');
  if (!figure) return;

  const F_MAX = 15000;
  const mel = hz => 2595 * Math.log10(1 + hz / 700), M0 = mel(50), M1 = mel(15000);
  const hzOfMel = m => 700 * (Math.pow(10, m / 2595) - 1);
  // y as a fraction from the top: linear rows from 0 Hz, or the centres of 256 mel bands
  const yLinear = f => 1 - f / F_MAX, fLinear = y => (1 - y) * F_MAX;
  const yMel = f => 1 - ((mel(f) - M0) / (M1 - M0) * 257 - 0.5) / 256;
  const fMel = y => hzOfMel(M0 + ((1 - y) * 256 + 0.5) / 257 * (M1 - M0));
  const fmtHz = f => f >= 1000 ? `${(f / 1000).toFixed(f < 10000 ? 1 : 0)} kHz` : `${Math.round(f)} Hz`;
  const fmtDb = db => `${db > 0 ? '+' : db < 0 ? '−' : ''}${Math.abs(db).toFixed(0)}`;
  const NS = 'http://www.w3.org/2000/svg';

  fetch(figure.dataset.src).then(r => r.json()).then(start).catch(err => console.error('spectrogram views:', err));

  function start(D) {
    const SPAN = D.to - D.from, WIN = D.window;
    const views = {
      stft: { x: t => (t - D.from) / SPAN, t: x => D.from + x * SPAN, y: yLinear, f: fLinear },
      logfloor: { x: t => (t - D.from) / SPAN, t: x => D.from + x * SPAN, y: yMel, f: fMel },
      window: { x: t => (t - WIN) / 3, t: x => WIN + x * 3, y: yMel, f: fMel },
    };
    for (const el of document.querySelectorAll('[data-sv="hi"]')) el.textContent = D.hi.toFixed(0);
    for (const el of document.querySelectorAll('[data-sv="band"]')) el.textContent = fmtHz(D.band.hz);

    // Axes
    const ticks = [500, 1000, 2000, 4000, 8000, 12000];
    for (const el of document.querySelectorAll('.sv-axis')) {
      const isMel = el.dataset.axis === 'mel', y = isMel ? yMel : yLinear;
      el.innerHTML = ticks.filter(f => isMel || f >= 1000).map(f => `<span style="top:${(y(f) * 100).toFixed(2)}%">${f >= 1000 ? f / 1000 + 'k' : f}</span>`).join('');
    }
    for (const el of document.querySelectorAll('.sv-time')) {
      const [a, b, step] = el.dataset.axis === 'window' ? [WIN, WIN + 3, 1] : [D.from, D.to, 1];
      const marks = [];
      for (let t = a; t <= b + 1e-9; t += step) marks.push(`<span style="left:${((t - a) / (b - a) * 100).toFixed(2)}%">${+t.toFixed(1)} s</span>`);
      el.innerHTML = marks.join('');
    }

    // Overlays, with one cross for all images
    const stages = [...document.querySelectorAll('.sv-stage')].map(el => ({ el, view: views[el.dataset.view], canvas: el.querySelector('canvas') }));
    const showBoxes = document.getElementById('sv-boxes'), showWindow = document.getElementById('sv-window');
    let hover = null;
    function draw() {
      for (const s of stages) {
        const r = s.el.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
        s.canvas.width = Math.round(W * dpr);
        s.canvas.height = Math.round(H * dpr);
        const g = s.canvas.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (showWindow?.checked && s.el.dataset.view !== 'window') {
          const x0 = s.view.x(WIN) * W, x1 = s.view.x(WIN + 3) * W;
          g.fillStyle = 'rgba(0,0,4,0.35)';
          g.fillRect(0, 0, x0, H);
          g.fillRect(x1, 0, W - x1, H);
          g.strokeStyle = '#f6a723';
          g.lineWidth = 1.5;
          g.setLineDash([5, 4]);
          g.strokeRect(x0 + 0.75, 0.75, x1 - x0 - 1.5, H - 1.5);
          g.setLineDash([]);
        }
        if (!showBoxes || showBoxes.checked) {
          g.strokeStyle = '#5ee0f5';
          g.lineWidth = 1.5;
          for (const b of D.boxes) {
            const x0 = Math.max(-2, s.view.x(b.t0) * W), x1 = Math.min(W + 2, s.view.x(b.t1) * W);
            if (x1 < 0 || x0 > W) continue;
            const y0 = s.view.y(b.f1) * H, y1 = s.view.y(b.f0) * H;
            g.strokeRect(x0, y0, x1 - x0, y1 - y0);
          }
        }
        if (hover) {
          const x = s.view.x(hover.t) * W, y = s.view.y(hover.f) * H, inside = x >= 0 && x <= W;
          g.strokeStyle = 'rgba(255,255,255,0.85)';
          g.lineWidth = 1;
          g.beginPath();
          if (inside) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, H); }
          g.moveTo(0, Math.round(y) + 0.5);
          g.lineTo(W, Math.round(y) + 0.5);
          g.stroke();
          const text = `${fmtHz(hover.f)} · ${hover.t.toFixed(2)} s`;
          g.font = '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
          const tw = g.measureText(text).width + 12, ax = inside ? x : W / 2;
          let lx = ax + 8, ly = y - 24;
          if (lx + tw > W) lx = ax - tw - 8;
          if (ly < 4) ly = y + 8;
          g.fillStyle = 'rgba(0,0,4,0.78)';
          g.fillRect(lx, ly, tw, 18);
          g.fillStyle = '#ffffff';
          g.fillText(text, lx + 6, ly + 13);
        }
      }
    }
    const clamp = v => Math.min(1, Math.max(0, v));
    for (const s of stages) {
      s.el.addEventListener('pointermove', e => {
        const r = s.el.getBoundingClientRect();
        hover = { t: s.view.t(clamp((e.clientX - r.left) / r.width)), f: Math.min(F_MAX, Math.max(50, s.view.f(clamp((e.clientY - r.top) / r.height)))) };
        draw();
      });
      s.el.addEventListener('pointerleave', () => { hover = null; draw(); });
    }
    showBoxes?.addEventListener('change', draw);
    showWindow?.addEventListener('change', draw);
    window.addEventListener('resize', draw);
    for (const img of document.querySelectorAll('.sv-stage img')) if (!img.complete) img.addEventListener('load', draw);
    draw();

    drawLevels(D);
    drawFrequencies();
  }

  function svg(el, viewBox, html) {
    el.setAttribute('viewBox', viewBox);
    el.innerHTML = html;
  }

  /** Share of a band's frames below `threshold` dB above its 20th percentile (1 dB bins from -10 dB). */
  function below(hist, threshold) {
    let s = 0;
    hist.forEach((p, b) => { const lo = b - 10; if (lo + 1 <= threshold) s += p; else if (lo < threshold) s += p * (threshold - lo); });
    return s;
  }

  function drawLevels(D) {
    const el = document.getElementById('sv-levels');
    if (!el) return;
    const L = 110, R = 730, dbMin = -10, dbMax = 50, top = 22, base = 120;
    const X = db => L + (db - dbMin) / (dbMax - dbMin) * (R - L);
    const hist = D.band.mel.hist.slice(0, dbMax - dbMin), peak = Math.max(...hist);
    let path = `M${X(dbMin)},${base}`;
    hist.forEach((p, b) => { const y = base - p / peak * (base - top); path += ` L${X(dbMin + b)},${y} L${X(dbMin + b + 1)},${y}`; });
    path += ` L${X(dbMax)},${base} Z`;
    const stftBlack = D.band.stft.medianAboveP20 + 7, stftWhite = D.band.stft.medianAboveP20 + D.hi;
    const gradient = `<defs><linearGradient id="sv-inferno" x1="0" x2="1" y1="0" y2="0">${D.stops.map((c, i) => `<stop offset="${i * 10}%" stop-color="${c}"/>`).join('')}</linearGradient></defs>`;
    let grid = '';
    for (let db = dbMin; db <= dbMax; db += 10) grid += `<line class="sv-grid" x1="${X(db)}" x2="${X(db)}" y1="${top - 8}" y2="232"/><text x="${X(db)}" y="248" text-anchor="middle">${fmtDb(db)} dB</text>`;
    const bar = (y, a, b, label, share) => `
      <rect x="${X(dbMin)}" y="${y}" width="${X(a) - X(dbMin)}" height="20" fill="${D.stops[0]}"/>
      <rect x="${X(a)}" y="${y}" width="${X(b) - X(a)}" height="20" fill="url(#sv-inferno)"/>
      <rect x="${X(b)}" y="${y}" width="${X(dbMax) - X(b)}" height="20" fill="${D.stops[10]}"/>
      <text class="sv-label" x="${L - 10}" y="${y + 15}" text-anchor="end">${label}</text>
      <text x="${X(a)}" y="${y - 5}" text-anchor="middle">${fmtDb(a)}</text>
      <text x="${X(b)}" y="${y - 5}" text-anchor="middle">${fmtDb(b)}</text>
      <text class="sv-on-dark" x="${X(dbMin) + 6}" y="${y + 15}">${Math.round(share * 100)}% black</text>`;
    const median = X(D.band.mel.medianAboveP20);
    svg(el, '0 0 760 256', `${gradient}${grid}
      <path class="sv-hist" d="${path}"/>
      <line class="sv-ink" x1="${X(0)}" x2="${X(0)}" y1="${top - 8}" y2="${base}"/>
      <text class="sv-ink-text" x="${X(0) - 6}" y="${top + 4}" text-anchor="end">20th percentile</text>
      <line class="sv-ink sv-dashed" x1="${median}" x2="${median}" y1="${top + 14}" y2="${base}"/>
      <text class="sv-ink-text" x="${median + 6}" y="${top + 24}">median</text>
      ${bar(150, -3, 45, 'Model', below(D.band.mel.hist, -3))}
      ${bar(192, stftBlack, stftWhite, 'STFT view', below(D.band.stft.hist, stftBlack))}`);
  }

  function drawFrequencies() {
    const el = document.getElementById('sv-frequencies');
    if (!el) return;
    const top = 26, bottom = 276, xa = 230, xb = 530, H = bottom - top;
    const yl = f => top + yLinear(f) * H, ym = f => top + yMel(f) * H;
    let s = `<rect class="sv-half" x="${xa - 10}" y="${yl(2700)}" width="20" height="${yl(50) - yl(2700)}"/>
      <rect class="sv-half" x="${xb - 10}" y="${ym(2700)}" width="20" height="${ym(50) - ym(2700)}"/>
      <path class="sv-half sv-faint" d="M${xa + 10},${yl(2700)} L${xb - 10},${ym(2700)} L${xb - 10},${ym(50)} L${xa + 10},${yl(50)} Z"/>
      <line class="sv-ink" x1="${xa}" x2="${xa}" y1="${top}" y2="${bottom}"/>
      <line class="sv-ink" x1="${xb}" x2="${xb}" y1="${top}" y2="${bottom}"/>
      <text class="sv-label" x="${xa}" y="${top - 10}" text-anchor="middle">STFT view, linear</text>
      <text class="sv-label" x="${xb}" y="${top - 10}" text-anchor="middle">Log-floor, mel</text>`;
    let lastLinear = Infinity;   // from the bottom up: skip a linear label that would overlap the one below
    for (const f of [50, 500, 1000, 2000, 2700, 4000, 6000, 8000, 10000, 12000, 15000]) {
      const strong = f === 2700, label = f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, cls = strong ? ' class="sv-strong"' : '';
      s += `<line class="${strong ? 'sv-link sv-strong-line' : 'sv-link'}" x1="${xa}" x2="${xb}" y1="${yl(f)}" y2="${ym(f)}"/>
        <text${cls} x="${xb + 16}" y="${ym(f) + 4}">${label}</text>`;
      if (lastLinear - yl(f) >= 13 || strong) {
        s += `<text${cls} x="${xa - 16}" y="${yl(f) + 4}" text-anchor="end">${label}</text>`;
        lastLinear = yl(f);
      }
    }
    svg(el, '0 0 760 300', s);
  }
})();
