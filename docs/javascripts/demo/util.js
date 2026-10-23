// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Small helpers for the browser demo: formatting, icons, downloads, toasts.
 */

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ESC[c]);

/** Fill {name} placeholders. */
export const fill = (text, params) => Object.entries(params || {}).reduce((s, [k, v]) => s.split(`{${k}}`).join(v), text);

export function fmtTime(sec, decimals = 0) {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  // Round first so 59.999 s becomes 1:00.00, not 0:60.00
  const f = Math.pow(10, decimals);
  sec = decimals ? Math.round(sec * f) / f : Math.floor(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  let s = sec % 60;
  s = decimals ? s.toFixed(decimals).padStart(decimals + 3, '0') : String(Math.floor(s)).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

export function fmtNumber(value, digits = 0) {
  return Number(value).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export const fmtPct = (p, digits = 0) => `${fmtNumber(p * 100, digits)}%`;

export function fmtBytes(bytes) {
  if (bytes < 1048576) return `${fmtNumber(bytes / 1024, 0)} KB`;
  return `${fmtNumber(bytes / 1048576, bytes < 10 * 1048576 ? 1 : 0)} MB`;
}

/** One line of key figures: [[value, label], ...] -> "8 boxes · 2 verified". */
export function statLine(stats) {
  return stats.map(([v, l]) => `<span><b>${esc(typeof v === 'number' ? fmtNumber(v) : v)}</b> ${esc(l)}</span>`).join('');
}

export function csvField(value) {
  const s = String(value ?? '');
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function downloadFile(name, content, type = 'text/plain;charset=utf-8') {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

export function debounce(fn, ms) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

/** Inline SVG of a Material Design icon from the page's #bbd-icons template. */
const icons = new Map();
export function icon(name) {
  if (!icons.has(name)) {
    const el = document.getElementById('bbd-icons')?.content.querySelector(`[data-icon="${name}"]`);
    icons.set(name, el ? el.innerHTML.trim() : '');
  }
  return `<span class="bbd-icon" aria-hidden="true">${icons.get(name)}</span>`;
}

/** A short message at the bottom of the page, optionally with one action (e.g. Undo). */
export function toast(message, action) {
  let host = document.getElementById('bbd-toasts');
  if (!host) {
    host = document.createElement('div');
    host.id = 'bbd-toasts';
    host.className = 'bbd-toasts';
    host.setAttribute('aria-live', 'polite');
    document.body.appendChild(host);
  }
  const el = document.createElement('div');
  el.className = 'bbd-toast';
  const text = document.createElement('span');
  text.textContent = message;
  el.appendChild(text);
  const close = () => { el.classList.remove('is-shown'); setTimeout(() => el.remove(), 300); };
  if (action) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = action.label;
    btn.addEventListener('click', () => { action.onClick(); close(); });
    el.appendChild(btn);
  }
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add('is-shown'));
  setTimeout(close, action ? 6500 : 3800);
}
