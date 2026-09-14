/**
 * Browser PCEN spectrogram renderer for the BirdBox ONNX demo.
 *
 * Mirrors the Streamlit visualization in src/streamlit/app.py:
 * flattop STFT -> HTK mel -> PCEN -> inferno colormap, then detection boxes
 * in time x mel-normalized coordinates on a horizontally scrollable canvas.
 */
(function (global) {
  "use strict";

  var SAMPLE_RATE = 32000;
  var N_FFT = 2048;
  var HOP_LENGTH = 375;
  var N_MELS = 256;
  var FMIN = 50;
  var FMAX = 15000;
  var LEFT_PAD_SECONDS = 0.5;
  var WARMUP_FRAMES = 100;
  var PCEN_GAIN = 0.75;
  var PCEN_BIAS = 1.0;
  var PCEN_POWER = 0.35;
  var PCEN_TIME_CONSTANT = 1.0;
  var PCEN_EPS = 1e-6;
  var VMIN = 0;
  var VMAX = 100;
  var PIXELS_PER_SECOND = 100;
  var CANVAS_HEIGHT = 420;

  var INFERNO = new Uint8Array([0,0,4,1,0,5,1,1,6,1,1,8,2,1,10,2,2,12,2,2,14,3,2,16,4,3,18,4,3,20,5,4,23,6,4,25,7,5,27,8,5,29,9,6,31,10,7,34,11,7,36,12,8,38,13,8,41,14,9,43,16,9,45,17,10,48,18,10,50,20,11,52,21,11,55,22,11,57,24,12,60,25,12,62,27,12,65,28,12,67,30,12,69,31,12,72,33,12,74,35,12,76,36,12,79,38,12,81,40,11,83,41,11,85,43,11,87,45,11,89,47,10,91,49,10,92,50,10,94,52,10,95,54,9,97,56,9,98,57,9,99,59,9,100,61,9,101,62,9,102,64,10,103,66,10,104,68,10,104,69,10,105,71,11,106,73,11,106,74,12,107,76,12,107,77,13,108,79,13,108,81,14,108,82,14,109,84,15,109,85,15,109,87,16,110,89,16,110,90,17,110,92,18,110,93,18,110,95,19,110,97,19,110,98,20,110,100,21,110,101,21,110,103,22,110,105,22,110,106,23,110,108,24,110,109,24,110,111,25,110,113,25,110,114,26,110,116,26,110,117,27,110,119,28,109,120,28,109,122,29,109,124,29,109,125,30,109,127,30,108,128,31,108,130,32,108,132,32,107,133,33,107,135,33,107,136,34,106,138,34,106,140,35,105,141,35,105,143,36,105,144,37,104,146,37,104,147,38,103,149,38,103,151,39,102,152,39,102,154,40,101,155,41,100,157,41,100,159,42,99,160,42,99,162,43,98,163,44,97,165,44,96,166,45,96,168,46,95,169,46,94,171,47,94,173,48,93,174,48,92,176,49,91,177,50,90,179,50,90,180,51,89,182,52,88,183,53,87,185,53,86,186,54,85,188,55,84,189,56,83,191,57,82,192,58,81,193,58,80,195,59,79,196,60,78,198,61,77,199,62,76,200,63,75,202,64,74,203,65,73,204,66,72,206,67,71,207,68,70,208,69,69,210,70,68,211,71,67,212,72,66,213,74,65,215,75,63,216,76,62,217,77,61,218,78,60,219,80,59,221,81,58,222,82,56,223,83,55,224,85,54,225,86,53,226,87,52,227,89,51,228,90,49,229,92,48,230,93,47,231,94,46,232,96,45,233,97,43,234,99,42,235,100,41,235,102,40,236,103,38,237,105,37,238,106,36,239,108,35,239,110,33,240,111,32,241,113,31,241,115,29,242,116,28,243,118,27,243,120,25,244,121,24,245,123,23,245,125,21,246,126,20,246,128,19,247,130,18,247,132,16,248,133,15,248,135,14,248,137,12,249,139,11,249,140,10,249,142,9,250,144,8,250,146,7,250,148,7,251,150,6,251,151,6,251,153,6,251,155,6,251,157,7,252,159,7,252,161,8,252,163,9,252,165,10,252,166,12,252,168,13,252,170,15,252,172,17,252,174,18,252,176,20,252,178,22,252,180,24,251,182,26,251,184,29,251,186,31,251,188,33,251,190,35,250,192,38,250,194,40,250,196,42,250,198,45,249,199,47,249,201,50,249,203,53,248,205,55,248,207,58,247,209,61,247,211,64,246,213,67,246,215,70,245,217,73,245,219,76,244,221,79,244,223,83,244,225,86,243,227,90,243,229,93,242,230,97,242,232,101,242,234,105,241,236,109,241,237,113,241,239,117,241,241,121,242,242,125,242,244,130,243,245,134,243,246,138,244,248,142,245,249,146,246,250,150,248,251,154,249,252,157,250,253,161,252,255,164]);

  var melBands = null;
  var flattopWindow = null;

  function hzToMel(hz) {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
  }

  function melToHz(mel) {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
  }

  function hzToMelNormalized(freqHz) {
    var mel = hzToMel(freqHz);
    var minMel = hzToMel(FMIN);
    var maxMel = hzToMel(FMAX);
    return (mel - minMel) / (maxMel - minMel);
  }

  function buildFlattop(n) {
    var a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368];
    var w = new Float32Array(n);
    var i;
    for (i = 0; i < n; i += 1) {
      var k = (2 * Math.PI * i) / n;
      w[i] =
        a[0] -
        a[1] * Math.cos(k) +
        a[2] * Math.cos(2 * k) -
        a[3] * Math.cos(3 * k) +
        a[4] * Math.cos(4 * k);
    }
    return w;
  }

  function buildMelBands() {
    var nFreqs = N_FFT / 2 + 1;
    var minMel = hzToMel(FMIN);
    var maxMel = hzToMel(FMAX);
    var melPoints = new Float64Array(N_MELS + 2);
    var i;
    for (i = 0; i < melPoints.length; i += 1) {
      melPoints[i] = minMel + (i / (N_MELS + 1)) * (maxMel - minMel);
    }
    var bins = new Float64Array(melPoints.length);
    for (i = 0; i < melPoints.length; i += 1) {
      bins[i] = ((N_FFT + 1) * melToHz(melPoints[i])) / SAMPLE_RATE;
    }
    var bands = new Array(N_MELS);
    var m;
    for (m = 0; m < N_MELS; m += 1) {
      var left = bins[m];
      var center = bins[m + 1];
      var right = bins[m + 2];
      var entries = [];
      var f;
      var f0 = Math.max(0, Math.floor(left));
      var f1 = Math.min(nFreqs - 1, Math.ceil(right));
      for (f = f0; f <= f1; f += 1) {
        var w = 0;
        if (f >= left && f <= center && center > left) {
          w = (f - left) / (center - left);
        } else if (f > center && f <= right && right > center) {
          w = (right - f) / (right - center);
        }
        if (w > 0) entries.push(f, w);
      }
      bands[m] = new Float32Array(entries);
    }
    return bands;
  }

  function ensureAssets() {
    if (!flattopWindow) flattopWindow = buildFlattop(N_FFT);
    if (!melBands) melBands = buildMelBands();
  }

  function fftRadix2(re, im) {
    var n = re.length;
    var i;
    var j = 0;
    for (i = 1; i < n; i += 1) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        var ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    var len;
    for (len = 2; len <= n; len <<= 1) {
      var half = len >> 1;
      var ang = (-2 * Math.PI) / len;
      var wlenRe = Math.cos(ang);
      var wlenIm = Math.sin(ang);
      for (i = 0; i < n; i += len) {
        var wr = 1;
        var wi = 0;
        for (j = 0; j < half; j += 1) {
          var uRe = re[i + j];
          var uIm = im[i + j];
          var vRe = re[i + j + half] * wr - im[i + j + half] * wi;
          var vIm = re[i + j + half] * wi + im[i + j + half] * wr;
          re[i + j] = uRe + vRe;
          im[i + j] = uIm + vIm;
          re[i + j + half] = uRe - vRe;
          im[i + j + half] = uIm - vIm;
          var nextWr = wr * wlenRe - wi * wlenIm;
          wi = wr * wlenIm + wi * wlenRe;
          wr = nextWr;
        }
      }
    }
  }

  function pcenSmoothing() {
    var ratio = (PCEN_TIME_CONSTANT * SAMPLE_RATE) / HOP_LENGTH;
    var b = (Math.sqrt(1.0 + 4.0 * ratio * ratio) - 1.0) / (2.0 * ratio * ratio);
    return Math.min(b, 1.0);
  }

  function computePcen(samples) {
    ensureAssets();
    var scaled = new Float32Array(samples.length);
    var i;
    for (i = 0; i < samples.length; i += 1) {
      scaled[i] = samples[i] * 2147483648.0;
    }

    var padLen = Math.floor(LEFT_PAD_SECONDS * SAMPLE_RATE);
    var padded = new Float32Array(padLen + scaled.length);
    padded.set(scaled.subarray(0, Math.min(padLen, scaled.length)), 0);
    padded.set(scaled, padLen);

    var nFreqs = N_FFT / 2 + 1;
    var maxFrames = 1 + Math.floor((padded.length - N_FFT) / HOP_LENGTH);
    if (maxFrames <= 0) {
      throw new Error("Audio too short for spectrogram.");
    }

    var mel = new Float32Array(N_MELS * maxFrames);
    var re = new Float64Array(N_FFT);
    var im = new Float64Array(N_FFT);
    var frame;
    for (frame = 0; frame < maxFrames; frame += 1) {
      var start = frame * HOP_LENGTH;
      for (i = 0; i < N_FFT; i += 1) {
        re[i] = padded[start + i] * flattopWindow[i];
        im[i] = 0;
      }
      fftRadix2(re, im);
      var m;
      for (m = 0; m < N_MELS; m += 1) {
        var band = melBands[m];
        var sum = 0;
        var k;
        for (k = 0; k < band.length; k += 2) {
          var bin = band[k];
          var weight = band[k + 1];
          sum += weight * (re[bin] * re[bin] + im[bin] * im[bin]);
        }
        mel[m * maxFrames + frame] = sum;
      }
    }

    var loopLength = Math.min(WARMUP_FRAMES, Math.floor(maxFrames / 4));
    var totalFrames = maxFrames + loopLength;
    var looped = new Float32Array(N_MELS * totalFrames);
    var m;
    for (m = 0; m < N_MELS; m += 1) {
      for (i = 0; i < loopLength; i += 1) {
        looped[m * totalFrames + i] = mel[m * maxFrames + i];
      }
      for (i = 0; i < maxFrames; i += 1) {
        looped[m * totalFrames + loopLength + i] = mel[m * maxFrames + i];
      }
    }

    var smoothCoef = pcenSmoothing();
    var decay = 1.0 - smoothCoef;
    var smoothed = new Float32Array(N_MELS * totalFrames);
    for (m = 0; m < N_MELS; m += 1) {
      var row = m * totalFrames;
      smoothed[row] = smoothCoef * looped[row];
      for (frame = 1; frame < totalFrames; frame += 1) {
        smoothed[row + frame] =
          smoothCoef * looped[row + frame] + decay * smoothed[row + frame - 1];
      }
    }

    var padFrames = Math.floor(padLen / HOP_LENGTH);
    var trim = loopLength + padFrames;
    var kept = maxFrames - padFrames;
    if (kept <= 0) {
      throw new Error("Spectrogram empty after padding trim.");
    }
    var pcen = new Float32Array(N_MELS * kept);
    for (m = 0; m < N_MELS; m += 1) {
      for (frame = 0; frame < kept; frame += 1) {
        var src = m * totalFrames + trim + frame;
        var value = looped[src];
        var ref = smoothed[src];
        pcen[m * kept + frame] =
          Math.pow(value / Math.pow(ref + PCEN_EPS, PCEN_GAIN) + PCEN_BIAS, PCEN_POWER) -
          Math.pow(PCEN_BIAS, PCEN_POWER);
      }
    }

    return { pcen: pcen, nMels: N_MELS, nFrames: kept };
  }

  function speciesColor(classId) {
    var id = Math.abs(Math.round(Number(classId)) || 0);
    var hue = (id * 47) % 360;
    return "hsl(" + hue + " 85% 45%)";
  }

  function drawSpectrogram(canvas, samples, duration, detections) {
    var spec = computePcen(samples);
    var width = Math.max(100, Math.round(duration * PIXELS_PER_SECOND));
    var height = CANVAS_HEIGHT;
    canvas.width = width;
    canvas.height = height;

    var ctx = canvas.getContext("2d");
    var image = ctx.createImageData(width, height);
    var data = image.data;
    var x;
    var y;
    for (x = 0; x < width; x += 1) {
      var frame = Math.min(
        spec.nFrames - 1,
        Math.floor(((x + 0.5) * spec.nFrames) / width)
      );
      for (y = 0; y < height; y += 1) {
        var melBin = Math.min(
          spec.nMels - 1,
          Math.floor((((height - 1 - y) + 0.5) * spec.nMels) / height)
        );
        var value = spec.pcen[melBin * spec.nFrames + frame];
        var norm = (value - VMIN) / (VMAX - VMIN);
        if (norm < 0) norm = 0;
        if (norm > 1) norm = 1;
        var idx = Math.min(255, (norm * 255) | 0);
        var o = (y * width + x) * 4;
        data[o] = INFERNO[idx * 3];
        data[o + 1] = INFERNO[idx * 3 + 1];
        data[o + 2] = INFERNO[idx * 3 + 2];
        data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);

    detections.forEach(function (det) {
      var flo = hzToMelNormalized(det.freq_low_hz);
      var fhi = hzToMelNormalized(det.freq_high_hz);
      if (flo < 0) flo = 0;
      if (fhi > 1) fhi = 1;
      if (flo > fhi) {
        var tmp = flo;
        flo = fhi;
        fhi = tmp;
      }
      var x0 = (det.time_start / duration) * width;
      var x1 = (det.time_end / duration) * width;
      var y0 = (1 - fhi) * height;
      var y1 = (1 - flo) * height;
      var color = speciesColor(det.class_id);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0));

      var conf = det.avg_confidence != null ? det.avg_confidence : det.confidence;
      var label =
        (det.species || "class_" + Math.round(det.class_id)) +
        " " +
        Number(conf).toFixed(2);
      ctx.font = "bold 12px sans-serif";
      var textWidth = ctx.measureText(label).width + 10;
      var labelY = Math.max(14, y0 - 4);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x0, labelY - 14, textWidth, 16);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x0 + 5, labelY - 2);
    });
  }

  function bindWheelScroll(container) {
    if (container.dataset.wheelBound === "1") return;
    container.dataset.wheelBound = "1";
    container.addEventListener(
      "wheel",
      function (event) {
        if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
        event.preventDefault();
        container.scrollLeft += event.deltaY;
      },
      { passive: false }
    );
  }

  function render(root, audio, detections) {
    var section = root.querySelector("#onnx-spec");
    var canvas = root.querySelector("#onnx-spec-canvas");
    var scroll = root.querySelector("#onnx-spec-scroll");
    var meta = root.querySelector("#onnx-spec-meta");
    if (!section || !canvas || !scroll || !audio) return;

    meta.textContent =
      "Audio duration: " +
      audio.duration.toFixed(1) +
      "s | Detections: " +
      (detections ? detections.length : 0) +
      " | Scroll horizontally to navigate the timeline";
    section.hidden = false;
    section.classList.add("is-visible");
    bindWheelScroll(scroll);
    drawSpectrogram(canvas, audio.samples, audio.duration, detections || []);
    scroll.scrollLeft = 0;
  }

  function clear(root) {
    var section = root.querySelector("#onnx-spec");
    if (!section) return;
    section.hidden = true;
    section.classList.remove("is-visible");
  }

  global.BirdBoxSpectrogram = {
    render: render,
    clear: clear,
    hzToMelNormalized: hzToMelNormalized,
  };
})(window);
