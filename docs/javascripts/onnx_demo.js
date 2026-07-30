/**
 * In-browser BirdBox ONNX demo for Material for MkDocs.
 *
 * Expects a root element #onnx-demo on the page. Models are listed in
 * ../models/manifest.json (relative to this getting-started page).
 */
(function () {
  "use strict";

  var SAMPLE_RATE = 32000;
  var MIN_SECONDS = 3;
  var MAX_SECONDS = 60;
  var DEFAULT_CONF = 0.18;
  var DEFAULT_SONG_GAP = 0.1;

  var state = {
    session: null,
    modelUrl: null,
    modelCatalog: [],
    names: {},
    audio: null,
    fileName: null,
    detections: [],
  };

  function demoRoot() {
    return document.getElementById("onnx-demo");
  }

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function setStatus(root, message, tone) {
    var el = $(".onnx-demo__status", root);
    if (!el) return;
    el.textContent = message;
    if (tone) el.setAttribute("data-tone", tone);
    else el.removeAttribute("data-tone");
  }

  function setProgress(root, fraction) {
    var bar = $(".onnx-demo__progress", root);
    var fill = bar && bar.querySelector("span");
    if (!bar || !fill) return;
    if (fraction == null) {
      bar.classList.remove("is-active");
      fill.style.width = "0%";
      return;
    }
    bar.classList.add("is-active");
    fill.style.width = Math.max(0, Math.min(100, fraction * 100)).toFixed(1) + "%";
  }

  function resolveAsset(relativePath) {
    var script = document.querySelector('script[src*="onnx_demo.js"]');
    var base;
    if (script && script.src) {
      base = script.src.replace(/javascripts\/onnx_demo\.js.*$/, "");
    } else {
      base = new URL("../", window.location.href).href;
    }
    return new URL(relativePath.replace(/^\.\.\//, ""), base).href;
  }

  function formatSeconds(value) {
    return Number(value).toFixed(3);
  }

  function formatHz(value) {
    return Math.round(Number(value)).toLocaleString();
  }

  function speciesName(classId) {
    var key = String(classId);
    if (Object.prototype.hasOwnProperty.call(state.names, key)) {
      return state.names[key];
    }
    if (Object.prototype.hasOwnProperty.call(state.names, classId)) {
      return state.names[classId];
    }
    return "class_" + classId;
  }

  function ensureOrt() {
    if (typeof ort === "undefined") {
      throw new Error(
        "ONNX Runtime Web failed to load. Check your network connection and reload the page."
      );
    }
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    ort.env.wasm.simd = true;
    if (!ort.env.wasm.wasmPaths) {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    }
  }

  async function loadManifest(root) {
    var select = $("#onnx-model", root);
    select.innerHTML = "";
    var url = resolveAsset("models/manifest.json");
    var response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error("Could not load model list from " + url);
    }
    var data = await response.json();
    var models = data.models || [];
    if (!models.length) {
      throw new Error("No ONNX models listed in docs/models/manifest.json");
    }
    state.modelCatalog = models;
    models.forEach(function (model, index) {
      var option = document.createElement("option");
      option.value = model.file;
      option.textContent = model.label || model.file;
      if (index === 0) option.selected = true;
      select.appendChild(option);
    });
    applyCatalogNames(models[0]);
  }

  function applyCatalogNames(modelEntry) {
    if (modelEntry && modelEntry.names && typeof modelEntry.names === "object") {
      state.names = modelEntry.names;
      return;
    }
    state.names = {};
  }

  function selectedCatalogEntry(fileName) {
    var i;
    for (i = 0; i < state.modelCatalog.length; i += 1) {
      if (state.modelCatalog[i].file === fileName) {
        return state.modelCatalog[i];
      }
    }
    return null;
  }

  function parseMetadata(session) {
    // ONNX Runtime Web does not expose custom metadata_props on the session.
    // Prefer names from docs/models/manifest.json, then fall back if present.
    var meta = session.metadata || {};
    if (meta.names) {
      try {
        state.names = JSON.parse(meta.names);
      } catch (err) {
        /* keep catalog names */
      }
    }
    var conf = parseFloat(meta.default_conf);
    var gap = parseFloat(meta.default_song_gap);
    return {
      conf: Number.isFinite(conf) ? conf : DEFAULT_CONF,
      songGap: Number.isFinite(gap) ? gap : DEFAULT_SONG_GAP,
    };
  }

  async function loadModel(root, fileName) {
    ensureOrt();
    applyCatalogNames(selectedCatalogEntry(fileName));
    var url = resolveAsset("models/" + fileName);
    if (state.session && state.modelUrl === url) {
      return state.session;
    }

    setStatus(root, "Downloading model " + fileName + "…");
    setProgress(root, 0.05);

    var response = await fetch(url);
    if (!response.ok) {
      throw new Error("Failed to download " + fileName + " (" + response.status + ")");
    }

    var total = Number(response.headers.get("Content-Length")) || 0;
    var reader = response.body && response.body.getReader();
    var chunks = [];
    var received = 0;

    if (reader) {
      while (true) {
        var step = await reader.read();
        if (step.done) break;
        chunks.push(step.value);
        received += step.value.byteLength;
        if (total > 0) setProgress(root, Math.min(0.9, received / total));
        else setProgress(root, 0.35);
      }
    } else {
      var bufferFallback = await response.arrayBuffer();
      chunks = [new Uint8Array(bufferFallback)];
      received = bufferFallback.byteLength;
    }

    var modelBuffer = new Uint8Array(received);
    var offset = 0;
    chunks.forEach(function (chunk) {
      modelBuffer.set(chunk, offset);
      offset += chunk.byteLength;
    });

    setStatus(root, "Initializing ONNX Runtime…");
    setProgress(root, 0.95);

    if (state.session) {
      try {
        await state.session.release();
      } catch (err) {
        /* ignore */
      }
      state.session = null;
    }

    var session = await ort.InferenceSession.create(modelBuffer.buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    state.session = session;
    state.modelUrl = url;

    var defaults = parseMetadata(session);
    var confInput = $("#onnx-conf", root);
    var gapInput = $("#onnx-song-gap", root);
    confInput.value = String(defaults.conf);
    gapInput.value = String(defaults.songGap);
    $("#onnx-conf-value", root).value = Number(defaults.conf).toFixed(2);
    $("#onnx-song-gap-value", root).value = Number(defaults.songGap).toFixed(2);

    setProgress(root, 1);
    setStatus(root, "Model ready: " + fileName, "ok");
    return session;
  }

  function mixToMono(audioBuffer) {
    var channels = audioBuffer.numberOfChannels;
    var length = audioBuffer.length;
    if (channels === 1) {
      return audioBuffer.getChannelData(0).slice();
    }
    var mono = new Float32Array(length);
    var c;
    for (c = 0; c < channels; c += 1) {
      var data = audioBuffer.getChannelData(c);
      var i;
      for (i = 0; i < length; i += 1) {
        mono[i] += data[i];
      }
    }
    var inv = 1 / channels;
    for (i = 0; i < length; i += 1) {
      mono[i] *= inv;
    }
    return mono;
  }

  async function decodeAndResample(file) {
    var arrayBuffer = await file.arrayBuffer();
    var probeCtx = new (window.AudioContext || window.webkitAudioContext)();
    var decoded;
    try {
      decoded = await probeCtx.decodeAudioData(arrayBuffer.slice(0));
    } finally {
      await probeCtx.close();
    }

    var mono = mixToMono(decoded);
    var duration = mono.length / decoded.sampleRate;
    if (duration < MIN_SECONDS) {
      throw new Error(
        "Audio is only " +
          duration.toFixed(2) +
          " s. The model needs at least " +
          MIN_SECONDS +
          " seconds."
      );
    }

    var useDuration = Math.min(duration, MAX_SECONDS);
    var targetLength = Math.round(useDuration * SAMPLE_RATE);
    var offline = new OfflineAudioContext(1, targetLength, SAMPLE_RATE);
    var buffer = offline.createBuffer(1, Math.round(useDuration * decoded.sampleRate), decoded.sampleRate);
    buffer.copyToChannel(mono.subarray(0, buffer.length), 0);
    var source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start(0);
    var rendered = await offline.startRendering();
    return {
      samples: rendered.getChannelData(0).slice(),
      duration: useDuration,
      originalDuration: duration,
      truncated: duration > MAX_SECONDS,
      sampleRate: SAMPLE_RATE,
    };
  }

  function detectionsToRows(tensor) {
    var data = tensor.data;
    var rows = tensor.dims[0] || 0;
    var cols = tensor.dims[1] || 8;
    var out = [];
    var r;
    for (r = 0; r < rows; r += 1) {
      var o = r * cols;
      out.push({
        time_start: data[o],
        time_end: data[o + 1],
        freq_low_hz: data[o + 2],
        freq_high_hz: data[o + 3],
        avg_confidence: data[o + 4],
        max_confidence: data[o + 5],
        detections_merged: data[o + 6],
        class_id: data[o + 7],
        species: speciesName(Math.round(data[o + 7])),
      });
    }
    return out;
  }

  function renderMetrics(root, detections) {
    var box = $(".onnx-demo__metrics", root);
    box.classList.add("is-visible");
    $("#onnx-metric-count", root).textContent = String(detections.length);
    var species = {};
    var sum = 0;
    detections.forEach(function (det) {
      species[det.species] = true;
      sum += det.avg_confidence;
    });
    $("#onnx-metric-species", root).textContent = String(Object.keys(species).length);
    $("#onnx-metric-conf", root).textContent = detections.length
      ? (sum / detections.length).toFixed(3)
      : "—";
  }

  function renderTable(root, detections) {
    var wrap = $(".onnx-demo__results", root);
    var tbody = $("#onnx-results-body", root);
    tbody.innerHTML = "";
    detections.forEach(function (det) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" +
        formatSeconds(det.time_start) +
        "</td><td>" +
        formatSeconds(det.time_end) +
        "</td><td>" +
        formatHz(det.freq_low_hz) +
        "</td><td>" +
        formatHz(det.freq_high_hz) +
        "</td><td>" +
        det.avg_confidence.toFixed(3) +
        "</td><td>" +
        det.max_confidence.toFixed(3) +
        "</td><td>" +
        Math.round(det.detections_merged) +
        "</td><td>" +
        det.species +
        "</td>";
      tbody.appendChild(tr);
    });
    wrap.classList.add("is-visible");
    $("#onnx-download-csv", root).disabled = detections.length === 0;
  }

  function clearResults(root) {
    state.detections = [];
    $(".onnx-demo__metrics", root).classList.remove("is-visible");
    $(".onnx-demo__results", root).classList.remove("is-visible");
    $("#onnx-results-body", root).innerHTML = "";
    $("#onnx-download-csv", root).disabled = true;
    if (window.BirdBoxSpectrogram) {
      window.BirdBoxSpectrogram.clear(root);
    }
  }

  function toCsv(detections) {
    var header = [
      "time_start",
      "time_end",
      "freq_low_hz",
      "freq_high_hz",
      "avg_confidence",
      "max_confidence",
      "detections_merged",
      "class_id",
      "species",
    ];
    var lines = [header.join(",")];
    detections.forEach(function (det) {
      lines.push(
        [
          det.time_start.toFixed(6),
          det.time_end.toFixed(6),
          det.freq_low_hz.toFixed(3),
          det.freq_high_hz.toFixed(3),
          det.avg_confidence.toFixed(6),
          det.max_confidence.toFixed(6),
          Math.round(det.detections_merged),
          Math.round(det.class_id),
          JSON.stringify(det.species),
        ].join(",")
      );
    });
    return lines.join("\n");
  }

  async function runDetection(root) {
    var detectBtn = $("#onnx-detect", root);
    var modelSelect = $("#onnx-model", root);
    var conf = parseFloat($("#onnx-conf", root).value);
    var songGap = parseFloat($("#onnx-song-gap", root).value);

    if (!state.audio) {
      setStatus(root, "Choose an audio file first.", "error");
      return;
    }

    detectBtn.disabled = true;
    clearResults(root);

    try {
      var session = await loadModel(root, modelSelect.value);
      setStatus(root, "Running detection in WebAssembly…");
      setProgress(root, 0.2);

      // ORT Web only accepts names listed in session.inputNames. Models that
      // attach default initializers for conf / song_gap may omit them here.
      var feeds = {
        audio: new ort.Tensor("float32", state.audio.samples, [
          state.audio.samples.length,
        ]),
      };
      if (session.inputNames.indexOf("conf") !== -1) {
        feeds.conf = new ort.Tensor("float32", Float32Array.from([conf]), [1]);
      }
      if (session.inputNames.indexOf("song_gap") !== -1) {
        feeds.song_gap = new ort.Tensor(
          "float32",
          Float32Array.from([songGap]),
          [1]
        );
      }

      var t0 = performance.now();
      var results = await session.run(feeds);
      var elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      setProgress(root, 1);

      var outputName = session.outputNames[0];
      var detections = detectionsToRows(results[outputName]);
      state.detections = detections;
      renderMetrics(root, detections);
      renderTable(root, detections);

      setStatus(root, "Rendering PCEN spectrogram…");
      setProgress(root, 0.85);
      await new Promise(function (resolve) {
        setTimeout(resolve, 20);
      });
      if (window.BirdBoxSpectrogram) {
        window.BirdBoxSpectrogram.render(root, state.audio, detections);
      }

      var note = detections.length
        ? "Found " + detections.length + " song segment(s) in " + elapsed + " s."
        : "No detections above the confidence threshold (" + elapsed + " s).";
      if (state.audio.truncated) {
        note +=
          " Audio was truncated to the first " + MAX_SECONDS + " seconds for the browser demo.";
      }
      setStatus(root, note, detections.length ? "ok" : undefined);
    } catch (err) {
      console.error(err);
      setStatus(root, err.message || String(err), "error");
      setProgress(root, null);
    } finally {
      detectBtn.disabled = false;
      setTimeout(function () {
        setProgress(root, null);
      }, 400);
    }
  }

  async function onFileChosen(root, file) {
    if (!file) return;
    clearResults(root);
    setStatus(root, "Decoding and resampling audio to 32 kHz mono…");
    try {
      var prepared = await decodeAndResample(file);
      state.audio = prepared;
      state.fileName = file.name;
      var player = $("#onnx-player", root);
      player.src = URL.createObjectURL(file);
      player.classList.add("is-visible");
      player.hidden = false;
      $("#onnx-detect", root).disabled = false;
      var msg =
        "Ready: " +
        file.name +
        " (" +
        prepared.duration.toFixed(1) +
        " s at 32 kHz).";
      if (prepared.truncated) {
        msg +=
          " Original length was " +
          prepared.originalDuration.toFixed(1) +
          " s. Truncated to " +
          MAX_SECONDS +
          " s.";
      }
      setStatus(root, msg, "ok");
    } catch (err) {
      state.audio = null;
      $("#onnx-detect", root).disabled = true;
      setStatus(root, err.message || String(err), "error");
    }
  }

  async function loadSample(root) {
    setStatus(root, "Loading sample audio…");
    var url = resolveAsset("audio/sample_15s.wav");
    var response = await fetch(url);
    if (!response.ok) {
      throw new Error("Could not load sample audio.");
    }
    var blob = await response.blob();
    var file = new File([blob], "sample_15s.wav", { type: "audio/wav" });
    var input = $("#onnx-audio", root);
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (err) {
      /* some browsers block programmatic FileList assignment */
    }
    await onFileChosen(root, file);
  }

  function bind(root) {
    if (root.dataset.bound === "1") return;
    root.dataset.bound = "1";

    var conf = $("#onnx-conf", root);
    var gap = $("#onnx-song-gap", root);
    var confOut = $("#onnx-conf-value", root);
    var gapOut = $("#onnx-song-gap-value", root);

    conf.addEventListener("input", function () {
      confOut.value = Number(conf.value).toFixed(2);
    });
    gap.addEventListener("input", function () {
      gapOut.value = Number(gap.value).toFixed(2);
    });

    $("#onnx-audio", root).addEventListener("change", function (event) {
      var file = event.target.files && event.target.files[0];
      onFileChosen(root, file);
    });

    $("#onnx-detect", root).addEventListener("click", function () {
      runDetection(root);
    });

    $("#onnx-sample", root).addEventListener("click", function () {
      loadSample(root).catch(function (err) {
        setStatus(root, err.message || String(err), "error");
      });
    });

    $("#onnx-download-csv", root).addEventListener("click", function () {
      if (!state.detections.length) return;
      var blob = new Blob([toCsv(state.detections)], { type: "text/csv;charset=utf-8" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (state.fileName || "detections").replace(/\.[^.]+$/, "") + "_songs.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    });

    $("#onnx-model", root).addEventListener("change", function () {
      state.session = null;
      state.modelUrl = null;
      applyCatalogNames(selectedCatalogEntry($("#onnx-model", root).value));
      clearResults(root);
      setStatus(root, "Model selection changed. Run detection to load it.");
    });
  }

  async function init() {
    var root = demoRoot();
    if (!root) return;

    bind(root);
    try {
      await loadManifest(root);
      setStatus(
        root,
        "Select a model, load audio (or the sample), then run detection. Everything stays in your browser."
      );
    } catch (err) {
      setStatus(root, err.message || String(err), "error");
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
