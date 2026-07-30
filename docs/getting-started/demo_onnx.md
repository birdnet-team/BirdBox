# Demo ONNX

Run BirdBox detection entirely in your browser. The graphs under `docs/models/` bake in PCEN preprocessing, YOLO inference, NMS, and song reconstruction. No Python install and no server-side GPU queue.

!!! warning "Browser limits"
    - First model download is large (about 50 MB for Just-Bird fp32). The file is cached by your browser afterward.
    - Audio is decoded locally, resampled to 32 kHz, and truncated to **60 seconds**.
    - Requires a recent browser with WebAssembly. Safari can be slower than Chromium or Firefox.
    - This showcase is for quick checks. Use the [Streamlit demo](demo_streamlit.md) or the [CLI](../cli/detect-birds.md) for long files and batch jobs.

---

## Interactive Demo

<div class="onnx-demo" id="onnx-demo" markdown="0">
  <div class="onnx-demo__grid">
    <div>
      <label for="onnx-model">Select Model</label>
      <select id="onnx-model" aria-label="Select Model"></select>
      <p class="onnx-demo__hint">Models are loaded from <code>docs/models/</code> via <code>manifest.json</code>.</p>
    </div>
    <div>
      <label for="onnx-audio">Audio file</label>
      <input id="onnx-audio" type="file" accept="audio/*,.wav,.flac,.ogg,.mp3" />
      <p class="onnx-demo__hint">WAV or FLAC recommended. Minimum 3 seconds. Maximum 60 seconds in this demo.</p>
    </div>
    <div>
      <label for="onnx-conf">Confidence Threshold</label>
      <div class="onnx-demo__slider-row">
        <input id="onnx-conf" type="range" min="0.01" max="0.80" step="0.01" value="0.18" />
        <output id="onnx-conf-value" for="onnx-conf">0.18</output>
      </div>
    </div>
    <div>
      <label for="onnx-song-gap">Song Gap Threshold (seconds)</label>
      <div class="onnx-demo__slider-row">
        <input id="onnx-song-gap" type="range" min="0" max="2" step="0.01" value="0.1" />
        <output id="onnx-song-gap-value" for="onnx-song-gap">0.10</output>
      </div>
    </div>
    <div class="onnx-demo__full onnx-demo__actions">
      <button type="button" class="md-button md-button--primary" id="onnx-detect" disabled>Detect Bird Vocalizations</button>
      <button type="button" class="md-button" id="onnx-sample">Load 15 s sample</button>
      <button type="button" class="md-button" id="onnx-download-csv" disabled>Download CSV</button>
    </div>
  </div>

  <div class="onnx-demo__progress" aria-hidden="true"><span></span></div>
  <p class="onnx-demo__status" role="status">Loading demo…</p>
  <audio id="onnx-player" class="onnx-demo__player" controls hidden></audio>

  <div class="onnx-demo__spec" id="onnx-spec" hidden>
    <div class="onnx-demo__spec-title">PCEN Spectrogram with Detections</div>
    <p class="onnx-demo__spec-meta" id="onnx-spec-meta"></p>
    <div class="onnx-demo__spec-scroll" id="onnx-spec-scroll">
      <canvas id="onnx-spec-canvas" aria-label="PCEN spectrogram with detection boxes"></canvas>
    </div>
  </div>

  <div class="onnx-demo__metrics" aria-live="polite">
    <div class="onnx-demo__metric">
      <span class="onnx-demo__metric-label">Songs</span>
      <span class="onnx-demo__metric-value" id="onnx-metric-count">0</span>
    </div>
    <div class="onnx-demo__metric">
      <span class="onnx-demo__metric-label">Species</span>
      <span class="onnx-demo__metric-value" id="onnx-metric-species">0</span>
    </div>
    <div class="onnx-demo__metric">
      <span class="onnx-demo__metric-label">Avg Confidence</span>
      <span class="onnx-demo__metric-value" id="onnx-metric-conf">—</span>
    </div>
  </div>

  <div class="onnx-demo__results">
    <table>
      <thead>
        <tr>
          <th>Start (s)</th>
          <th>End (s)</th>
          <th>Low (Hz)</th>
          <th>High (Hz)</th>
          <th>Avg Conf</th>
          <th>Max Conf</th>
          <th>Merged</th>
          <th>Species</th>
        </tr>
      </thead>
      <tbody id="onnx-results-body"></tbody>
    </table>
  </div>
</div>

---

## Adding models

1. Export for the docs demo.

    === "Linux / macOS"
        ```bash
        python src/inference/onnx_export.py \
            --output-path docs/models/Just-Bird_fp32.onnx \
        ```

    === "Windows (PowerShell)"
        ```powershell
        python src/inference/onnx_export.py `
            --output-path docs/models/Just-Bird_fp32.onnx `
        ```

    === "Windows (CMD)"
        ```cmd
        python src/inference/onnx_export.py ^
            --output-path docs/models/Just-Bird_fp32.onnx ^
        ```

    `conf` and `song_gap` are required graph inputs. The demo sliders feed them on every run. Suggested defaults are stored in ONNX metadata (`default_conf`, `default_song_gap`).

2. Place the `.onnx` file under `docs/models/` if the export path was elsewhere.

3. Add an entry to `docs/models/manifest.json`.

    ```json
    {
      "models": [
        {
          "file": "Just-Bird_fp32.onnx",
          "label": "Just-Bird_fp32.onnx",
          "names": { "0": "bird" }
        },
        {
          "file": "Hawaii_fp32.onnx",
          "label": "Hawaii_fp32.onnx",
          "names": { "0": "hawama", "1": "ercfra" }
        }
      ]
    }
    ```

The dropdown reads that manifest on page load. Put class labels in the optional `names` map (ONNX Runtime Web does not expose ONNX metadata props).

---

## Output columns

| Column | Meaning |
| :--- | :--- |
| Start / End | Song bounds in seconds from the start of the (possibly truncated) audio |
| Low / High | Frequency bounds in Hz |
| Avg Conf / Max Conf | Mean and peak confidence over merged clip detections |
| Merged | Number of clip-level boxes folded into the song |
| Species | Label from model metadata (`names`) |

For the full set of BirdBox file formats outside the browser, see [Detection Output Formats](../data/outputs.md).
