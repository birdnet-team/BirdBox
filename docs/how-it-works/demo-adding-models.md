# Adding Models to the Demo

The model menu lists the models in `docs/models/manifest.json`. The first one is the default. The [instant sliders](demo-detection.md#instant-sliders) need the graph's per-window boxes (`raw_all`) and a JavaScript port of its merge: `docs/javascripts/demo/fusion.js` for window fusion (`fusion`, the export's default for both models) and `docs/javascripts/demo/songs.js` for song reconstruction (`reconstruct`, for a model exported with `--merge reconstruct`). The Just-Bird frequency gate (`--merge reconstruct` with Just-Bird) has no port. The demo skips such entries.

1. Export the model.

    === "Linux / macOS"
        ```bash
        python src/inference/onnx_export.py \
            --output-path docs/models/Just-Bird_fp16.onnx \
            --precision fp16
        ```

    === "Windows (PowerShell)"
        ```powershell
        python src/inference/onnx_export.py `
            --output-path docs/models/Just-Bird_fp16.onnx `
            --precision fp16
        ```

    === "Windows (CMD)"
        ```cmd
        python src/inference/onnx_export.py ^
            --output-path docs/models/Just-Bird_fp16.onnx ^
            --precision fp16
        ```

    For another model, add `--pt-model`, for example `--pt-model models/Birds-With-Labels_fp32.pt --output-path docs/models/Birds-With-Labels_fp16.onnx`.

    `--precision fp16` stores the network weights in fp16, which halves the file (20 MB instead of 38 MB for Just-Bird, 51 MB instead of about 100 MB for Birds-With-Labels). Feature extraction and merging stay in fp32. On 13 test recordings (`tests/onnx_graph_parity.py`), the fp16 graphs reproduced the boxes of `detect_birds.py` with the `.pt` model: 99.8% for Just-Bird (mean IoU 0.995) and 99.6% for Birds-With-Labels, with the same species (mean IoU 0.997). The export prints the merge it baked in (`Merge (baked in)`, set with `--merge`) and the suggested defaults (`Default conf` and `Default song_gap`, set with `--conf` and `--song-gap`). It also stores them in the ONNX metadata. With window fusion, `Default song_gap` is the phrase gap.

2. Place the `.onnx` file under `docs/models/` if the export path was elsewhere.

3. Add an entry to `docs/models/manifest.json`, first in the list if the demo should start with it.

    ```json
    {
      "models": [
        {
          "file": "Just-Bird_fp16.onnx",
          "label": "Just-Bird_fp16.onnx (recommended)",
          "merge_method": "fusion",
          "conf": 0.15,
          "phrase_gap": 0.05,
          "species_mapping": "Just-Bird",
          "names": { "0": "bird" }
        },
        {
          "file": "Birds-With-Labels_fp16.onnx",
          "label": "Birds-With-Labels_fp16.onnx",
          "merge_method": "fusion",
          "conf": 0.21,
          "phrase_gap": 0.05,
          "species_mapping": "Birds-With-Labels",
          "bird_class_id": 633,
          "label_conf": 0.25,
          "names": { "0": "????", "1": "amabaw1", "…": "…", "633": "bird" },
          "common_names": { "0": "Unknown", "1": "Amazonian Barred-Woodcreeper", "…": "…", "633": "Bird (species unsure)" }
        }
      ]
    }
    ```

    | Field | Meaning |
    | :--- | :--- |
    | `file` | The ONNX file, relative to `docs/models/` |
    | `label` | Name in the model menu |
    | `merge_method` | `fusion` or `reconstruct`, as printed by the export. Other entries are skipped. |
    | `conf` | Where the confidence slider starts: usually the printed `Default conf` (0.15 for Just-Bird, 0.21 for Birds-With-Labels, 0.2 for other models). |
    | `phrase_gap` | `fusion` only: where the phrase gap slider starts, the printed `Default song_gap`, which is the phrase gap here (0.05 for both models) |
    | `song_gap` | `reconstruct` only: where the song gap slider starts, the printed `Default song_gap` (0.1) |
    | `species_mapping` | Written to `model_config` of the exported JSON |
    | `bird_class_id` | Species models with a `bird` class only: the index of that class, the last one (633 for Birds-With-Labels). It turns on species naming. The merge then ignores species, and each merged box is named after the best species inside it. |
    | `label_conf` | Species models with a `bird` class only: the score the best species needs to name a box. Below it, the box is named `bird`. 0.25 if left out. |
    | `names` | eBird codes by class index, for the table and the exports (ONNX Runtime Web does not expose ONNX metadata) |
    | `common_names` | Optional: common names by class index, shown in the table instead of the codes |

    For a species model, `names` and `common_names` come from `id_to_ebird` and `ebird_to_name` of its mapping in `src/config.py`. With more than one class, the table gets a species column.

4. Check that the demo's merge still matches the graph. The tests need [Node.js](https://nodejs.org){ target="_blank" rel="noopener noreferrer" } 22, the version the CI uses.

    ```bash
    cd tests/js && npm install
    node demo_merge_parity.mjs ../../docs/models/Just-Bird_fp16.onnx
    ```

    The script compares `fusion.js` or `songs.js`, whichever the manifest entry's `merge_method` names, with the graph's own detections for 48 slider settings (32 for song reconstruction) on `tests/test.wav` and fails on any difference. The network runs once. Each setting then runs only the graph's merge (`mergeGraph` in `onnx-graph.js`), so the check takes one to three minutes. Run it after exporting a new model or changing a merge in `src/inference/onnx_export.py`.

    The other demo tests in `tests/js` check the chunked and streamed runs, the boxes while recording and which boxes are dashed. They run in parallel processes, each on one ONNX Runtime thread like the demo, so the rows are the browser's:

    - **`npm test`:** every test on the manifest's first model, the demo's default. On the other models it runs the tests of their merge and one chunked run, because the models share all other code. About 4 minutes on a 20-core machine.
    - **`npm run test:all`:** every test on every model, for a new or newly exported model. About 10 minutes.
    - **`npm run test:serial`:** the tests one after another.

    The memory a machine has free decides how many processes run at once, at about 2 GB each, and `--jobs=N` sets that number. The `demo tests` workflow runs the tests on pull requests that change the demo, its models or these tests, and every test on every model when `docs/models` changes.

5. Preview the demo. Install the docs tools with `pip install -r requirements-docs.txt`, run `mkdocs serve` from the repository root and pick the new model in the menu.
