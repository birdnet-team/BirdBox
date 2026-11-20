# Adding Models to the Demo

The model menu lists the models in `docs/models/manifest.json`; the first one is the default. The [instant sliders](demo-detection.md#instant-sliders) need the graph's per-window boxes (`raw_all`) and a JavaScript port of its merge: `docs/javascripts/demo/fusion.js` for window fusion (`fusion`, the export's default for Just-Bird) and `docs/javascripts/demo/songs.js` for song reconstruction (`reconstruct`, the default for Birds-With-Labels). The Just-Bird frequency gate (`--merge reconstruct` with Just-Bird) has no port; such entries are skipped.

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

    `--precision fp16` stores the network weights in fp16, which halves the file (20 MB instead of 38 MB for Just-Bird, 52 MB instead of about 100 MB for Birds-With-Labels). Feature extraction and merging stay in fp32. On 13 test recordings (`tests/onnx_graph_parity.py`), the fp16 graphs reproduced the boxes of `detect_birds.py` with the `.pt` model: 99.8% for Just-Bird (mean IoU 0.995) and 99.6% for Birds-With-Labels, with the same species (mean IoU 0.997). The export prints the merge it baked in (`Merge (baked in)`) and the suggested defaults (`Default conf`, `Default song_gap`), which it also stores in the ONNX metadata.

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
          "merge_method": "reconstruct",
          "conf": 0.25,
          "song_gap": 0.1,
          "species_mapping": "Birds-With-Labels",
          "names": { "0": "????", "1": "amabaw1", "…": "…" },
          "common_names": { "0": "Unknown", "1": "Amazonian Barred-Woodcreeper", "…": "…" }
        }
      ]
    }
    ```

    | Field | Meaning |
    | :--- | :--- |
    | `file` | The ONNX file, relative to `docs/models/` |
    | `label` | Name in the model menu |
    | `merge_method` | `fusion` or `reconstruct`, as printed by the export. Other entries are skipped. |
    | `conf` | Where the confidence slider starts: usually the printed `Default conf` (0.15 for Just-Bird, 0.2 for the other models). Birds-With-Labels starts at 0.25. |
    | `phrase_gap` | `fusion` only: where the phrase gap slider starts, the printed `Default song_gap` (0.05 for Just-Bird, 0 for Birds-With-Labels) |
    | `song_gap` | `reconstruct` only: where the song gap slider starts, the printed `Default song_gap` (0.1) |
    | `species_mapping` | Written to `model_config` of the exported JSON |
    | `names` | eBird codes by class index, for the table and the exports (ONNX Runtime Web does not expose ONNX metadata) |
    | `common_names` | Optional: common names by class index, shown in the table instead of the codes |

    For a species model, `names` and `common_names` come from `id_to_ebird` and `ebird_to_name` of its mapping in `src/config.py`. With more than one class, the table gets a species column.

4. Check that the demo's merge still matches the graph:

    ```bash
    cd tests/js && npm install
    node demo_merge_parity.mjs ../../docs/models/Just-Bird_fp16.onnx
    ```

    The script compares `fusion.js` or `songs.js`, whichever the manifest entry's `merge_method` names, with the graph's own detections for 48 slider settings (32 for song reconstruction) on `tests/test.wav` and fails on any difference. The network runs once; each setting then runs only the graph's merge (`mergeGraph` in `onnx-graph.js`), so the check takes one to three minutes. Run it after exporting a new model or changing a merge in `src/inference/onnx_export.py`.

    The other demo tests in `tests/js` check the chunked and streamed runs, the boxes while recording and which boxes are dashed. They run in parallel processes, each on one ONNX Runtime thread like the demo, so the rows are the browser's:

    - **`npm test`:** every test on the manifest's first model, the demo's default. On the other models it runs the tests of their merge and one chunked run, because the models share all other code. About 4 minutes on a 20-core machine.
    - **`npm run test:all`:** every test on every model, for a new or newly exported model. About 10 minutes.
    - **`npm run test:serial`:** the tests one after another.

    The memory a machine has free decides how many processes run at once, at about 2 GB each, and `--jobs=N` sets that number. The `demo tests` workflow runs the tests on pull requests that change the demo, its models or these tests, and every test on every model when `docs/models` changes.
