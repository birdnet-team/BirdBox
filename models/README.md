## YOLO-Models

Trained YOLO-Models for this task can be found on **[TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL)**, which always has the current files.
Each **[GitHub release](https://github.com/birdnet-team/BirdBox/releases)** has the model files it was released with attached, with a `SHA256SUMS.txt` to check them. Use those to work with exactly the weights of a given BirdBox version.
**[Just-Bird](../docs/models-and-metrics/just-bird.md) is the recommended model.** It detects bird vocalizations with a single `bird` class and is the default in the pipeline scripts. For species-level detection, use `Birds-With-Labels_fp32.pt`.
The models are YOLO11 models trained with Ultralytics and are therefore licensed under the [AGPL-3.0](../LICENSE-AGPL-3.0), in every format (`.pt`, `.onnx`, `.tflite`, `.engine`). See [NOTICE.md](../NOTICE.md).
Alternatively, you can train your own model on a custom dataset by using the code available in the **[BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train)** repository (currently only accessible from within the BirdNET-Team).

To specify the model using the CLI, just pass the relative path of the model as the `--model` command-line argument. 
If you use the code as a package, you can specify the `model` function parameter to match the relative path of the model file. 

**Important:** The species mapping in the `conf.yaml` file the model is trained with and the model's `id_to_ebird` dictionary in `SPECIES_MAPPING` in [`src/config.py`](../src/config.py) have to match.
