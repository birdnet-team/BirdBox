## YOLO-Models

Trained YOLO-Models for this task can be found on **[TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL)**.
Alternatively, you can train your own model on a custom dataset by using the code available in the **[BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train)** repository (currently private).

To specify the model using the CLI, just pass the relative path of the model as the `--model` command-line argument. 
If you use the code as a package, you can specify the `model` function parameter to match the relative path of the model file. 

If you use the Streamlit WebApp but haven't downloaded a model, a default model can be retrieved directly via the PWA.

**Important:** The species mapping in the `conf.yaml` file the model is trained with and the [`ID_TO_EBIRD_CODES` dictionary in `src/config.py`](src/config.py#L32-L60) have to match.
