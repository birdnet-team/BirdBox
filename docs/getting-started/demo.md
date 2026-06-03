# Demo

BirdBox offers an [interactive demo](https://birdnet.cornell.edu/birdbox/){ target="_blank" rel="noopener noreferrer" } to run model inference on single audio files.
The environment is capable of CUDA, has all dependencies pre-installed and can access the most recent BirdBox YOLO-models.

!!! warning "Demo Limitations"
    The hosted demo runs on an RTX 3080 Ti (12 GB VRAM). Due to this shared hardware, the following hard limits apply:

    - **Max. concurrent users:** 10
    - **Max. file size:** 200 MB
    - **Max. file length:** 10 minutes

    For unrestricted use, run the CLI on your own system (see [CLI Reference](../cli/index.md)) or host the demo yourself.

---

## Self-hosted demo

To host the demo yourself you have to setup the [BirdBox environment](installation.md) beforehand.
If the installation is complete just run:

```bash
streamlit run src/streamlit/app.py
```

---

## Interface

If everything works as intended you should see an interface similar to this one:

![Streamlit UI](../img/streamlit_ui_screenshot.png)

---

## Output Formats

For detailed explanations of the output formats, see [Data In/Out](../data/index.md).
