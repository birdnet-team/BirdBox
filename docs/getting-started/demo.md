# Demo

BirdBox offers an [interactive demo](https://birdnet.cornell.edu/birdbox/){ target="_blank" rel="noopener noreferrer" } to run model inference on single audio files.
The environment is capable of CUDA, has all dependencies pre-installed and can access the most recent BirdBox YOLO-models.

---

## Limitations

The host uses an RTX 3080 Ti with 12 GB of VRAM.
This is sufficient to effectively run inference with the provided YOLO-models.
But due to this hardware limit the interactive demo enforces:

- max. concurrent users: 10
- max. file size: 200 MB
- max. file length: 10 minutes

For any other purpose use the CLI on your own system as described in [CLI Reference](../cli/index.md).
Alternatively, you can also host the interactive demo yourself.

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
