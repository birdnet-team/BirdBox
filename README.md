<div align="center">
  <h1>BirdBox</h1>
  <img src="img/logo_birdbox.png" width="300" alt="BirdBox-Logo" />
  
  <p><strong>Deep Learning Bird Call Detection & Evaluation System</strong></p>
  
  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
  
</div>

BirdBox is a comprehensive system for detecting and evaluating bird calls in audio recordings using deep learning. It leverages YOLO (You Only Look Once) object detection on spectrogram images to identify and localize bird vocalizations in time and frequency.

⚠️ **Note**: This project is still in active development. Performance may vary.

## Key Features

**Multiple Audio Formats** - Supports WAV, FLAC, OGG, MP3 (WAV/FLAC recommended for best results)  
**Arbitrary-Length Audio Processing** - Handle audio from seconds to hours  
**Song Reconstruction** - Automatically merge temporally adjacent detections into continuous bird songs  
**Batch Processing** - Process entire directories of audio files  
**PCEN Normalization** - Per-Channel Energy Normalization for robust spectral features  
**Comprehensive Evaluation** - F-beta analysis, confusion matrices, optimal threshold finding  
**Multiple Output Formats** - JSON, CSV (compatible with annotation formats)  
**Model Agnostic** - Works with `.pt`, `.onnx`, `.engine` model formats

## YOLO-Models

Trained YOLO-Models for this task can be found on the **[TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL)**.
Alternatively, you can train your own model on a custom dataset by using the code available in the **[BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train)** repository (currently only available for the BirdNET Team).

To specify the model using the CLI, just pass the relative path of the model as the `--model` command-line argument. 
If you use the code as a package, you can specify the `model` function parameter to match the relative path of the model file.

**Important:** The species mapping in the `conf.yaml` file the model is trained with and the `DATASETS[model_name]` dictionary in [`src/config.py`](src/config.py#L17) have to match.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/birdnet-team/BirdBox.git
cd BirdBox

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage, i. e. run detection on single audio files

#### Option 1: Web Interface (Streamlit App)

The easiest way to use BirdBox is through the interactive web interface:

```bash
streamlit run src/streamlit/app.py
```

Then open your browser to `http://localhost:8501` and:
- Upload audio files (WAV, FLAC, OGG, MP3)
- Select a model from the dropdown
- Adjust detection parameters with sliders
- Click "Detect Bird Calls"
- View PCEN spectrograms with bounding boxes
- Download results as JSON or CSV

If done correctly, the Streamlit Web Interface will look like this:

![Streamlit app screenshot](img/streamlit_ui_screenshot.png)

#### Option 2: Command Line Interface

```bash
# Detect birds in a single audio file (supports WAV, FLAC, OGG, MP3)
python src/inference/detect_birds.py \
    --audio path/to/recording.wav \
    --model models/best.pt \
    --species-mapping species_mapping

# Or process entire directory (batch processing)
python src/inference/detect_birds.py \
    --audio path/to/audio/folder \
    --model models/best.pt \
    --species-mapping species_mapping
```

## Typical Workflow

### Complete Detection & Evaluation Pipeline

```bash
# Step 1: Run inference with low confidence and --no-merge to get raw detections
python src/inference/detect_birds.py \
    --audio path/to/audio/folder \
    --model models/model_name.pt \
    --species-mapping mapping_name \
    --output-path results/raw_detections \
    --output-format json \
    --conf 0.001 \
    --no-merge

# Step 2: Analyze F-beta scores to find optimal threshold
python src/evaluation/f_beta_score_analysis.py \
    --detections results/raw_detections.json \
    --labels path/to/labels.csv \
    --output-path results/f_beta_analysis \
    --beta 2.0 \
    --iou-threshold 0.25 \
    --song-gap 0.1

# Step 3: Filter raw detections to optimal threshold and merge
python src/evaluation/filter_and_merge_detections.py \
    --input results/raw_detections.json \
    --output-path results/filtered_detections \
    --output-format json \
    --conf 0.2 \
    --song-gap 0.1

# Step 4: Generate confusion matrix
python src/evaluation/confusion_matrix_analysis.py \
    --detections results/filtered_detections.csv \
    --labels path/to/labels.csv \
    --output-path results/confusion_matrix

# Step 5: Examine results in results/ directory
```

Feel free to adapt **[\`run_pipeline.sh\`](run_pipeline.sh)** to your use case. This script serves as an example for a typical workflow.

## Performance Optimization

### For Detection
- Use GPU acceleration (automatically detected)
- Adjust song gap threshold based on species vocalization patterns
- Adjust ìou threshold to fit the specific use-case

### For Evaluation
- Tune the β-Parameter for the Fβ-Analysis to fit the specific use-case
- β < 1 leads to more weight on precision
- β > 1 leads to more weight on recall

## Troubleshooting

### Common Issues

**"No detections found"**
- Lower confidence threshold (e.g. `--conf 0.1`)
- Check if audio file is in a supported format (WAV, FLAC, OGG, MP3)
- Verify model is trained on similar species
- If using MP3/OGG, try with WAV/FLAC version of same recording

**"Poor detection performance"**
- Use lossless formats (WAV/FLAC) instead of lossy (MP3/OGG)
- Model was trained on WAV files - lossy compression can affect accuracy
- Ensure MP3/OGG files use high bitrate (≥256 kbps) if you must use them

**"Out of memory errors"**
- Process shorter audio files
- Reduce PCEN segment length in config

**"No matching files in evaluation"**
- Verify ground truth CSV has correct column names
- Ensure audio filenames match between detections and labels


## Citation

Feel free to use BirdBox for your acoustic analyses and research. If you do, please cite as:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)

