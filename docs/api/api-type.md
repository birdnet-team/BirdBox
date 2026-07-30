
BirdBox is a **Python library**. There is no server, no HTTP endpoint, and no REST API. You import the classes directly into your own Python scripts and call them as regular functions.

```python
from inference.detect_birds import BirdCallDetector   # ← local Python import
```

If you are looking for a point-and-click interface or want to run detection without writing code, use the [CLI Reference](../cli/workflows.md) or the [Streamlit demo](../getting-started/demo_streamlit.md) instead.

---

## Module Overview

| Module | Key class / function | What it does |
| :--- | :--- | :--- |
| `inference.detect_birds` | [`BirdCallDetector`](detect-birds.md) | Load a model and run detection on audio files |
| `evaluation.f_beta_score_analysis` | [`FBetaScoreAnalyzer`](f-beta-score-analysis.md) | Sweep confidence thresholds, compute F-beta scores |
| `evaluation.filter_and_merge_detections` | [`DetectionFilter`](filter-and-merge-detections.md) | Filter raw detections by confidence and merge into song segments |
| `evaluation.confusion_matrix_analysis` | [`ConfusionMatrixAnalyzer`](confusion-matrix-analysis.md) | Build a per-species confusion matrix |

---

## Setup

Make sure BirdBox is installed and your working directory is the project root before importing:

```python
import sys
sys.path.insert(0, "src")   # only needed when running outside the installed package

from inference.detect_birds import BirdCallDetector
```

See [Installation](../getting-started/installation.md) for the full setup instructions.
