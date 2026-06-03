# Training vs Inference PCEN

Training and inference intentionally differ in clip-boundary policy.

## Training

Training consumes pre-segmented examples. Boundary handling is conservative to avoid cross-chunk leakage:

- clips are not allowed to cross chunk boundaries
- edge clips may be dropped
- this preserves split integrity and avoids hidden context bleed

## Inference

Inference treats each recording as continuous. Internal segmentation is a memory strategy, not a semantic boundary:

- clip extraction targets full temporal coverage
- segment boundaries are bridged logically
- overlap (`3 s` clips, `1.5 s` hop) reduces boundary misses

## Why It Matters

!!! info "Expected and Intentional Difference"
    The distinction between training and inference clip handling is expected and desirable:

    - **Conservative training** supports a robust evaluation setup with clean split integrity.
    - **Dense inference coverage** improves recall on long soundscapes by bridging segment boundaries.

    Implementation references:

    - Training side: BirdBox-Train preprocessing
    - Inference side: `src/inference/utils/pcen_inference.py`
