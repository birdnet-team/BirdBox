# Birds-With-Labels Model

The Birds-With-Labels model combines all five zenodo datasets mentioned in the [introduction](overview.md#available-models) with Bird-Sounds-Global, annotated soundscapes from [xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" }.
The five datasets bring almost 407 hours of soundscape recordings with 155,584 bounding box labels of 281 bird species. Bird-Sounds-Global adds 111 hours with 62,030 labels of 765 species from many countries.
The model has 634 classes: the 281 regional species, the 351 Bird-Sounds-Global species with at least 10 training labels in at least 3 recordings, the unknown class `????` for the rest, and `bird` for calls of any species.

It finds calls as [Just-Bird](just-bird.md) does, with one tight box per call, and names each box after its species when it is sure. Otherwise the box is named `bird`. [Species Names](../how-it-works/overview.md#species-names) explains how.

!!! info "Call boxes and a bird class (November 2026)"
    Until November 2026 the model drew one box per species over a whole song, and found far fewer calls than Just-Bird. The current model was trained on call boxes and has a `bird` class that every box trains, whatever its species. Its classes 0 to 632 keep their ids, so the species mapping `Birds-With-Labels` serves both models. BirdBox recognises the new model by its last class, `bird`.

!!! info "Formerly All-In-One"
    Until 28 October 2026 this model was called All-In-One. Files made before then still carry the old name: rename `All-In-One.pt` to `Birds-With-Labels_fp32.pt` and pass `--species-mapping Birds-With-Labels`. BirdBox no longer accepts the old name.

More on Birds-With-Labels:

- [Training](birds-with-labels-training.md): how the call boxes and the bird class were trained, and boxes per species and split.
- [Model Types](birds-with-labels-model-types.md): the export formats and how closely they match the PyTorch model.

All scores on this page come from BirdBox as it ships: the rolling noise floor (each test minute after the audio before it), window fusion with a 0.05 s phrase gap, and names at a species score of 0.25. The thresholds are chosen on the val splits.

---

## Finding Calls

With species ignored, the model finds calls as well as Just-Bird. Every box and every label counts as `bird`, so F1 measures finding calls alone. Both models use Just-Bird's NMS (IoU 0.5) and the merge and threshold that score best on val.

| Test set | Just-Bird | Birds-With-Labels |
| :--- | :---: | :---: |
| Five datasets, IoU 0.25 | 0.404 | 0.412 |
| Five datasets, IoU 0.5 | 0.223 | 0.235 |
| Bird-Sounds-Global, IoU 0.25 | 0.367 | 0.377 |
| Bird-Sounds-Global, IoU 0.5 | 0.195 | 0.197 |
| Hawai'i and Southern Sierra Nevada (labelled per call), IoU 0.5 | 0.558 | 0.576 |
| Hawai'i and Southern Sierra Nevada (labelled per call), IoU 0.75 | 0.297 | 0.291 |

Just-Bird was trained on the test splits of Birds-With-Labels too, so it has seen these recordings. On 14 soundscapes from other places, Just-Bird's boxes (confidence 0.15) serve as the reference. The model draws about as many boxes, covers 76% of Just-Bird's (each by at least half of its duration), and 7% of its boxes lie elsewhere. The previous model covered 31%.

---

## Species

Two kinds of test tell apart how well the model names species and how well its boxes match the labels. Most labels of the five datasets and Bird-Sounds-Global box a whole song, while the model boxes each call.

**Which species are there, and where.** These tests do not depend on the shape of the boxes. Presence asks whether a species is detected in each 3 s segment or recording where it is labelled. Bout-aware matching counts a label as found when boxes of its species lie inside it, and a box as correct when it lies inside a label of its species. Macro F1 over the labelled species of the five datasets and Bird-Sounds-Global together:

| Test | Previous model | Birds-With-Labels |
| :--- | :---: | :---: |
| Presence, 3 s segments | 0.515 | 0.512 |
| Presence, whole recordings | 0.610 | 0.621 |
| Bout-aware, boxes inside a label | 0.440 | 0.477 |
| Bout-aware, either way | 0.477 | 0.486 |

The model names species as well as the previous one, and better within songs.

**Box by box.** Each box has to overlap a label of its species at the given IoU:

| Test set | Previous model | Birds-With-Labels |
| :--- | :---: | :---: |
| Five datasets, IoU 0.25 | 0.524 | 0.389 |
| Bird-Sounds-Global, IoU 0.25 | 0.337 | 0.304 |
| Hawai'i and Southern Sierra Nevada (labelled per call), IoU 0.5 | 0.488 | 0.492 |
| Hawai'i and Southern Sierra Nevada (labelled per call), IoU 0.75 | 0.247 | 0.260 |

On labels per call the model is as good as the previous one, and its boxes are tighter. On song labels it scores lower: one song label meets several call boxes, and at most one of them can match it.

Boxes named `bird` are not counted in these tests. They claim a call, not a species.

---

## Precision, Recall and F1-Score

The graph below shows precision, recall and F1 on the **test splits** against the confidence threshold, each averaged over the species with test labels (macro average), at IoU 0.25.
The upper panel is the test split of the five datasets, the lower one the test split of Bird-Sounds-Global. The confidence is the bird score, so the threshold decides which calls are found. Which of them get a species name is decided by the species score.
Both test splits are labelled mostly per song, so the scores are lower than on call labels (see [Species](#species)). The default threshold is 0.21.

<img src="../../img/metrics/birds_with_labels_F1.png#only-light" width="1000" alt="Macro precision, recall and F1 for every confidence threshold, best F1 0.394 at 0.21 on the regional test split and 0.292 at 0.15 on the Bird-Sounds-Global test split">
<img src="../../img/metrics/birds_with_labels_F1_dark.png#only-dark" width="1000" alt="Macro precision, recall and F1 for every confidence threshold, best F1 0.394 at 0.21 on the regional test split and 0.292 at 0.15 on the Bird-Sounds-Global test split">

!!! info "Average over the labelled species"
    The F-beta analysis (`f_beta_score_analysis.py`) also averages over species that have detections but no test labels, each with an F1 of 0 (`Overall_Macro`). A model with more classes has more of them, so this page averages over the labelled species only: 184 in the regional test split, 357 in Bird-Sounds-Global.

!!! info "Starting at 0.05"
    The test detections were made at a confidence of 0.05, so the curves start there.

---

## Top Performing Species

The figure shows the F1-score against the confidence threshold for the 12 best species among the 82 with at least 50 labels in the regional test split.
Species with fewer labels are left out: with only a few labels, a high F1 is mostly chance. The 12 species reach a best F1 of 0.78 to 0.95.

<img src="../../img/metrics/birds_with_labels_top.png#only-light" width="1000" alt="F1 against the confidence threshold for the 12 best species with at least 50 regional test labels">
<img src="../../img/metrics/birds_with_labels_top_dark.png#only-dark" width="1000" alt="F1 against the confidence threshold for the 12 best species with at least 50 regional test labels">
