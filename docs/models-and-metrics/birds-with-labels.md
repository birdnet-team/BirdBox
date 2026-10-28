# Birds With Labels Model

The Birds-With-Labels model combines all five zenodo datasets mentioned in the [introduction](overview.md#available-models).
It therefore utilized almost 407 hours of soundscape recordings with 155,584 bounding box labels spanning across 282 different bird species.

!!! info "Formerly All-In-One"
    Until 28 October 2026 this model was called All-In-One. Files made before then still carry the old name: rename `All-In-One.pt` to `Birds-With-Labels.pt` and pass `--species-mapping Birds-With-Labels`. BirdBox no longer accepts the old name.

More on Birds-With-Labels:

- [Training](birds-with-labels-training.md): training curves and boxes per species and split.
- [Model Types](birds-with-labels-model-types.md): the export formats and how closely they match the PyTorch model.

---

## Precision, Recall and F1-Score

The graph below shows the performance of the model on the **test split** under different confidence thresholds: precision, recall and the F1-score, each averaged over the species (macro average).
To optimize the F1-score, pick a confidence threshold around 0.27.

<img src="../../img/metrics/birds_with_labels_F1.png#only-light" width="1000" alt="Macro precision, recall and F1 on the test split for every confidence threshold, best F1 0.537 at 0.27">
<img src="../../img/metrics/birds_with_labels_F1_dark.png#only-dark" width="1000" alt="Macro precision, recall and F1 on the test split for every confidence threshold, best F1 0.537 at 0.27">

!!! info "Rising Recall at Low Confidence"
    The rising recall from confidence values 0 to ~0.1 is unusual but by design. At very low confidence thresholds, the merging algorithm produces imprecise merged boxes that can accidentally overlap more ground truth labels. See [How it works](../how-it-works/overview.md) for details.

---

## Top Performing Species

The figure shows the F1-score against the confidence threshold for the 12 best species among the 82 with at least 50 labels in the test split.
Species with fewer labels are left out: with only a few labels, a high F1 is mostly chance. Three species reach an F1 of 1.0 on two to four test labels.
At a confidence threshold of 0.4, all 12 species reach an F1 of at least 0.79.

<img src="../../img/metrics/birds_with_labels_top.png#only-light" width="1000" alt="F1 against the confidence threshold for the 12 best species with at least 50 test labels">
<img src="../../img/metrics/birds_with_labels_top_dark.png#only-dark" width="1000" alt="F1 against the confidence threshold for the 12 best species with at least 50 test labels">
