# Just Bird Model

The Just Bird model has a single class, `bird`. It finds bird vocalizations of any species and draws a tight box around each call. Detections are merged with [window fusion](../cli/detect-birds.md#window-fusion-just-bird-default) by default, which also joins same-band boxes less than 0.05 s apart into one phrase box (`--phrase-gap`, `0` keeps one box per call).

The current model was trained on 1 November 2026 and replaces the model of 20 October 2026. It is the first Just-Bird model that sees [log-floor spectrograms](../how-it-works/spectrogram.md#stage-5-log-floor-normalization) instead of PCEN, and its labels were rebuilt with two changes to the label refinement.

More on Just-Bird:

- [Training](just-bird-training.md): the training recipe and why each step is there, training curves and boxes per split.
- [Label Refinement](just-bird-label-refinement.md): how the bout labels of four datasets were split into call labels.
- [Model Types](just-bird-model-types.md): the export formats and how closely they match the PyTorch model.

---

## Training Data

The model was trained on all five regional datasets and on Bird-Sounds-Global:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }
- Bird-Sounds-Global: annotated soundscapes from [xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } (Lifeplan and SystemShift projects) from many countries

**Labels.** Hawai'i and Southern Sierra Nevada are annotated per call. The other datasets mostly put one box around a whole song or bout. Their boxes were replaced by single-call boxes in two rounds of self-training: a model trained on call-level labels found the calls inside each human box. A clip with a human box it could not replace was discarded, together with all its other boxes. Two rules are new in this model: a call-sized human box is kept when a teacher box covers at least half of it ([looser rule](just-bird-label-refinement.md#testing-a-looser-rule-for-call-sized-boxes)), and of two nested boxes only the larger one is kept ([nested boxes](just-bird-label-refinement.md#nested-boxes-in-the-refined-labels)). Both teachers were trained on log-floor spectrograms, like the model itself. [Training](just-bird-training.md#training-recipe) shows why, and [Just-Bird Label Refinement](just-bird-label-refinement.md) how.

**Recipe.** YOLO11s, started from the COCO weights [[14]](#ref-coco), trained for 10 epochs on log-floor spectrograms. The learning rate decays to its final value within those 10 epochs, and mixup augmentation [[31]](#ref-mixup) is switched off for the last 3 (mosaic augmentation is not used). There is no early stopping.

---

## Performance on an Unseen Site

Because the model was trained on every dataset, none of them is left to test it on recordings from a site it has not seen. The transfer was therefore measured with the same recipe trained without Southern Sierra Nevada and then run on the full Southern Sierra Nevada recordings. This is a test on a domain the model never saw, like the camera-trap test of the WILDS benchmark, where models are evaluated on camera locations that are not in their training data [[11]](#ref-koh). The recipes of the two previous models were retrained and tested in the same way.

Event F1 on full recordings with window fusion (including the removal of nested boxes, phrase gap off):

| Recipe | IoU 0.25 | IoU 0.5 | IoU 0.75 | IoU 0.5 at the threshold chosen on Hawai'i val |
| :--- | :---: | :---: | :---: | :---: |
| Current (November 2026) | **0.568** | **0.387** | **0.127** | **0.384** |
| October 2026 | 0.540 | 0.385 | 0.116 | **0.384** |
| March 2026 | 0.463 | 0.278 | 0.059 | 0.249 |

Against the October recipe, the current one gains where the fit of a box counts: +0.009 at IoU 0.75 (95% interval +0.001 to +0.017, paired bootstrap over the 100 recordings) and +0.028 at IoU 0.25. At IoU 0.5 the two tie. A second training run with another seed scores within 0.001 of these numbers. The gain at IoU 0.75 comes from the new labels, not from the spectrogram: log-floor with the October labels does not gain (−0.003), and the new labels on PCEN with the level removed gain as much (+0.012).

The graph shows precision, recall and F1 at IoU 0.5 for every confidence threshold. The curves come from the model trained without Southern Sierra Nevada, not from the released weights, which were trained on it. The dashed and dotted lines are the October and March recipes, retrained the same way, not the released models.

<img src="../../img/metrics/just_bird_F1.png#only-light" width="1000" alt="Precision, recall and F1 at IoU 0.5 on the Southern Sierra Nevada hold-out, current recipe against previous recipe">
<img src="../../img/metrics/just_bird_F1_dark.png#only-dark" width="1000" alt="Precision, recall and F1 at IoU 0.5 on the Southern Sierra Nevada hold-out, current recipe against previous recipe">

At IoU 0.75 a box has to fit the call closely. Here the current recipe is ahead of the October one at almost every threshold:

<img src="../../img/metrics/just_bird_F1_iou075.png#only-light" width="1000" alt="F1 at IoU 0.75 on the Southern Sierra Nevada hold-out over the confidence threshold, current recipe against the October and March recipes">
<img src="../../img/metrics/just_bird_F1_iou075_dark.png#only-dark" width="1000" alt="F1 at IoU 0.75 on the Southern Sierra Nevada hold-out over the confidence threshold, current recipe against the October and March recipes">

- **Confidence threshold:** **0.15**, the default in `detect_birds.py` and the [Demo](../demo.md). The best threshold on the Hawai'i validation recordings is 0.13 to 0.14, and it carries over to the unseen site with little loss (0.384 against the best 0.387).
- **Phrase gap:** some Southern Sierra Nevada labels box a whole phrase where the model boxes single syllables, which lowers the scores at IoU 0.5 and above. The current model already draws phrase boxes more often than the October one, so the default phrase gap of 0.05 s adds little: F1 at IoU 0.5 from 0.387 to 0.390 there, and on song-labelled Western-US validation minutes from 0.230 to 0.234 (October model: 0.385 to 0.396 and 0.170 to 0.197). On call-level labels it costs about 0.06 (Hawai'i val: 0.576 to 0.518). If your annotations box single calls, pass `--phrase-gap 0`.
- **Western-US validation:** on these song-labelled minutes, which no model trained on, the released model reaches 0.230 at IoU 0.5 with the phrase gap off, against 0.170 for the October model (each at its best threshold; 0.207 against 0.197 at the defaults). [Phrase boxes](../cli/detect-birds.md#window-fusion-just-bird-default) has the numbers for other gaps.
- **In-domain:** on the full Hawai'i validation recordings, the released model reaches F1 0.568 at IoU 0.5 (threshold 0.14, phrase gap off), against 0.590 for the October model. At IoU 0.75, each at its best threshold, the two tie (0.282 against 0.284). The loss is the price of a spectrogram that does not change with the recording level: PCEN kept part of the level in the image, and the October model apparently used it to tell the faint background calls of the Hawai'i recordings, which the annotators mostly left unlabelled, from the louder labelled ones (played 12 dB quieter, a PCEN model makes many more faint false alarms). Most of the extra false alarms are faint sounds 10–20 dB above the noise floor, many of them visible calls without a label. A model trained on PCEN images with the level removed loses the same way, so it is not specific to log-floor.

---

## References

<a id="ref-koh"></a>
**[11]** Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). "WILDS: A benchmark of in-the-wild distribution shifts." Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:5637–5664.

<a id="ref-coco"></a>
**[14]** Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). "Microsoft COCO: Common objects in context." Computer Vision – ECCV 2014, 740–755.

<a id="ref-mixup"></a>
**[31]** Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond empirical risk minimization." International Conference on Learning Representations (ICLR).
