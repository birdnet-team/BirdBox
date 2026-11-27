# Just-Bird Model

The Just-Bird model has a single class, `bird`. It finds bird vocalizations of any species and draws a tight box around each call. Detections are merged with [window fusion](../how-it-works/overview.md#window-fusion) by default, which also joins same-band boxes less than 0.05 s apart into one phrase box (`--phrase-gap`, `0` keeps one box per call).

The model was trained on 1 November 2026. It sees [log-floor spectrograms](../how-it-works/spectrogram.md#stage-5-log-floor-normalization), and so do the teacher models that made its training labels.

!!! info "Rolling noise floor (November 2026)"
    The model was trained on spectrograms whose noise floor was taken over each 60 s segment. BirdBox now takes it over the 60 s up to each clip's end ([rolling floor](../how-it-works/spectrogram.md#stage-5-log-floor-normalization)), so boxes are final as soon as the audio reaches them. The weights are the same, and the two floors score the same. Change in F1 at IoU 0.5 from the segment floor to the rolling floor, at thresholds 0.13 and 0.15:

    - Released model, Hawai'i validation recordings: −0.001 and −0.003 (both 95% intervals include zero).
    - Recipe trained without Southern Sierra Nevada, Hawai'i validation recordings: −0.002 and +0.000.
    - The same recipe, the unseen 10-minute Southern Sierra Nevada recordings: +0.003 and +0.005.

    The scores below were measured with the segment floor.

More on Just-Bird:

- [Training](just-bird-training.md): the training recipe and why each step is there, training curves and boxes per split.
- [Label Refinement](just-bird-label-refinement.md): how the bout labels of four datasets were split into call labels.
- [Model Types](just-bird-model-types.md): the export formats and how closely they match the PyTorch model.
- [Species from BirdNET](just-bird-species-from-birdnet.md): naming the species of the boxes with BirdNET V3.0, and where its window should sit.

---

## Training Data

The model was trained on six datasets:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }
- Bird-Sounds-Global: annotated soundscapes from [xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } (Lifeplan and SystemShift projects) from many countries

**Labels.** Hawai'i and Southern Sierra Nevada are annotated per call. The other datasets mostly put one box around a whole song or bout. Their boxes were replaced by single-call boxes in two rounds of self-training: a model trained on call-level labels found the calls inside each human box. A clip with a human box it could not replace was discarded, together with all its other boxes. Two further rules refine the labels. A call-sized human box is kept when a teacher box covers at least half of it ([looser rule](just-bird-label-refinement.md#the-looser-rule-for-call-sized-boxes)). Of two nested boxes, only the larger one is kept ([nested boxes](just-bird-label-refinement.md#nested-boxes-in-the-refined-labels)). Both teachers were trained on log-floor spectrograms, like the model itself. [Training](just-bird-training.md#training-recipe) shows why, and [Just-Bird Label Refinement](just-bird-label-refinement.md) how.

**Recipe.** YOLO11s, started from the COCO weights [[18]](#ref-coco), trained for 10 epochs on log-floor spectrograms. The learning rate starts at 0.001 and decays linearly to 3% of it (0.00003) at epoch 10, after a 3-epoch warm-up. Mixup augmentation [[36]](#ref-mixup) (strength 0.3) is switched off for the last 3 epochs. Mosaic augmentation is not used. There is no early stopping.

---

## Performance on an Unseen Site

Because the model was trained on every dataset, none of them is left to test it on recordings from a site it has not seen. The transfer was therefore measured with the same recipe trained without Southern Sierra Nevada and then run on the full Southern Sierra Nevada recordings. This is a test on a domain the model never saw, like the camera-trap test of the WILDS benchmark, where models are evaluated on camera locations that are not in their training data [[14]](#ref-koh). For comparison, the recipes of the two earlier Just-Bird models (October and March 2026) were retrained and tested in the same way.

The score is F1 over boxes on full recordings: a box counts as found when it overlaps a label by at least the given IoU, each label matched once. Boxes are merged with window fusion as BirdBox runs it (nested boxes dropped, phrase gap off). The first three columns use the threshold that is best on these recordings, separately for each column. A user cannot know that threshold in advance, so it is a best case. The last column uses the threshold chosen on Hawai'i val, as a user would.

| Recipe | IoU 0.25, best threshold | IoU 0.5, best threshold | IoU 0.75, best threshold | IoU 0.5, threshold chosen on Hawai'i val |
| :--- | :---: | :---: | :---: | :---: |
| Current (November 2026) | **0.568** | **0.387** | **0.127** | **0.385** |
| October 2026 | 0.541 | 0.384 | 0.116 | 0.383 |
| March 2026 | 0.465 | 0.281 | 0.059 | 0.250 |

Against the October recipe, the current one gains where the fit of a box counts. With each recipe at its Hawai'i-val threshold, it gains +0.009 at IoU 0.75 (95% interval +0.000 to +0.016, paired bootstrap over the 100 recordings) and +0.027 at IoU 0.25. At IoU 0.5 the two tie. A second training run with another seed scores within 0.001 of these numbers.

The gain at IoU 0.75 comes from the new labels, not from the spectrogram. Log-floor with the October labels does not gain (−0.003). The new labels on PCEN with gain 1 gain as much (+0.009). PCEN with gain 1 is a variant of PCEN that, like log-floor, removes the recording level from the image completely.

The graph shows precision, recall and F1 at IoU 0.5 for every confidence threshold. The curves come from the model trained without Southern Sierra Nevada, not from the released weights, which were trained on it. The dashed and dotted lines are the October and March recipes, retrained the same way, not the released models.

<img src="../../img/metrics/just_bird_F1.png#only-light" width="1000" alt="Precision, recall and F1 at IoU 0.5 on the Southern Sierra Nevada hold-out, current recipe against previous recipe">
<img src="../../img/metrics/just_bird_F1_dark.png#only-dark" width="1000" alt="Precision, recall and F1 at IoU 0.5 on the Southern Sierra Nevada hold-out, current recipe against previous recipe">

At IoU 0.75 a box has to fit the call closely. Here the current recipe is ahead of the October one at almost every threshold:

<img src="../../img/metrics/just_bird_F1_iou075.png#only-light" width="1000" alt="F1 at IoU 0.75 on the Southern Sierra Nevada hold-out over the confidence threshold, current recipe against the October and March recipes">
<img src="../../img/metrics/just_bird_F1_iou075_dark.png#only-dark" width="1000" alt="F1 at IoU 0.75 on the Southern Sierra Nevada hold-out over the confidence threshold, current recipe against the October and March recipes">

- **Confidence threshold:** **0.15**, the default in `detect_birds.py` and the [Demo](../demo.md). It was the best threshold of the October model and was kept. For the current recipe, the best threshold on the Hawai'i validation recordings is 0.13 to 0.14. It carries over to the unseen site with little loss (0.385 against the best 0.387). At 0.15 the recipe scores 0.382 there, 0.003 less than at 0.13.

- **Phrase gap:** some Southern Sierra Nevada labels box a whole phrase where the model boxes single syllables. This lowers the scores at IoU 0.5 and above. The default phrase gap of 0.05 s adds little, F1 at IoU 0.5 at the best threshold:
    - Unseen site, recipe trained without Southern Sierra Nevada: 0.387 without the gap, 0.391 with it (October recipe: 0.384 and 0.389).
    - Song-labelled Western-US validation minutes, released model: 0.230 with and without it (October model: 0.171 and 0.172).
    - Call-level labels (Hawai'i val): it costs 0.005 (0.579 and 0.574).

    If your annotations box single calls, pass `--phrase-gap 0`.

- **Western-US validation:** on these song-labelled minutes, which no model trained on, the released model reaches 0.230 at IoU 0.5 with the phrase gap off, against 0.171 for the October model, each at its best threshold. At the defaults it is 0.209 against 0.168. [Phrase Gap](../how-it-works/overview.md#phrase-gap) shows the other gaps.

- **In-domain:** on the full Hawai'i validation recordings, the released model reaches F1 0.570 at IoU 0.5 (threshold 0.14, phrase gap off), against 0.593 for the October model. At IoU 0.75, each at its best threshold, the two tie (0.284 against 0.285). The loss is the price of a spectrogram that does not change with the recording level. PCEN kept part of the level in the image. The October model apparently used it to tell the faint background calls of the Hawai'i recordings, which the annotators mostly left unlabelled, from the louder labelled ones. When the Hawai'i recordings are played 12 dB quieter, a PCEN model makes many more faint false alarms. Most of the extra false alarms are faint sounds 10–20 dB above the noise floor, many of them visible calls without a label. A model trained on PCEN images with gain 1, which also removes the level, loses the same way. So the loss is not specific to log-floor.

---

## References

<a id="ref-koh"></a>
**[14]** Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). "WILDS: A benchmark of in-the-wild distribution shifts." Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:5637–5664.

<a id="ref-coco"></a>
**[18]** Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). "Microsoft COCO: Common objects in context." Computer Vision – ECCV 2014, 740–755.

<a id="ref-mixup"></a>
**[36]** Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond empirical risk minimization." International Conference on Learning Representations (ICLR).
