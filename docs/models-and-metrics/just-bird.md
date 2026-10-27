# Just Bird Model

The Just Bird model has a single class, `bird`. It finds bird vocalizations of any species and draws a tight box around each call. Detections are merged with [window fusion](../cli/detect-birds.md#window-fusion-just-bird-default) by default, which also joins same-band boxes less than 0.05 s apart into one phrase box (`--phrase-gap`, `0` keeps one box per call).

The current model was released on 20 October 2026 and replaces the March 2026 model.

---

## Training Data

The model was trained on all five regional datasets and on Bird-Sounds-Global:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }
- Bird-Sounds-Global: annotated soundscapes from [xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } (Lifeplan and SystemShift projects) from many countries

**Labels.** Hawai'i and Southern Sierra Nevada are annotated per call. The other datasets mostly put one box around a whole song or bout. Their boxes were replaced by single-call boxes in two rounds of self-training: a model trained on call-level labels found the calls inside each human box. A clip with a human box it could not replace was discarded, together with all its other boxes. [Training Recipe](#training-recipe) shows how this was done and why.

**Recipe.** YOLO11s, started from the COCO weights, trained for 10 epochs on PCEN spectrograms. The learning rate decays to its final value within those 10 epochs, and mixup augmentation is switched off for the last 3 (mosaic augmentation is not used). There is no early stopping.

---

## Performance on an Unseen Site

Because the model was trained on every dataset, none of them is left to test it on recordings from a site it has not seen. The transfer was therefore measured with the same recipe trained without Southern Sierra Nevada and then run on the full Southern Sierra Nevada recordings, a zero-shot test in the sense of [[2]](#ref-bansal). The previous model's recipe was retrained and tested in the same way.

Event F1 on full recordings with window fusion (including the removal of nested boxes, phrase gap off):

| Recipe | IoU 0.25 | IoU 0.5 | IoU 0.75 | IoU 0.5 at the threshold chosen on Hawai'i val |
| :--- | :---: | :---: | :---: | :---: |
| Current | **0.540** | **0.385** | **0.116** | **0.384** |
| Previous (March 2026) | 0.463 | 0.278 | 0.059 | 0.249 |

The graph shows precision, recall and F1 at IoU 0.5 for every confidence threshold. The curves come from the model trained without Southern Sierra Nevada, not from the released weights, which were trained on it. The dashed line is the previous recipe retrained the same way, not the March 2026 model.

<img src="../../img/metrics/just_bird_F1.png" width="1000" alt="Precision, recall and F1 at IoU 0.5 on the Southern Sierra Nevada hold-out, current recipe against previous recipe">

- **Confidence threshold:** **0.15**, the default in `detect_birds.py` and the [Demo](../demo.md). The best threshold on the Hawai'i validation recordings is 0.13 to 0.15, and it carries over to the unseen site with almost no loss (0.384 against the best 0.385).
- **Phrase gap:** some Southern Sierra Nevada labels box a whole phrase where the model boxes single syllables, which lowers the scores at IoU 0.5 and above. The default phrase gap of 0.05 s raises F1 at IoU 0.5 there from 0.385 to 0.396, and on song-labelled Western-US validation minutes from 0.170 to 0.197. On call-level labels it costs about 0.05 (Hawai'i val: 0.581 to 0.531). If your annotations box single calls, pass `--phrase-gap 0`. [Phrase boxes](../cli/detect-birds.md#window-fusion-just-bird-default) has the numbers for other gaps.
- **In-domain:** on the full Hawai'i validation recordings, the released model reaches F1 0.590 at IoU 0.5 (threshold 0.14, phrase gap off).

---

## Training Recipe

The recipe was worked out on the same hold-out: every model in this section was trained without Southern Sierra Nevada and tested on its full recordings, as in [Performance on an Unseen Site](#performance-on-an-unseen-site). The released model repeats the last step with Southern Sierra Nevada added.

```mermaid
flowchart LR
  hawaii["Call labels<br/>Hawai'i"]
  ssn["Call labels<br/>Southern Sierra Nevada"]
  bouts["Bout labels<br/>Northeastern US, Western US,<br/>Amazon Basin, Bird-Sounds-Global"]
  t1["Teacher 1<br/>trained on Hawai'i only"]
  r1["Round-1 labels"]
  t2["Teacher 2<br/>trained on Hawai'i + round-1 labels"]
  r2["Round-2 labels"]
  final["Just Bird<br/>10 epochs from COCO weights"]
  hawaii --> t1 --> r1
  bouts --> r1 --> t2 --> r2
  bouts --> r2
  hawaii --> t2
  hawaii --> final
  ssn --> final
  r2 --> final
```

### Splitting Bouts into Calls

Trained on the human labels of all datasets, the model learns the bout style and draws boxes of several seconds on a new site. It then scores lower there than a model trained on Hawai'i alone (see the first two bars [below](#each-step-on-the-unseen-site)). The bout labels are therefore replaced by call labels before the final training. A teacher model looks for calls inside each human box, and the call boxes it finds replace the box:

- **Round 1** uses a teacher trained on Hawai'i only, which never saw a bout box.
- **Round 2** uses a teacher trained on all sets with the round-1 labels, and replaces the original human boxes again.

<img src="../../img/metrics/just_bird_label_example.png" width="1000" alt="A Northeastern US clip with one human bout box, two call boxes from round 1 and five call boxes from round 2">

[How the Labels Are Refined](#how-the-labels-are-refined) explains the rule, what each round changes, and which boxes and clips are discarded.

### Each Step on the Unseen Site

<img src="../../img/metrics/just_bird_recipe_steps.png" width="1000" alt="F1 at IoU 0.5 on the full Southern Sierra Nevada recordings for each step of the recipe">

- **More data needs better labels.** The bout-labelled datasets with their human labels lower F1 from 0.280 to 0.122. With the teacher's call labels the same data raises it to 0.380–0.384.
- **Round 2** ties round 1 on full recordings. It is kept because it discards less data, scores higher on 3 s clips (F1 0.368 against 0.343 at IoU 0.5, both trained on a quarter of the data), and its threshold transfers: the one chosen on Hawai'i val loses almost nothing on the new site (0.384 against the best 0.385; round 1: 0.380 against 0.393).
- **The previous recipe** trained on the human labels for up to 100 epochs, with early stopping and augmentation from a hyperparameter search.

### Ten Epochs

<img src="../../img/metrics/just_bird_epochs.png" width="1000" alt="Best F1 at IoU 0.5 per epoch on the unseen site and on Hawai'i val, for a 30-epoch run and the 10-epoch run">

Training longer does not help a new site. In a 30-epoch run, F1 on the unseen site is 0.361 after the first epoch and about 0.38 from epoch 6 on, then slowly declines, while Hawai'i val holds at 0.585–0.588. The checkpoints before epoch 30 are taken before that run's learning rate has decayed. The 10-epoch run decays fully within its 10 epochs and scores best on the unseen site (0.385). Early stopping on Hawai'i val cannot see this, so the recipe uses a fixed 10 epochs.

### Training Curves of the Rounds

<img src="../../img/metrics/just_bird_rounds_train.png" width="1000" alt="Hawai'i validation mAP per epoch for teacher 1, teacher 2 and the final recipe">

Teacher 1 was trained for up to 100 epochs with early stopping and kept epoch 24. Teacher 2 and the final recipe run a fixed schedule that decays fully and keep the last epoch. On Hawai'i val the three end within about 0.03 mAP50 of each other (0.63–0.66). Only the unseen site separates them.

---

## How the Labels Are Refined

The refinement changes only the four bout-labelled datasets: Northeastern US, Western US, Amazon Basin and Bird-Sounds-Global. Hawai'i and Southern Sierra Nevada keep their human call labels. It works on the training images, 3 s clips that overlap by half. An annotation longer than a clip is cut into one box per clip, and each of these boxes is handled on its own.

Terms used in this section:

- **Human box:** a box an annotator drew. In these four datasets it usually covers a whole song or bout of several seconds.
- **Replaced:** the teacher found at least one call box for the human box, and those call boxes take its place. A human box can be replaced by many call boxes or by one.
- **Discarded:** the teacher found no call box that qualifies (see the rule below), so the human box cannot be replaced. It is not used as the annotator drew it either: **the human box and its whole 3 s clip are discarded from training**, including any other boxes in that clip. Training on long bout boxes made the model worse on a new site (F1 0.122 against 0.380, see [Each Step on the Unseen Site](#each-step-on-the-unseen-site)), so none is kept.
- **Call-sized:** the whole annotation, before it was cut into clips, is at most 0.5 s long, the median length of the Hawai'i call labels. Here the annotator boxed a single call. Longer annotations cover a phrase, song or bout; they are 96% of the boxes in these four datasets.

### The Rule

For every clip:

1. The teacher proposes boxes with a confidence of at least 0.05. The cutoff is low on purpose: inside a box a human drew, even a weak detection is likely a call.
2. **Half-inside test:** a teacher box qualifies for a human box when at least half of the teacher box lies within the human box. Small call boxes inside a bout box pass easily.
3. The qualifying teacher boxes replace the human box. They are trimmed to its edges, so a new box never leaves the time and frequency range the annotator marked.
4. Near-duplicates (IoU ≥ 0.6, for example one call inside two overlapping human boxes) are merged, keeping the more confident box.
5. If no teacher box qualifies, the human box is discarded, and with it the whole clip.

<img src="../../img/metrics/just_bird_half_inside.png" width="1000" alt="Drawing of the half-inside test: a bout box replaced by three call boxes; a tight call box whose larger teacher box fails the test; the same box under the looser rule, where the teacher box is trimmed to it">

The test is measured on the teacher box, which makes it strict with tight call boxes. The teacher often draws a slightly larger box around the same call. Then less than half of the teacher box lies inside the tight human box (middle), so a good label is discarded with its clip. The looser rule on the right was tested in [Testing a Looser Rule](#testing-a-looser-rule-for-call-sized-boxes).

Real clips from the training data:

<img src="../../img/metrics/just_bird_refine_cases.png" width="1000" alt="Six training clips: a bout box replaced by five call boxes, a good call box replaced by an almost identical box, a loose call box replaced by a tighter one, and three boxes that are discarded">

In the top row the human box is replaced: a bout box by five call boxes, a call box that already fits its call by an almost identical box, and a loose call box by a tighter one. In the bottom row it is discarded with its clip, for the reasons in [Why Boxes Are Discarded](#why-boxes-are-discarded).

Two things the rule cannot do:

- **It never adds a label.** A call outside every human box stays unlabelled, as in the original data.
- **A call the teacher misses inside a bout box loses its label**, because the call boxes replace the whole bout box. In the bout clip above, the short, low sound at about 0.45 s gets no box in either round and now counts as background. A lower confidence cutoff would find more of these calls, but would also accept more noise.

### What Each Round Changes

<img src="../../img/metrics/just_bird_label_rounds.png" width="1000" alt="Median box length per dataset for human, round-1 and round-2 labels, and the share of human boxes discarded in each round">

Round 1 cuts the median box length from 1.15–3.0 s to 0.43–1.16 s, and round 2 to 0.38–0.82 s, close to the 0.50 s of the call-level Hawai'i labels. The larger effect of round 2 is that it replaces more of the bout boxes, so far fewer clips are discarded:

| Dataset | Training clips | Discarded, round 1 | Discarded, round 2 | Used for training |
| :--- | :---: | :---: | :---: | :---: |
| Northeastern US | 116,780 | 25,847 (22%) | 10,617 (9%) | 106,163 |
| Western US | 36,435 | 13,605 (37%) | 9,014 (25%) | 27,421 |
| Amazon Basin | 31,264 | 23,730 (76%) | 18,101 (58%) | 13,163 |
| Bird-Sounds-Global | 84,404 | 23,826 (28%) | 15,334 (18%) | 69,070 |

Amazon Basin loses the most. Its recordings are dense choruses with long, overlapping annotations, and even after round 2 more than half of its clips are discarded.

### Why Boxes Are Discarded

A discarded box is not one the teacher judged to be correct: when the teacher finds the call in a good call box, it replaces it with about the same box (the second real clip above). A human box is discarded only when no teacher box passes the half-inside test, for one of two reasons:

- **No teacher box.** The teacher finds nothing in the box. Usually this is a stretch of a long annotation where the bird is silent or barely audible in this clip, or a faint call.
- **The teacher box sticks out.** Teacher boxes touch the human box, but less than half of each lies inside it. This happens where boxes of several birds overlap, as in the Amazon Basin choruses, and where the teacher draws a larger box around a call than the annotator did (the drawing above).

To see which boxes these are, every human box was traced back to the whole annotation it was cut from, and the teacher was run again on every clip with a discarded box:

<img src="../../img/metrics/just_bird_discarded.png" width="1000" alt="Left: share of human boxes discarded, by annotation length, per dataset. Right: the discarded boxes, divided into call-sized and longer annotations, and by whether the teacher found nothing or its box stuck out">

| Dataset | Human boxes | Call-sized | Discarded, all boxes | Discarded, call-sized | Call-sized share of the discarded boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Northeastern US | 151,983 | 5,423 (3.6%) | 7% | 22% | 11% |
| Western US | 54,951 | 316 (0.6%) | 19% | 70% | 2% |
| Amazon Basin | 77,792 | 733 (0.9%) | 34% | 74% | 2% |
| Bird-Sounds-Global | 125,370 | 8,467 (6.8%) | 15% | 31% | 14% |
| All four | 410,096 | 14,939 (3.6%) | 16% | 31% | 7% |

Round 2, training split, boxes counted per 3 s clip.

- **Call-sized boxes are discarded more often** than longer ones: 31% against 16% over all four datasets. In 62% of these cases the teacher did find something, but its box sticks out, as in the drawing.
- **But call-sized boxes are rare.** These datasets were annotated per bout, and only 3.6% of their boxes are call-sized. They make up 7% of the discarded boxes.
- **Most discarded boxes come from long annotations.** In 48% of them the teacher found nothing, mostly silent or faint stretches. In the rest its boxes stick out, above all in Amazon Basin (65% of its discarded boxes).

### Testing a Looser Rule for Call-Sized Boxes

Because the half-inside test rejects good call boxes whenever the teacher draws a larger box, a looser rule was tested on the same hold-out. For call-sized human boxes only, a teacher box also qualifies when it covers at least half of the human box. It is trimmed to the human box, so the label stays about the same as the annotator drew it, and the clip stays in training. In short: if the teacher confirms there is a call, the clip is kept.

This keeps 1,604 clips that the current rule discards, 0.6% more bout-labelled training data. A model trained with them, with the recipe otherwise unchanged, scores:

| Rule | Unseen site, IoU 0.25 / 0.5 / 0.75 | Unseen site, IoU 0.5 at the Hawai'i-val threshold | Hawai'i val, IoU 0.5 |
| :--- | :---: | :---: | :---: |
| Current | 0.540 / 0.385 / 0.116 | 0.384 | 0.581 |
| Looser for call-sized boxes | 0.541 / 0.389 / 0.120 | 0.387 | 0.583 |

Every score is 0.001–0.005 higher, less than a second training run with another seed would move it. Discarding these clips costs nothing measurable, and keeping them does no harm. The released model keeps the current rule. The looser one remains an option (`--call-cover` in `refine_coarse_boxes.py`) for the next time the labels are rebuilt.

---

## Training Results

The figure shows the validation mAP and the box loss after each epoch. The validation set is the Hawai'i and Southern Sierra Nevada val splits, the two datasets with call-level labels. The drop in training loss at epoch 8 is where mixup augmentation is switched off.

<img src="../../img/metrics/just_bird_train.png" width="1000" alt="Validation mAP and box loss per epoch">

!!! info "Model Checkpoint"
    The checkpoint with the best validation fitness was kept. That was the last epoch (10).

---

## Species Distribution Across Splits

The following table shows the number of boxes in the training and validation images, counted per 3 s training image. Consecutive images overlap by 50%, so most calls are counted twice. The training split includes the test splits of all datasets, because the transfer was measured on a held-out site instead (see above).

| Species | Train | Val | Test | Total | 70/15/15 Quality {: data-sort-method="mixed-split" } |
| :--- | :---: | :---: | :---: | :---: | :--- |
| bird | 1,162,238 | 16,755 | 0 | 1,178,993 | 98.6/1.4/0.0 |

---

## References

<a id="ref-bansal"></a>
[2] Bansal, A., Sikka, K., et al. (2018). "Zero-shot object detection." Proceedings of the European Conference on Computer Vision (ECCV), 384–400.
