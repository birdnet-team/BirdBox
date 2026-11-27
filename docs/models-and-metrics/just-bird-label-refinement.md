# Just-Bird Label Refinement

Most training datasets of the [Just-Bird model](just-bird.md) put one box around a whole song or bout, while the model should draw one box per call. Such boxes also cover the pauses between the calls. A study on crowd-sourced bird recordings found the same problem with labels that cover whole clips, and turned them into call-level labels [[2]](#ref-ayers). Before training, those boxes are replaced by call boxes from a teacher model, in two rounds (see [Training Recipe](just-bird-training.md#training-recipe)). This page explains the rule, what each round changes, and which boxes and clips are discarded.

The page describes the labels of the released model. Its teachers see [log-floor spectrograms](../how-it-works/spectrogram.md#stage-5-log-floor-normalization), like the model itself. Three design decisions were tested with an earlier version of the recipe, whose teachers and models saw PCEN spectrograms: the [looser rule for call-sized boxes](#the-looser-rule-for-call-sized-boxes), [dropping nested boxes](#nested-boxes-in-the-refined-labels), and [stopping after two rounds](#why-there-is-no-third-round). These tests are marked as such. [PCEN or Log-Floor Teachers](#pcen-or-log-floor-teachers) compares the two kinds of teacher.

---

## Overview

The refinement changes only the four bout-labelled datasets: Northeastern US, Western US, Amazon Basin and Bird-Sounds-Global. Hawai'i and Southern Sierra Nevada keep their human call labels ([Which Datasets Are Refined](#which-datasets-are-refined) explains why). It works on the training images, 3 s clips that overlap by half. An annotation longer than a clip is cut into one box per clip, and each of these boxes is handled on its own.

Terms used on this page:

- **Human box:** a box an annotator drew. In these four datasets it usually covers a whole song or bout of several seconds.
- **Replaced:** the teacher found at least one call box for the human box, and those call boxes take its place. A human box can be replaced by many call boxes or by one.
- **Discarded:** the teacher found no call box that qualifies (see the rule below), so the human box cannot be replaced. It is not used as the annotator drew it either: **the human box and its whole 3 s clip are discarded from training**, including any other boxes in that clip. Training on long bout boxes made the model worse on a new site (F1 0.124 against 0.383 with call labels, see [Each Step on the Unseen Site](just-bird-training.md#each-step-on-the-unseen-site)), so none is kept. Leaving out training samples whose labels cannot be trusted is a common way to learn from noisy labels (sample selection, [[31]](#ref-song)).
- **Call-sized:** the whole annotation, before it was cut into clips, is at most 0.5 s long. That is the median length of the Hawai'i call boxes in the 3 s training clips. Whole Hawai'i annotations have a median of 0.60 s (table below), because an annotation cut at a clip edge leaves shorter boxes. Here the annotator boxed a single call. Longer annotations cover a phrase, song or bout. They are 96% of the boxes in these four datasets.

---

## Which Datasets Are Refined

The datasets differ in what one human box covers. In each, every box was traced back to the whole annotation it was cut from (training split, boxes counted per 3 s clip):

| Dataset | What one box covers | Median annotation length | Boxes from call-sized annotations | Refined |
| :--- | :--- | :---: | :---: | :---: |
| Hawai'i | a call | 0.60 s | 47% | no |
| Southern Sierra Nevada | a call | 0.40 s | 55% | no |
| Bird-Sounds-Global | mixed: calls, songs, bouts | 2.08 s | 7% | yes |
| Western US | a bout | 3.30 s | 1% | yes |
| Northeastern US | a bout | 5.50 s | 4% | yes |
| Amazon Basin | a bout | 12.80 s | 1% | yes |

The annotation guidelines of Northeastern US, Western US and Amazon Basin let annotators join consecutive calls with pauses under 5 s into one box. The guidelines of Southern Sierra Nevada have no such clause, and Hawai'i is labelled per call as well. Bird-Sounds-Global, soundscapes from xeno-canto, mixes both styles. Most of its boxes cover more than one call.

Hawai'i and Southern Sierra Nevada are left as they are, for four reasons:

- **There is nothing to split.** About half of their boxes are call-sized, and the median annotation is at most 0.6 s. The rule exists to cut long bout boxes into calls.
- **The rule would lose good labels.** A call box the teacher finds is replaced by about the same box, so nothing is gained. A call box the teacher misses is discarded with its whole clip, like the [valid faint call](#the-rule) below. On call-level labels the rule can only cost data.
- **Hawai'i defines what a call box is.** Teacher 1 learned from the Hawai'i labels. Relabelling them with a model trained on them would only feed the teacher's own mistakes back into the training data.
- **Southern Sierra Nevada was the unseen test site** while the recipe was developed (see [Training Recipe](just-bird-training.md#training-recipe)). Its labels had to stay human to measure the teachers and models against. The released model adds them unchanged.

---

## The Rule

For every clip:

1. The teacher proposes boxes with a confidence of at least 0.05. The cutoff is low on purpose: inside a box a human drew, even a weak detection is likely a call. Self-training for object detection usually keeps only confident boxes, because low-confidence boxes are often false positives and would become false labels anywhere in the image [[19]](#ref-liu), [[29]](#ref-sohn). Here the human box limits where a teacher box can go, so a weak box inside it is much less likely to be wrong.
2. **Half-inside test:** a teacher box qualifies for a human box when at least half of the teacher box lies within the human box. Small call boxes inside a bout box pass easily.
3. **Looser rule for call-sized boxes:** if no teacher box passes the half-inside test and the human box is call-sized, the teacher box that covers the largest share of the human box qualifies, if it covers at least half of it.
4. The qualifying teacher boxes replace the human box. They are trimmed to its edges, so a new box never leaves the time and frequency range the annotator marked.
5. Near-duplicates (IoU ≥ 0.6, for example one call inside two overlapping human boxes) are merged, keeping the more confident box.
6. If no teacher box qualifies, the human box is discarded, and with it the whole clip.
7. **Nested boxes:** after both rounds, of two boxes where one lies at least 80% inside the other, the inner one is dropped ([Nested Boxes](#nested-boxes-in-the-refined-labels)).

<img src="../../img/metrics/just_bird_half_inside.png#only-light" width="1000" alt="Drawing of the half-inside test: a bout box replaced by three call boxes; a tight call box whose larger teacher box fails the test; the same box under the looser rule, where the teacher box is trimmed to it">
<img src="../../img/metrics/just_bird_half_inside_dark.png#only-dark" width="1000" alt="Drawing of the half-inside test: a bout box replaced by three call boxes; a tight call box whose larger teacher box fails the test; the same box under the looser rule, where the teacher box is trimmed to it">

The half-inside test is measured on the teacher box, which makes it strict with tight call boxes. The teacher often draws a slightly larger box around the same call. Then less than half of the teacher box lies inside the tight human box (middle), and without the looser rule a good label would be discarded with its clip. The looser rule (bottom) keeps it.

Real clips from the training data. All teacher boxes in the figure are from **teacher 2**, the second round, whose labels the released model was trained on:

<img src="../../img/metrics/just_bird_refine_cases.png#only-light" width="1000" alt="Seven training clips: a bout box replaced by its call boxes, a good call box replaced by about the same box, a loose call box replaced by a tighter one, a tight call box replaced under the looser rule, and three boxes that are discarded">
<img src="../../img/metrics/just_bird_refine_cases_dark.png#only-dark" width="1000" alt="Seven training clips: a bout box replaced by its call boxes, a good call box replaced by about the same box, a loose call box replaced by a tighter one, a tight call box replaced under the looser rule, and three boxes that are discarded">

In the first four clips the human box is replaced:

- a bout box by its call boxes (the nested ones are dropped later, see [Nested Boxes](#nested-boxes-in-the-refined-labels))
- a call box that already fits its call by about the same box
- a loose call box by a tighter one
- a tight call box by the trimmed teacher box, under the looser rule

In the last three it is discarded with its clip, for the reasons in [Why Boxes Are Discarded](#why-boxes-are-discarded):

- **Teacher boxes stick out** (Western US): discarded in round 1 and in round 2. The annotation is 1.9 s long, so the looser rule does not apply.
- **Valid faint call** (Amazon Basin): the human box is correct, but discarded in round 1 and in round 2. Neither teacher finds the call, even with the confidence cutoff lowered from 0.05 to 0.001: no box of teacher 1 or teacher 2 even touches the human box. So the clip and its good label are lost to training. This is the price of the rule: a call that no teacher detects cannot be told apart from a stretch of silence inside a long annotation, and both are discarded.
- **Quiet stretch** (Amazon Basin): a clip from a 12 s annotation. Neither teacher finds a call in this part of it, so it is discarded in both rounds.

Two things the rule cannot do:

- **It never adds a label.** A call outside every human box stays unlabelled, as in the original data.
- **A call the teacher misses inside a bout box loses its label**, because the call boxes replace the whole bout box. In the bout clip above, the short, low sound at about 0.45 s gets no box in either round and now counts as background. A lower confidence cutoff would find more of these calls, but would also accept more noise.

---

## What Each Round Changes

<img src="../../img/metrics/just_bird_label_rounds.png#only-light" width="1000" alt="Median box length per dataset for human, round-1 and round-2 labels, and the share of human boxes discarded in each round">
<img src="../../img/metrics/just_bird_label_rounds_dark.png#only-dark" width="1000" alt="Median box length per dataset for human, round-1 and round-2 labels, and the share of human boxes discarded in each round">

Boxes are counted per 3 s clip here, so no box is longer than 3 s. Round 1 cuts the median box length from 1.15–3.0 s to 0.42–1.10 s, and round 2 to 0.38–0.84 s, close to the 0.50 s of the call-level Hawai'i labels. The larger effect of round 2 is that it replaces more of the bout boxes, so far fewer clips are discarded. Round 2 also applies the looser rule for call-sized boxes.

| Dataset | Training clips | Discarded, round 1 | Discarded, round 2 | Used for training |
| :--- | :---: | :---: | :---: | :---: |
| Northeastern US | 116,780 | 16,731 (14%) | 7,525 (6%) | 109,255 |
| Western US | 36,435 | 10,724 (29%) | 7,639 (21%) | 28,796 |
| Amazon Basin | 31,264 | 22,488 (72%) | 16,315 (52%) | 14,949 |
| Bird-Sounds-Global | 84,404 | 21,838 (26%) | 13,638 (16%) | 70,766 |

Round 2 does not build on the labels of round 1. Both rounds start again from the original human boxes, so teacher 2 gets a new chance at every clip that round 1 discarded. Teacher 2 never trained on those clips (it was trained on the round-1 labels, without the discarded clips), but it learned from far more data than teacher 1, which saw only Hawai'i. Comparing the discarded clips of the two rounds:

| Dataset | Discarded, round 1 | Kept again in round 2 | Discarded in both rounds | Kept in round 1, discarded in round 2 |
| :--- | :---: | :---: | :---: | :---: |
| Northeastern US | 16,731 | 9,461 | 7,270 | 255 |
| Western US | 10,724 | 3,175 | 7,549 | 90 |
| Amazon Basin | 22,488 | 6,404 | 16,084 | 231 |
| Bird-Sounds-Global | 21,838 | 8,690 | 13,148 | 490 |
| All four | 71,781 | 27,730 (39%) | 44,051 | 1,066 |

Teacher 2 finds the calls in 39% of the clips that round 1 discarded. Only 1,066 clips go the other way. A call that neither teacher finds, like the faint call above, stays discarded.

### Why There Is No Third Round

A third round would train teacher 3 on the round-2 labels and relabel again. It was not run, because round 2 already brought no gain on full recordings of the unseen site. This was decided with the PCEN version of the recipe, before the looser rule and dropping nested boxes:

| Labels of the bout-labelled datasets | F1 at IoU 0.5, unseen site, at the Hawai'i-val threshold | Discarded clips |
| :--- | :---: | :---: |
| Human boxes | 0.124 | none |
| Round 1 | 0.381* | 87,008 |
| Round 2 | 0.383 | 53,066 |

\* This model (teacher 2) was trained on a quarter of the data, the other two on all of it.

- **Round 1 brought the big step**, from bout boxes to call boxes.
- **Round 2 ties round 1** on full recordings. Its gains lie elsewhere: 33,942 more clips are kept (19% more bout-labelled training data), 3 s clips score higher, and the threshold chosen on Hawai'i val carries over to the new site.
- **A third round could only win back clips from the 52,174 discarded in both rounds** (44,051 with the released model's labels), many of them silent stretches of long annotations. Winning back 1,604 of them with the [looser rule](#the-looser-rule-for-call-sized-boxes) moved no score beyond what another training seed would.

A third round was not tested, so this is a judgment from the trend, not a measurement. Each round needs a new teacher, a relabelling of all four datasets and a final training to measure it.

Amazon Basin loses the most. Its recordings are dense choruses with long, overlapping annotations, and even after round 2 about half of its clips are discarded.

---

## Why Boxes Are Discarded

A discarded box is not one the teacher judged to be correct: when the teacher finds the call in a good call box, it replaces it with about the same box (the second real clip above). A human box is discarded only when no teacher box qualifies, for one of two reasons:

- **No teacher box.** The teacher finds nothing in the box. Usually this is a stretch of a long annotation where the bird is silent or barely audible in this clip, or a faint call.
- **The teacher box sticks out.** Teacher boxes touch the human box, but less than half of each lies inside it. This happens where boxes of several birds overlap, as in the Amazon Basin choruses. For a call-sized box it also means that no teacher box covers half of it.

To see which boxes these are, every human box was traced back to the whole annotation it was cut from, and the teacher was run again on every clip with a discarded box:

<img src="../../img/metrics/just_bird_discarded.png#only-light" width="1000" alt="Left: share of human boxes discarded, by annotation length, per dataset. Right: the discarded boxes, divided into call-sized and longer annotations, and by whether the teacher found nothing or its box stuck out">
<img src="../../img/metrics/just_bird_discarded_dark.png#only-dark" width="1000" alt="Left: share of human boxes discarded, by annotation length, per dataset. Right: the discarded boxes, divided into call-sized and longer annotations, and by whether the teacher found nothing or its box stuck out">

| Dataset | Human boxes | Call-sized | Discarded, all boxes | Discarded, call-sized | Call-sized share of the discarded boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Northeastern US | 151,983 | 5,423 (3.6%) | 5% | 10% | 7% |
| Western US | 54,951 | 316 (0.6%) | 16% | 48% | 2% |
| Amazon Basin | 77,792 | 733 (0.9%) | 30% | 44% | 1% |
| Bird-Sounds-Global | 125,370 | 8,467 (6.8%) | 13% | 15% | 8% |
| All four | 410,096 | 14,939 (3.6%) | 14% | 15% | 4% |

Round 2, training split, boxes counted per 3 s clip.

- **Call-sized boxes are discarded about as often as longer ones** over all four datasets (15% against 14%). The looser rule keeps most call-sized boxes whose teacher box sticks out. In 70% of the call-sized boxes that are still discarded, the teacher found nothing.
- **In Western US and Amazon Basin, call-sized boxes are still discarded more often** (48% and 44%). There, too, the teacher finds nothing in most of them (83–84%).
- **Call-sized boxes are rare.** These datasets were annotated per bout, and only 3.6% of their boxes are call-sized. They make up 4% of the discarded boxes.
- **Most discarded boxes come from long annotations.** In 48% of them the teacher found nothing, mostly silent or faint stretches. In the rest its boxes stick out, above all in Amazon Basin (60% of its discarded boxes).

---

## The Looser Rule for Call-Sized Boxes

The half-inside test rejects good call boxes whenever the teacher draws a larger box. For call-sized human boxes only, the looser rule (step 3 of [the rule](#the-rule)) also accepts a teacher box that covers at least half of the human box. It is trimmed to the human box, so the label stays about the same as the annotator drew it, and the clip stays in training. In short: if the teacher confirms there is a call, the clip is kept.

The rule was tested with the PCEN version of the recipe, on the Southern Sierra Nevada hold-out. There it keeps 1,604 clips that the half-inside test alone discards, 0.6% more bout-labelled training data. A model trained with them, with the recipe otherwise unchanged, scores:

| Rule | Unseen site, IoU 0.25 / 0.5 / 0.75, best threshold | Unseen site, IoU 0.5 at the Hawai'i-val threshold | Hawai'i val, IoU 0.5, best threshold |
| :--- | :---: | :---: | :---: |
| Half-inside test only | 0.541 / 0.384 / 0.116 | 0.383 | 0.584 |
| With the looser rule | 0.541 / 0.389 / 0.120 | 0.387 | 0.586 |

Every score is the same or up to 0.005 higher, no more than a second training run with another seed moves it (0.001–0.006, see [Nested Boxes](#nested-boxes-in-the-refined-labels)). Discarding these clips costs nothing measurable, and keeping them does no harm: a manual check of some of the kept clips found real calls in the human boxes, with teacher boxes far larger than the call around them. The released model uses the looser rule (`--call-cover 0.5` in `refine_coarse_boxes.py`).

---

## Nested Boxes in the Refined Labels

In the round-2 labels, about half of the boxes of the four refined datasets lie at least 80% inside a larger box of the same clip. Nearly all of these pairs lie inside one human box: teacher 2 often draws a phrase box and the syllable boxes inside it, and both qualify. The merge in step 5 of [the rule](#the-rule) only catches near-duplicates (IoU ≥ 0.6), not a small box inside a large one. [Window fusion](../how-it-works/overview.md#window-fusion) removes nested boxes from the model's output, and step 7 removes them from the training labels in the same way: it drops the inner box of each pair (drop inner).

| Dataset | Human labels: boxes inside a larger box | Round 2: boxes dropped as nested |
| :--- | :---: | :---: |
| Hawai'i | 2% | not refined |
| Southern Sierra Nevada | 1% | not refined |
| Northeastern US | 13% | 57% |
| Western US | 3% | 63% |
| Amazon Basin | 26% | 51% |
| Bird-Sounds-Global | 3% | 56% |

Training split, boxes counted per 3 s clip. A box counts as nested when at least 80% of it lies inside a larger box.

Drop inner keeps the phrase box of each pair. The other choice, **drop outer**, keeps the syllable boxes. Both were tested with the PCEN version of the recipe, each against the round-2 labels with no nested boxes dropped. That recipe was also trained a second time with another seed, to measure the noise of one run. Full recordings, window fusion as BirdBox runs it, phrase gap off:

| Labels | Unseen site, IoU 0.25 / 0.5 / 0.75, best threshold | Hawai'i val, IoU 0.5, best threshold |
| :--- | :---: | :---: |
| Round 2, nested boxes kept | 0.541 / 0.384 / 0.116 | 0.584 |
| Round 2, nested boxes kept, second seed | 0.543 / 0.390 / 0.115 | 0.587 |
| Drop inner | 0.550 / 0.392 / 0.130 | 0.587 |
| Drop outer | 0.513 / 0.341 / 0.081 | 0.591 |

- **The second seed moves the scores by 0.001–0.006.** Differences of that size are noise.
- **Drop inner** gains most at IoU 0.75 (+0.014), where the fit of the box counts most.
- **Drop outer** learns syllable fragments: on 3 s clips its median box is 0.19 s, against 0.24 s with nested boxes kept and 0.40 s for the median annotation of the unseen site. Its confidences also fall on the unseen site, so a threshold chosen on Hawai'i val does not carry over.

### Measured on Calls as a User Wants Them

The unseen site's labels do not box one unit: some species are boxed per song element, others per phrase. Box F1 then penalises element boxes on a phrase label as much as a miss, so it cannot tell whether finer boxes are better. To measure that, 260 five-second stretches of the unseen site (21.7 min) were relabelled by hand by one person: missed calls added, phrase boxes split where the elements stand alone, and kept together where they only make sense together (458 boxes before, 567 after). F1 at IoU 0.5 on these labels, with a 95% interval from a paired bootstrap over the stretches:

| Comparison | Best threshold and phrase gap | Defaults (threshold 0.15, phrase gap 0.05 s) |
| :--- | :---: | :---: |
| Drop outer vs drop inner | −0.010 [−0.054, +0.031] | −0.092 [−0.147, −0.035] |
| Drop inner vs nested boxes kept | +0.027 [−0.005, +0.059] | +0.030 [−0.000, +0.059] |
| Second seed vs nested boxes kept | +0.001 [−0.023, +0.025] | |

- **Finer boxes are not what is wanted.** Drop outer's element boxes, joined by the phrase gap, match the hand labels slightly worse than drop inner, and at the default threshold clearly worse.
- **Drop inner is ahead of keeping nested boxes in every measurement**, on the unseen site and on the hand labels, but never beyond the noise (on the hand labels just at its edge).

The released model uses drop inner (`make_nest.py inner`). Like the looser rule, it is cheap and never scored below the labels without it.

---

## PCEN or Log-Floor Teachers

The released model sees [log-floor spectrograms](../how-it-works/spectrogram.md), and so do its teachers, so the whole chain uses one kind of spectrogram. The teachers of the earlier PCEN recipe saw PCEN spectrograms. To compare the two, the same rules (with the looser rule and drop inner) were also run with the PCEN teachers. Training split:

| Dataset | Clips used, PCEN teachers | Clips used, log-floor teachers (released) | Boxes, PCEN teachers | Boxes, log-floor teachers | Median round-2 box, PCEN / log-floor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Northeastern US | 106,405 | 109,255 | 199,531 | 200,231 | 0.42 s / 0.48 s |
| Western US | 27,450 | 28,796 | 41,193 | 46,053 | 0.56 s / 0.51 s |
| Amazon Basin | 13,266 | 14,949 | 27,756 | 33,346 | 0.82 s / 0.84 s |
| Bird-Sounds-Global | 69,864 | 70,766 | 123,049 | 129,815 | 0.38 s / 0.38 s |
| All four | 216,985 | 223,766 | 391,529 | 409,445 | |

Boxes are counted after drop inner. The log-floor teachers find a qualifying call box in more clips, so 6,781 fewer clips are discarded (3% more bout-labelled training data, most of it in Northeastern US and Amazon Basin), and the median box length barely changes.

A student was trained on each set of labels, on log-floor images and with the hold-out recipe, twice with different seeds. Full recordings of the unseen site, window fusion as BirdBox runs it, phrase gap off:

| Teachers | Unseen site, IoU 0.25 / 0.5 / 0.75, best threshold | Unseen site, IoU 0.5 at the Hawai'i-val threshold | Hand-labelled stretches, best F1 |
| :--- | :---: | :---: | :---: |
| PCEN | 0.549 / 0.392 / 0.128 | 0.389 | 0.462 |
| PCEN, second seed | 0.557 / 0.396 / 0.131 | 0.396 | 0.463 |
| Log-floor (released recipe) | 0.568 / 0.387 / 0.127 | 0.385 | 0.484 |
| Log-floor, second seed | 0.569 / 0.388 / 0.128 | 0.385 | 0.483 |

The differences below are log-floor minus PCEN teachers, first seed against first seed and second against second, at the Hawai'i-val threshold unless noted:

- **IoU 0.75 is a tie** (−0.003 and −0.005, both within the noise), and IoU 0.25 is 0.012–0.019 higher with the log-floor teachers (best threshold).
- **IoU 0.5 is slightly lower**: −0.004 [−0.016, +0.007] and −0.011 [−0.021, −0.002]. Against the October recipe (PCEN throughout), both seeds tie at IoU 0.5 (+0.002 and +0.003) and gain at IoU 0.75 (+0.009 and +0.007).
- **On the [hand-labelled stretches](#measured-on-calls-as-a-user-wants-them)** both seeds score highest of all variants, +0.063 [+0.020, +0.107] over the October recipe.

The two log-floor seeds agree to within 0.001, while the two PCEN seeds differ by up to 0.008, so the IoU 0.5 gap is between 0.004 and 0.011. The log-floor teachers cost little, and in exchange no model in the chain depends on PCEN any more.

---

## References

<a id="ref-ayers"></a>
**[2]** Ayers, J., Perry, S., Prestrelski, S., et al. (2024). "A deep learning approach to the automated segmentation of bird vocalizations from weakly labeled crowd-sourced audio." NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning.

<a id="ref-liu"></a>
**[19]** Liu, Y.-C., Ma, C.-Y., He, Z., Kuo, C.-W., Chen, K., Zhang, P., Wu, B., Kira, Z., & Vajda, P. (2021). "Unbiased teacher for semi-supervised object detection." International Conference on Learning Representations (ICLR).

<a id="ref-sohn"></a>
**[29]** Sohn, K., Zhang, Z., Li, C.-L., Zhang, H., Lee, C.-Y., & Pfister, T. (2020). "A simple semi-supervised learning framework for object detection." arXiv preprint arXiv:2005.04757.

<a id="ref-song"></a>
**[31]** Song, H., Kim, M., Park, D., Shin, Y., & Lee, J.-G. (2023). "Learning from noisy labels with deep neural networks: A survey." *IEEE Transactions on Neural Networks and Learning Systems*, 34(11):8135–8153.
