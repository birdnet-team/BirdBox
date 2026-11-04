This section describes the algorithmic ideas and the concepts utilized within BirdBox during inference.

For user-facing parameters see [CLI Reference](../cli/workflows.md) or [API Reference](../api/api-type.md).
To compare different models, take a look at [Models and Metrics](../models-and-metrics/overview.md).
If you are interested in the file flow see [File Flows](../data/fileflows.md).

## Inference Algorithm Concept

The key idea of the BirdBox inference is to use YOLO[[23]](#ref-redmon) from Ultralytics[[9]](#ref-jocher) with a sliding window approach.
Both BirdBox models are YOLO11 object detectors: [Just-Bird](../models-and-metrics/just-bird.md) (YOLO11s, one `bird` class) and [Birds-With-Labels](../models-and-metrics/birds-with-labels.md) (YOLO11l, 282 species).
YOLO cannot run on continuous audio, so the recording is cut into short, overlapping images:

1. **Load the audio.** It is mixed to mono and resampled to 32 kHz. Silence is added at the end so that the last 3 s window is complete; boxes in that silence are cut at the real end of the file.
2. **Compute a log-floor mel spectrogram** with STFT[[1]](#ref-allen): log power minus each band's noise floor, one minute at a time to bound the memory. It does not change with the recording level ([details](spectrogram.md#stage-5-log-floor-normalization)).
3. **Cut 3 s clips every 1.5 s** (50% overlap) and render each as a 256×256 pixel image. BirdNET analyzes audio in 3 s segments as well [[10]](#ref-kahl).
4. **Run YOLO on the clips** in batches. Boxes below the confidence threshold are dropped (default 0.15 for Just-Bird, 0.2 for Birds-With-Labels), and overlapping boxes within a clip are reduced by non-maximum suppression [[8]](#ref-hosang) (`--nms-iou`, default 0.7).
5. **Convert the boxes** from pixels to seconds and Hz. The frequency axis follows the HTK mel scale from 50 Hz to 15 kHz, the inverse of how the training labels were drawn.
6. **Merge the boxes of overlapping windows** into one set of detections per file, with window fusion (Just-Bird) or the song merge (Birds-With-Labels), see below.
7. **Write the detections** in the selected output formats.

To examine the generation of the spectrograms in detail, see [Spectrogram Generation](spectrogram.md).
The various output data formats provided by BirdBox can be examined in [Detection Output Formats](../data/outputs.md).

## Merging the Windows

Consecutive windows overlap by half, so almost every call is seen by two windows. Any vocalization of 1.5 s or less lies completely inside at least one window; only longer vocalizations can be cut into parts. The two views of a call rarely give exactly the same box, and the merge decides what is kept. BirdBox has two merges, and each model has its default (`--merge` overrides it).

The figure shows both merges on six seconds of the demo sample. The song merge unions the two views of a call and joins calls less than 0.1 s apart, so the two calls at 9.5–10.1 s become one box. Window fusion keeps one box for each of them.

<img src="../../img/how-it-works/window_fusion.png#only-light" width="1000" alt="Six seconds of the demo sample: the raw Just-Bird boxes of the overlapping windows, the song merge with six boxes, and window fusion with seven call boxes">
<img src="../../img/how-it-works/window_fusion_dark.png#only-dark" width="1000" alt="Six seconds of the demo sample: the raw Just-Bird boxes of the overlapping windows, the song merge with six boxes, and window fusion with seven call boxes">

### Window Fusion (Just-Bird)

Window fusion keeps one box per call, or per phrase with a phrase gap. It works in four steps. The figure below shows each step on another six seconds of the demo sample. Solid boxes are what a step keeps, and dashed boxes what it removes or merges into others.

1. **Ownership.** Each window owns its middle half (the thick lines at the top). The first window also owns the start of the file, the last one the end. A box is kept only from the window that owns its centre, which is the window that sees the call most centrally. The middle halves tile the file, so every call is kept once. The calls at 43.8 and 44.2 s were boxed by both windows, and only the boxes of the window that owns them stay. A call that only the other window found is lost: the faint call at 44.7 s and 2.5–4.5 kHz was boxed only by a window whose middle half ends at 44.25 s.
2. **Weighted box fusion.** YOLO's non-maximum suppression within a window leaves boxes that overlap less than IoU 0.7. Boxes of one window that overlap at 2D IoU ≥ 0.4 are averaged, weighted by confidence [[27]](#ref-solovyev), and keep the strongest confidence. At 47.2–49.5 s two boxes of one window become one.
3. **Drop nested boxes.** A box that lies at least 80% inside a larger box is removed, for example a syllable inside the box of its whole call. At 46.3–47.2 s the box of the first call goes, because it lies inside the longer box that starts with it. The step keeps the larger box even when it is the less confident one. Nested boxes are dropped before the join, so a removed box cannot change a joined box later. This removes 10–25% of the boxes without lowering F1.
4. **Join.** Boxes in the same frequency band (frequency IoU ≥ 0.8) that overlap in time by at least 0.05 s are linked, for example a long call that two windows cut differently. Every group of linked boxes becomes one box, and this repeats on the joined boxes until nothing joins. With a phrase gap above 0, boxes closer than the gap are linked instead, so syllables become phrase boxes. Just-Bird uses 0.05 s by default; `--phrase-gap 0` keeps one box per call. In the figure the boxes from 46.3 s on become one phrase box, which continues to 62.4 s, and the boxes before 43 s become one.

<img src="../../img/how-it-works/window_fusion_steps.png#only-light" width="1000" alt="Six seconds of the demo sample after each window fusion step: 25 raw boxes, 12 after ownership, 11 after weighted box fusion, 10 after dropping nested boxes and 7 after the join">
<img src="../../img/how-it-works/window_fusion_steps_dark.png#only-dark" width="1000" alt="Six seconds of the demo sample after each window fusion step: 25 raw boxes, 12 after ownership, 11 after weighted box fusion, 10 after dropping nested boxes and 7 after the join">

On test sets labelled per call, fusion raised micro F1 at IoU 0.5 over the song merge, from 0.515 to 0.582 on Hawai'i and from 0.615 to 0.637 on Southern Sierra Nevada (models trained on each dataset alone). For the flags, see [Merge Method](../cli/detect-birds.md#merge-method).

#### Phrase Gap

Some datasets box every syllable, others one box per phrase or song. The phrase gap sets how far apart boxes in one band may be and still be joined. Only this time rule changes with the gap, so the boxes change gradually as it grows. Just-Bird defaults to 0.05 s because it is run on recordings of any convention.

<img src="../../img/metrics/just_bird_phrase_gap.png#only-light" width="1000" alt="Best F1 at IoU 0.5 over the phrase gap on Hawai'i val, Southern Sierra Nevada and Western-US val, October against November model. F1 falls with the gap on Hawai'i val and changes little on the other two sets">
<img src="../../img/metrics/just_bird_phrase_gap_dark.png#only-dark" width="1000" alt="Best F1 at IoU 0.5 over the phrase gap on Hawai'i val, Southern Sierra Nevada and Western-US val, October against November model. F1 falls with the gap on Hawai'i val and changes little on the other two sets">

Each point is the best F1 over all thresholds on full recordings, and the y-axes differ between panels. The first two panels use the Just-Bird recipe trained without Southern Sierra Nevada, and the third the released models on validation minutes they did not train on.

On call labels the score falls steadily as the gap grows, while on phrase and song labels it barely moves. The 0.05 s default costs 0.005 on Hawai'i and adds at most 0.005 on the other two sets. The current model (November 2026) is well ahead on song labels at every gap, because it draws phrase boxes where its training labels did. For the values to pass, see [Phrase Gap](../cli/detect-birds.md#phrase-gap).

### Song Merge (Birds-With-Labels)

The song merge joins detections of the same species whose gap is at most the song gap (`--song-gap`, default 0.1 s) into one song box that spans all of them. Most of the Birds-With-Labels training labels box whole songs, and the wide song boxes match them better than single calls (fusion scored 0.02–0.05 lower). For Just-Bird (`--merge reconstruct`), boxes are only joined if they also share the frequency band, so calls of different birds are not chained.

For details, see [Song Gap Threshold](../cli/detect-birds.md#song-gap-threshold).

### Filter, Then Merge

The confidence threshold is applied to the raw boxes of each window, before merging. The evaluation tools work the same way: [`detect_birds.py --no-merge`](../cli/detect-birds.md#evaluation-mode) writes the raw boxes once, and [F-beta analysis](../cli/f-beta-score-analysis.md) or [Filter and Merge Detections](../cli/filter-and-merge-detections.md) filter them at any threshold and merge them as detection would.

## References

<a id="ref-allen"></a>
**[1]** Allen, J. B., & Rabiner, L. R. (1977). "A unified approach to short-time Fourier analysis and synthesis." Proceedings of the IEEE, 65(11), 1558–1564.

<a id="ref-hosang"></a>
**[8]** Hosang, J., Benenson, R., & Schiele, B. (2017). "Learning non-maximum suppression." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4507–4515.

<a id="ref-jocher"></a>
**[9]** Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* (Version 11.0.0) [Computer software]. AGPL-3.0. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics){ target="_blank" rel="noopener noreferrer" }. There is no YOLO11 paper; this is the citation given in the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgements){ target="_blank" rel="noopener noreferrer" }.

<a id="ref-kahl"></a>
**[10]** Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). "BirdNET: A deep learning solution for avian diversity monitoring." *Ecological Informatics*, 61:101236.

<a id="ref-redmon"></a>
**[23]** Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection.“ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

<a id="ref-solovyev"></a>
**[27]** Solovyev, R., Wang, W., & Gabruseva, T. (2021). "Weighted boxes fusion: Ensembling boxes from different object detection models." *Image and Vision Computing*, 104117.
