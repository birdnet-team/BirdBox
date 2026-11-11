# Species from BirdNET

Just-Bird draws a box around every bird vocalization but does not name the species. To name it, the boxes can be combined with BirdNET [[10]](#ref-kahl): BirdNET V3.0 scores a 3 s window around a box, and the box takes the bird species with the highest score. The [BirdNET website demo](https://birdnet.cornell.edu/demo/#birdbox){ target="_blank" rel="noopener noreferrer" } does this on its BirdBox tab.

This page compares where that window can sit, measured on annotated boxes with a known species. It is a study, not a BirdBox feature: BirdBox itself does not run BirdNET. The scripts are in `scripts/birdnet_labels/`.

---

## The Question

Until November 2026 the website demo scored one window per box, centred on it, once the box could no longer change. The species then came 1.5 s after the box in a file and about 4 s after the sound while recording. Now it scores the windows BirdBox detects on, one every 1.5 s, alongside the detection, so a box usually has its species when it appears. A box that grows, or changes with the sliders, needs no new scoring.

Those windows are not centred on the box. Does that cost accuracy?

## Window Placements

| Placement | Windows for a box | Used |
| :--- | :--- | :--- |
| `centred` | One 3 s window centred on the box. A box over 3 s: windows from its start, every 1.5 s. | Demo until November 2026 |
| `grid15` | BirdBox's windows, one every 1.5 s. A box up to 1.5 s takes the window that owns its centre (the middle 1.5 s of a window, as in the [window fusion](../how-it-works/overview.md#window-fusion-just-bird)). A longer box takes the windows from the one around its first 1.5 s to the one around its last. | Demo since November 2026 |
| `grid075` | As `grid15`, on windows every 0.75 s. Twice the BirdNET work. | |
| `inside15` | As `grid15`, and a box up to 1.5 s also takes a neighbouring window it lies wholly in. | |
| `fit15` | As `grid15`, but a box up to 3 s takes only the window that owns its centre when it lies wholly inside it. | |
| `near15` | As `grid15`, but a box up to 3 s always takes only the window that owns its centre. | |

With several windows, each species keeps its highest score. The window at the end of a recording is padded with silence, as BirdBox pads its last clip.

## Data

- **Recordings:** 116 recordings of four datasets, spread evenly over the file names:
    - [Southern Sierra Nevada](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }: 40 of 100 recordings of 10 min
    - [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }: 60 of 635 recordings of 5 min
    - [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }: 4 of 33 recordings of 1 h
    - [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }: 12 of 285 recordings of 1 h
- **Boxes:** the 11,552 annotated boxes of these recordings whose species is a bird of BirdNET V3.0. 1,216 further boxes were left out: 504 with an unidentified species (`????`), and 712 of three species whose eBird code has no BirdNET V3.0 bird (`amepip` 401, `pasfly` 171, `akepa1` 140).
- **Model:** BirdNET+ V3.0-preview3.1 Global 10K-pruned (FP16), the model of the BirdNET Live app and the website demo, run with ONNX Runtime 1.27 on the CPU.
- **Rules of the demo:** only bird species count, and no location filter is applied. **Top-1** means the highest-scoring bird is the annotated species, **top-5** that it is among the five highest-scoring birds, which the demo's species menu lists when they score at least 0.05. The demo names a species only from a score of 0.35 on, otherwise the box stays unidentified ("Aves").

The annotated boxes stand in for the boxes BirdBox finds. Detected boxes differ from them, for example where calls are joined into phrases, so a demo run can score differently. Whether BirdNET V3.0 was trained on these recordings was not checked: that could make the absolute accuracies optimistic, but not change the comparison between placements, which all score the same recordings.

## Results

All four datasets, 11,552 boxes. Differences in top-1 to `centred` with 95% intervals of a paired bootstrap over recordings:

| Placement | Top-1 | Top-5 | Named right | Named wrong | Unnamed | Top-1 minus `centred` |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `centred` | 0.507 | 0.752 | 37.5% | 22.6% | 39.9% |  |
| `grid15` | 0.506 | 0.751 | 37.7% | 22.7% | 39.5% | −0.001 (−0.008 to +0.007) |
| `grid075` | 0.508 | 0.752 | 37.6% | 22.6% | 39.8% | +0.001 (−0.004 to +0.006) |
| `inside15` | 0.505 | 0.768 | 40.3% | 27.3% | 32.4% | −0.002 (−0.009 to +0.006) |
| `fit15` | 0.507 | 0.751 | 37.7% | 22.7% | 39.6% | ±0.000 (−0.007 to +0.007) |
| `near15` | 0.507 | 0.751 | 37.6% | 22.4% | 40.0% | ±0.000 (−0.007 to +0.008) |

**Overall the placements tie.** The species of a box changes from `centred` to `grid15` for 22% of the boxes, but as often to the right species (553) as away from it (559). `inside15` names more boxes, because a second window can only raise the top score, but it names more of them wrongly too.

Per dataset, `centred` against `grid15`:

| Dataset | Boxes | Recordings | Top-1 `centred` | Top-1 `grid15` | Difference | Top-5 `centred` | Top-5 `grid15` |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Southern Sierra Nevada | 3,475 | 40 | 0.466 | 0.462 | −0.004 (−0.014 to +0.007) | 0.667 | 0.666 |
| Island of Hawai'i | 5,650 | 60 | 0.463 | 0.470 | +0.007 (−0.004 to +0.016) | 0.744 | 0.742 |
| Western United States | 998 | 4 | 0.471 | 0.451 | −0.020 (−0.034 to +0.012) | 0.786 | 0.792 |
| Northeastern United States | 1,429 | 12 | 0.802 | 0.796 | −0.006 (−0.027 to +0.020) | 0.966 | 0.965 |
| **All four** | 11,552 | 116 | 0.507 | 0.506 | −0.001 (−0.008 to +0.007) | 0.752 | 0.751 |

No interval excludes zero. Western United States has only 4 recordings, so its interval is wide.

By box length, top-1:

| Box length | Boxes | `centred` | `grid15` | `grid075` | `inside15` | `fit15` | `near15` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Up to 1.5 s | 8,709 | 0.471 | 0.477 | 0.475 | 0.475 | 0.477 | 0.477 |
| 1.5 to 3 s | 1,654 | 0.613 | 0.588 | 0.599 | 0.588 | 0.591 | 0.592 |
| Over 3 s | 1,189 | 0.621 | 0.611 | 0.618 | 0.611 | 0.611 | 0.611 |

## Boxes of 1.5 to 3 s

This is the one place where BirdBox's windows lose: 0.025 in top-1 (95% interval −0.043 to −0.004). A centred window holds such a box whole, in its middle. On the grid, 1,313 of the 1,654 boxes lie wholly inside the window that owns their centre, and the rest take two windows.

Taking only the one window that holds the box (`fit15`), or always only the window that owns its centre (`near15`), recovers almost nothing. A window twice as dense (`grid075`) recovers about half, for twice the work:

| Placement, boxes of 1.5 to 3 s | Top-1 | Minus `centred` (95% interval) | Top-5 |
| :--- | ---: | :--- | ---: |
| `centred` | 0.613 |  | 0.834 |
| `grid15` | 0.588 | −0.025 (−0.043 to −0.004) | 0.838 |
| `fit15` | 0.591 | −0.022 (−0.039 to −0.001) | 0.839 |
| `near15` | 0.592 | −0.021 (−0.036 to −0.003) | 0.834 |
| `grid075` | 0.599 | −0.014 (−0.026 to +0.001) | 0.836 |

So even a grid window that holds the whole box names it less often than a window centred on it: what else the window holds matters. The right species is still among the five in the menu as often as with the centred window.

On the BirdNET website demo's sample soundscape, 4 of the 56 boxes BirdBox finds are 1.5 to 3 s long.

## How Good Is the Species at All?

The placement matters much less than how often BirdNET V3.0 names the annotated species at all: about half of the boxes in top-1, and three quarters in top-5. The Northeastern United States reaches 0.80, the other three datasets about 0.47.

A likely reason, not measured here: BirdNET V3.0 hears the whole 3 s window at all frequencies. Where several birds call at once, as in the dense choruses of Hawai'i and the Sierra Nevada, it may name the loudest one, not necessarily the one in the box.

## Ideas to Try

None of these are tested yet.

- **Centred window afterwards for boxes of 1.5 to 3 s:** keep the fast species from the grid, then score a centred window and replace the species. That should recover the 0.025 for these boxes, but some boxes would switch species a moment after they appear.
- **Let BirdNET hear only the box:** band-pass the window to the box's frequency band, or silence the window outside the box's time span, before BirdNET scores it.
- **Location and date:** the demo's geomodel filter was off here. It removes species that do not occur at the place and time.
- **Combine windows differently:** the mean instead of the highest score, or windows weighted by how much of the box they hold.
- **Detected instead of annotated boxes:** run Just-Bird, match its boxes to the annotations, and score the species of the matched boxes, as a user of the demo would see them.
- **Which species are confused:** a confusion table per dataset would show whether the errors are neighbours in the chorus or similar-sounding species.

## Reproduce

The BirdNET V3.0 files come from [birdnet-live-app](https://github.com/birdnet-team/birdnet-live-app/tree/351310119bd17fe9c2149051055e810903e007e1/assets/models){ target="_blank" rel="noopener noreferrer" } (`assets/models`): the FP16 model, its labels CSV and `taxonomy.csv`, which gives the eBird codes. Each dataset folder holds `annotations.csv` and `soundscape_data_wav/` with the recordings at 32 kHz. Needs `numpy`, `soundfile` and `onnxruntime`.

```bash
M=BirdNET+_V3.0-preview3.1_Global_10K-pruned_FP16.onnx
L=BirdNET+_V3.0-preview3.1_Global_10K-pruned_Labels.csv
python scripts/birdnet_labels/label_windows.py $M $L taxonomy.csv Southern-Sierra-Nevada ssn.json --files=40
python scripts/birdnet_labels/label_windows.py $M $L taxonomy.csv Hawaii hawaii.json --files=60
python scripts/birdnet_labels/label_windows.py $M $L taxonomy.csv Western-US western.json --files=4
python scripts/birdnet_labels/label_windows.py $M $L taxonomy.csv Northeastern-US northeast.json --files=12
python scripts/birdnet_labels/summarize.py ssn.json hawaii.json western.json northeast.json
```

The four runs take about 30 minutes side by side on 20 CPU cores.

---

## References

<a id="ref-kahl"></a>
**[10]** Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). "BirdNET: A deep learning solution for avian diversity monitoring." *Ecological Informatics*, 61:101236.
