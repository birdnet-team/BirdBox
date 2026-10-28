
This chapter describes the available models, gives insight into the underlying dataset splits and demonstrates their performance based on individual evaluation and testsets with various metrics.

BirdBox accepts YOLO models in multiple formats such as `.pt`, `.onnx` and `.engine`.
Trained models for this task can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Alternatively, you can train your own model on a custom dataset by using the code available in the [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } repository.

!!! info "Restricted Access"
    BirdBox-Train is currently only accessible to members of the BirdNET-Team.

To run benchmarks with a model of your choice and a custom dataset see [CLI Reference](../cli/workflows.md).

!!! tip "Recommended model"
    [Just-Bird.pt](just-bird.md) is the recommended model and the default in the [Demo](../demo.md) and the pipeline scripts. It detects bird vocalizations with a single `bird` class. For species-level detection, use the [Birds-With-Labels](birds-with-labels.md) model.

---

## Available Models and Respective Datasets { #available-models }

The available models are trained with datasets from Zenodo.org.
The utilized datasets are:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }

BirdBox provides two models, both trained on all five datasets:

- [Just-Bird.pt](just-bird.md) (recommended), a single `bird` class, trained additionally on Bird-Sounds-Global soundscapes from xeno-canto
- [Birds-With-Labels.pt](birds-with-labels.md)

---

## Dataset Splits

Each dataset is split once into a training, a validation and a test split. The split aims at 70% training, 15% validation and 15% test data for every species.
The counted quantity is the number of labels within a split, not the number of files.
For example, a file with five aldfly annotations is counted as five, not one.

The two models use these splits differently:

| | Birds-With-Labels | Just-Bird |
| :--- | :--- | :--- |
| Training | train splits | train and test splits of all datasets, and the val split of Bird-Sounds-Global |
| Validation | val splits | val splits of Hawai'i and Southern Sierra Nevada only (labelled per call) |
| Test | test splits | none: tested on a site it never saw, with a model trained without that site |
| Train/val/test share of the boxes | 69.4/14.9/15.7 | 98.6/1.4/0 |

Birds-With-Labels is a species model and is tested in-domain on the test splits, so it keeps the 70/15/15 split. Just-Bird is measured on an unseen site instead, so the test splits are more useful as training data. It validates on call labels only, and with over a million training boxes a small validation set is enough.
See [Just-Bird Data Splits](just-bird-training.md#data-splits) and [Birds-With-Labels Species Distribution](birds-with-labels-training.md#species-distribution-across-splits) for the counts.

### Limitations

There exist multiple reasons why the 70/15/15 split of each dataset is not trivial:

1. it is strictly forbidden to split overlapping data into different data-splits (but 50% overlap by default)
2. temporally adjacent recordings may record the same bird with the same background noise twice [[24]](#ref-roberts)
3. single three-second clips often contain multiple annotations from different species

The first and second problem can be solved by **dividing** the soundscape recordings **into one-minute chunks** before any data is split.
Three-second clips with overlap are then only generated within these one-minute files.
The resulting three-second clips can then only be sorted into the same dataset split.
Therefore, no direct overlap between splits is possible and the problem of temporal data leakage is mitigated.

However, combined with the third problem, this limits the capabilities of the dataset-splitter.
From now on, only one minute chunks can be sorted into a distinct split.

### Rare Species

Yet some rare species only occur in one or two minutes of the original audio.
This makes it impossible to split them into three datasets.

One could think about rather complex data augmentation techniques like cutting out the rare vocalizations and inserting them into multiple different background noises.
This newly generated synthetic data could then be split.
But this data scaling can still be employed after a first successful deployment of BirdBox.

Since this project is in its early stages, it has been decided to just consider every species with **less than 100 annotations** as **rare**.
For Birds-With-Labels, those rare species are only used for training.
The metrics during training and hyperparameter tuning (evaluation dataset) and for testing (test dataset) are computed only with splittable, often occurring species.
The splits that Just-Bird uses were made later and fold rare species into the unknown class (`????`) before splitting, so their boxes are split like all others. For Just-Bird every box is a `bird` anyway.

### Clipping

Within each dataset, the amount of labels of a species has been reduced if it exceeded **10,000**. This applies to both models. Bird-Sounds-Global, used only by Just-Bird, was not clipped.
However, this **limit** is **soft** because we can only discard or keep entire three second clips.
If a spectrogram contains a species which should be clipped, but also a rare one, the entire spectrogram is kept.
Thus adding another annotation above 10,000.

### Increase of Annotation Amount

Additionally, the amount of annotations in the dataset exceed the amount of original annotations by far due to two reasons:

- 50% overlap leads to label duplicates within the same dataset-split
- 3 second clips cut long annotations (e.g. 3 minutes) into multiple small ones

This behavior is intended. See [How it works](../how-it-works/overview.md).

---

## ID to Species Mapping Compatibility

Because BirdBox is capable of running different models for the detection, it has to manage multiple different id to species mappings.
The possible mappings are defined in [src/config.py](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L17){ target="_blank" rel="noopener noreferrer" }.
Those mappings directly refer to the utilized conf.yaml during the training of the corresponding YOLO-model.

The class-id decoding depends on the selected mapping.
This mapping is set like this:

- CLI: pass explicit `--species-mapping` ([details](../data/audio-and-model.md#species-mapping-species-mapping))
- `onnx_export.py` without `--species-mapping`: mapping is inferred from the model file name ([details](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L889){ target="_blank" rel="noopener noreferrer" })

!!! danger "Mapping/Model Mismatch"
    If the species mapping does not match the selected model, the species labels in the output will be silently invalid. Always pass the `--species-mapping` value that corresponds to the model you are running.

---

## References

<a id="ref-roberts"></a>
**[24]** Roberts, D. R., et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography*, 40(8):913–929.
