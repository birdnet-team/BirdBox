# Models and Metrics

This chapter describes the available models, gives insight into the underlying dataset splits and demonstrates their performance based on individual evaluation and testsets with various metrics.

BirdBox accepts YOLO models in multiple formats such as `.pt`, `.onnx` and `.engine`.
Trained models for this task can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Alternatively, you can train your own model on a custom dataset by using the code available in the [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } repository.

!!! info "Restricted Access"
    BirdBox-Train is currently only accessible to members of the BirdNET-Team.

To run benchmarks with a model of your choice and a custom dataset see [CLI Reference](../cli/index.md).

---

## Available Models

The available models are trained with datasets from Zenodo.org.
The utilized datasets are:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }

A model has been trained for each of those datasets individually.
Additionally, there are two models that were trained on an orchestration of all five datasets:

- [All-in-One.pt](all-in-one.md)
- [Just-Bird.pt](just-bird.md)

---

## Dataset Splits

Each split of a dataset aims to disjunctively divide the data into 70% training, 15% validation, and 15% test data.
The counted quantity is the number of labels within a split, not the number of files.
For example, a file with five aldfly annotations is counted as five, not one.

The exact quantities for each model can be examined in the corresponding `Species Distribution Across Splits` sections (e.g. see [All-in-One](all-in-one.md#species-distribution-across-splits) or [Just-Bird](just-bird.md#species-distribution-across-splits)).

### Limitations

There exist multiple reasons why this 70/15/15 split is not trivial:

1. it is strictly forbidden to split overlapping data into different data-splits (but 50% overlap by default)
2. temporally adjacent recordings may record the same bird with the same background noise twice [[1]](#ref-roberts)
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
Those rare species are only used for training.
The metrics during training and hyperparameter tuning (evaluation dataset) and for testing (test dataset) are computed only with splittable, often occurring species.

### Clipping

The amount of labels within a species has been reduced if the total amount of labels exceeded **10,000**.
However, this **limit** is **soft** because we can only discard or keep entire three second clips.
If a spectrogram contains a species which should be clipped, but also a rare one, the entire spectrogram is kept.
Thus adding another annotation above 10,000.

### Increase of Annotation Amount

Additionally, the amount of annotations in the dataset exceed the amount of original annotations by far due to two reasons:

- 50% overlap leads to label duplicates within the same dataset-split
- 3 second clips cut long annotations (e.g. 3 minutes) into multiple small ones

This behavior is intended; see [How it works](../how-it-works/index.md).

---

## ID to Species Mapping Compatibility

Because BirdBox is capable of running different models for the detection, it has to manage multiple different id to species mappings.
The possible mappings are defined in [src/config.py](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L19){ target="_blank" rel="noopener noreferrer" }.
Those mappings directly refer to the utilized conf.yaml during the training of the corresponding YOLO-model.

The class-id decoding depends on the selected mapping.
This mapping is set like this:

- CLI: pass explicit `--species-mapping` ([details](../cli/index.md))
- Streamlit: mapping is inferred from model file name ([details](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L1901){ target="_blank" rel="noopener noreferrer" })

!!! danger "Mapping/Model Mismatch"
    If the species mapping does not match the selected model, the species labels in the output will be silently invalid. Always pass the `--species-mapping` value that corresponds to the model you are running.

---

## References

<a id="ref-roberts"></a>
**[1]** Roberts, D. R., Bahn, V., et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography*, 40(8):913–929.

