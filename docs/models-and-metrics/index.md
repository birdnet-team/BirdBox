# Models and Metrics

This chapter describes the available models and demonstrates their performance based on individual testsets with various metrics.

BirdBox accepts YOLO models in multiple formats such as `.pt`, `.onnx` and `.engine`.
Trained models for this task can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Alternatively, you can train your own model on a custom dataset by using the code available in the [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } repository (currently only accessible from within the BirdNET-Team).

To run benchmarks with a model of your choice and a custom dataset see [CLI Reference](../cli/index.md).

---

## Available Models

The available models are trained with datasets from Zenodo.org.
The utilized dataset are:

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

Each split of a dataset aims to disjunctively divide the data into 70% training-, 15% validation- and 15% test-data.
However, there exist multiple reasons why this split is not trivial:

1. it is strictly forbidden to split overlapping data into different data-splits (but 50% overlap by default)
2. temporal adjacent recordings may record the same bird with the same background noice twice [[2]](#ref-roberts)
3. single three second clips often contain multiple annotations from different species

The first and second problem can be solved by dividing the soundscape recordings into one minute chunks before any data is split.
Three second clips with overlap are then only generated within these one minute files.
The resulting three second clips can then only be sorted into a mutual dataset-split.
Therefore, no direct overlap between splits is possible and the problem of temporal data leakage is mitigated.

However, combined with the third problem, this limits the capabilities of the dataset-splitter.
From now on, only one minute chunks can be sorted into a distinct split.
Combined with the fact, that some rare species just occur in about one minute of the given data, makes it impossible to split them into three datasets.

As the dataset splitting would be strongly affected by rare species, every species that has **less than 100 annotations** is **considered rare** and is just used for training.
The metrics during training and hyperparameter tuning (evaluation-dataset) and for testing (test-dataset) are just computed with robust, often occuring species.
If a model performs nicely for those species, we can expect the rare species to be detected at least reasonable [[1]](#ref-geng).

---

## ID to Species Mapping Compatibility

Because BirdBox is capable of running different models for the detection, it has to manage multiple different id to species mappings.
The possible mappings are defined in [src/config.py](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L19){ target="_blank" rel="noopener noreferrer" }.
Those mappings directly refer to the utilized conf.yaml during the training of the corresponding YOLO-model.

The class-id decoding depends on the selected mapping.
This mapping is set like this:

- CLI: pass explicit `--species-mapping` ([details](../cli/index.md))
- Streamlit: mapping is inferred from model file name ([details](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L1901){ target="_blank" rel="noopener noreferrer" })

If the species mapping does not fit to the selected model, then the species labels in the output will be invalid.

---

## References

<a id="ref-geng"></a>
**[1]** Geng, C., Huang, S. J., and Chen, S. (2020). "Recent advances in open set recognition: A survey." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(10):3614–3631.

<a id="ref-roberts"></a>
**[2]** Roberts, D. R., Bahn, V., et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography*, 40(8):913–929.

