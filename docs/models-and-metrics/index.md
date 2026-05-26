# Models and Metrics

This chapter describes the available models and demonstrates their performance based on individual testsets with various metrics.

BirdBox accepts YOLO models in multiple formats such as `.pt`, `.onnx` and `.engine`.
Trained models for this task can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Alternatively, you can train your own model on a custom dataset by using the code available in the [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } repository (currently only accessible from within the BirdNET-Team).

## Available Models

The available models are trained with datasets from Zenodo.org.
The utilized dataset are:

- [Southwestern Amazon Basin](https://doi.org/10.5281/zenodo.7079124){ target="_blank" rel="noopener noreferrer" }
- [Island of Hawai'i](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }
- [Northeastern United States](https://doi.org/10.5281/zenodo.7079380){ target="_blank" rel="noopener noreferrer" }
- [Southern Sierra Nevada mountain range](https://doi.org/10.5281/zenodo.7525805){ target="_blank" rel="noopener noreferrer" }
- [Western United States](https://doi.org/10.5281/zenodo.7050014){ target="_blank" rel="noopener noreferrer" }

Each dataset has it's corresponding model trained with a 70% train-, 15% validation- and 15% test-dataset.
For more detail, see the individuel model descriptions (e.g. [Northeastern United States](northeastern-us.md)).

Additionally, there are two models that were trained on an orchestration of all five datasets:

- [All-in-One.pt](all-in-one.md)
- [Just-Bird.pt](just-bird.md)


## ID to Species Mapping Compatibility

Because BirdBox is capable of running different models for the detection, it has to manage multiple different id to species mappings.
The possible mappings are defined in [src/config.py](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L19){ target="_blank" rel="noopener noreferrer" }.
Those mappings directly refer to the utilized conf.yaml during the training of the corresponding YOLO-model.

The class-id decoding depends on the selected mapping.
This mapping is set like this:

- CLI: pass explicit `--species-mapping` ([details](../cli/index.md))
- Streamlit: mapping is inferred from model file name using `config.get_species_mapping_for_model(...)`

If model and mapping disagree, species labels in output are invalid.

## Best Practice

For reproducibility, keep the following tuple together in experiment records:

- model file path
- dataset file path
- mapping name
- confidence threshold
- song-gap
- NMS IoU

