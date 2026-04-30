## Taxonomy Conversion

To contribute annotations to Xeno-Canto, users must follow the Annota-JSON format specified in [XC Article 321](https://xeno-canto.org/article/321).

### The Taxonomic Gap

BirdBox builds upon datasets from the BirdNET ecosystem such as [Northeastern United States](https://zenodo.org/records/7079380), [Southwestern Amazon Basin](https://zenodo.org/records/7079124) and [Hawaii](https://zenodo.org/records/7078499).
The species labels used in those datasets follows the [Clements 2021](https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/august-2021/2021-citation-checklist-download/) taxonomy, whereas Xeno-Canto nowadays uses the [AviList](https://www.avilist.org/).

### The Solution: Custom Mapping

To provide a separate output format that complies with the Annota-JSON format, BirdBox uses the created [`Cornell-to-AviList-mapping.json`](taxonomies/Cornell-to-AviList-mapping.json).
The python script [`create_cornell_to_AviList_mapping.py`](taxonomies/create_cornell_to_AviList_mapping.py) uses the **[AviList v2025 - Extended](https://www.avilist.org/checklist/v2025/)** to generate a mapping from Clements 2021 to this AviList. Thus making it possible to contribute annotations to Xeno-Canto.
