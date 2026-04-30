## Taxonomy Conversion

To contribute annotations to Xeno-Canto, users must follow the **Annota-JSON** format specified in [XC Article 321](https://xeno-canto.org/article/321).

### The Taxonomic Gap

BirdBox builds upon datasets from the **BirdNET ecosystem** such as [Northeastern United States](https://zenodo.org/records/7079380), [Southwestern Amazon Basin](https://zenodo.org/records/7079124) and [Hawaii](https://zenodo.org/records/7078499).
The species labels used in those datasets follow the **Cornell/Clements (eBird)** taxonomy ([Clements 2021](https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/august-2021/2021-citation-checklist-download/)), whereas Xeno-Canto nowadays uses **AviList** ([avilist.org](https://www.avilist.org/)).

### The Solution: Custom Mapping

To provide a separate output format that complies with **Annota-JSON**, BirdBox uses the generated mapping `Cornell-to-AviList-mapping.json`.
The python script `create_cornell_to_AviList_mapping.py` uses the **[AviList v2025 - Extended](https://www.avilist.org/checklist/v2025/)** workbook to generate a mapping from **Clements 2021** to **AviList**, making it possible to contribute annotations to Xeno-Canto.

### Species Splits and Merges

Taxonomies change over time. When comparing Cornell/Clements (used by BirdNET datasets) to AviList, you can encounter:

- **Splits (one Cornell code → multiple AviList “species” rows)**: the AviList workbook may contain multiple rows for the same `Species_code_Cornell_Lab` that disagree on `Scientific_name` and/or `English_name_AviList`.  
  In that case, `create_cornell_to_AviList_mapping.py` will **warn on stderr** and then still **keep the first row found** for that Cornell code when writing `Cornell-to-AviList-mapping.json`.

- **Merges / synonyms (multiple Cornell codes → one AviList scientific name)**: multiple Cornell codes can end up pointing to the same AviList `Scientific_name`.  
  This is not necessarily wrong, but the script prints an **informational note on stderr** listing the affected scientific names and the set of Cornell codes.

In practice, for the current `AviList-v2025-11Jun-extended.xlsx`, the script often reports **no split/merge warnings**, so this typically does not materially affect BirdBox’s workflow. If AviList updates introduce such cases later, these warnings serve as a heads-up that the mapping contains ambiguities and may need manual review.
