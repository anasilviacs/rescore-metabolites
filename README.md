# ReSCORE METASPACE

Take annotations obtained from the METASPACE spatial metabolomics annotation engine and use a semi-supervised method to re-score them.

...

### install

Requirements:
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

#### note
for a re-scoring approach to work on target-decoy searches, both targets and decoys must be obtained from the [METASPACE engine](https://github.com/METASPACE2020/sm-engine). Furthermore, improvements are obtained if additional features are extracted from the spectra-metabolite matches. Currently, to do so, you need to run a modified version of the engine that can be found at [this fork](https://github.com/anasilviacs/sm-engine/tree/extra_features), which relies on an outdated version of the engine (v0.4).

## re-scoring annotations

After running a search, the results must be exported as a `.csv` file. To do so, use `export_search_results.py` as:

```
python export_search_results.py [dataset name] [path to csv file]

__dataset name__ is the name of the dataset of interest in the engine database
__path to csv file__ is where the exported results will be stored
```

With the exported file, run `semisupervised.py`:

```
python semisupervised.py [path to csv file]
 __path to csv file__ is where the exported csv is
```

This version of the re-scoring method saves a `csv` identical to the one supplied with an additional "final score" column, a figure with the model's feature importances and a `log.txt` which can be used to monitor the iterative re-scoring procedure.
