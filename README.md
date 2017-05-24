# ReSCORE METASPACE

Take annotations obtained from the METASPACE spatial metabolomics annotation engine and use a semi-supervised method to re-score them.

This is achieved by taking the most confident target hits along with the decoy hits and training a classifier. This classifier is then used to re-score all of the annotations.

This version uses [Percolator](https://github.com/percolator/percolator).

### install

Requirements:
- numpy
- pandas
- [Percolator](https://github.com/percolator/percolator)

#### note
for a re-scoring approach to work on target-decoy searches, both targets and decoys must be obtained from the [METASPACE engine](https://github.com/METASPACE2020/sm-engine). Furthermore, improvements are obtained if additional features are extracted from the spectra-metabolite matches. Currently, to do so, you need to run a modified version of the engine that can be found at [this fork](https://github.com/anasilviacs/sm-engine/tree/extra_features), which relies on an outdated version of the engine (v0.4).

## re-scoring annotations

After running a search, the results must be exported as a `.csv` file. To do so, use `export_search_results.py` as:

```
python export_search_results.py [dataset name] [path to csv file]

"dataset name" is the name of the dataset of interest in the engine database
"path to csv file" is where the exported results will be stored
```

With the exported file, run `semisupervised.py`:

```
python semisupervised.py [path to csv file] [-d] [-k]
 "path to csv file" is where the exported csv file is
 "-d" is a flag which should be used for getting the decoys' q-values
 "-k" is a flag which should be used to keep the intermediate files
```

This version of the re-scoring method saves a `csv` with the q-values for each hit over 10 iterations plus the average. There is a possibility to also save the decoy hits' q-values, although give the nature of the sampling process it is unlikely that all of them will have values.
