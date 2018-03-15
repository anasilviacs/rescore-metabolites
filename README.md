# ReSCORE METASPACE

Take annotations obtained from the [METASPACE](http://metaspace2020.eu/) spatial metabolomics annotation engine and use a semi-supervised method to re-score them.

This is achieved by taking the most confident target hits along with the decoy hits and training a classifier. This classifier is then used to re-score all of the annotations.

#### note
for the re-scoring approach to work on target-decoy searches, additional features for both targets and decoys must be obtained and stored for each match. Currently, to do so, you need to run a modified version of the engine that can be found [here](https://github.com/anasilviacs/sm-engine). This is a modified version of version 0.4 of the annotation engine. 

This rescoring pipeline uses [Percolator](https://github.com/percolator/percolator).

### install

Requirements:
- numpy
- pandas
- [Percolator](https://github.com/percolator/percolator)


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

This version of the re-scoring method saves a `csv` with the q-values for each hit over 10 iterations plus the median. There is a possibility to also save the decoy hits' q-values, although give the nature of the sampling process it is unlikely that all of them will have values.

## License

This project is licensed under Apache 2.0 license.
