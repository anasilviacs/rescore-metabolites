# ReSCORE METASPACE

Take annotations obtained from the [METASPACE](http://metaspace2020.eu/) spatial metabolomics annotation engine and use a semi-supervised method to re-score them. This is achieved by taking the most confident target hits along with the decoy hits and training a classifier. This classifier is then used to re-score all of the annotations.

**IMPORTANT** Currently, the [METASPACE](http://metaspace2020.eu/) result export includes hits to the target database and the three MSM features, plus the MSM score. For this re-scoring approach to work as described in the accompanying publication, target *and* decoy hits are necessary, plus the additional features described in the paper. To use this rescoring approach, you must execute your search on a modified version of the METASPACE engine that can be found at [anasilviacs/sm-engine](https://github.com/anasilviacs/sm-engine). It should be noted that this is a transitory measure, i.e. a proof of concept rather than a definitive tool, that will be implemented in future [METASPACE](http://metaspace2020.eu/) development cycles.

### Reproducing results
To test this pipeline, we provide an example file:
[MTBLS415 exported search results](http://genesis.ugent.be/uvpublicdata/silvia/MTBLS415/120901101000.csv).

This file is the result from searching one experiment obtained from the MetaboLights repository (accession number [MTBLS415](https://www.ebi.ac.uk/metabolights/MTBLS415)) against [HMDB](http://www.hmdb.ca/) with the modified version of the METASPACE engine found in [anasilviacs/sm-engine](https://github.com/anasilviacs/sm-engine), and exported into a tab-separated file through use of the `export_search_results.py` script provided in this repository.

----

## Installation

To execute this pipeline the following tools and packages are required:

- [Percolator](https://github.com/percolator/percolator)
- Python >= 3
  - NumPy >= 1.13.3
  - Pandas >= 0.20.2

If you'd like to execute your own search, please refer to [anasilviacs/sm-engine](https://github.com/anasilviacs/sm-engine/wiki) for the modified engine's installation and usage instructions.

## Usage

##### Exporting search results
After running a search with the modified version of the METASPACE engine found in [anasilviacs/sm-engine](https://github.com/anasilviacs/sm-engine/wiki), the results must be exported as a tab-separated, `.tsv` file. To do so, use the `export_search_results.py` script as follows:

```
python export_search_results.py [dataset name] [path to tsv file]

"dataset name" is the name of the dataset of interest in the engine database
"path to tsv file" is where the exported results will be stored
```
An example of what an exported file looks like can be downloaded from our servers through the following link:
[MTBLS415 exported search results](http://genesis.ugent.be/uvpublicdata/silvia/MTBLS415/120901101000.csv). This file looks as follows:

```
formula_db	db_ids	sf_name	sf	adduct	chaos	spatial	spectral	image_corr_01	image_corr_02	image_corr_03	image_corr_12	image_corr_13	image_corr_23	snr	percent_0s	peak_int_diff_0	peak_int_diff_1	peak_int_diff_2	peak_int_diff_3	quart_1	quart_2	quart_3	ratio_peak_01	ratio_peak_02	ratio_peak_03	ratio_peak_12	ratio_peak_13	ratio_peak_23	percentile_10	percentile_20	percentile_30	percentile_40	percentile_50	percentile_60	percentile_70	percentile_80	percentile_90	fdr	isocalc_sigma	isocalc_charge	isocalc_pts_per_mz	first_peak_mz	targets	target	msm
HMDB	{31173}	[u'6-Hydroxy-1H-indole-3-acetamide']	C10H10N2O2	+Nd	0.0	0.0	0.0	0.0	0.0	0.0	-0.000410693	0.0	0.0	0.0	0.0	-0.700482	0.955152	-0.0837069	-0.614721	0.0	0.0	0.0	0.0	0.0	0.0	3.4445	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.001238-1	4039	331.982499	[u'-H', u'+Cl']	0	0.0
HMDB	{31173}	[u'6-Hydroxy-1H-indole-3-acetamide']	C10H10N2O2	+Ru	0.0	0.0	0.878995	-0.00128352	-0.00047319	-0.00179479	-0.00126883	-0.00481261	-0.00177425	0.0218785	0.999519	-0.888743	0.491929	-0.0492674	0.545825	0.0	0.0	0.0	0.108129	1.01837	0.0624696	9.41809	0.577733	0.0613429	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.001238	-1	4039	285.982374	[u'-H', u'+Cl']	0	0.0
HMDB	{31173}	[u'6-Hydroxy-1H-indole-3-acetamide']	C10H10N2O2	+Th	0.0	0.0	0.0	0.0	0.0	-0.000240442	0.0	0.0	0.0	0.0155062	0.99976	-0.285585	-0.00726385	-0.107809	0.6996660.0	0.0	0.0	0.0	0.0	1.00407	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.001238-1	4039	422.112831	[u'-H', u'+Cl']	0	0.0
HMDB	{31173}	[u'6-Hydroxy-1H-indole-3-acetamide']	C10H10N2O2	+Rh	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	-0.000339876	0.0	0.0	-0.994127	-0.00726385	0.747369	0.512321	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.64987	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.001238-1	4039	292.98
```

#### Rescoring metabolite annotations
Execute `semisupervised.py` on the exported tab-separated file::

```
python semisupervised.py [path to tsv file] [-d] [-k]

 "path to tsv file" is where the exported tsv file is
 "-d" is a flag which should be used for getting the decoys' q-values
 "-k" is a flag which should be used to keep the intermediate files
```

This version of the re-scoring method saves a comma-separated, `results.csv` file with the final (median, called `combined` in the file) q-value for each metabolite ion. It is possible to also save the decoy hits' q-values, although give the nature of the sampling process it is unlikely that all of them will have values, and the intermediate q-values (i.e. the ones obtained in each iteration of the method). The simplest version of the output file looks as follows:

```
SpecId,combined
C10H10N7O2S-H,0.5886835
C10H10S3-H,0.5578665
C10H11ClFN5O3-H,0.7680545
C10H12ClN5O3-H,0.736791
```

## License

This project is licensed under Apache 2.0 license.
