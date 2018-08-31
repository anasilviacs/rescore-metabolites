import argparse
from itertools import compress
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(42)
FDR_LEVEL = 0.1

"""
This script takes in a tsv file which is the export of a sm-engine search done
with the additional feature extraction. From this output (i.e. annotations),
we build a linear SVM model which is used to re-score all the annotations.
This version uses Percolator and treats each target adduct individually, trying
to reproduce the way the engine does the searches by sampling different sets of
decoys and optimizing the score for each sampled set. To aggregate all these
different scores we take the q-values instead of the score itself.
"""

parser = argparse.ArgumentParser(description='Semi-supervised improvement of sm-engine scores')
parser.add_argument('dataset', type=str, help='path to dataset')
parser.add_argument('-k', dest='keep', help='keep intermediate files (default FALSE)', action='store_true', default=False)
parser.add_argument('-d', dest='decoys', help='return decoy q-values (default FALSE)', action='store_true', default=False)

args = parser.parse_args()

sys.stdout.write("\n*ReSCORE METASPACE*\n")

# Load data
sys.stdout.write('loading data...\n')
name = args.dataset.split('/')[-1].rstrip('.tsv')
data = pd.read_csv(args.dataset, sep='\t')

# Output directory
savepath = f'data/thesis_v2/new/{name}/'
if not os.path.isdir(savepath): os.mkdir(savepath)

sys.stdout.write(f'dataset {name} loaded; results will be saved at {savepath}\n')

# Add columns of interest to the dataframe
target_adducts = [t.lstrip('[').lstrip('"').lstrip("u'").rstrip(",").rstrip(']').rstrip("\'") for t in data.targets[0].split(' ')]
sys.stdout.write('target adducts are {}\n'.format(target_adducts))
data['target'] = [1 if data.adduct[r] in target_adducts else 0 for r in range(len(data))]
data['above_fdr'] = [1 if data.fdr[r] in np.arange(0, FDR_LEVEL+0.01, 0.01) else 0 for r in range(len(data))]
data['msm'] = data['chaos'] * data['spatial'] * data['spectral']
ids_init = data.above_fdr.value_counts()[1]
sys.stdout.write('there are {} targets and {} decoys. of all the targets, {} are above the {} FDR threshold.\n'.format(data.target.value_counts()[1], data.target.value_counts()[0], ids_init, FDR_LEVEL))

# List with all the features used to build the model
features = ['chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02',
        'image_corr_03','image_corr_12', 'image_corr_13', 'image_corr_23',
        'percent_0s','peak_int_diff_0', 'peak_int_diff_1', 'peak_int_diff_2',
        'peak_int_diff_3', 'percentile_10', 'percentile_20', 'percentile_30',
        'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70',
        'percentile_80', 'percentile_90', 'quart_1', 'quart_2', 'quart_3',
        'ratio_peak_01', 'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12',
        'ratio_peak_13', 'ratio_peak_23', 'snr', 'msm', 'above_fdr']

# EMBL features:
# features = ['chaos', 'spatial', 'spectral', 'msm']

sys.stdout.write('using following features:\n')
sys.stdout.write(", ".join(features))
sys.stdout.write("\n")

# Percolator requires very specific columns:
data['SpecId'] = data['sf'] + data['adduct']
data['Label'] = [1 if data.target[r]==1 else -1 for r in range(len(data))]
data['ScanNr'] = np.arange(len(data))
data['Peptide'] = ['R.'+sf+'.T' for sf in data['sf']]
data['Proteins'] = data['sf']

if args.keep: sys.stdout.write('attention! will keep all intermediate files\n')
else: sys.stdout.write('intermediate files will be deleted\n')
if args.decoys: sys.stdout.write("attention! saving decoys' q-values\n")

niter = 10

agg_df = pd.DataFrame()
if args.decoys: decoy_df = pd.DataFrame()

# build a decoy DataFrame
# NOTE would be faster to here define sets of decoys and fetch the actual features later
data_neg = pd.DataFrame(columns=data.columns)
for sf in np.unique(data.sf):
    tmp = data[(data.target == 0) & (data.sf == sf)]
    if len(tmp) > 0:
        data_neg = data_neg.append(tmp.iloc[np.random.randint(0, len(tmp), niter),:])
    else: continue
data_neg['n'] = (list(np.arange(1,niter+1)) * int((len(data_neg)/niter)+1))[:len(data_neg)]

# Split by target & run percolator 20 times, once with each decoy set
for target in target_adducts:
    sys.stdout.write('processing target adduct {}. initial #ids at {} FDR: {}\n'.format(target, FDR_LEVEL, np.sum(data[data.adduct == target].above_fdr)))
    data_pos = data[data.adduct == target]

    tmp = pd.DataFrame(index=data_pos.SpecId)
    tmp_dec = pd.DataFrame(index=data_neg.SpecId)

    for decoy in range(1,niter+1):
        sys.stdout.write('iteration #{}\n'.format(decoy))

        data_perc = pd.concat([data_pos, data_neg[data_neg.n == decoy]])

        data_perc['Label'] = data_perc['Label'].astype(int)
        data_perc['ScanNr'] = data_perc['ScanNr'].astype(int)

        data_perc = data_perc[['SpecId', 'Label', 'ScanNr'] + features + ['Peptide', 'Proteins']]

        pin_path = os.path.join(savepath, "{}_{}.pin".format(target, decoy))
        pout_path = os.path.join(savepath, "{}_{}.pout".format(target, decoy))
        if args.decoys: pout_decoys = os.path.join(savepath, "{}_{}_decoys.pout".format(target, decoy))

        data_perc.to_csv(pin_path, index=False, sep='\t')

        # Send to Percolator
        if args.decoys:
            command = "percolator -v 0 -t {} -F {} -U {} -r {} -B {}\n".format(FDR_LEVEL, FDR_LEVEL, pin_path, pout_path, pout_decoys)
        else:
            command = "percolator -v 0 -t {} -F {} -U {} -r {}\n".format(FDR_LEVEL, FDR_LEVEL, pin_path, pout_path)

        sys.stdout.write('executing: {}'.format(command))
        os.system(command)

# Read Results: qs is a dict where spec_id are the keys and the values are the q-values
qs = {}
for target in target_adducts:
    for decoy in range(1,niter+1):
        # Check if Percolator was able to run
        pout_path = os.path.join(savepath, "{}_{}.pout".format(target, decoy))
        if not os.path.isfile(pout_path):
            sys.stdout.write("Percolator wasn't able to re-score adduct {} (iteration {})\n".format(target, decoy))
            continue
        else:
            pout = open(pout_path)
            for line in pout:
                if line.startswith('PSMId'): continue
                split_line = line.strip().split('\t')
                if split_line[0] in qs.keys():
                    qs[split_line[0]].append(float(split_line[2]))
                else:
                    qs[split_line[0]] = [float(split_line[2])]
            if args.decoys:
                pout_path = os.path.join(savepath, "{}_{}_decoys.pout".format(target, decoy))
                pout = open(pout_path)
                for line in pout:
                    if line.startswith('SpecId'): continue
                    split_line = line.strip().split('\t')
                    if split_line[0] in qs.keys():
                        qs[split_line[0]].append(float(split_line[2]))
                    else:
                        qs[split_line[0]] = [float(split_line[2])]

# Calculate median q-value & write results
out = open(savepath + 'results.csv', 'w')
out.write('sf_adduct,combined\n')

ids_end = 0

for k in qs.keys():
    v = np.median(qs[k])
    out.write(k + ',' + str(v) + '\n')
    if v <= FDR_LEVEL: ids_end += 1

sys.stdout.write('final number of identifications at {} FDR: {} ({}% difference)\n'.format(FDR_LEVEL, ids_end, (1.0*ids_end/ids_init)*100))

# Delete files:
# if not args.keep:
#     os.remove(pin_path)
#     os.remove(pout_path)
#     if args.decoys: os.remove(pout_decoys)
