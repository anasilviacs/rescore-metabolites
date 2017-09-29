import argparse
from itertools import compress
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
# np.random.seed(42)

"""
This script takes in a csv file which is the export of a sm-engine search done
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

def get_FDR_threshold(pos, neg, thr=0.10):
    """
    Gets the score threshold that permits a defined FDR. FDR is calculated as
    ((#decoys above threshold/#decoys) / (#targets above threshold/#targets).
    :param pos: pandas DF column [label] elements where [label] is [positive]
    :param neg: pandas DF column [label] elements where [label] is [negative]
    :param thr: the permitted FDR. default 10%
    :return the score of the positive instance that allows the specified
                            percentage of false discoveries to be scored higher
    """
    # order scores in ascending order
    spos = sorted(pos)
    sneg = sorted(neg)
    # counters
    c_pos = 0
    c_neg = 0
    # total number of each
    len_pos = len(spos)
    len_neg = len(sneg)
    while True:
        if c_pos >= len_pos:
            break
        if c_neg >= len_neg:
            break
        d = (1.0 * len_neg-c_neg)/len_neg
        t = (1.0 * len_pos-c_pos)/len_pos
        # sys.stdout.write len_pos, c_pos, t
        fdr = d/t
        # sys.stdout.write(fdr, c_pos, c_neg)
        if fdr < thr:
            return spos[c_pos]
        if spos[c_pos] < sneg[c_neg]:
            c_pos += 1
        elif (c_pos + 1 < len_pos and spos[c_pos] == spos[c_pos + 1]):
            c_pos += 1
        else:
            c_neg += 1
    return 999

# Loading data
sys.stdout.write('loading data...\n')
name = args.dataset.split('/')[-1].rstrip('.csv')
data = pd.read_csv(args.dataset, sep='\t')

# Output directory
# savepath = args.dataset.split('/')[0] + '/rescored/' + args.dataset.split('/')[-2] + '/'
savepath = args.dataset.split('/')[0] + '/rescored/' + args.dataset.split('/')[-2] + '/' + name + '/'
if not os.path.isdir(savepath): os.mkdir(savepath)

sys.stdout.write('dataset {} loaded; results will be saved at {}\n'.format(name, savepath))

# Adding columns of interest to the dataframe
target_adducts = [t.lstrip('[').lstrip('"').lstrip("u'").rstrip(",").rstrip(']').rstrip("\'") for t in data.targets[0].split(' ')]
sys.stdout.write('target adducts are {}\n'.format(target_adducts))

data['target'] = [1 if data.adduct[r] in target_adducts else 0 for r in range(len(data))]
data['above_fdr'] = [1 if data.fdr[r] in [0.01, 0.05, 0.10] else 0 for r in range(len(data))]
data['msm'] = data['chaos'] * data['spatial'] * data['spectral']
ids_init = data.above_fdr.value_counts()[1]
sys.stdout.write('there are {} targets and {} decoys. of all the targets, {} are above the 10% FDR threshold.\n'.format(data.target.value_counts()[1], data.target.value_counts()[0], ids_init))

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

# fdrs = np.linspace(0.01, 0.30, 30)
niter = 20

agg_df = pd.DataFrame()

# Split by target
for target in target_adducts:
    sys.stdout.write('processing target adduct {}. initial #ids at 10% FDR: {}\n'.format(target,np.sum(data[data.adduct == target].above_fdr)))
    data_pos = data[data.adduct == target]

    # build a decoy DataFrame
    data_neg = pd.DataFrame(columns=data_pos.columns)
    for sf in np.unique(data_pos.sf):
        tmp = data[(data.target == 0) & (data.sf == sf)]
        if len(tmp) > 0:
            data_neg = data_neg.append(tmp.iloc[np.random.randint(0, len(tmp), niter),:])
        else: continue
    data_neg['n'] = (list(np.arange(1,niter+1)) * int((len(data_neg)/niter)+1))[:len(data_neg)]

    tmp = pd.DataFrame(index=data_pos.SpecId)
    tmp_dec = pd.DataFrame(index=data_neg.SpecId)

    for decoy in range(1,niter+1):
        sys.stdout.write('iteration #{}\n'.format(decoy))

        data_perc = pd.concat([data_pos, data_neg[data_neg.n == decoy]])

        data_perc['Label'] = data_perc['Label'].astype(int)
        data_perc['ScanNr'] = data_perc['ScanNr'].astype(int)

        """
        threshs = [get_FDR_threshold(data_perc[data_perc.target == 1]['msm'], data_perc[data_perc.target == 0]['msm'], thr=i) for i in fdrs]
        nids = [len(data_perc[(data_perc.target == 1) & (data_perc.msm > score)]) for score in threshs]
        nids_threshs = [a > 20 for a in nids]

        # we select the threshold for the minimum of all fdr levels tested that allows for at least 10 identifications
        # thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
        nid = list(compress(nids, nids_threshs))[0]
        thresh = list(compress(threshs, nids_threshs))[0]
        fdr_level = list(compress(fdrs, nids_threshs))[0]


        #threshs = [get_FDR_threshold(data_perc[data_perc.target == 1]['msm'], data_perc[data_perc.target == 0]['msm'], thr=i) for i in fdrs]
        #thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
        #fdr_level = list(compress(fdrs, [t != 999 for t in threshs]))[0]
        """

        data_perc = data_perc[['SpecId', 'Label', 'ScanNr'] + features + ['Peptide', 'Proteins']]

        pin_path = os.path.join(savepath, "{}_{}.pin".format(target, decoy))
        pout_path = os.path.join(savepath, "{}_{}.pout".format(target, decoy))
        if args.decoys: pout_decoys = os.path.join(savepath, "{}_{}_decoys.pout".format(target, decoy))

        data_perc.to_csv(pin_path, index=False, sep='\t')

        # Send to Percolator
        fdr_level = 0.1
        if args.decoys:
            command = "percolator -v 0 -t {} -F {} -U {} -r {} -B {}\n".format(fdr_level, fdr_level, pin_path, pout_path, pout_decoys)
        else:
            command = "percolator -v 0 -t {} -F {} -U {} -r {}\n".format(fdr_level, fdr_level, pin_path, pout_path)

        sys.stdout.write('executing: {}'.format(command))
        os.system(command)

        # Check if Percolator was able to run
        if os.path.isfile(pout_path):
            # Read results
            perc_out = pd.read_csv(pout_path, sep='\t')
            tmp[str(decoy)] = [perc_out[perc_out.PSMId == sf]['q-value'].values[0] for sf in tmp.index]
            if args.decoys:
                perc_out = pd.read_csv(pout_decoys, sep='\t')
                tmp_dec[str(decoy)] = [None]*len(tmp_dec)
                for sf in perc_out.PSMId:
                    tmp_dec.loc[sf, str(decoy)] = perc_out[perc_out.PSMId == sf]['q-value'].values[0]
        else:
            sys.stdout.write("Percolator wasn't able to re-score adduct {} (iteration {})\n".format(target, decoy))
            if not args.keep: os.remove(pin_path)
            tmp[str(decoy)] = [None]*len(tmp) # unnecessary, i think
            continue

        if not args.keep:
            os.remove(pin_path)
            os.remove(pout_path)
            if args.decoys: os.remove(pout_decoys)

    # take median q-value per hit
    tmp['combined'] = tmp.median(axis=1)
    if args.decoys: tmp_dec['combined'] = tmp_dec.median(axis=1)
    sys.stdout.write("#ids at FDR < 10%: {}\n".format(len(tmp[tmp['combined'] <= 0.1])))

    # aggregate results for all adducts
    agg_df = pd.concat([agg_df, tmp])
    if args.decoys: decoy_df = pd.concat([decoy_df, tmp_dec])

ids_end = len(agg_df[agg_df['combined'] <= 0.1])

sys.stdout.write('final number of identifications at 10% FDR: {} ({}% difference)\n'.format(ids_end, (1.0*ids_end/ids_init)*100))

# Write out results
if args.decoys:
    agg_df = pd.concat([agg_df, decoy_df])

agg_df.to_csv(savepath +'/results.csv', index=True)
