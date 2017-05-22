# probably won't need all of these
# import time
import argparse
from itertools import compress
# import random
import os
import warnings
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# # from sklearn.grid_search import GridSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
# from sklearn import metrics

warnings.filterwarnings("ignore")
np.random.seed(42)

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

args = parser.parse_args()

print("\n*ReSCORE METASPACE*\n")

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
        # print len_pos, c_pos, t
        fdr = d/t
        # print(fdr, c_pos, c_neg)
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
print('loading data...\n')
name = args.dataset.split('/')[-1].rstrip('.csv')
data = pd.read_csv(args.dataset, sep='\t')

# Output directories
savepath = args.dataset.split('/')[0] + '/tests/' + args.dataset.split('/')[-2] + '/'
if not os.path.exists(savepath + name + '/'):
    print('creating folders... \n')
    os.makedirs(savepath + name + '/')
    os.makedirs(savepath + name + '/tmp/')
    os.makedirs(savepath + name + '/data/')

log = open(savepath + name + '/' +name+ '_log.txt', 'w')

print('dataset {} loaded; results will be saved at {}\n'.format(name, savepath))

# Adding columns of interest to the dataframe
target_adducts = [t.lstrip('[').lstrip('"').lstrip("u'").rstrip(",").rstrip(']').rstrip("\'") for t in data.targets[0].split(' ')]
print('target adducts are {}\n'.format(target_adducts))

data['target'] = [1 if data.adduct[r] in target_adducts else 0 for r in range(len(data))]
data['above_fdr'] = [1 if data.fdr[r] in [0.01, 0.05, 0.10] else 0 for r in range(len(data))]
data['msm'] = data['chaos'] * data['spatial'] * data['spectral']
print('there are {} targets and {} decoys. of all the targets, {} are above the 10% FDR threshold.\n'.format(data.target.value_counts()[1], data.target.value_counts()[0], data.above_fdr.value_counts()[1]))

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

print('using following features:\n')
print(features)

# HERE STARTS Percolator

# Add columns that Percolator needs: ['SpecId', 'Label', 'ScanNr'] + features + ['Peptide', 'Proteins']
data['SpecId'] = data['sf'] + data['adduct']
data['Label'] = [1 if data.target[r]==1 else -1 for r in range(len(data))]
data['ScanNr'] = np.arange(len(data))
data['Peptide'] = ['R.'+sf+'.T' for sf in data['sf']]
data['Proteins'] = data['sf']

fdrs = np.linspace(0.01, 0.30, 30)

# Split by target
for target in target_adducts:
    print('\nprocessing target adduct {}\n'.format(target))
    data_pos = data[data.adduct == target]

    for decoy in range(1):
        print('iteration #{}'.format(decoy))
        data_neg = pd.DataFrame(columns=data_pos.columns)
        for sf in np.unique(data_pos.sf):
            tmp = data[(data.target == 0) & (data.sf == sf)]
            if len(tmp) > 0:
                data_neg = data_neg.append(tmp.iloc[np.random.randint(0, len(tmp)),:])
            else: continue

        """
        neg_idx = data[data.target == 0].index.values
        np.random.seed(42)
        np.random.shuffle(neg_idx)
        data_neg = data.loc[neg_idx[:len(data_pos)]]
        """
        data_perc = pd.concat([data_pos, data_neg])
        # data_perc['Label'] = data_perc['Label'].fillna(0)
        data_perc['Label'] = data_perc['Label'].astype(int)
        data_perc['ScanNr'] = data_perc['ScanNr'].astype(int)

        # print(data_perc.head())
        # print(data_perc.Label.value_counts())
        """
        threshs = [get_FDR_threshold(data_perc[data_perc.target == 1]['msm'], data_perc[data_perc.target == 0]['msm'], thr=i) for i in fdrs]
        nids = [len(data_perc[(data_perc.target == 1) & (data_perc.msm > score)]) for score in threshs]
        nids_threshs = [a > 20 for a in nids]

        # we select the threshold for the minimum of all fdr levels tested that allows for at least 10 identifications
        # thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
        nid = list(compress(nids, nids_threshs))[0]
        thresh = list(compress(threshs, nids_threshs))[0]
        fdr_level = list(compress(fdrs, nids_threshs))[0]
        """
        threshs = [get_FDR_threshold(data_perc[data_perc.target == 1]['msm'], data_perc[data_perc.target == 0]['msm'], thr=i) for i in fdrs]
        # thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
        # fdr_level = list(compress(fdrs, [t != 999 for t in threshs]))[0]

        data_perc = data_perc[['SpecId', 'Label', 'ScanNr'] + features + ['Peptide', 'Proteins']]

        pin_path = os.path.join(savepath, name, "{}_{}.pin".format(target, decoy))
        pout_path = os.path.join(savepath, name, "{}_{}.pout".format(target, decoy))

        data_perc.to_csv(pin_path, index=False, sep='\t')

        # Send to Percolator
        fdr_level = 0.1
        command = "percolator -v 0 -t {} -F {} -U {} > {}".format(fdr_level, fdr_level, pin_path, pout_path)
        print('running percolator: {}\n'.format(command))
        os.system(command)

        # Read results
        print('reading percolator results from {}\n'.format(pout_path))
        perc_out = pd.read_csv(pout_path, sep='\t')
        """
        fout2 = open(args.spec_file + ".msgfout", "w")
        with open("%s.out" % (args.spec_file + ".target.pin")) as f:
            row = f.readline()
            fout2.write('\t'.join(header) + '\t' + row)
            for row in f:
                l = row.rstrip().split('\t')
                l0 = l[0]
                tmp = '_'.join(l[0].split('_')[-6:-3])
                if tmp in id_map:
                    l[0] = id_map[tmp]
                    prot = l[5]
                    if len(l) > 6:
                        for i in range(6, len(l)):
                            prot += "|" + l[i]
                        l[5] = prot
                        fout2.write('\t'.join(pin_map[l0]) + '\t' + '\t'.join(l[:6]) + '\n')

    # Aggregate results on q-value for each adduct
    # Write all results
"""
