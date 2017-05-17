# probably won't need all of these
# import time
import argparse
# from itertools import compress
# import random
import os
# import warnings
# import pandas as pd
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
    os.makedirs(savepath + name + '/')
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
        'ratio_peak_13', 'ratio_peak_23', 'snr', 'msm']

# EMBL features:
# features = ['chaos', 'spatial', 'spectral', 'msm']

print('using following features:\n')
print(features)

# HERE STARTS Percolator

# Add columns that Percolator needs

# Split by target
# Sample eq. number of decoys
# Send to Percolator
# Read results
# Aggregate results
# Write results
