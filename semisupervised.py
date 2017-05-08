import time
import argparse
from itertools import compress
import random
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics

warnings.filterwarnings("ignore")
np.random.seed(42)

"""
This script takes in a csv file which is the export of a sm-engine search done
with the additional feature extraction. From this output (i.e. annotations),
we build a linear SVM model which is used to re-score all the annotations.
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
print('target adducts are {}\n'.format(data.targets[0]))
data['target'] = [1 if data.adduct[r] in data.targets[0] else 0 for r in range(len(data))]
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

print('\nsplitting and scaling the data\n')

# splitting the data:
# all the target hits
data_pos = data[data.target == 1]
# same number of decoys
neg_idx = data[data.target == 0].index.values
np.random.seed(42)
np.random.shuffle(neg_idx)
# data_neg = data.loc[neg_idx[:len(data_pos)]]
# data_out = data.loc[neg_idx[len(data_pos):]]
# twicethe targets
data_neg = data.loc[neg_idx[:len(data_pos)*2]]
data_out = data.loc[neg_idx[len(data_pos)*2:]]

# data is now all targets + same number of decoys
data = pd.concat([data_pos, data_neg])

# identifier columns
X = data[features + ['sf_name', 'sf', 'adduct', 'target', 'above_fdr']]
X_out = data_out[features + ['sf_name', 'sf', 'adduct', 'target', 'above_fdr']]
X_out['target'] = [0] * len(X_out)

# Scaling the data
scaler = StandardScaler()
X.loc[:, features] = scaler.fit_transform(X.loc[:, features].values)
X_out.loc[:, features] = scaler.transform(X_out.loc[:, features].values)

# When selecting the next batch of positive cases we test several FDR values
# fdrs = np.linspace(0.01, 0.60, num=60)
fdrs = np.linspace(0.01, 0.30, 30)
# initial FDR level is 10%
fdr_level = 0.10

print('starting iterative process:\n')
for it in range(10):
    # set cv scheme - to guarantee that there are positive training examples on each fold,
    # we select a random permutation of all unique sum formulas that are above and below fdr
    cv_abovefdr = np.random.permutation(data[X.above_fdr == 1].sf.unique())
    cv_others = np.random.permutation(data[X.above_fdr == 0].sf.unique())

    # for 5 folds in each iteration, we split the sf lists from above in 5.
    n_af = len(cv_abovefdr)/5
    n_o = len(cv_others)/5
    # updating the column "fold"
    for i in range(5):
        # 1/5th of the sf which are above the fdr
        test_sf = cv_abovefdr[i*n_af:(i+1)*n_af]
        # 1/5th of the sf below fdr
        test_sf = np.append(test_sf, cv_others[i*n_o:(i+1)*n_o])
        X.loc[X.sf.isin(test_sf), 'fold'] = i

    for f in range(5):
        start = time.time()

        # test and train set are defined by the folds
        # test set is only needed for the model to be applied. y_test is not used
        data_test = X[X.fold == f]
        X_test = data_test[features]
        # y_test = data_test['target']

        data_train = X[X.fold != f]
        # from the points in this training fold, we select the high confidence
        # positives + all the negatives
        X_train = data_train[(data_train.above_fdr == 1) | (data_train.target == 0)][features]
        y_train = data_train[(data_train.above_fdr == 1) | (data_train.target == 0)]['target']
        # the ratio of positive to negative labels in the fold
        log.write("Iteration {}, fold {}: {} neg to {} pos \n".format(it+1, f+1, y_train.value_counts()[0],  y_train.value_counts()[1]))

        # train the model. class_weight is balanced to heavily penalize
        # misclassifications when we have few examples for a class
        bst = LinearSVC(class_weight='balanced', random_state=42)
        bst.fit(X_train, y_train)

        # use this model to re-score the test set
        X.loc[X.fold == f, 'fold_score'] = bst.decision_function(X_test)


    # Selecting positive instances for next fold:
    # compute FDR for new score
    threshs = [get_FDR_threshold(X[X.target == 1]['fold_score'], X[X.target == 0]['fold_score'], thr=i) for i in fdrs]
    nids = [len(X[(X.target == 1) & (X.fold_score > score)]) for score in threshs]
    nids_threshs = [a > 10 for a in nids]

    # we select the threshold for the minimum of all fdr levels tested that allows for at least 10 identifications
    # thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
    nid = list(compress(nids, nids_threshs))[0]
    thresh = list(compress(threshs, nids_threshs))[0]
    fdr_level = list(compress(fdrs, nids_threshs))[0]
    # print(thresh, fdr_level, nid)

    # update X['above_fdr'] in accordance to the new score/fdr level
    X.loc[:, 'above_fdr'] = [1 if ((X.loc[i, 'fold_score'] > thresh) & X.loc[i, 'target'] == 1) else 0 for i in X.index]

    end = time.time()
    print("Iteration {} took {}s; {} ids at {} FDR".format(it+1, str(end-start).split('.')[0], nid, fdr_level))
    log.write("Iteration {} took {}s; {} ids at {} FDR".format(it+1, end-start, nid, fdr_level))
    print(" -------------------")
    log.write(" -------------------\n")


# Final model
print('training final model\n')
X_train = X[(X.above_fdr == 1) | (X.target == 0)][features]
y_train = X[(X.above_fdr == 1) | (X.target == 0)]['target']

final = LinearSVC(class_weight='balanced', random_state=42)
final.fit(X_train, y_train)

X_all = pd.concat([X, X_out])
X_all['target'] = pd.concat([X['target'], X_out.target])
X_all['final_label'] = [None] * len(X_all)

# final re-scoring of all annotations (including left out decoys)
X_all['final_score'] = final.decision_function(X_all[features])

print('un-scaling features and saving results\n')
# De-scaling the features; saving dataframe with results
to_save = X_all[['sf_name', 'sf', 'adduct', 'target'] + features + ['final_score']]
to_save[features] = scaler.inverse_transform(to_save[features])
to_save.to_csv(savepath + name + '/data/' +name+'_rescored.csv', index=False)


# Saving feature weights
importances = final.coef_
indices = np.argsort(importances[::-1])

feat_imp, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.set_title("Feature importances")
ax.bar(range(len(features)), importances[0][indices][0], align="center")
ax.set_xticks(range(len(features)))
ax.set_xticklabels([features[i] for i in indices[0]], rotation='vertical')
ax.set_xlim([-1, len(features)])
feat_imp.savefig(savepath + name + '/' +name+'_feat_importances.png')

log.close()
print('finished!')
