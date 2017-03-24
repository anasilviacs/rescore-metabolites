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
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
warnings.filterwarnings("ignore")

"""
This script takes in a csv file which is the export of a sm-engine search done
with the additional feature extraction. From this output (i.e. annotations), we
we build a linear SVM model which is used to re-score all the annotations.
"""

parser = argparse.ArgumentParser(description='Semi-supervised improvement of sm-engine scores')
parser.add_argument('dataset', type=str, help='path to dataset')

args = parser.parse_args()


def get_FDR_threshold(pos, neg, thr=0.10):
    """
    Gets the score threshold that permits a defined FDR. FDR is calculated as
    ((#decoys abov threshold/#decoys) / (#targets above threshold/#targets).
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

name = args.dataset.split('/')[-1].rstrip('.csv')
data = pd.read_csv(args.dataset, sep='\t')
savepath = args.dataset.split('/')[0] + '/new_fdr/' + args.dataset.split('/')[-2] + '/'
print('dataset {}. result will be saved at {}'.format(name, savepath))

# adding boolean "target" column which is 1 if target, 0 if decoy. depends on adduct.
data['target'] = [1 if data.adduct[r] in ['+Na', '+K', '+H'] else 0 for r in range(len(data))]
# adding boolean above_fdr column, 1 if fdr level is 10% or less, 0 otherwise. decoys have fdr = 0
data['above_fdr'] = [1 if data.fdr[r] in [0.01, 0.05, 0.10] else 0 for r in range(len(data))]
# adding column with msm score: moc * spatial * spectral
data['msm'] = data['chaos'] * data['spatial'] * data['spectral']

# list with all the features
features = ['chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02',
        'image_corr_03','image_corr_12', 'image_corr_13', 'image_corr_23',
        'percent_0s','peak_int_diff_0', 'peak_int_diff_1', 'peak_int_diff_2',
        'peak_int_diff_3', 'percentile_10', 'percentile_20', 'percentile_30',
        'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70',
        'percentile_80', 'percentile_90', 'quart_1', 'quart_2', 'quart_3',
        'ratio_peak_01', 'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12',
        'ratio_peak_13', 'ratio_peak_23', 'snr', 'msm']

# EMBL features
# features = ['chaos', 'spatial', 'spectral', 'msm']

# splitting the data
# all the target hits:
data_pos = data[data.target == 1]
# # shuffle indices for decoy hits and split in half
# neg_idx = data[data.target == 0].index.values
# np.random.seed(42)
# np.random.shuffle(neg_idx)
# n = len(neg_idx)
# data_neg = data.loc[neg_idx[:n/2]]
# data_out = data.loc[neg_idx[n/2:]]
# instead, shuffle and select equal number of decoys (compared to targets).
# neg_idx = data[data.target == 0].index.values
# np.random.seed(42)
# np.random.shuffle(neg_idx)
# data_neg = data.loc[neg_idx[:len(data_pos)]]
# data_out = data.loc[neg_idx[len(data_pos):2*len(data_pos)]]
# Not splitting the decoys: using everything!
data_neg = data[data.target == 0]
# data is now all targets + same number of decoys
data = pd.concat([data_pos, data_neg])

# we must keep track of the sf/sf_name/adduct for the final output.
# the model will need info on if an annotation is above_fdr and if it's a target
X = data[features + ['sf_name', 'sf', 'adduct', 'above_fdr', 'target']]
# X_out = data_out[features + ['sf_name', 'sf', 'adduct', 'above_fdr', 'target']]
# X_out['target'] = [0] * len(X_out)

# the data must be scaled. will use sklearn.preprocessing.StandardScaler
scaler = StandardScaler()
X.loc[:, features] = scaler.fit_transform(X.loc[:, features].values)
# the left out decoys must be scaled by the same factor
# X_out.loc[:, features] = scaler.transform(X_out.loc[:, features].values)

# when trying out FDR values we try everything from 0.01 to 0.60 in 0.01 steps
fdrs = np.linspace(0.01, 0.60, num=60)
# initial FDR level is 10%
fdr_level = 0.10

# create the directories where outputs are saved
if not os.path.exists(savepath + name + '/'):
    os.makedirs(savepath + name + '/')
    os.makedirs(savepath + name + '/data/')
# log file
log = open(savepath + name + '/' +name+ '_log.txt', 'w')
# log.write("Initial number of identifications at 10% FDR: {} \n".format(X.above_fdr.value_counts()[1]))

# start the figure with the ROC curves
roc_fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], 'k--', label='Luck')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves for each model iteration')

for it in range(10):
    # how many annotations are above the FDR threshold:
    print("Iteration {}: # identified at {} FDR: {}".format(it+1, fdr_level, np.sum(X['above_fdr'])))
    log.write("Iteration {}: # identified at {} FDR: {} \n".format(it+1, fdr_level, np.sum(X['above_fdr'])))

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

        # use this model to re-score the test set. the
        X.loc[X.fold == f, 'fold_score'] = bst.decision_function(X_test)

    # After the 5 folds are done, every annotation has been re-scored. we save
    # the roc plots for the fold scores:
    fpr, tpr, _ = metrics.roc_curve(X.target, X.fold_score.astype('float'), pos_label=1)
    ax.plot(fpr, tpr, label='it. {}, auc={:.2f}'.format(it+1, metrics.auc(fpr, tpr)))

    # compute FDR for new threshold. we test different fdr levels, defined in fdrs
    threshs = [get_FDR_threshold(X[X.target == 1]['fold_score'], X[X.target == 0]['fold_score'], thr=i) for i in fdrs]
    # we select the threshold for the minimum of all fdr levels tested
    thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
    fdr_level = list(compress(fdrs, [t != 999 for t in threshs]))[0]
    # update X['above_fdr'] in accordance to the new score/fdr level
    X.loc[:, 'above_fdr'] = [1 if ((X.loc[i, 'fold_score'] > thresh) & X.loc[i, 'target'] == 1) else 0 for i in X.index]

    end = time.time()
    print("Iteration {} took {}s; model AUC = {}".format(it+1, end-start, metrics.auc(fpr, tpr)))
    log.write("Iteration {} took {:.3f}s; model AUC = {:.4f} \n".format(it+1, end-start, metrics.auc(fpr, tpr)))
    print(" -------------------")
    log.write(" -------------------\n")

# save roc_fig
ax.legend(loc='lower right', prop={'size':10})
roc_fig.savefig(savepath + name + '/roc_' +name+'.png')


# now for the "final model" where the left-out decoys are brought back in.
# these are only re-scored, to validate the model's performance

X_train = X[(X.above_fdr == 1) | (X.target == 0)][features]
y_train = X[(X.above_fdr == 1) | (X.target == 0)]['target']

final = LinearSVC(class_weight='balanced', random_state=42)
final.fit(X_train, y_train)


# I am using all the data now, so this isn't necessary
# X_all = pd.concat([X, X_out])
# X_all['target'] = pd.concat([X['target'], X_out.target])
# X_all['final_label'] = [None] * len(X_all)

# final re-scoring of all annotations
# X_all['final_score'] = final.decision_function(X_all[features])
X['final_score'] = final.decision_function(X[features])

"""
# computing FDR threshold for final score
threshs = [get_FDR_threshold(X_all[X_all.target == 1]['final_score'], X_all[X_all.target == 0]['final_score'], thr=i) for i in fdrs]
thresh = list(compress(threshs, [t != 999 for t in threshs]))[0]
fdr_level = list(compress(fdrs, [t != 999 for t in threshs]))[0]
X_all.loc[:, 'above_fdr'] = [1 if X_all.loc[i, 'final_score'] > thresh else 0 for i in X_all.index]

# labeling annotations as:
# "N": decoys, the negative class
# "TP": targets above FDR threshold
# "FP": targets below FDR threshold
X_all['final_label'] = [None] * len(X_all)
X_all.loc[X_all.target == 0, 'final_label'] = 'N'
X_all.loc[(X_all.target == 1) & (X_all.above_fdr == 1), 'final_label'] = 'TP'
X_all.loc[(X_all.target == 1) & (X_all.above_fdr == 0), 'final_label'] = 'FP'
# histogram an boxplot of scores divided by the final label.
# y-axis is log-scaled
score_hist = sns.FacetGrid(X_all, hue='final_label', size=7)
score_hist.map(sns.distplot, 'final_score', bins=50, kde=False, rug=False).add_legend()
score_hist.set(xlabel='Final SVM score', ylabel='Frequency', yscale='log', title='Distribution of final scores by final label')
score_hist.savefig(savepath + name + '/final_scores_hist.png')

plt.figure()
score_box = sns.boxplot(data=X_all, x='final_label', y='final_score')
score_box_fig = score_box.get_figure()
score_box_fig.savefig(savepath + name + '/final_scores_box.png')

print("Final FDR level = {}. # identifications: {} \n".format(fdr_level, X_all.final_label.value_counts()['TP']))
log.write("Final FDR level = {}. # identifications: {} \n".format(fdr_level, X_all.final_label.value_counts()['TP']))

# roc curve for the final model
final_roc, ax = plt.subplots(1, 1, figsize=(7, 7))
fpr, tpr, _ = metrics.roc_curve(X_all['target'], X_all['final_score'], pos_label=1)
ax.plot(fpr, tpr, label='SVM')
ax.set_title('Final ROC. AUC = {:.2f}'.format(metrics.auc(fpr, tpr)))
ax.plot([0, 1], [0, 1], 'k--', label='Luck')
fpr, tpr, _ = metrics.roc_curve(X_all['target'], X_all['msm'], pos_label=1)
ax.plot(fpr, tpr, label='MSM')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', prop={'size':15})
final_roc.savefig(savepath + name + '/' +name+'_final_roc.png')
"""

# I want to save the annotation identifiers, the features, if it's a target, above_fdr,
# the final score and the final label. I want the features to not be scaled!
# to_save = X_all[['sf_name', 'sf', 'adduct', 'target'] + features + ['above_fdr', 'final_score', 'final_label']]
to_save = X[['sf_name', 'sf', 'adduct', 'target'] + features + ['final_score']]
to_save[features] = scaler.inverse_transform(to_save[features])
to_save.to_csv(savepath + name + '/data/' +name+'_rescored.csv', index=False)

# Also, a plot of the features' weights
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
