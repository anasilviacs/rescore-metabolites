import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from collections import Counter
import seaborn as sns
import sys
import argparse


parser = argparse.ArgumentParser(description='Semi-supervised improvement of sm-engine scores')
parser.add_argument('orig', type=str, help='path to original search results')
parser.add_argument('resc', type=str, help='path to rescored results')

sys.stdout.write("\n*Plot ReScored results*\n")

args = parser.parse_args()
name = args.orig.split('/')[-1].rstrip('.csv')
savepath = '/'.join(args.resc.split('/')[:-1])
sys.stdout.write("reults will be saved in {}\n".format(savepath))

def get_FDR_threshold(pos, neg, thr=0.10):
    """
    Gets the score threshold that permits a defined FDR. FDR is calculated as
    ((# decoys > X/# decoys)/(#targets > X / # targets)).
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

sys.stdout.write("Loading original search results\n")
orig = pd.read_csv(args.orig, sep='\t')

target_adducts = [t.lstrip('[').lstrip('"').lstrip("u'").rstrip(",").rstrip(']').rstrip("\'")
                  for t in orig.targets[0].split(' ')]

orig['sf_add'] = orig['sf'] + orig['adduct']
orig['target'] = [1 if orig.adduct[r] in target_adducts else 0 for r in range(len(orig))]
orig['above_fdr'] = [1 if orig.fdr[r] in [0.01, 0.05, 0.10] else 0 for r in range(len(orig))]
orig['target_adduct'] = [r.adduct if r.target == 1 else 'decoy' for _, r in orig.iterrows()]
orig['msm'] = orig['spatial'] * orig['spectral'] * orig['chaos']

# MSM score distribution
g = sns.FacetGrid(orig, hue='target', size=7)
g.map(sns.distplot, 'msm', kde=False)
g.set(yscale='log')
g.add_legend()
g.fig.suptitle('MSM score for targets and decoys')
g.savefig(savepath + '/' + name + '_msmdistribution.png')
sys.stdout.write("Saving MSM score distribution\n")

sys.stdout.write("Loading rescored results\n")
resc = pd.read_csv(args.resc)

# FDR plot
fdr_levels = np.linspace(0, 1, 101)

f, ax = plt.subplots(1,1, figsize=(15,5))

nids_engine = []
nids_msm = []
nids_resc = []

for fdr in fdr_levels:
    score = get_FDR_threshold(orig[orig.target==1]['msm'], orig[orig.target==0]['msm'], fdr)
    nids_msm.append(len(orig[(orig.target == 1) & (orig.msm > score)]))
    nids_resc.append(len(resc[resc['combined'] <= fdr]))

for fdr in np.unique(orig.fdr):
    nids_engine.append(len(orig[(orig.target == 1) & (orig.fdr <= fdr)]))

maxlim = np.max([np.max(nids_msm), np.max(nids_engine), np.max(nids_resc)])

ax.plot(np.linspace(0, maxlim, 100), [0.1]*100, label='10% FDR line', color='black', ls='--')
ax.plot(np.linspace(0, maxlim, 100), [0.01]*100, label='10% FDR line', color='black', ls='-.')

ax.plot(nids_engine, np.unique(orig.fdr), label='engine (normalized FDR)')
ax.plot(nids_msm, fdr_levels, label='engine (discretized FDR)')
ax.plot(nids_resc, fdr_levels, label='rescored')

ax.set_xlabel('# of annotations')
ax.set_ylabel('FDR estimate')
ax.legend(loc='best')
ax.set_title("Number of annotations vs FDR trade-off")
ax.set_xlim([0,maxlim])
ax.set_ylim([0,1])

plt.savefig(savepath + '/' + name + '_fdrplot.png')
sys.stdout.write("Saving FDR plot\n")

# Subset FDR plot
f, ax = plt.subplots(1,1, figsize=(15,5))

rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=resc.columns[1:])

for r in rescored_nids.iterrows():
    fdr = r[0]
    for c in rescored_nids.columns:
        rescored_nids.loc[fdr, c] = np.sum(resc[c] < fdr)

ax.plot(rescored_nids.combined, rescored_nids.index.values, label='median')

for c in rescored_nids.columns[:-1]:
    ax.plot(rescored_nids[c], rescored_nids.index.values, label='subset '+c, alpha=0.5)


ax.plot(np.linspace(0,np.max(rescored_nids.combined), 100), [0.1]*100,
            label='10% FDR line', color='black', ls='--')
ax.plot(np.linspace(0,np.max(rescored_nids.combined), 100), [0.01]*100,
            label='1% FDR line', color='black', ls='-.')

ax.set_xlim([0,np.max(rescored_nids.combined)])
ax.set_ylim([0,1])
ax.legend(loc='best')
ax.set_title("Number of annotations vs FDR trade-off, each subset")

plt.savefig(savepath + '/' + name + '_subsetsfdrplot.png')
sys.stdout.write("Saving subsets FDR plot\n")

# Venn diagrams
f, ax = plt.subplots(1,1)#, figsize=(15,5))
local_ids = set(orig[orig.above_fdr == 1].sf_add)

rescore_id = set(resc[resc.combined<=0.10].SpecId)

s = (
    len(local_ids.difference(rescore_id)),    # Ab
    len(rescore_id.difference(local_ids)),    # aB
    len(set.intersection(local_ids, rescore_id))    # AB
)

v = venn2(subsets=s, set_labels=('engine', 'rescore'), ax=ax)
ax.set_title('Overlap in annotations: METASPACE engine and ReScore')

plt.savefig(savepath + '/' + name + '_annotationoverlap.png')
sys.stdout.write("Saving Venn diagram for annotations\n")

# Split by target adduct
f, ax = plt.subplots(1,len(target_adducts), figsize=(15,5))
f.suptitle('Overlap in annotations per target adduct', fontsize=14)

resc['adduct'] = ['+'+r.SpecId.split('+')[1] if '+' in r.SpecId else '-'+r.SpecId.split('-')[1] for _,r in resc.iterrows()]
for i, t in enumerate(target_adducts):
    local_ids = set(orig[(orig.above_fdr == 1) & (orig.adduct == t)].sf_add)
    rescore_id = set(resc[(resc.adduct == t) & (resc.combined <= 0.10)].SpecId)

    s = (
        len(local_ids.difference(rescore_id)),    # Ab
        len(rescore_id.difference(local_ids)),    # aB
        len(set.intersection(local_ids, rescore_id))    # AB
    )

    v = venn2(subsets=s, set_labels=('engine '+t, 'rescored'+t), ax=ax[i])

plt.savefig(savepath + '/' + name + '_annotationoverlappertarget.png')
sys.stdout.write("Saving Venn diagram for annotations split by adduct\n")

# number of ids at different FDR levels
def std_dev_median(values):
    return np.mean(np.absolute(values - np.median(values)))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)

rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=resc.columns[1:-1])

for r in rescored_nids.iterrows():
    fdr = r[0]
    for c in rescored_nids.columns:
        rescored_nids.loc[fdr, c] = np.sum(resc[c] < fdr)

fdrs = [rescored_nids.index[9], rescored_nids.index[49], rescored_nids.index[99],
        rescored_nids.index[149], rescored_nids.index[199]]
nids = [rescored_nids.loc[fdr, 'combined'] for fdr in fdrs]
dev_med = [int(std_dev_median(rescored_nids.loc[fdr, rescored_nids.columns[:-2]])) for fdr in fdrs]

bars = ax.bar(np.arange(len(fdrs)), nids, yerr=dev_med, ecolor='crimson', tick_label=fdrs, align='center')

ax.set_ylabel('# ids')
ax.set_ylim([0, np.max(nids)*1.2])
ax.set_xlabel('FDR')

def autolabel(rects, dev_med):
    """
    Attach a text label above each bar displaying its height
    """
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.1*height,
                '{} +- {}'.format(int(height), dev_med[i]),
                ha='center', va='bottom')

autolabel(bars, dev_med)

ax.set_title("Number of identifications at different FDR thresholds")
plt.savefig(savepath + '/' + name + '_nids.png')
sys.stdout.write("Saving barplot with number of identifications at different FDRs\n")

# Split by target adduct
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)

n = len(np.unique(resc.adduct))
width = (1-0.1) / 3
space = 0

colors = ['royalblue', 'green', 'orange']

leg_prep = ()

for i, target in enumerate(np.unique(resc.adduct)):
    rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=resc.columns[1:-1])

    for r in rescored_nids.iterrows():
        fdr = r[0]
        for c in rescored_nids.columns:
            rescored_nids.loc[fdr, c] = np.sum(resc[resc.adduct == target][c] <= fdr)

    fdrs = [rescored_nids.index[9], rescored_nids.index[49], rescored_nids.index[99],
            rescored_nids.index[149], rescored_nids.index[199]]
    inds = np.arange(len(fdrs))
    nids = [rescored_nids.loc[fdr, 'combined'] for fdr in fdrs]
    dev_med = [int(std_dev_median(rescored_nids.loc[fdr, rescored_nids.columns[:-2]])) for fdr in fdrs]

    bars = ax.bar(inds+space, nids, width, yerr=dev_med, ecolor='crimson', tick_label=fdrs,
                  align='center', color=colors[i])

    autolabel(bars, dev_med)

    leg_prep = leg_prep + (bars[0],)
    space = space+(1.0/n)

ax.legend(leg_prep, tuple(np.unique(resc.adduct)), loc='best')

ax.set_ylabel('# ids')
ax.set_ylim([0, np.max(nids)+250])

ax.set_xticks(inds + (1.0/n))
ax.set_xticklabels(fdrs)
ax.set_xlabel('FDR')

ax.set_title('Number of identifications per target adduct at different FDRs', fontsize=14)
plt.savefig(savepath + '/' + name + '_nidspertarget.png')
sys.stdout.write("Saving barplot with number of identifications at different FDRs split by adduct\n")
