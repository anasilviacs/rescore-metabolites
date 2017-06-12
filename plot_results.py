import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from collections import Counter
import seaborn as sns
import os

#TODO un-hard-code files, replace prints with stdout

parser = argparse.ArgumentParser(description='Semi-supervised improvement of sm-engine scores')
parser.add_argument('orig', type=str, help='path to original search results')
parser.add_argument('resc', type=str, help='path to rescored results')

sys.stdout.write("\nPlotting rescored results\n")

args = parser.parse_args()
name = args.orig.split('/')[-1].rstrip('.csv')

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
g = sns.FacetGrid(orig, hue='target', size=5)
g.map(sns.distplot, 'msm', kde=False)
g.set(yscale='log')
g.add_legend()
g.set_title('MSM score for targets and decoys')
g.savefig(name + '.png')

# Load rescore results
rescored_new = pd.read_csv('results.csv')

# FDR plot
fdr_levels = np.linspace(0, 1, 101)
plt.figure(figsize=(15,5))

plt.plot(nids_msm, fdr_levels, label='msm')
plt.plot(nids_svm, fdr_levels, label='all-at-once')
plt.plot(nids_engine, np.unique(orig.fdr), label='msm_engine')

fdrs_new = []
nids_new = []

rescored_new['combined'] = np.median(rescored_new[rescored_new.columns[1:-1]], axis=1)
for v in rescored_new.sort_values(by='combined')['combined']:
    fdrs_new.append(v)
    nids_new.append(len(rescored_new[rescored_new['combined'] <= v]))

plt.plot(nids_new, fdrs_new, label='sampling & aggregating')

maxlim = np.max([np.max(nids_msm), np.max(nids_svm), np.max(nids_new)])
plt.plot(np.linspace(0, maxlim, maxlim), [0.1]*maxlim, label='10% FDR line', color='black', ls='--')
plt.plot(np.linspace(0, maxlim, maxlim), [0.01]*maxlim, label='10% FDR line', color='black', ls='-.')

plt.xlabel('# of annotations')
plt.ylabel('FDR estimate')
plt.legend(loc='best')
plt.title("Number of annotations vs FDR trade-off, MSM, SVM and new ReScoring")
plt.xlim([0,maxlim])
plt.ylim([0,1])

# Subset FDR plot
plt.figure(figsize=(15,5))

rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=rescored_new.columns[1:])

for r in rescored_nids.iterrows():
    fdr = r[0]
    for c in rescored_nids.columns:
        rescored_nids.loc[fdr, c] = np.sum(rescored_new[c] < fdr)

plt.plot(rescored_nids.combined, rescored_nids.index.values, label='median')

for c in rescored_nids.columns:
    plt.plot(rescored_nids[c], rescored_nids.index.values, label='subset '+c, alpha=0.5)


plt.plot(np.linspace(0,np.max(rescored_nids.combined), np.max(rescored_nids.combined)),
         [0.1]*np.max(rescored_nids.combined), label='10% FDR line', color='black', ls='--')
plt.plot(np.linspace(0,np.max(rescored_nids.combined), np.max(rescored_nids.combined)),
         [0.01]*np.max(rescored_nids.combined), label='1% FDR line', color='black', ls='-.')

plt.xlim([0,np.max(rescored_nids.combined)])
plt.ylim([0,1])
plt.legend(loc='best')
plt.title("Number of annotations vs FDR trade-off, each subset in the new re-scoring approach")

# Venn diagrams
local_ids = set(orig[orig.above_fdr == 1].sf_add)

rescore_id = set(rescored_new[rescored_new.combined<=0.10].SpecId)

s = (
    len(local_ids.difference(rescore_id)),    # Ab
    len(rescore_id.difference(local_ids)),    # aB
    len(set.intersection(local_ids, rescore_id))    # AB
)

v = venn2(subsets=s, set_labels=('engine', 'rescore'))
plt.title('Overlap in annotations: METASPACE engine and ReScore')

# Split by target adduct
f, ax = plt.subplots(1,3, figsize=(15,5))

f.suptitle('Overlap in annotations per target adduct: Local and Remote METASPACE engine', fontsize=14)
for i, t in enumerate(target_adducts):
    local_ids = set(orig[(orig.above_fdr == 1) & (orig.adduct == t)].sf_add)

    rescore_id = set(rescored[(rescored.adduct == t) & (rescored.final_score > thresh)].sf_add)

    s = (
        len(local_ids.difference(rescore_id)),    # Ab
        len(rescore_id.difference(local_ids)),    # aB
        len(set.intersection(local_ids, rescore_id))    # AB
    )

    v = venn2(subsets=s, set_labels=('engine '+t, 'rescored'+t), ax=ax[i])

plt.show()

# number of ids at different FDR levels
def std_dev_median(values):
    return np.mean(np.absolute(values - np.median(values)))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)

rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=rescored_new.columns[1:])

for r in rescored_nids.iterrows():
    fdr = r[0]
    for c in rescored_nids.columns:
        rescored_nids.loc[fdr, c] = np.sum(rescored_new[c] < fdr)

fdrs = [rescored_nids.index[9], rescored_nids.index[49], rescored_nids.index[99],
        rescored_nids.index[149], rescored_nids.index[199]]
nids = [rescored_nids.loc[fdr, 'combined'] for fdr in fdrs]
dev_med = [int(std_dev_median(rescored_nids.loc[fdr, rescored_nids.columns[:-2]])) for fdr in fdrs]

bars = ax.bar(np.arange(len(fdrs)), nids, yerr=dev_med, ecolor='crimson', tick_label=fdrs, align='center')

ax.set_ylabel('# ids')
ax.set_ylim([0, np.max(nids)+500])
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

plt.title("number of identifications at different FDR thresholds")

# Split by target adduct
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)

n = len(np.unique(rescored_new.adduct))
width = (1-0.1) / 3
space = 0

colors = ['royalblue', 'green', 'orange']

leg_prep = ()

for i, target in enumerate(np.unique(rescored_new.adduct)):
    rescored_nids = pd.DataFrame(index=np.linspace(0.001,1,1000), columns=rescored_new.columns[1:-1])

    for r in rescored_nids.iterrows():
        fdr = r[0]
        for c in rescored_nids.columns:
            rescored_nids.loc[fdr, c] = np.sum(rescored_new[rescored_new.adduct == target][c] <= fdr)

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

ax.legend(leg_prep, tuple(np.unique(rescored_new.adduct)), loc='best')

ax.set_ylabel('# ids')
ax.set_ylim([0, np.max(nids)+250])

ax.set_xticks(inds + (1.0/n))
ax.set_xticklabels(fdrs)
ax.set_xlabel('FDR')

ax.set_title('Number of identifications per target adduct at different FDRs', fontsize=14)
