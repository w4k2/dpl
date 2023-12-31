"""
Classification of synthetic data -- plot

"""

import numpy as np
from utils import synthetic_dataset_names
import matplotlib.pyplot as plt

dataset_names = synthetic_dataset_names()
method_names = ['KNN', 'GNB', 'SVM', 'MLP', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std']

cols = ['b','b','b','b','r','r','r','r']
markers = ['o','x','*','s','o','x','*','s']
    
res = np.load('results/E2_syn.npy')
print(res.shape) # datasets x folds x methods

res = np.mean(res, axis=1)

dataset_names = [d.split('_rs')[0] for d in dataset_names]

order = np.argsort(dataset_names)
res = res[order]
dataset_names = np.array(dataset_names)[order]

res_others = res[:99]
res_others = res_others.reshape((3,-1,8))
res_others = np.mean(res_others, axis=0)

res_ours = res[99:]

dataset_names_others = (np.array(dataset_names))[:99][::3]
dataset_names_ours = (np.array(dataset_names))[99:]

print(dataset_names_others)
print(dataset_names_ours)

fig, ax = plt.subplots(2,1,figsize=(13,8))

# other synthetic -- 33
for data_id, data in enumerate(dataset_names_others):
    for method_id, method in enumerate(method_names):
        if data_id==0:
            ax[0].scatter(data_id, res_others[data_id,method_id], marker=markers[method_id], c=cols[method_id], label=method)
        else:
            ax[0].scatter(data_id, res_others[data_id,method_id], marker=markers[method_id], c=cols[method_id])

ax[0].set_xticks(np.arange(len(dataset_names_others)), dataset_names_others, rotation=90)
ax[0].set_xlim(-1,36)
ax[0].set_ylim(0.3, 1.01)
ax[0].set_ylabel('BAC')
ax[0].grid(ls=':')
ax[0].legend(frameon=False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# ours synthetic -- 48
for data_id, data in enumerate(dataset_names_ours):
    for method_id, method in enumerate(method_names):
        if data_id==0:
            ax[1].scatter(data_id, res_ours[data_id,method_id], marker=markers[method_id], c=cols[method_id], label=method)
        else:
            ax[1].scatter(data_id, res_ours[data_id,method_id], marker=markers[method_id], c=cols[method_id])

ax[1].set_xticks(np.arange(len(dataset_names_ours)), dataset_names_ours, rotation=90)
ax[1].set_xlim(-1,53)
ax[1].set_ylim(0.3, 1.01)
ax[1].set_ylabel('BAC')
ax[1].grid(ls=':')
ax[1].legend(frameon=False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/E2_syn.png')
plt.savefig('figures/E2_syn.eps')
plt.savefig('foo.png')
    
