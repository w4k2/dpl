"""
Density estimation - 1D 

Plot
"""

import numpy as np
import matplotlib.pyplot as plt

##
n_centroids = [2, 3, 5, 7]
factors = [1, 2, 3, 4]

dataset_names = []

for n_c in n_centroids:
    for f in factors:
        dataset_names.append('snake_c%i_f%i' % (n_c, f))

print(dataset_names, len(dataset_names))

pred_space = np.linspace(-5,5,1000).reshape(-1, 1)

###

res_pred = np.load('results/E3_1d_v.npy')

res_pred = np.mean(res_pred, axis=0)

# Plot imgs
labels = ['source', 'KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std']

fig, ax = plt.subplots(4,4,figsize=(18,12), sharex=True, sharey=True)
ax = ax.ravel()

ls = [':', '-', '-', '-' , '-', '-', '-', '-']
# cols=['black', 'cornflowerblue','cornflowerblue','cornflowerblue','tomato','tomato','tomato','tomato']
cols=plt.cm.coolwarm(np.linspace(0,1,8))
cols[0] = [0., 0., 0., 1.]


for d_id, d in enumerate(dataset_names):
    for est_id, est in enumerate(labels):
        # if est_id in [2,1,4,6,7]:
        #     continue
        ax[d_id].plot(pred_space, res_pred[d_id, est_id], label=est, ls=ls[est_id], c=cols[est_id])
    
    ax[d_id].set_title(d)
    ax[d_id].grid(ls=':')
    ax[d_id].spines['top'].set_visible(False)
    ax[d_id].spines['right'].set_visible(False)

ax[3].legend(frameon=False)
            
plt.tight_layout()
plt.savefig('figures/E3_1d_supp.png')
plt.savefig('foo.png')
plt.savefig('figures/E3_1d_supp.eps')
