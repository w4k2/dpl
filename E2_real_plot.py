"""
Klasyfikacja danych syntetycznych

"""

import numpy as np
import os
import matplotlib.pyplot as plt

dataset_names = os.listdir('datasets')
try:
    dataset_names.remove('.DS_Store')
except:
    pass

method_names = ['KNN', 'GNB', 'SVM', 'MLP', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std']

cols = ['b','b','b','b','r','r','r','r']
markers = ['o','x','*','s','o','x','*','s']
    
res = np.load('results/E2_real.npy')
print(res.shape) # datasets x folds x methods

res = np.mean(res, axis=1)

dataset_names = [d.split('.')[0] for d in dataset_names]

order = np.argsort(dataset_names)
res = res[order]
dataset_names = np.array(dataset_names)[order]


fig, ax = plt.subplots(1,1,figsize=(15,5))

for method_id, method in enumerate(method_names):
    ax.scatter(np.arange(len(dataset_names)), res[:, method_id], marker=markers[method_id], c=cols[method_id], label=method, alpha=0.5)

ax.set_xticks(np.arange(len(dataset_names)), dataset_names, rotation=90)
ax.set_xlim(-1,68)
ax.set_ylim(0.3, 1.01)
ax.set_ylabel('BAC')
ax.grid(ls=':')
ax.legend()

plt.tight_layout()
plt.savefig('figures/E2_real.png')
    
