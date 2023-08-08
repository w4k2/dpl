import numpy as np
import matplotlib.pyplot as plt
from utils import snakeway, ns2pdf, zeto

datasets = {
    's1000': snakeway(n_samples=5000, n_centroids=3),
    's2000': snakeway(n_samples=5000, n_centroids=5, translation=2),
    's3000': snakeway(n_samples=5000, n_centroids=7, sigma=3),
    's4000': snakeway(n_samples=5000, n_centroids=9, factor=8),
    'z1000': zeto(n_samples=5000, m_centroids=3),
    'z2000': zeto(n_samples=5000, m_centroids=5, translation=2),
    'z3000': zeto(n_samples=5000, m_centroids=7, sigma=3),
    'z4000': zeto(n_samples=5000, m_centroids=9, factor=8),
}

# Figure it out
fig, ax = plt.subplots(4,4,figsize=(10,10))
ax = ax.ravel()
irange = (-5,5)

# Snakeway linspace
sils = np.linspace(*irange, 1000)

# Zeto linspace
zils = np.array(np.meshgrid(sils, sils)).reshape(2,-1).T

for i, (dataset, (X, y, ns)) in enumerate(datasets.items()):
    print(i)
    aa = ax[i]
    if i >= 4:
        aa = ax[i+4]
    ab = ax[i+4]
    if i >= 4:
        ab = ax[i+4+4]
        
    aa.set_title(dataset)
    
    if dataset[0] == 's':
        # Snakeway
        spdf = ns2pdf(sils, ns)
              
        aa.hist(X[y==0], bins=64, color='r', range=irange)
        aa.hist(X[y==1], bins=64, color='b', range=irange)

        ab.plot(sils, spdf, c='k')

    elif dataset[0] == 'z':
        # Zeto
        zpdf = np.product(ns2pdf(zils, ns), axis=1)
        
        aa.scatter(*X[y==0].T, c='r', alpha=.01)
        aa.scatter(*X[y==1].T, c='b', alpha=.01)
        
        zimg = zpdf.reshape(1000,1000)
        ab.imshow(zimg, cmap='bwr')
    
plt.tight_layout()
plt.savefig('figures/datasets.png')