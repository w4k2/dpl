from utils import snakeway, zeto, ns2pdf, norm_0_1
import matplotlib.pyplot as plt
import numpy as np

# Our synthetic
n_centroids = [3, 5, 7]
factors = [1, 3, 8]
n_samples = 400

fig, ax = plt.subplots(3,3,figsize=(8,8), sharex=True, sharey=True)
plt.suptitle('Snakeway', fontsize=15)

for n_c_id, n_c in enumerate(n_centroids):
    for f_id, f in enumerate(factors):
        X, y, ns = snakeway(n_samples=n_samples, n_centroids=n_c, factor=f)
        
        pred_space = np.linspace(-3,3,1000).reshape(-1, 1)
        spdf = ns2pdf(pred_space, (ns[0], np.ones_like(ns[1]))).flatten()
        n_spdf = norm_0_1(spdf)
        
        if n_c==3:
            ax[n_c_id, f_id].set_title('factor = %i' % f, fontsize=12)
        if f==1:
            ax[n_c_id, f_id].set_ylabel('%i centroids' % n_c, fontsize=12)
        ax[n_c_id, f_id].scatter(X, np.random.rand(len(y)), c=y, cmap='coolwarm', alpha=0.23)
        ax[n_c_id, f_id].plot(pred_space, n_spdf, color='black', ls=':')
        ax[n_c_id, f_id].spines['top'].set_visible(False)
        ax[n_c_id, f_id].spines['right'].set_visible(False)
        ax[n_c_id, f_id].set_yticks([])

        
plt.tight_layout()
plt.savefig('snake.png')


fig, ax = plt.subplots(3,3,figsize=(8,8), sharex=True, sharey=True)
plt.suptitle('Zeto', fontsize=15)

for n_c_id, n_c in enumerate(n_centroids):
    for f_id, f in enumerate(factors):
        X, y, ns = zeto(n_samples=n_samples, m_centroids=n_c, factor=f)
        
        if n_c==3:
            ax[n_c_id, f_id].set_title('factor = %i' % f, fontsize=12)
        if f==1:
            ax[n_c_id, f_id].set_ylabel('%i centroids' % n_c, fontsize=12)
        ax[n_c_id, f_id].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.23)
        ax[n_c_id, f_id].spines['top'].set_visible(False)
        ax[n_c_id, f_id].spines['right'].set_visible(False)

        
plt.tight_layout()
plt.savefig('zeto.png')