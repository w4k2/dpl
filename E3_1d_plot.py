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
labels = ['true', 'KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']

fig, ax = plt.subplots(4,4,figsize=(15,10), sharex=True, sharey=True)
ax = ax.ravel()

ls = [':', '-', '-', '-' , '-', '-', '-', '-']
cols=['black', 'cornflowerblue','cornflowerblue','cornflowerblue','tomato','tomato','tomato','tomato']

for d_id, d in enumerate(dataset_names):
    for est_id, est in enumerate(labels):
        if est_id in [2,1,4,6,7]:
            continue
        ax[d_id].plot(pred_space, res_pred[d_id, est_id], label=est, ls=ls[est_id], c=cols[est_id], alpha=0.7)
    
    ax[d_id].set_title(d)
    ax[d_id].grid(ls=':')

ax[0].legend(frameon=False)
            
plt.tight_layout()
plt.savefig('figures/E3_1d.png')


###

res = np.load('results/E3_1d.npy')
res = np.mean(res, axis=0)[:,:,-1,] # last iteration


# Plot imgs
labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']

fig, ax = plt.subplots(1,3,figsize=(10,7), sharex=True, sharey=True)

for m_id, m in enumerate(['Statistic', 'MSE', 'MAE']):
    ax[m_id].imshow(res[:,:,m_id], cmap='coolwarm')
    
    ax[m_id].set_title(m)
    ax[m_id].grid(ls=':')
    ax[m_id].set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax[m_id].set_yticks(np.arange(len(dataset_names)), dataset_names)
            
plt.tight_layout()
plt.savefig('figures/E3_1d_err.png')