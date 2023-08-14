import numpy as np
import matplotlib.pyplot as plt

###

n_centroids = [2, 3, 5, 7]
factors = [1, 2, 3, 4]

dataset_names = []
for n_c in n_centroids:
    for f in factors:
        dataset_names.append('zeto_c%i_f%i' % (n_c, f))


pred_space = np.linspace(-5,5,100).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T

###

res_pred = np.load('results/E3_2d_v.npy')
res_pred = np.mean(res_pred, axis=0)

# Plot imgs
labels = ['true', 'KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']

fig, ax = plt.subplots(4,5,figsize=(15,12), sharex=True, sharey=True)

for d_plot_id, dataset_id in enumerate([1,5,9,13]):
    for plot_id, est_id in enumerate([0,1,4,5,6]):
        ax[d_plot_id, plot_id].scatter(*pred_mesh.T, c=res_pred[dataset_id, est_id], cmap='coolwarm')
    
        if d_plot_id==0:
            ax[d_plot_id, plot_id].set_title(labels[est_id])
        if plot_id ==0:
            ax[d_plot_id, plot_id].set_ylabel(dataset_names[dataset_id])
        ax[d_plot_id, plot_id].set_xlim(-5,5)
        ax[d_plot_id, plot_id].set_ylim(-5,5)

            
plt.tight_layout()
plt.savefig('figures/E3_2d.png')

###

res = np.load('results/E3_2d.npy')
res = np.mean(res, axis=0)[:,:,-1] # last iteration
res = res[:,:,1:] #skip Statistic

# Plot imgs
labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']

fig, ax = plt.subplots(1,2,figsize=(7,8), sharex=True, sharey=True)
for m_id, m in enumerate(['MSE', 'MAE']):
    ax[m_id].imshow(res[:,:,m_id], cmap='coolwarm')
    
    ax[m_id].set_title(m)
    ax[m_id].grid(ls=':')
    ax[m_id].set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax[m_id].set_yticks(np.arange(len(dataset_names)), dataset_names)
            
plt.tight_layout()
plt.savefig('figures/E3_2d_err.png')