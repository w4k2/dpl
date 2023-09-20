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

fig, ax = plt.subplots(4,5,figsize=(12,9), sharex=True, sharey=True)

for d_plot_id, dataset_id in enumerate([1,5,9,13]):
    for plot_id, est_id in enumerate([0,3,4,5,6]):
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
res = res[:,:,1] #just MSE

# Plot imgs
labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']


fig, ax = plt.subplots(1,1,figsize=(4,8), sharex=True, sharey=True)
ax.imshow(res, cmap='coolwarm')
ax.set_title('MSE')
# ax.grid(ls=':')
ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
ax.set_yticks(np.arange(len(dataset_names)), dataset_names)
        
for _a, __a in enumerate(dataset_names):
    for _b, __b in enumerate(labels):
        ax.text(_b, _a, "%.3f" % (
            res[_a, _b]
            ) , va='center', ha='center', c='white', fontsize=8)
        
plt.tight_layout()
plt.savefig('figures/E3_2d_err.png')