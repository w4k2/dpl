"""
Density estimation - 2D 

Plot
"""
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

# print(res_pred.shape)
# exit()

# Plot imgs
labels = ['source', 'KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std']

fig, ax = plt.subplots(4,5,figsize=(9,7), sharex=True, sharey=True)

for d_plot_id, dataset_id in enumerate([1,5,9,13]):
    for plot_id, est_id in enumerate([0,3,4,5,6]):
        
        print(pred_mesh.shape)
        print(res_pred[dataset_id, est_id].shape)
        
        q = np.sqrt(pred_mesh.shape[0]).astype(int)
        img = res_pred[dataset_id, est_id].reshape(q,q)
        print(img.shape)
        
        #ax[d_plot_id, plot_id].scatter(*pred_mesh.T, c=res_pred[dataset_id, est_id], cmap='coolwarm')
        
        ax[d_plot_id, plot_id].imshow(img, cmap='coolwarm')
    
        if d_plot_id==0:
            ax[d_plot_id, plot_id].set_title(labels[est_id])
        if plot_id ==0:
            ax[d_plot_id, plot_id].set_ylabel(dataset_names[dataset_id])
        ax[d_plot_id, plot_id].set_xlim(0,q-1)
        ax[d_plot_id, plot_id].set_ylim(0,q-1)
        ax[d_plot_id, plot_id].set_xticks(np.linspace(0,q-1,3),
                                          np.linspace(-5,5,3))
        ax[d_plot_id, plot_id].set_yticks(np.linspace(0,q-1,5),
                                          np.linspace(-5,5,5))


            
plt.tight_layout()
plt.savefig('figures/E3_2d.png')
plt.savefig('figures/E3_2d.eps')
plt.savefig('foo.png')

exit()

###

res = np.load('results/E3_2d.npy')
res = np.mean(res, axis=0)[:,:,-1] # last iteration
res = res[:,:,1] #just MSE

# Plot imgs
labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std']


fig, ax = plt.subplots(1,1,figsize=(4.5,7), sharex=True, sharey=True)
ax.imshow(res, cmap='coolwarm',vmin=0,vmax=0.3, aspect='auto')
ax.set_title('2D')
# ax.grid(ls=':')
ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
ax.set_yticks(np.arange(len(dataset_names)), dataset_names)
        
for _a, __a in enumerate(dataset_names):
    for _b, __b in enumerate(labels):
        ax.text(_b, _a, "%.3f" % (
            res[_a, _b]
            ) , va='center', ha='center', c='white' if res[_a, _b]<0.068 else 'black', fontsize=8)
        
plt.tight_layout()
plt.savefig('figures/E3_2d_err.png')
plt.savefig('foo.png')
plt.savefig('figures/E3_2d_err.eps')