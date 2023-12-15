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

fig, ax = plt.subplots(4,8,figsize=(11,5.5), sharex=True, sharey=True)

count = 0
for ee in range(4):
    for d_plot_id in range(4):
        for plot_id in range(8):
            
            q = np.sqrt(pred_mesh.shape[0]).astype(int)
            img = res_pred[count, plot_id].reshape(q,q)

            #ax[d_plot_id, plot_id].scatter(*pred_mesh.T, c=res_pred[count, plot_id], cmap='coolwarm')
        
            ax[d_plot_id, plot_id].imshow(img, cmap='coolwarm')


            if d_plot_id==0:
                ax[d_plot_id, plot_id].set_title(labels[plot_id])
            if plot_id ==0:
                ax[d_plot_id, plot_id].set_ylabel(dataset_names[count])
            # ax[d_plot_id, plot_id].set_xlim(-5,5)
            # ax[d_plot_id, plot_id].set_ylim(-5,5)
            
            
            ax[d_plot_id, plot_id].set_xlim(0,q-1)
            ax[d_plot_id, plot_id].set_ylim(0,q-1)
            ax[d_plot_id, plot_id].set_xticks(np.linspace(0,q-1,3),
                                            np.linspace(-5,5,3))
            ax[d_plot_id, plot_id].set_yticks(np.linspace(0,q-1,5),
                                            np.linspace(-5,5,5))
            
        count+=1

    plt.tight_layout()
    plt.savefig('figures/E3_2d_supp_%i.png' % ee)
    plt.savefig('figures/E3_2d_supp_%i.eps' % ee)
    plt.savefig('foo.png')

