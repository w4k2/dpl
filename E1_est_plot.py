"""
Hyperparameters / configuration

Syntehtic snakeway - density estimation -- plot

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

factors = [1, 3, 5, 10]
n_centroids = [2, 3, 5, 7, 9]

n_iter = 256
curve_quants = [5,10,20,50]
base_regressors = [
    'MLP10',
    'MLP100',
    'MLP10-10',
    'MLP100-100',
    'MLP10-10-10',
    'MLP100-100-100'
]
transforms = ['none', 'sqrt', 'log', 'std']
test_space = np.linspace(-5,5,1000).reshape(-1, 1)

res = np.load('results/E1_est.npy')
res = res[:, :, :,0, :, :, :, 1] #get MSE only + rm integrator axis

print(res.shape) # factors x centroids x cq  x reg x transforms x iters

res = res[..., -1] # last iteration
print(res.shape) # cq x integrators x reg x transforms

# for factor_id, factor in enumerate(factors):

#     fig, ax = plt.subplots(5,6,figsize=(12,10), sharex=True, sharey=True)
#     # plt.suptitle('factor: %i' % factor)
    
#     for cent_id, cent in enumerate(n_centroids):
#         for r_id, r in enumerate(base_regressors):
            
#             if cent_id==0:
#                 ax[cent_id, r_id].set_title('%s' % (r))
#                 ax[-1, r_id].set_xlabel('curve quants')

#             if r_id==0:
#                 ax[cent_id, r_id].set_ylabel('%i centroids \n transform' % cent)
                
#             ax[cent_id, r_id].imshow(res[factor_id, cent_id, :, r_id, :],cmap='coolwarm', vmin=0, vmax=0.5)#np.max(res[factor_id]))
            
#             ax[cent_id, r_id].set_xticks(np.arange(4), curve_quants)
#             ax[cent_id, r_id].set_yticks(np.arange(4), transforms)
            
#             for _a, __a in enumerate(curve_quants):
#                 for _b, __b in enumerate(transforms):
#                     ax[cent_id, r_id].text(_b, _a, "%.3f" % (
#                         res[factor_id, cent_id, _a, r_id, _b]
#                         ) , va='center', ha='center', c='white', fontsize=8)
            
#         plt.tight_layout()
#         plt.savefig('figures/E1_est_mse_f%i.png' % factor)
        
        
#one row

for factor_id, factor in enumerate(factors):

    fig, ax = plt.subplots(1,6,figsize=(12,3), sharex=True, sharey=True)
    # plt.suptitle('factor: %i' % factor)
    
    for cent_id, cent in enumerate(n_centroids):
        if cent!=3:
            continue
        
        for r_id, r in enumerate(base_regressors):
            
            ax[r_id].set_title('%s' % (r))
            ax[r_id].set_xlabel('curve quants')

            if r_id==0:
                ax[r_id].set_ylabel('%i centroids \n transform' % cent)
                
            ax[r_id].imshow(res[factor_id, cent_id, :, r_id, :],cmap='coolwarm', vmin=0, vmax=0.5)#np.max(res[factor_id]))
            
            ax[r_id].set_xticks(np.arange(4), curve_quants)
            ax[r_id].set_yticks(np.arange(4), transforms)
            
            for _a, __a in enumerate(curve_quants):
                for _b, __b in enumerate(transforms):
                    ax[r_id].text(_b, _a, "%.3f" % (
                        res[factor_id, cent_id, _a, r_id, _b]
                        ) , va='center', ha='center', c='white', fontsize=8)
            
        plt.tight_layout()
        plt.savefig('figures/E1_est_mse_f%i_2.png' % factor)
        plt.savefig('figures/E1_est_mse_f%i_2.eps' % factor)
                

# # preds
# res_pred = np.load('results/E1_est_v.npy')
# res_pred = res_pred[:, :, :,0] # rm integrator axis

# originals = res_pred[:,:,0,0,0]
# res_pred = res_pred[:,:,:,:,1:]
# res_pred = np.mean(res_pred, axis=0)

# print(res_pred.shape) # factors x centroids x cq x reg x transforms

# cq_id=1
# # cols=plt.cm.magma(np.linspace(0,1,len(base_regressors)))
# cols=['blue', 'pink', 'forestgreen', 'orange', 'cyan', 'red']

# fig, ax = plt.subplots(5, 4, figsize=(13,13), sharex=True, sharey=True)

# for cent_id, cent in enumerate(n_centroids):
#     for tr_id, tr in enumerate(transforms):

#         if tr_id==0:
#             ax[cent_id, tr_id].set_ylabel('%i centroids' % cent)
#         if cent_id==0:
#             ax[cent_id, tr_id].set_title('%s' % (tr))
        
#         # for cq_id, cq in enumerate(curve_quants):
#         for r_id, r in enumerate(base_regressors):
#             ax[cent_id, tr_id].plot(test_space, res_pred[cent_id, cq_id, r_id, tr_id], label='%s' % (r), alpha=0.5, c=cols[r_id])
        
#         ax[cent_id, tr_id].grid(ls=':')
#         ax[cent_id, tr_id].plot(test_space, np.mean(originals[:, cent_id], axis=0), color='black', ls=':')


# plt.legend(loc='lower center', bbox_to_anchor=(-1.4, -0.6), ncol=3, frameon=False)
# plt.tight_layout()
# plt.savefig('figures/E1_est.png')
                
        
# learning
res = np.load('results/E1_est.npy')
res = res[:, :, :,0, :, :, :, 1] #get MSE only + rm integrator axis

cols=plt.cm.jet(np.linspace(0,0.5,4))
s=3

print(res.shape) # factors x centroids x cq  x reg x transforms x iters

for reg_id, reg in enumerate(base_regressors):
    for tr_id, tr in enumerate(transforms):
        
        fig, ax = plt.subplots(1, len(n_centroids), figsize=(13,4), sharex=True, sharey=True)
        plt.suptitle('reg: %s | T: %s' % (reg, tr))
        
        for f_id, f in enumerate(factors):
            if f!=3:
                continue
            
            for c_id, c in enumerate(n_centroids):
                
                ax[c_id].set_title('centroids: %s' % c)
                ax[c_id].set_xlabel('iteration')
                if c_id==0:
                    ax[c_id].set_ylabel('factor: %s \n Mean Squared Error' % f)
                
                for cq_id, cq in enumerate(curve_quants):
                    temp = gaussian_filter1d(res[f_id, c_id, cq_id, reg_id, tr_id], s)
                    ax[c_id].plot(temp, label='q: %i' % cq, c=cols[cq_id])
            
                ax[c_id].grid(ls=':')
                ax[c_id].spines['top'].set_visible(False)
                ax[c_id].spines['right'].set_visible(False)
                
        
        plt.legend()      
        plt.tight_layout()
        plt.savefig('figures/E1_L_r%s_t%s.png' % (reg_id, tr_id))
        plt.savefig('figures/E1_L_r%s_t%s.eps' % (reg_id, tr_id))
        