import numpy as np
import matplotlib.pyplot as plt


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
transforms = ['none', 'sqrt', 'log', 'std_norm']
test_space = np.linspace(-5,5,1000).reshape(-1, 1)

res = np.load('results/E1_est.npy')
res = res[:, :, :,0, :, :, :, 1] #get MSE only + rm integrator axis

print(res.shape) # factors x centroids x cq  x reg x transforms x iters

res = res[..., -1] # last iteration
print(res.shape) # cq x integrators x reg x transforms

for factor_id, factor in enumerate(factors):

    fig, ax = plt.subplots(5,6,figsize=(12,12), sharex=True, sharey=True)
    plt.suptitle('factor: %i' % factor)
    
    for cent_id, cent in enumerate(n_centroids):
        for r_id, r in enumerate(base_regressors):
            
            if cent_id==0:
                ax[cent_id, r_id].set_title('%s' % (r))
                ax[-1, r_id].set_xlabel('curve quants')

            if r_id==0:
                ax[cent_id, r_id].set_ylabel('%i centroids \n transform' % cent)
                
            ax[cent_id, r_id].imshow(res[factor_id, cent_id, :, r_id, :],cmap='coolwarm')
            
            ax[cent_id, r_id].set_xticks(np.arange(4), curve_quants, rotation=90)
            ax[cent_id, r_id].set_yticks(np.arange(4), transforms)
            
            for _a, __a in enumerate(curve_quants):
                for _b, __b in enumerate(transforms):
                    ax[cent_id, r_id].text(_b, _a, "%.3f" % (
                        res[factor_id, cent_id, _a, r_id, _b]
                        ) , va='center', ha='center', c='white', fontsize=8)
            
        plt.tight_layout()
        plt.savefig('figures/E1_est_mse_f%i.png' % factor)
        
#preds
# res_pred = np.load('results/E1_est_v.npy')
# res_pred = res_pred[:, :, :,0] # rm integrator axis

# originals = res_pred[:,:,0,0,0]
# res_pred = res_pred[:,:,:,:,1:]

# print(res_pred.shape) # factors x centroids x cq x reg x transforms

# cq_id=2
# cols=plt.cm.Reds(np.linspace(0.2,1,len(base_regressors)))

# for cent_id, cent in enumerate(n_centroids):
#     for f_id, f in enumerate(factors):
        
#         fig, ax = plt.subplots(2, 2, figsize=(12,12), sharex=True, sharey=True)
#         ax = ax.ravel()
        
#         for tr_id, tr in enumerate(transforms):

#             ax[tr_id].set_title(tr)
            
#             # for cq_id, cq in enumerate(curve_quants):
#             for r_id, r in enumerate(base_regressors):
#                 ax[tr_id].plot(res_pred[f_id, cent_id, cq_id, r_id, tr_id], label='%s' % (r), alpha=0.5, c=cols[r_id])
            
#             ax[tr_id].grid(ls=':')
#             ax[tr_id].plot(originals[f_id, cent_id], color='blue', ls='--')


#         ax[0].legend()
#         plt.tight_layout()
#         plt.savefig('figures/E1_est_c%i_f%i.png' % (cent, f))
                        
            
        
        
        