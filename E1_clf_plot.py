"""
Hyperparameters / configuration

synthetic data from make_classification -- classification -- plot

"""
import numpy as np
import matplotlib.pyplot as plt

mval = .53

weights = [
    [0.5, 0.5],
    [0.75, 0.25],
    [0.9, 0.1]
    ]

curve_quants = [5,10,20,50]
integrators = [
    'GNB',
    'MLP10',
    'MLP100',
    'MLP10-10',
    'MLP100-100'
]
base_regressors = [
    'MLP10',
    'MLP100',
    'MLP10-10',
    'MLP100-100',
    'MLP10-10-10'
]
transforms = ['none', 'sqrt', 'log', 'std']


res = np.load('results/E1_clf.npy')
print(res.shape) #weights x reps x cq x integrators x regressors x transforms x folds
                            
res_mean = np.mean(res, axis=(1,-1))
print(res_mean.shape)

# res_mean[:, 0, 0, 3, 0] = 0 # integrator 0, regressor 3
# res_mean[:, 0, 0, 0, 3] = 0 # cq 0, transform 3

for w_id, (_, w) in enumerate(weights):
    
    fig, ax = plt.subplots(4,4,figsize=(15,15), sharex=True, sharey=True)
    plt.suptitle('weight: %0.2f' % w)
    
    for cq_id, cq in enumerate(curve_quants):
        for t_id, t in enumerate(transforms):
            
            if cq_id==0:
                ax[cq_id, t_id].set_title('%s' % (t))
                ax[-1, t_id].set_xlabel('regressor')

            if t_id==0:
                ax[cq_id, t_id].set_ylabel('cq:%i \n integrator' % (cq))
                
            ax[cq_id, t_id].imshow(res_mean[w_id, cq_id, :, :, t_id], vmin=0.5, vmax=1, cmap='coolwarm')
            
            ax[cq_id, t_id].set_xticks(np.arange(5), base_regressors, rotation=90)
            ax[cq_id, t_id].set_yticks(np.arange(5), integrators)
            
            for _a, __a in enumerate(integrators):
                for _b, __b in enumerate(base_regressors):
                    ax[cq_id, t_id].text(_b, _a, "%.3f" % (
                        res_mean[w_id, cq_id, _a, _b, t_id]
                        ) , va='center', ha='center', c='white', fontsize=11)
        
    fig, ax = plt.subplots(5,5,figsize=(12,12), sharex=True, sharey=True)
    # plt.suptitle('weight: %0.2f' % w)

    for itg_id, itg in enumerate(integrators):
        for r_id, r in enumerate(base_regressors):
            
            if itg_id==0:
                ax[itg_id, r_id].set_title('regressor: %s' % (r))
                ax[-1, r_id].set_xlabel('curve quants')

            if r_id==0:
                ax[itg_id, r_id].set_ylabel('integrator: %s \n transform' % (itg))
                
            ax[itg_id, r_id].imshow(res_mean[w_id, :, itg_id, r_id, :].T, vmin=0.5, vmax=1, cmap='coolwarm')
            
            ax[itg_id, r_id].set_xticks(np.arange(4), curve_quants)
            ax[itg_id, r_id].set_yticks(np.arange(4), transforms)
            
            mval = .6
            for _a, __a in enumerate(curve_quants):
                for _b, __b in enumerate(transforms):
                    
                    val = res_mean[w_id, _a, itg_id, r_id, _b]
                    ax[itg_id, r_id].text(_b, _a, "%.3f" % (
                        val
                        ) , va='center', ha='center', c='black' if val > mval else 'white', fontsize=8)
            
    plt.tight_layout()
    plt.savefig('figures/E1_clf_w%i_2.png' % w_id)
    plt.savefig('figures/E1_clf_w%i_2.eps' % w_id)
    plt.savefig('foo.png')
    
    #exit()
                            

# just one row

    fig, ax = plt.subplots(1,5,figsize=(12,3), sharex=True, sharey=True)
    # plt.suptitle('weight: %0.2f' % w)

    for itg_id, itg in enumerate(integrators):
        if itg!='MLP100':
            continue

        for r_id, r in enumerate(base_regressors):
            
            ax[r_id].set_title('regressor: %s' % (r))
            ax[r_id].set_xlabel('curve quants')

            if r_id==0:
                ax[r_id].set_ylabel('integrator: %s \n transform' % itg)
                
            ax[r_id].imshow(res_mean[w_id, :, itg_id, r_id, :].T, vmin=0.5, vmax=1, cmap='coolwarm')
            
            ax[r_id].set_xticks(np.arange(4), curve_quants)
            ax[r_id].set_yticks(np.arange(4), transforms)
            
            for _a, __a in enumerate(transforms):
                for _b, __b in enumerate(curve_quants):
                    val = res_mean[w_id, _a, itg_id, r_id, _b]
                    ax[r_id].text(_a, _b, "%.3f" % (
                        val
                        ) , va='center', ha='center', c='black' if val > mval else 'white', fontsize=8)
            
    plt.tight_layout()
    plt.savefig('figures/E1_clf_w%i_3.png' % w_id)
    plt.savefig('figures/E1_clf_w%i_3.eps' % w_id)
    plt.savefig('foo.png')
    
    #exit()
                

