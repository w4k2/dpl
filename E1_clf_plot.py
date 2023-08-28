import numpy as np
import matplotlib.pyplot as plt


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
            

            
    plt.tight_layout()
    plt.savefig('figures/E1_clf_w%i.png' % w_id)
                            
                            
                

