import numpy as np
import matplotlib.pyplot as plt
from methods import DPL
from utils import zeto, ns2pdf
from sklearn.neural_network import MLPRegressor
from os import system


# Processing parameters
n_iters = 1024
n_samples = 1000
d = 128
m_reg = MLPRegressor(hidden_layer_sizes=(1000),learning_rate_init=1e-2)
s_reg = MLPRegressor(hidden_layer_sizes=(1000),learning_rate_init=1e-3)

# Generate data
X, y, ns = zeto(n_samples=n_samples, m_centroids=5)

aprober = np.linspace(-5,5, d).reshape(d, 1)
prober = np.array(np.meshgrid(aprober, aprober)).reshape(2,-1).T

zpdf = np.product(ns2pdf(prober, ns), axis=1)

# Initialize models

cc = {'curve_quants': 16}
models = {
    'monotonic': DPL(s_reg, **cc, monotonic=True),
    'dychotomic': DPL(m_reg, **cc, monotonic=False)
}

# Prepare plot
fig, ax = plt.subplots(5,len(models),figsize=(10,10*1.2))

az = ax[0,0]
az.scatter(*X.T, c=y, cmap='bwr_r', s=1)
az.set_ylim(-5,5)
az.set_xlim(-5,5)
az.grid(ls=":")
az.set_title('Problem scatter plot')

ay = ax[0,1]
ay.scatter(*prober.T, c=zpdf, cmap='bwr', s=1)
ay.grid(ls=":")
ay.set_title('Distribution image')

for iter in range(n_iters):
    print('Iteration %i' % iter)
    for idx, (name, model) in enumerate(models.items()):
        print(idx, name, model)
        model.partial_fit(X, y)
        
        pred = model.decfunc(prober)
        abspred = np.abs(pred)
        
        pred_img = pred.reshape(d,d, -1)
        
        aa = ax[1,idx]
        ab = ax[2,idx]
        ac = ax[3,idx]
        ad = ax[4,idx]
        
        # Slice curves
        aa.set_title('.25, .5 and .75 slice [%s]' % name)
        aa.plot(aprober, 
                np.abs(pred_img[:,int(d*.25),:]),
                c='blue', alpha=.05)
        aa.plot(aprober, 
                np.abs(pred_img[:,int(d*.75),:]),
                c='green', alpha=.05)
        aa.plot(aprober, 
                np.abs(pred_img[:,int(d*.5),:]),
                c='red', alpha=.05)
        aa.set_yscale('log')
        aa.grid(ls=":")
        
        # Mean log view
        ab.set_title('Absolute mean log view')
        ab.scatter(*prober.T, 
                   c=np.log(np.abs(np.mean(abspred, axis=-1))), 
                   cmap='bwr', 
                   s=5)
                
        # Log view
        ac.set_title('Mean view [-1,1]')
        ac.scatter(*prober.T, c=np.mean(abspred, axis=-1), cmap='bwr', s=5, vmin=-1, vmax=1)
        
        # Loss curve
        ad.plot(model.clf.loss_curve_, color='black', lw=1, alpha=1)
        ad.grid(ls=":")
        ad.set_yscale('log')
        ad.set_title('Log loss')
    
    for aa in ax[:-1].ravel():
        aa.set_ylim(-5,5)
        aa.set_xlim(-5,5)
    
    plt.tight_layout()
    
    plt.savefig('frames/%04i.png' % model.epoch)
    system('cp frames/%04i.png frames/last.png' % model.epoch)