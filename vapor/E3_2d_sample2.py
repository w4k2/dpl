import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from methods import DPL
from utils import zeto
import matplotlib.pyplot as plt
from scipy.stats import hmean

# Data configuration
n_samples = 1000

# Our synthetic
n_c = 5
f = 2

X, y, ns = zeto(n_samples=n_samples, m_centroids=n_c, factor=f)

# Method configuration
n_methods = 7
iterations = 64

# Experiment configuration
pred_space = np.linspace(-5,5,100).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T


#### Experiment ####

clf = DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100, 100)), integrator=MLPClassifier(), 
          curve_quants=30, monotonic=False, transform='sqrt')

for i in range(iterations):
    pred_clf = clf.partial_fit(X,y).predict_proba(pred_mesh)
    dist = clf.decfunc(pred_mesh)
    
    dist = np.abs(dist)
    
    d_std = np.std(dist, axis=1)
    d_hmean = hmean(dist, axis=1)
    
    fig, ax = plt.subplots(1,3,figsize=(10,4))
    ax = ax.ravel()

    ax[0].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    ax[0].set_xlim(-5,5)
    ax[0].set_ylim(-5,5)
    ax[0].set_title('classification data')
    
    ax[1].imshow(d_std.reshape(100,100), cmap='coolwarm')
    ax[1].set_title('std')

  
    ax[2].imshow(d_hmean.reshape(100,100), cmap='coolwarm')
    ax[2].set_title('hmean')

   
    plt.tight_layout()
    plt.savefig('foo2.png')

