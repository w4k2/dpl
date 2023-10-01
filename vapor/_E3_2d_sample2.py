import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from methods import DPL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import f_oneway
from tqdm import tqdm
from utils import norm_0_1, zeto, ns2pdf
import matplotlib.pyplot as plt
from scipy.stats import hmean

# Data configuration
n_samples = 1000

# Our synthetic
n_c = 5
f = 2

X, y, ns = zeto(n_samples=n_samples, m_centroids=n_c, factor=f)

# Method configuration
iterations = 120

# Experiment configuration
pred_space = np.linspace(-10,10,500).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T


#### Experiment ####

clf = DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100, 100), learning_rate_init=0.0001), 
          integrator=MLPClassifier(), 
          curve_quants=30, monotonic=False, transform='sqrt')

for i in range(iterations):
    clf.partial_fit(X, y)
    dist = clf.decfunc(pred_mesh)
    
    dist = np.abs(dist)
    
    d_std = np.std(dist, axis=1)
    d_hmean = hmean(dist, axis=1)
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
  
    ax.imshow(d_std.reshape(500,500), cmap='magma_r')
    ax.set_xticks([])
    ax.set_yticks([])
   
    plt.tight_layout()
    plt.savefig('foo2.png')
    plt.savefig('temp/temp2/%04d.png' % i)

