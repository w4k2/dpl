import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from methods import DPL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import f_oneway
from tqdm import tqdm
from utils import norm_0_1, zeto, ns2pdf
import matplotlib.pyplot as plt

# Data configuration
n_samples = 1000

# Our synthetic
n_c = 5
f = 2

X, y, ns = zeto(n_samples=n_samples, m_centroids=n_c, factor=f)

# Method configuration
n_methods = 7
iterations = 320

# Experiment configuration
q = 500
pred_space = np.linspace(-5,5,q).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T


#### Experiment ####

clf = DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100), learning_rate_init=0.0001), integrator=MLPClassifier(), 
          curve_quants=30, monotonic=False, transform='sqrt')

for i in range(iterations):
    pred_clf = clf.partial_fit(X,y).predict_proba(pred_mesh)
    std_pred_clf = clf.partial_fit(X,y).std_predict_proba(pred_mesh)
    mean_pred_clf = clf.partial_fit(X,y).mean_predict_proba(pred_mesh)
    n_pred_clf = norm_0_1(pred_clf)
    n_std_pred_clf = norm_0_1(std_pred_clf)
    n_mean_pred_clf = norm_0_1(mean_pred_clf)
    
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax = ax.ravel()
    
    cmap = 'twilight'

    ax[0].imshow(n_std_pred_clf[:,0].reshape(q,q), cmap=cmap)
    ax[1].imshow(n_std_pred_clf[:,1].reshape(q,q), cmap=cmap)
    
    ax[2].imshow(n_mean_pred_clf[:,0].reshape(q,q), cmap=cmap)
    ax[3].imshow(n_mean_pred_clf[:,1].reshape(q,q), cmap=cmap)

    for aa in ax:
        aa.set_xticks([])
        aa.set_yticks([])
        aa.spines['top'].set_visible(False)
        aa.spines['bottom'].set_visible(False)
        aa.spines['right'].set_visible(False)
        aa.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('temp/%04d.png' % i)
    plt.savefig('foo2.png')

