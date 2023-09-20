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
iterations = 32

# Experiment configuration
pred_space = np.linspace(-5,5,100).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T


#### Experiment ####

clf = DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), 
          curve_quants=30, monotonic=False, transform='sqrt')

for i in range(iterations):
    pred_clf = clf.partial_fit(X,y).predict_proba(pred_mesh)
    _pred_clf = clf.partial_fit(X,y)._predict_proba(pred_mesh)
    n_pred_clf = norm_0_1(pred_clf)
    _n_pred_clf = norm_0_1(_pred_clf)
    
    fig, ax = plt.subplots(2,3,figsize=(10,7))
    ax = ax.ravel()

    ax[0].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    ax[0].set_xlim(-5,5)
    ax[0].set_ylim(-5,5)
    ax[0].set_title('classification data')
    ax[0].set_ylabel('original predict proba', fontsize=12)
    
    ax[1].imshow(n_pred_clf[:,0].reshape(100,100), cmap='coolwarm')
    ax[1].set_title('negative class support')

    ax[3].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    ax[3].set_xlim(-5,5)
    ax[3].set_ylim(-5,5)
    ax[3].set_ylabel('modified predict proba', fontsize=12)

    ax[2].imshow(n_pred_clf[:,1].reshape(100,100), cmap='coolwarm')
    ax[2].set_title('positive class support')

    ax[4].imshow(_n_pred_clf[:,0].reshape(100,100), cmap='coolwarm')
    ax[5].imshow(_n_pred_clf[:,1].reshape(100,100), cmap='coolwarm')

    plt.tight_layout()
    plt.savefig('foo.png')

