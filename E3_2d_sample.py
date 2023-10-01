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
iterations = 32

# Experiment configuration
pred_space = np.linspace(-5,5,100).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T

#### Experiment ####

clf = DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), 
          curve_quants=30, monotonic=False, transform='sqrt')

for i in range(iterations):
    pred_clf = clf.partial_fit(X,y).predict_proba(pred_mesh)
    dist = clf.decfunc(pred_mesh)
    
    s_pred_clf = pred_clf/np.std(dist, axis=1).reshape(-1,1)
    
    o_mi, o_ma = np.min(pred_clf), np.max(pred_clf)
    m_mi, m_ma = np.min(s_pred_clf), np.max(s_pred_clf)

    fig, ax = plt.subplots(2,3,figsize=(10.5,7))
    ax = ax.ravel()

    ax[0].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.2)
    ax[0].set_xlim(-5,5)
    ax[0].set_ylim(-5,5)
    ax[0].set_title('classification data')
    ax[0].set_ylabel('original predict proba', fontsize=12)
    
    ax[3].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.2)
    ax[3].set_xlim(-5,5)
    ax[3].set_ylim(-5,5)
    ax[3].set_ylabel('modified predict proba', fontsize=12)

    ax[1].set_title('negative class support')
    ax[2].set_title('positive class support')

    o = ax[1].imshow(pred_clf[:,0].reshape(100,100), cmap='coolwarm', vmin=o_mi, vmax=o_ma)
    ax[2].imshow(pred_clf[:,1].reshape(100,100), cmap='coolwarm', vmin=o_mi, vmax=o_ma)

    m = ax[4].imshow(s_pred_clf[:,0].reshape(100,100), cmap='coolwarm', vmin=m_mi, vmax=m_ma)
    ax[5].imshow(s_pred_clf[:,1].reshape(100,100), cmap='coolwarm', vmin=m_mi, vmax=m_ma)

    cax_2 = ax[2].inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(o, ax=ax[2], cax=cax_2)  
    
    cax_5 = ax[5].inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(m, ax=ax[5], cax=cax_5)  
      
    for aa in ax:
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
    
    for aa in ax[[1,2,4,5]]: 
        aa.set_xticks([])
        aa.set_yticks([])
        aa.spines['bottom'].set_visible(False)
        aa.spines['left'].set_visible(False)
            
        
    plt.tight_layout()
    plt.savefig('foo2.png')

