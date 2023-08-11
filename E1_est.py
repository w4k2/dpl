"""
Hiperparametry / konfiguracja

Dane syntetyczne z snakeway

"""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import f_oneway
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from utils import norm_0_1, ns2pdf, snakeway


# Data configuration
n_samples = 1000
factors = [1, 3, 5, 10]
sigmas = [0.5, 1, 3, 5]
n_centroids = [2, 3, 5, 7, 9]

# Method configuration

n_iter = 256
curve_quants = [5,10,20,50]
integrators = [
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(10), random_state=1432),
    MLPClassifier(hidden_layer_sizes=(100), random_state=1432),
    MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1432),
    MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1432)
]
base_regressors = [
    MLPRegressor(hidden_layer_sizes=(10), random_state=1432),
    MLPRegressor(hidden_layer_sizes=(100), random_state=1432),
    MLPRegressor(hidden_layer_sizes=(10, 10), random_state=1432),
    MLPRegressor(hidden_layer_sizes=(100, 100), random_state=1432),
    MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=1432),
    MLPRegressor(hidden_layer_sizes=(100, 100, 100), random_state=1432)
]
transforms = ['none', 'sqrt', 'log', 'std_norm']

# Experiment configuration
test_space = np.linspace(-5,5,1000).reshape(-1, 1)

res = np.zeros((len(factors), len(sigmas), len(n_centroids), 
                len(curve_quants), len(integrators), len(base_regressors), 
                len(transforms), n_iter, 3))

res_pred = np.zeros((len(factors), len(sigmas), len(n_centroids), 
                len(curve_quants), len(integrators), len(base_regressors), 
                len(transforms)+1, len(test_space)))

pbar = tqdm(total=len(factors)*len(sigmas)*len(n_centroids)*\
                len(curve_quants)*len(integrators)*len(base_regressors)*\
                len(transforms)*n_iter)

#### Experiment ####

# Generate data

for f_id, f in enumerate(factors):
    for s_id, s in enumerate(sigmas):
        for n_c_id, n_c in enumerate(n_centroids):
            
            X, y, ns = snakeway(n_samples=n_samples,
                                factor=f,
                                sigma=s,
                                n_centroids=n_c)
            
            spdf = ns2pdf(test_space, (ns[0], np.ones_like(ns[1]))).flatten()
            n_spdf = norm_0_1(spdf)
            
            # Configure method
            for cq_id, cq in enumerate(curve_quants):
                for int_id, itg in enumerate(integrators):
                    for br_id, br in enumerate(base_regressors):
                        for t_id, t in enumerate(transforms):
                            
                            dpl = DPL(
                                base_clf=clone(br),
                                curve_quants=cq,
                                max_iter=n_iter,
                                monotonic=True,
                                transform=t,
                                integrator=clone(itg)
                            )
                            
                            for i in range(n_iter):
                                pred = dpl.partial_fit(X, y).score_samples(test_space)
                                
                                stat, p = f_oneway(spdf, pred)
                                res[f_id, s_id, n_c_id, cq_id, int_id, br_id, t_id, i, 0] = stat

                                n_pred = norm_0_1(pred)
                                res[f_id, s_id, n_c_id, cq_id, int_id, br_id, t_id, i, 1] = mean_squared_error(n_spdf, n_pred)
                                res[f_id, s_id, n_c_id, cq_id, int_id, br_id, t_id, i, 2] = mean_absolute_error(n_spdf, n_pred)
                                    
                                pbar.update(1)
                                
                                if i==n_iter-1:
                                    res_pred[f_id, s_id, n_c_id, cq_id, int_id, br_id, 0] = n_spdf
                                    res_pred[f_id, s_id, n_c_id, cq_id, int_id, br_id, t_id+1] = n_pred
                                    
            np.save('results/E1_est.npy', res)            
            np.save('results/E1_est_v.npy', res_pred)