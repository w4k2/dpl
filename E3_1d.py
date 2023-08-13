import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from methods import DPL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import f_oneway
from tqdm import tqdm
from utils import norm_0_1, snakeway, ns2pdf
from sklearn.neighbors import KernelDensity

# Data configuration
n_samples = 1000

# Our synthetic
n_centroids = [2, 3, 5, 7]
factors = [1, 2, 3, 4]

datasets = []
dataset_names = []

for n_c in n_centroids:
    for f in factors:
        datasets.append(snakeway(n_samples=n_samples, n_centroids=n_c, factor=f))
        dataset_names.append('snake_c%i_f%i' % (n_c, f))


# Method configuration
n_methods = 7
iterations = 256
repeats=10

# Experiment configuration
pred_space = np.linspace(-5,5,1000).reshape(-1, 1)

res = np.zeros((repeats, len(datasets), n_methods, iterations, 3))
res_pred = np.zeros((repeats, len(datasets), n_methods+1, len(pred_space)))

pbar = tqdm(total=len(datasets)*4*iterations*repeats)

#### Experiment ####

for r in range(repeats):
        
    for dataset_id, (X, y, ns) in enumerate(datasets):
        
        # Actual density
        spdf = ns2pdf(pred_space, (ns[0], np.ones_like(ns[1]))).flatten()
        n_spdf = norm_0_1(spdf)
        res_pred[r, dataset_id, 0] = n_spdf
            

        methods = [
            KernelDensity(kernel='gaussian', bandwidth=0.2),
            KernelDensity(kernel='tophat', bandwidth=0.2),
            KernelDensity(kernel='epanechnikov', bandwidth=0.2),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), curve_quants=10, monotonic=True, transform='none'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), curve_quants=10, monotonic=True, transform='sqrt'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), curve_quants=10, monotonic=True, transform='log'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100, 100)), integrator=MLPClassifier(), curve_quants=10, monotonic=True, transform='std_norm'),
        ]
        
        for estim_id, estim in enumerate(methods):

            if estim_id < 3:
            # KDE
                pred = estim.fit(X).score_samples(pred_space)
                pred = np.exp(pred)
                # print(pred)
                
                stat, p = f_oneway(spdf, pred)
                res[r, dataset_id, estim_id, -1, 0] = stat

                n_pred = norm_0_1(pred)
                res[r, dataset_id, estim_id, -1, 1] = mean_squared_error(n_spdf, n_pred)
                res[r, dataset_id, estim_id, -1, 2] = mean_absolute_error(n_spdf, n_pred)
                
                res_pred[r, dataset_id, estim_id+1] = n_pred
                
            else:
                for i in range(iterations):
                    pred = estim.partial_fit(X, y).score_samples(pred_space)
                    
                    stat, p = f_oneway(spdf, pred)
                    res[r, dataset_id, estim_id, i, 0] = stat

                    n_pred = norm_0_1(pred)
                    res[r, dataset_id, estim_id, i, 1] = mean_squared_error(n_spdf, n_pred)
                    res[r, dataset_id, estim_id, i, 2] = mean_absolute_error(n_spdf, n_pred)
                                        
                    if i==iterations-1:
                        res_pred[r, dataset_id, estim_id+1] = n_pred
                    
                    pbar.update(1)
                
        print(dataset_names[dataset_id], res[r,dataset_id,:,-1,-1])
        np.save('results/E3_1d.npy', res)
        np.save('results/E3_1d_v.npy', res_pred)
                        
                        
                            
                            
            

