import numpy as np
import matplotlib.pyplot as plt
from utils import snakeway, ns2pdf
from sklearn.neural_network import MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.neighbors import KernelDensity
from scipy.stats import f_oneway
from sklearn.metrics import mean_squared_error, mean_absolute_error


def norm_0_1(data):
    data -=np.nanmin(data)
    data /=np.nanmax(data)
    return data

datasets = {
    's3_0': snakeway(n_samples=2000, n_centroids=3),
    's5_0': snakeway(n_samples=2000, n_centroids=5),
    's7_0': snakeway(n_samples=2000, n_centroids=7),
    's9_0': snakeway(n_samples=2000, n_centroids=9),
    's3_2': snakeway(n_samples=2000, n_centroids=3, factor=2),
    's5_2': snakeway(n_samples=2000, n_centroids=5, factor=2),
    's7_2': snakeway(n_samples=2000, n_centroids=7, factor=2),
    's9_2': snakeway(n_samples=2000, n_centroids=9, factor=2),
}

exp = False

#####
if exp == True:

    n_est = 7
    iters = 400
    res = np.zeros((len(datasets), n_est, iters, 4)) # (stat, p, mse, mae)

    base_reg = MLPRegressor(hidden_layer_sizes=(100, 10), random_state=2333)
    pred_space_small = np.linspace(-5, 5, 1000).reshape(-1, 1)


    for i, (d_name, (X, y, ns)) in enumerate(datasets.items()):
        print(d_name)
        
        # get actual density
        new_s = np.ones_like(ns[1])
        spdf = ns2pdf(pred_space_small, (ns[0], new_s)).flatten()
                
        # test estimated density

        estimators = [
            KernelDensity(kernel='gaussian', bandwidth=0.2),
            KernelDensity(kernel='tophat', bandwidth=0.2),
            KernelDensity(kernel='epanechnikov', bandwidth=0.2),
            DPL(base_clf=clone(base_reg), curve_quants=10,
                monotonic=True, transform='none'),
            DPL(base_clf=clone(base_reg), curve_quants=10,
                monotonic=True, transform='sqrt'),
            DPL(base_clf=clone(base_reg), curve_quants=10, 
                monotonic=True, transform='log'),
            DPL(base_clf=clone(base_reg), curve_quants=10, 
                monotonic=True, transform='std_norm'),
        ]
        
        for est_id, est in enumerate(estimators):
            print(est_id)

            if est_id<3:
                # KDE
                est.fit(X, y)
                pred = est.score_samples(pred_space_small) # Compute the log-likelihood of each sample under the model
                pred = np.exp(pred)
                
                # eval with anova
                stat, p = f_oneway(pred, spdf)
                res[i, est_id, :, 0] = stat
                res[i, est_id, :, 1] = p
                
                # eval mse
                n_spdf = norm_0_1(spdf)       
                n_pred = norm_0_1(pred)   
                n_pred[np.isinf(n_pred)]=0
                n_pred[np.isnan(n_pred)]=0
                                                    
                res[i, est_id, -1, 2] = mean_squared_error(n_spdf, n_pred)
                res[i, est_id, -1, 3] = mean_absolute_error(n_spdf, n_pred)
                
                # fig, ax = plt.subplots(1,1,figsize=(8,5))
                # ax2 = ax.twinx()
                # ax.plot(pred_space_small, n_spdf)
                # ax2.plot(pred_space_small, n_pred, color='r')
                # plt.savefig('foo.png')
                # plt.clf()
                # exit()
        
                
            else:
                #DPL
                for iter in range(iters):
                    est.partial_fit(X, y)                    
                    pred = est.score_samples(pred_space_small)
                    # exit()
                    
                    # eval with anova
                    stat, p = f_oneway(pred, spdf)
                    res[i, est_id, iter, 0] = stat
                    res[i, est_id, iter, 1] = p
                    
                    # eval mse
                    n_spdf = norm_0_1(spdf)       
                    n_pred = norm_0_1(pred)   
                    n_pred[np.isinf(n_pred)]=0
                    n_pred[np.isnan(n_pred)]=0
                
                    res[i, est_id, iter, 2] = mean_squared_error(n_spdf, n_pred)
                    res[i, est_id, iter, 3] = mean_absolute_error(n_spdf, n_pred)
                    
                    # fig, ax = plt.subplots(1,1,figsize=(8,5))
                    # ax2 = ax.twinx()
                    # ax.plot(n_spdf)
                    # ax2.plot(n_pred, color='r')
                    # plt.title('iter: %i, s=%0.3f, t=%0.3f' % (iter, stat, p))
                    # plt.savefig('foo.png')
                    # plt.clf()
    
        print(d_name, res[i,:,-1, -1])
        np.save('results/res_density_1d.npy', res)

else:
    
    #Plot
    res = np.load('results/res_density_1d.npy')
    print(res.shape) # datasets x estiators x iters x (stat, p, mse, mae)
        
    labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']
    cols = ['b','b','b','r','r','r','r']
    markers = ['o','x','*','s','d','*','o']
      
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    for m, metric in enumerate(['MSE', 'MAE']):   
        for est_id in range(6):
            ax[m].set_ylabel(metric)
            ax[m].scatter(np.arange(len(datasets)), res[:,est_id,-1,2+m], 
                        label=labels[est_id], marker=markers[est_id], color=cols[est_id],
                        alpha=0.7)
        ax[m].set_xticks(np.arange(len(datasets)), datasets.keys(), rotation=90)
        ax[m].grid(ls=':')
        
    ax[m].legend(ncol=2, frameon=False)

        
    plt.tight_layout()
    plt.savefig('figures/est_1d.png')  
    