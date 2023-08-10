import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from utils import ns2pdf, zeto
from sklearn.neural_network import MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KernelDensity

def norm_0_1(data):
    data -=np.nanmin(data)
    data /=np.nanmax(data)
    return data

datasets = {
    'z3_0': zeto(n_samples=2000, m_centroids=3),
    'z5_0': zeto(n_samples=2000, m_centroids=5),
    'z7_0': zeto(n_samples=2000, m_centroids=7),
    'z9_0': zeto(n_samples=2000, m_centroids=9),
    'z3_2': zeto(n_samples=2000, m_centroids=3, factor=2),
    'z5_2': zeto(n_samples=2000, m_centroids=5, factor=2),
    'z7_2': zeto(n_samples=2000, m_centroids=7, factor=2),
    'z9_2': zeto(n_samples=2000, m_centroids=9, factor=2)
}

exp = False

pred_space = np.linspace(-5, 5, 100).reshape(-1, 1)
pred_mesh = np.array(np.meshgrid(pred_space, pred_space)).reshape(2,-1).T

#####
if exp == True:

    n_est = 7
    iters = 400
    res = np.zeros((len(datasets), n_est, iters, 4)) # (stat, p, mse, mae)

    base_reg = MLPRegressor(hidden_layer_sizes=(100, 10), random_state=2333)

    for i, (d_name, (X, y, ns)) in enumerate(datasets.items()):
        print(d_name)
        
        # get actual density
        new_s = np.ones_like(ns[1])
        zpdf = np.product(ns2pdf(pred_mesh, (ns[0], new_s)), axis=1)
                
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
                pred = est.score_samples(pred_mesh) # Compute the log-likelihood of each sample under the model
                pred = np.exp(pred)
                
                # eval with anova
                stat, p = f_oneway(pred, zpdf)
                res[i, est_id, :, 0] = stat
                res[i, est_id, :, 1] = p
                
                # eval mse
                n_zpdf = norm_0_1(zpdf)       
                n_pred = norm_0_1(pred)   
                n_pred[np.isinf(n_pred)]=0
                n_pred[np.isnan(n_pred)]=0
                                                    
                res[i, est_id, -1, 2] = mean_squared_error(n_zpdf, n_pred)
                res[i, est_id, -1, 3] = mean_absolute_error(n_zpdf, n_pred)
                
                # fig, ax = plt.subplots(1,2,figsize=(11,5))
                # ax[0].scatter(*pred_mesh.T, c=n_zpdf, cmap='coolwarm')
                # ax[1].scatter(*pred_mesh.T, c=n_pred, cmap='coolwarm')
                # plt.savefig('foo.png')
                # plt.clf()
                # exit()
        
                
            else:
                #DPL
                for iter in range(iters):
                    est.partial_fit(X, y)                    
                    pred = est.score_samples(pred_mesh)
                    # exit()
                    
                    # eval with anova
                    stat, p = f_oneway(pred, zpdf)
                    res[i, est_id, iter, 0] = stat
                    res[i, est_id, iter, 1] = p
                    
                    # eval mse
                    n_zpdf = norm_0_1(zpdf)       
                    n_pred = norm_0_1(pred)   
                    n_pred[np.isinf(n_pred)]=0
                    n_pred[np.isnan(n_pred)]=0
                
                    res[i, est_id, iter, 2] = mean_squared_error(n_zpdf, n_pred)
                    res[i, est_id, iter, 3] = mean_absolute_error(n_zpdf, n_pred)
                    
                    # fig, ax = plt.subplots(1,2,figsize=(11,5))
                    # ax[0].scatter(*pred_mesh.T, c=n_zpdf, cmap='coolwarm')
                    # ax[1].scatter(*pred_mesh.T, c=n_pred, cmap='coolwarm')
                    # plt.suptitle(iter)
                    # plt.savefig('foo.png')
                    # plt.clf()
                    # exit()
    
        print(d_name, res[i,:,-1, -1])
        np.save('results/res_density_2d.npy', res)

else:
    
    #Plot
    res = np.load('results/res_density_2d.npy')
    print(res.shape) # datasets x estiators x iters x (stat, p, mse, mae)
        
    labels = ['KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']
    cols = ['b','b','b','r','r','r','r']
    markers = ['o','x','*','s','d','*','o']
      
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    for m, metric in enumerate(['MSE', 'MAE']):   
        for est_id in range(7):
            ax[m].set_ylabel(metric)
            ax[m].scatter(np.arange(len(datasets)), res[:,est_id,-1,2+m], 
                        label=labels[est_id], marker=markers[est_id], color=cols[est_id],
                        alpha=0.7)
        ax[m].set_xticks(np.arange(len(datasets)), datasets.keys(), rotation=90)
        ax[m].grid(ls=':')
        
    ax[0].legend(ncol=3, frameon=False)
        
    plt.tight_layout()
    plt.savefig('figures/est_2d.png')  
    