import numpy as np
import matplotlib.pyplot as plt
from utils import snakeway, ns2pdf
from sklearn.neural_network import MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.neighbors import KernelDensity
from scipy.stats import f_oneway
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter1d


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

pred_space_small = np.linspace(-5, 5, 1000).reshape(-1, 1)


#####
if exp == True:

    n_est = 7
    iters = 400
    res = np.zeros((len(datasets), n_est, iters, 4)) # (stat, p, mse, mae)
    res_pred = np.zeros((len(datasets), n_est+1, len(pred_space_small)))

    base_reg = MLPRegressor(hidden_layer_sizes=(100, 10), random_state=2333)


    for i, (d_name, (X, y, ns)) in enumerate(datasets.items()):
        print(d_name)
        
        # get actual density
        new_s = np.ones_like(ns[1])
        spdf = ns2pdf(pred_space_small, (ns[0], new_s)).flatten()
        res_pred[i, 0] = norm_0_1(spdf)   
        
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
                res_pred[i, est_id+1] = n_pred
                
            else:
                #DPL
                for iter in range(iters):
                    est.partial_fit(X, y)                    
                    pred = est.score_samples(pred_space_small)
                    
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
                    
                    if iter==iters-1:
                        res_pred[i, est_id+1] = n_pred

    
        print(d_name, res[i,:,-1, -1])
        np.save('results/res_density_1d.npy', res)
        np.save('results/res_density_1d_v.npy', res_pred)

else:
    
    #Plot
    res = np.load('results/res_density_1d.npy')
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
    plt.savefig('figures/est_1d.png')  
    
    # Plot imgs
    labels = ['true', 'KDE-g', 'KDE-t', 'KDE-e', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']

    res = np.load('results/res_density_1d_v.npy')
    print(res.shape)
    
    fig, ax = plt.subplots(8,8,figsize=(20,20), sharex=True, sharey=True)
    
    for d_id, d_name in enumerate(datasets.keys()):
        for est_id, est in enumerate(labels):
            ax[d_id, est_id].plot(res[d_id, est_id])
            
            if est_id==0:
                ax[d_id, est_id].set_ylabel(d_name)
            if d_id==0:
                ax[d_id, est_id].set_title(est)
                
    plt.tight_layout()
    plt.savefig('figures/imgs_1d.png')
    
    
    #Plot learning
    labels = ['DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']
    cols = ['gray', 'tomato', 'forestgreen', 'cornflowerblue']

    res = np.load('results/res_density_1d.npy')[:,-4:] #datasets, estimators, iters, (mse, mae)
    res = res[:,:,:,[0,2,3]]
    print(res.shape)
    
    fig, ax = plt.subplots(8,3,figsize=(14,14), sharex=True)
    
    for d_id, d_name in enumerate(datasets.keys()):
        for m_id, m in enumerate(['Statistic', 'MSE', 'MAE']):
            for e_id, e in enumerate(labels):
                temp = gaussian_filter1d(res[d_id, e_id, :, m_id], 3)
                ax[d_id, m_id].plot(temp, label=e, color=cols[e_id], alpha=0.75)
            
            if m_id==0:
                ax[d_id, m_id].set_ylabel(d_name)
            if d_id==0:
                ax[d_id, m_id].set_title(m)
            
            ax[d_id, m_id].grid(ls=':')
                
    ax[0,0].legend(ncol=2, frameon=False)
                
    plt.tight_layout()
    plt.savefig('figures/learning_1d.png')
    
    