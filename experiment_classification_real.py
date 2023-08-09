import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import matplotlib.pyplot as plt

exp = False
datasets = os.listdir('datasets')
datasets.remove('.DS_Store')

#####
if exp == True:

    n_clf = 6
    splits = 5
    res = np.zeros((len(datasets), splits, n_clf))

    base_reg = MLPRegressor(hidden_layer_sizes=(100), random_state=2333)

    for i, d_name in enumerate(datasets):
        print(d_name)

        data = np.loadtxt('datasets/%s' % d_name, delimiter=',')
        X, y = data[:,:-1], data[:,-1].astype(int)
        #cross val
        skf = StratifiedKFold(n_splits=splits)
        for fold, (train, test) in enumerate(skf.split(X, y)):
            
            classifiers = [
                GaussianNB(),
                KNeighborsClassifier(),
                MLPClassifier(hidden_layer_sizes=(100,10), random_state=2331),
                DPL(base_clf=clone(base_reg), curve_quants=10, 
                    monotonic=False, max_iter=25, norm='sqrt'),
                DPL(base_clf=clone(base_reg), curve_quants=10,
                    monotonic=False, max_iter=25, norm='log'),
                DPL(base_clf=clone(base_reg), curve_quants=10,
                    monotonic=False, max_iter=25, norm='norm'),
            ]
            
            for clf_id, clf in enumerate(classifiers):
                clf.fit(X[train], y[train])
                bac = balanced_accuracy_score(y[test], clf.predict(X[test]))
                res[i, fold, clf_id] = bac
        
        print(d_name, np.mean(res[i], axis=0))
        np.save('results/res.npy', res)
        
###
else:
    
    datasets = [d.split('.')[0] for d in datasets]
    
    #Plot
    res = np.load('results/res.npy')
    print(res.shape) # datasets x folds x clfs
    
    labels = ['GNB', 'KNN', 'MLP', 'DPL-sqrt', 'DPL-log', 'DPL-norm']
    cols = ['b','b','b','r','r','r']
    markers = ['o','x','*','s','d','*']
    
    res_mean = np.mean(res, axis=1)
    
    fig, ax = plt.subplots(2,1,figsize=(15,10), sharex=True)
    ax[0].imshow(res_mean.T, cmap='coolwarm')
    
    ax[0].set_yticks(np.arange(6), labels)
    
    for clf_id in range(6):
        ax[1].scatter(np.arange(len(datasets)), res_mean[:,clf_id], 
                      label=labels[clf_id], marker=markers[clf_id], color=cols[clf_id],
                      alpha=0.7)
    ax[1].set_xticks(np.arange(len(datasets)), datasets, rotation=90)
    ax[1].set_ylim(0.5,1)
    ax[1].grid(ls=':')

    ax[1].legend()

    
    plt.tight_layout()
    plt.savefig('figures/clf.png')
    
    
    
    
    