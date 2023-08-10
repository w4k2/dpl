import numpy as np
import matplotlib.pyplot as plt
from utils import snakeway, ns2pdf, zeto
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

datasets = {
    's1000': snakeway(n_samples=2000, n_centroids=3)[:2],
    's2000': snakeway(n_samples=2000, n_centroids=5, translation=2)[:2],
    's3000': snakeway(n_samples=2000, n_centroids=7, sigma=3)[:2],
    's4000': snakeway(n_samples=2000, n_centroids=9, factor=8)[:2],
    'z1000': zeto(n_samples=2000, m_centroids=3)[:2],
    'z2000': zeto(n_samples=2000, m_centroids=5, translation=2)[:2],
    'z3000': zeto(n_samples=2000, m_centroids=7, sigma=3)[:2],
    'z4000': zeto(n_samples=2000, m_centroids=9, factor=8)[:2],
    'mc_imb1': make_classification(n_samples=2000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0, class_sep=1., weights=[0.9,0.1]),
    'mc_imb05': make_classification(n_samples=2000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0, class_sep=.5, weights=[0.9,0.1]),
    'mc_imb025': make_classification(n_samples=2000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0, class_sep=.25, weights=[0.9,0.1]),
    'moons1': make_moons(n_samples=2000, noise=1),
    'moons05': make_moons(n_samples=2000, noise=0.5),
    'moons025': make_moons(n_samples=2000, noise=0.25),
    'circles1': make_circles(n_samples=2000, noise=1),
    'circles05': make_circles(n_samples=2000, noise=0.5),
    'circles025': make_circles(n_samples=2000, noise=0.25)
}

exp = False

#####
if exp == True:

    n_clf = 7
    splits = 5
    res = np.zeros((len(datasets), splits, n_clf))

    base_reg = MLPRegressor(hidden_layer_sizes=(100), random_state=2333)

    for i, (d_name, (X, y)) in enumerate(datasets.items()):
        print(d_name)

        #cross val
        skf = StratifiedKFold(n_splits=splits)
        for fold, (train, test) in enumerate(skf.split(X, y)):
            
            classifiers = [
                GaussianNB(),
                KNeighborsClassifier(),
                MLPClassifier(hidden_layer_sizes=(100,10), random_state=2331),
                DPL(base_clf=clone(base_reg), curve_quants=10,
                    monotonic=False, max_iter=25, transform='none'),
                DPL(base_clf=clone(base_reg), curve_quants=10,
                    monotonic=False, max_iter=25, transform='sqrt'),
                DPL(base_clf=clone(base_reg), curve_quants=10, 
                    monotonic=False, max_iter=25, transform='log'),
                DPL(base_clf=clone(base_reg), curve_quants=10, 
                    monotonic=False, max_iter=25, transform='std_norm'),
            ]
            
            for clf_id, clf in enumerate(classifiers):
                clf.fit(X[train], y[train])
                bac = balanced_accuracy_score(y[test], clf.predict(X[test]))
                res[i, fold, clf_id] = bac
        
        print(d_name, np.mean(res[i], axis=0))
        np.save('results/res_syn.npy', res)
        
###
else:
    
    #Plot
    res = np.load('results/res_syn.npy')
    print(res.shape) # datasets x folds x clfs
    
    labels = ['GNB', 'KNN', 'MLP', 'DPL-none', 'DPL-sqrt', 'DPL-log', 'DPL-std_norm']
    cols = ['b','b','b','r','r','r','r']
    markers = ['o','x','*','s','d','*','o']
    
    res_mean = np.mean(res, axis=1)
    
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    ax[0].imshow(res_mean.T, cmap='coolwarm')
    
    ax[0].set_yticks(np.arange(7), labels)
    
    for clf_id in range(7):
        ax[1].scatter(np.arange(len(datasets)), res_mean[:,clf_id], 
                      label=labels[clf_id], marker=markers[clf_id], color=cols[clf_id],
                      alpha=0.7)
    ax[1].set_xticks(np.arange(len(datasets)), datasets.keys(), rotation=90)
    ax[1].set_ylim(0.5,1)
    ax[1].grid(ls=':')
    
    ax[1].legend()

    
    plt.tight_layout()
    plt.savefig('figures/clf_syn.png')  
    