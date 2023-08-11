"""
Klasyfikacja danych syntetycznych

"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from methods import DPL
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils import snakeway, zeto

# Data configuration
n_samples = 1000
repeats_data = 3
random_states = np.random.randint(100, 1000, repeats_data)

# Our synthetic
n_centroids = [2, 3, 5, 7]
factors = [1, 3, 8]
sigmas = [1, 3]

# make classification synthetic
n_features = [10, 20, 30]
class_sep = [1., 0.5, 0.2]
weights = [
    [0.5, 0.5],
    [0.75, 0.25],
    [0.9, 0.1]
    ]

# moons/circles synthetic
noises = [0, 1, 3]

# Prepare datasets
datasets = []
dataset_names = []

for n_c in n_centroids:
    for f in factors:
        for s in sigmas:
            datasets.append(snakeway(n_samples=n_samples, n_centroids=n_c, factor=f, sigma=s)[:2])
            dataset_names.append('snake_c%i_f%i_s%i' % (n_c, f, s))
            
            datasets.append(zeto(n_samples=n_samples, m_centroids=n_c, factor=f, sigma=s)[:2])
            dataset_names.append('zeto_c%i_f%i_s%i' % (n_c, f, s))

for rs_id, rs in enumerate(random_states):
    for n_f in n_features:
        for cs in class_sep:
            for w in weights:
                datasets.append(make_classification(n_samples=n_samples, 
                                                    n_features=n_f, 
                                                    n_informative=n_f, 
                                                    n_redundant=0, 
                                                    n_repeated=0, 
                                                    class_sep=cs, 
                                                    weights=w,
                                                    random_state=rs))

                dataset_names.append('mc_f%i_cs%0.1f_w%0.1f_rs%i' % (n_f, cs, w[0], rs))
    
    for n in noises:
        datasets.append(make_moons(n_samples=n_samples, noise=n, random_state=rs))
        dataset_names.append('moons_n%0.1f_rs%i' % (n, rs))
        
        datasets.append(make_circles(n_samples=n_samples, noise=n, random_state=rs))
        dataset_names.append('circles_n%0.1f_rs%i' % (n, rs))


n_datasets = len(datasets)
print(dataset_names)

# Method configuration
n_methods = 8

# Experiment configuration
n_splits = 2
n_repeats = 5

res = np.zeros((n_datasets, n_splits*n_repeats, n_methods))
pbar = tqdm(total=n_datasets*n_methods*n_splits*n_repeats)

#### Experiment ####

# Configure method

for dataset_id, (X, y) in enumerate(datasets):
        
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=23422)
    for fold, (train, test) in enumerate(rskf.split(X, y)):

        methods = [
            KNeighborsClassifier(),
            GaussianNB(),
            SVC(),
            MLPClassifier(hidden_layer_sizes=(100,100,10)),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100,100)), curve_quants=10, max_iter=32, monotonic=False, transform='none'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100,100)), curve_quants=10, max_iter=32, monotonic=False, transform='sqrt'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100,100)), curve_quants=10, max_iter=32, monotonic=False, transform='log'),
            DPL(base_clf=MLPRegressor(hidden_layer_sizes = (100,100)), curve_quants=10, max_iter=32, monotonic=False, transform='std_norm'),
        ]
        
        for clf_id, clf in enumerate(methods):
            
            pred = clf.fit(X[train], y[train]).predict(X[test])
            res[dataset_id, fold, clf_id] = balanced_accuracy_score(y[test], pred)

            pbar.update(1)
            
    print(dataset_names[dataset_id], np.mean(res[dataset_id], axis=0))
    np.save('results/E2_syn.npy', res)
                    
                    
                        
                        
        

