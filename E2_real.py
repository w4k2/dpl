"""
Klasyfikacja danych rzeczywistych

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
import os

# Data configuration
datasets = os.listdir('datasets')
try:
    datasets.remove('.DS_Store')
except:
    pass


# Method configuration
n_methods = 8

# Experiment configuration
n_splits = 2
n_repeats = 5

res = np.zeros((len(datasets), n_splits*n_repeats, n_methods))
pbar = tqdm(total=len(datasets)*n_methods*n_splits*n_repeats)

#### Experiment ####

# Configure method

for dataset_id, dataset_name in enumerate(datasets):
    data = np.loadtxt('datasets/%s' % dataset_name, delimiter=',')
    X, y = data[:,:-1], data[:,-1].astype(int)
        
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
            
    print(dataset_name, np.mean(res[dataset_id], axis=0))
    np.save('results/E2_real.npy', res)
                    
                    
                        
                        
        

