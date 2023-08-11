"""
Hiperparametry / konfiguracja

Dane syntetyczne z make_classification

"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from methods import DPL
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

# Data configuration
n_samples = 1000
weights = [
    [0.5, 0.5],
    [0.75, 0.25],
    [0.9, 0.1]
    ]
repeats_data = 10

# Method configuration

n_iter = 32
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
n_splits = 2
n_repeats = 5
random_states = np.random.randint(100, 100000, repeats_data)

res = np.zeros((len(weights), repeats_data, 
                len(curve_quants), len(integrators), len(base_regressors), 
                len(transforms), n_splits*n_repeats))

pbar = tqdm(total=len(weights)*repeats_data*\
                len(curve_quants)*len(integrators)*len(base_regressors)*\
                len(transforms)*n_splits*n_repeats)

#### Experiment ####

# Generate data

for w_id, w in enumerate(weights):
    for rs_id, rs in enumerate(random_states):
        
        X, y = make_classification(
            n_samples = n_samples,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            class_sep=0.5,
            weights=w,
            flip_y=0.,
            random_state=rs
        )

        # Configure method
        for cq_id, cq in enumerate(curve_quants):
            for int_id, itg in enumerate(integrators):
                for br_id, br in enumerate(base_regressors):
                    for t_id, t in enumerate(transforms):
                        
                        dpl = DPL(
                            base_clf=clone(br),
                            curve_quants=cq,
                            max_iter=n_iter,
                            monotonic=False,
                            transform=t,
                            integrator=clone(itg)
                        )
                        
                        # K-fold
                        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rs)
                        for fold, (train, test) in enumerate(rskf.split(X, y)):
                            pred = dpl.fit(X[train], y[train]).predict(X[test])
                            res[w_id, rs_id, cq_id, int_id,br_id, t_id, fold] = balanced_accuracy_score(y[test], pred)
                            
                            pbar.update(1)
                            
        np.save('results/E1_clf.npy', res)
                            
                            
                            
                            
                

