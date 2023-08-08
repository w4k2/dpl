from sklearn.neural_network import MLPRegressor
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import cdist
import numpy as np

ENTROPY_MODES = ['inner', 'outer', 'both', 'none']
DEFAULT_MODEL = MLPRegressor(hidden_layer_sizes=(100,100,100),
                             learning_rate_init=1e-1,)

class EDPC(ClassifierMixin, BaseEstimator):
    """
    EDPC – Entropy Distance Profile Classifier
    """
    def __init__(self, 
                 base_clf=DEFAULT_MODEL, 
                 curve_quants='full',
                 max_iter=256,
                 monotonic=True,
                 batch_size=1., # it's a float
                 ):
        self.base_clf = base_clf
        self.curve_quants = curve_quants
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.monotonic = monotonic
        
    def _prepare(self, X, y=None):
        if not hasattr(self, 'clf'):
            # Gathering info and preparing parameters
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
            self.curve_quants = self.n_samples - 1 if self.curve_quants == 'full' else self.curve_quants
            self.epoch = 0
                        
            # Preparing model
            self.clf = clone(self.base_clf)
        
    def fit(self, X, y, debug=None):
        self._prepare(X, y)                
        
        # Fit model
        for epoch in range(self.max_iter):    
            self.partial_fit(X, y)
    
            # Debug
            debug(self) if debug is not None else None    
            
        # Train gaussian integrator
        self.integrator = GaussianNB().fit(self.clf.predict(X), y)
        
        return self
    
    def partial_fit(self, X, yy):
        y = np.ones_like(yy) if self.monotonic else yy

        self._prepare(X, y) 
        
        # Prepare distance representation
        representation = np.sort(cdist(X, X, 'euclidean'), axis=1)[:, 1:self.curve_quants+1]
        representation = np.sqrt(representation)
        representation[y==1] = -representation[y==1]
        
        # Evidence update
        mask = np.random.uniform(size=self.n_samples) < self.batch_size
        
        self.clf.partial_fit(X[mask], 
                             representation[mask])

        self.representation = representation
        self.epoch += 1
    
    def decfunc(self, X):
        return self.clf.predict(X)
    
    def predict(self, X):
        return self.integrator.predict(self.decfunc(X))
        
    def predict_proba(self, X):
        return self.integrator.predict_proba(self.decfunc(X))
        