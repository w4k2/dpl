from sklearn.neural_network import MLPRegressor
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import hmean

ENTROPY_MODES = ['inner', 'outer', 'both', 'none']
DEFAULT_MODEL = MLPRegressor(hidden_layer_sizes=(100,100,100),
                             learning_rate_init=1e-1,)
DEFAUILT_INTEGRATOR = GaussianNB()

#TODO zobaczyć czy translacja reprezentacji robi cokolwiek (w przypadku niezbalansowanych może)

class DPL(ClassifierMixin, BaseEstimator):
    """
    DPL – Distance Profile Layer
    """
    def __init__(self, 
                 base_clf=DEFAULT_MODEL,
                 curve_quants='full',
                 max_iter=256,
                 monotonic=True, # 1 = estymator gęstości rozkładu, 0 = klasyfikator
                 batch_size=1., # it's a float
                 transform = 'none',
                 integrator = DEFAUILT_INTEGRATOR
                 ):
        self.base_clf = base_clf
        self.curve_quants = curve_quants
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.monotonic = monotonic
        self.transform = transform
        self.integrator = integrator
        
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
            
        # Train integrator
        self.integrator.fit(self.clf.predict(X), y)
        
        return self
    
    def partial_fit(self, X, yy):
        y = np.ones_like(yy) if self.monotonic else yy # Nadzorowane/nie

        self._prepare(X, y) # ignorowane po pierwszym wywołaniu
        
        # Prepare distance representation
        representation = np.sort(cdist(X, X, 'euclidean'), axis=1)[:, 1:self.curve_quants+1] # curve_quants najmniejszych odległości do każdego obiektu
        if self.transform == 'sqrt':
            representation = np.sqrt(representation)
        elif self.transform == 'log':
            representation = np.log(representation+1)
        elif self.transform == 'std_norm':
            representation -= np.mean(representation)
            representation /= (np.std(representation)+0.001)
        elif self.transform == 'none':
            pass
        else:
            raise ValueError('transform undefined')

        representation[y==1] = -representation[y==1] # zmieniamy reprezentacje dla etykiety 1
        
        # Evidence update
        mask = np.random.uniform(size=self.n_samples) < self.batch_size # losowo wybrany batch size obiektów
        
        # print(representation[mask])
        self.clf.partial_fit(X[mask], representation[mask]) # Uczymy regresor reprezentacji (czyli sqrt z odległości do najbliższych)
        self.epoch += 1
    
    def decfunc(self, X):
        return self.clf.predict(X) # odpowiedzią jest estymowana odległość
    
    def predict(self, X):
        return self.integrator.predict(self.decfunc(X))
        
    def predict_proba(self, X):
        return self.integrator.predict_proba(self.decfunc(X))
    
    def score_samples(self, X):
        dist = -self.decfunc(X) # estimates the distance to curve_quants nearest neighbors
        dist[dist<0]=0.001
               
        dist_std = np.std(dist, axis=1)
        dist_hmean = hmean(dist, axis=1)
        
        fn = 1/(dist_hmean * dist_std)
        return fn
        
        # print(fn)
        # print(np.sum(np.isnan(fn)))
        # print(np.sum(np.isinf(fn)))
        
        # plt.title(self.epoch)
        # plt.scatter(*X.T, c=fn, cmap='coolwarm')
        # plt.savefig('foo.png')
        
