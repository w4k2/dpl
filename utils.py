import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def ns2pdf(x, ns):
    """
    Convert norms-signs tuple to probability distribution function
    """
    return np.sum(np.array([n.pdf(x)*s for n, s in zip(*ns)]), axis=0)
    

def zeto(n_samples=1000, m_centroids=3, factor=3, sigma=1,
         translation=0, normalize=True):
    base_centroids = np.linspace(-factor*(m_centroids-1), 
                            factor*(m_centroids-1), m_centroids)

    centroids = np.array(np.meshgrid(base_centroids, base_centroids)).reshape(2,-1).T

    if normalize:
        centroids -= np.mean(centroids)
        std = np.std(centroids)
        centroids /= std
    else:
        std = 1
    
    c_samples = n_samples // (m_centroids*m_centroids)
        
    norms = [norm(centroid+translation, 1/std) for centroid in centroids]
    signs = [1 if i%2 == 0 else -1 for i in range(m_centroids*m_centroids)]
    
    X = np.concatenate([np.random.normal(centroid, sigma/std, 
                                         size=(c_samples,2)) 
                        for centroid in centroids])

    y = np.concatenate([np.ones(c_samples) * int(i%2 == 0) 
                        for i in range(m_centroids*m_centroids)])

    X += translation

    return X, y, (norms, signs)
    
def snakeway(n_samples=1000, factor=3, 
             sigma=1, n_centroids=3,
             translation=0, normalize=True):
    centroids = np.linspace(-factor*(n_centroids-1), 
                            factor*(n_centroids-1), n_centroids)
    
    if normalize:
        centroids -= np.mean(centroids)
        std = np.std(centroids)
        centroids /= std
    else:
        std = 1
    
    c_samples = n_samples // n_centroids
    
    norms = [norm(centroid+translation, 1/std) for centroid in centroids]
    signs = [1 if i%2 == 0 else -1 for i in range(n_centroids)]
    # print(norms, signs)
    
    X = np.concatenate([np.random.normal(centroid, sigma/std, size=c_samples) 
                        for centroid in centroids])
    y = np.concatenate([np.ones(c_samples) * int(i%2 == 0) 
                        for i in range(n_centroids)])
        
    X += translation
    
    return X.reshape(-1,1), y, (norms, signs)

def norm_0_1(data):
    data -=np.nanmin(data)
    data /=np.nanmax(data)
    return data