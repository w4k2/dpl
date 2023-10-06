import numpy as np
import os


datasets = os.listdir('datasets')
try:
    datasets.remove('.DS_Store')
except:
    pass


for dataset_id, dataset_name in enumerate(datasets):
    data = np.loadtxt('datasets/%s' % dataset_name, delimiter=',')
    X, y = data[:,:-1], data[:,-1].astype(int)
    _, counts = np.unique(y, return_counts=True)
    ir = np.round(np.min(counts)/len(y),3)
    
    print(dataset_id+1, dataset_name.split('.')[0], X.shape[0], X.shape[1], ir)