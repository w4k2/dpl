import numpy as np
import matplotlib.pyplot as plt
from methods import DPL
from utils import snakeway, ns2pdf
from sklearn.neural_network import MLPRegressor
from os import system

n_iters = 1024

# Generate data
n_samples = 1000
X, y, ns = snakeway(n_samples=n_samples, n_centroids=6)

# Initialize models
m_reg = MLPRegressor(hidden_layer_sizes=(1000),learning_rate_init=1e-2)
s_reg = MLPRegressor(hidden_layer_sizes=(1000),learning_rate_init=1e-3)

cc = {
    'curve_quants': 16
}
models = {
    'monotonic': DPL(s_reg, **cc, monotonic=True),
    'dychotomic': DPL(m_reg, **cc, monotonic=False)
}

dd = 2

model = models['monotonic']

d = 4096
prober = np.linspace(-5,5, d).reshape(d, 1)


fig, ax = plt.subplots(7,2,figsize=(10,10*1.2))

az = ax[0,0]
az.hist(X[y==0], range=(-5,5), 
        bins=1000, log=True,
        color='black')
az.hist(X[y==1], range=(-5,5), 
        bins=1000, log=True,
        color='tomato')

az.set_ylim(1e-1,1e2)
az.grid(ls=":")
az.set_title('recognized log-distribution (snakeway%i)' % n_samples)

for iter in range(n_iters):
    print('Iteration %i' % iter)

    dd2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14][(iter//8) % 19]

    prober2 = np.linspace(-dd2 * 5, dd2 * 5, d).reshape(d, 1)
        
    for idx, (name, model) in enumerate(models.items()):
        print(idx, name, model)
        model.partial_fit(X, y)
        
        pred = model.decfunc(prober)
        pred2 = model.decfunc(prober2)
        
        abspred = np.abs(pred)
        abspred2 = np.abs(pred2)
        
        aa = ax[1, idx]
        aa.set_title('regression curve vertical close-up [%s]' % name)
        aa.set_ylim(-5, 5)
        aa.set_yticks([-4, -1, 0, 1, 4])
        
        ab = ax[2, idx]
        ab.set_title('regression curve close neighborhood')
        
        for a in [aa,ab]:
            a.grid(ls=":")
            a.plot(prober, pred, lw=1, alpha=.1, c='tomato')
            a.plot(prober, np.mean(pred, axis=1), color='white', lw=5, alpha=1)
            a.plot(prober, np.mean(pred, axis=1), color='black', lw=1, alpha=1)
            a.plot(prober, np.max(pred, axis=1), color='tomato', lw=1, alpha=1, ls=":")
            a.plot(prober, np.min(pred, axis=1), color='tomato', lw=1, alpha=1, ls=":")
                
        ac = ax[3,idx]
        ac.plot(prober, abspred, lw=1, alpha=.05, c='tomato')
        ac.plot(prober, np.mean(abspred, axis=1), color='white', lw=5, alpha=1)
        ac.plot(prober, np.mean(abspred, axis=1), color='black', lw=1, alpha=1)
        ac.plot(prober, np.max(abspred, axis=1), color='tomato', lw=1, alpha=1, ls=":")
        ac.grid(ls=":")
        ac.set_yscale('log')
        ac.set_ylim(1e-6, 1e7)
        ac.set_title('absolute log answer')

        cmap = plt.get_cmap('Dark2')
        anticolor = cmap(((iter//8)%19)/19)

        ac = ax[4,idx]
        ac.plot(prober2, abspred2, lw=1, alpha=.1, c=anticolor)
        ac.plot(prober2, np.mean(abspred2, axis=1), color='white', lw=5, alpha=1)
        ac.plot(prober2, np.mean(abspred2, axis=1), color=anticolor, lw=1, alpha=1)
        ac.plot(prober2, np.max(abspred2, axis=1), color=anticolor, lw=1, alpha=1, ls=":")
        ac.plot(prober2, np.min(abspred2, axis=1), color=anticolor, lw=1, alpha=1, ls=":")
        ac.grid(ls=":")
        ac.set_yscale('log')
        ac.set_ylim(1e-20, 1e20)
        ac.set_xlim(prober2[0], prober2[-1])
        ac.set_title('absolute log answer antipodes [%.0E]' % dd2)
            
        ad = ax[5,idx]
        ad.plot(prober2, pred2, lw=1, alpha=.05, c=anticolor)
        ad.plot(prober2, np.mean(pred2, axis=1), lw=5, alpha=1, c='white')
        ad.plot(prober2, np.mean(pred2, axis=1), lw=1, alpha=1, c=anticolor)
        ad.grid(ls=":")
        ad.set_title('regression curve antipodes')

        al = ax[6,idx]
        al.plot(model.clf.loss_curve_, color='black', lw=1, alpha=1)
        al.grid(ls=":")
        al.set_yscale('log')
        al.set_title('Log loss')
    
    fig.suptitle('Learning EDPC(full_curve) on static evidence', fontsize=20)
    plt.tight_layout()
    plt.savefig('frames/%04i.png' % model.epoch)
    system('cp frames/%04i.png frames/last.png' % model.epoch)
    
    for aa in ax[1:].ravel():
        aa.cla()
    


