import numpy as np
import matplotlib.pyplot as plt
from methods import EDPC
from utils import snakeway
from sklearn.neural_network import MLPClassifier, MLPRegressor
from os import system

n_iters = 256

# Generate data
n_samples = 1000
X, y = snakeway(n_samples=n_samples)
X += 2

# Initialize models
s = MLPRegressor(
    hidden_layer_sizes=(1000),
    learning_rate_init=1e-3
)

cc = {
    'curve_quants': 'full'
}
models = {
    'none': MLPClassifier(learning_rate_init=1e-2, hidden_layer_sizes=(100)),
    #'inner': EDPC(m, entropy_mode='inner', **cc),
    #'outer': EDPC(m, entropy_mode='outer', **cc),
    #'both': EDPC(l, entropy_mode='both', **cc),
}

dd = 2

_model = EDPC(s, entropy_mode='none', **cc)

d = 4096
prober = np.linspace(-dd * _model.inner_radius,
                        dd * _model.inner_radius, d).reshape(d, 1)



fig, ax = plt.subplots(4,2,figsize=(10,10*1.2))
ax = ax.ravel()

az = ax[-4]
az.hist(X[y==0], range=(-dd * _model.inner_radius,
                      dd * _model.inner_radius), 
            bins=1000, log=True,
            color='black')
az.hist(X[y==1], range=(-dd * _model.inner_radius,
                      dd * _model.inner_radius), 
            bins=1000, log=True,
            color='tomato')

means = np.array([-15, -9, -3, 3, 9, 15])

az.set_ylim(1e-1,1e2)
az.grid(ls=":")
az.set_title('recognized log-distribution (snakeway%i)' % n_samples)

for iter in range(n_iters):
    print('Iteration %i' % iter)

    dd2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14][iter//8 % 19]

    prober2 = np.linspace(-dd2 * _model.outer_radius,
                            dd2 * _model.outer_radius, d).reshape(d, 1)
    
        
    for idx, (name, model) in enumerate(models.items()):
        print(idx, name, model)
        model.partial_fit(X, y, classes=[0, 1])
        
        pred = model.predict_proba(prober)[:,1].reshape(-1, 1)
        pred2 = model.predict_proba(prober2)[:,1].reshape(-1, 1)
        
        abspred = np.abs(pred)
        abspred2 = np.abs(pred2)
        
        for aa in ax:
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
        
        #for lidx, color in enumerate(colorsr):
        aa = ax[0]
        aa.set_title('regression curve vertical close-up')
        aa.set_ylim(-2, 2)
        aa.set_yticks([-4, -1, 0, 1, 4])
        
        ab = ax[1]
        ab.set_title('regression curve close neighborhood')
        
        for a in [aa,ab]:
            a.grid(ls=":")
            a.plot(prober, pred, lw=1, alpha=.05, c='tomato')
            a.plot(prober, np.mean(pred, axis=1), color='white', lw=5, alpha=1)
            a.plot(prober, np.mean(pred, axis=1), color='black', lw=1, alpha=1)
            a.plot(prober, np.max(pred, axis=1), color='tomato', lw=1, alpha=1, ls=":")
            a.plot(prober, np.min(pred, axis=1), color='tomato', lw=1, alpha=1, ls=":")
                
        ac = ax[2]
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

        ac = ax[-3]
        ac.plot(prober2, abspred2, lw=1, alpha=.05, c=anticolor)
        ac.plot(prober2, np.mean(abspred2, axis=1), color='white', lw=5, alpha=1)
        ac.plot(prober2, np.mean(abspred2, axis=1), color=anticolor, lw=1, alpha=1)
        ac.plot(prober2, np.max(abspred2, axis=1), color=anticolor, lw=1, alpha=1, ls=":")
        ac.plot(prober2, np.min(abspred2, axis=1), color=anticolor, lw=1, alpha=1, ls=":")
        ac.grid(ls=":")
        ac.set_yscale('log')
        ac.set_ylim(1e-20, 1e20)
        # ac.hlines(dd2, -1000000000, 1000000000, color='black', alpha=1)
        # ac.hlines(dd2, -1000000, 1000000, color='black', alpha=1)
        # ac.hlines(dd2, -1000, 1000, color='red', alpha=1)
        # ac.hlines(dd2, -100, 100, color='black', alpha=1)
        # ac.hlines(dd2, -10, 10, color='red', alpha=1)
        # ac.hlines(dd2, -1, 1, color='black', alpha=1)
        ac.set_xlim(prober2[0], prober2[-1])
        ac.set_title('absolute log answer antipodes [%.0E]' % dd2)

            
        ad = ax[3]
        ad.plot(prober2, pred2, lw=1, alpha=.05, c=anticolor)
        ad.plot(prober2, np.mean(pred2, axis=1), lw=5, alpha=1, c='white')
        ad.plot(prober2, np.mean(pred2, axis=1), lw=1, alpha=1, c=anticolor)
        ad.grid(ls=":")
        ad.set_title('regression curve antipodes')

        al = ax[-1]
        al.plot(model.loss_curve_, color='black', lw=1, alpha=1)
        al.grid(ls=":")
        al.set_yscale('log')
        al.set_title('Log loss')
        
        ak = ax[-2]
        #ak.plot(model.dropout, color='black', lw=1, alpha=1)
        #ak.set_title('Dropout cycle')
        
        for aa in [al, ak]:
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.grid(ls=":")
        
        # ax[idx*mf+5].plot(np.array(model.dropout)[-60:])
        
        
    ax[-1].set_xlabel('epoch')
    ax[-2].set_xlabel('epoch')
    
    #ax[-2].set_ylabel('dropout strength')
    ax[-1].set_ylabel('loss value')
    
    
    fig.suptitle('Learning MLP on static evidence', fontsize=20)
    plt.tight_layout()
    #plt.savefig('1d.png')
    plt.savefig('frames/%04i.png' % iter)
    system('cp frames/%04i.png frames/last.png' % iter)
    ac.cla()
    
    for i, aa in enumerate(ax):
        if i != 4:
            aa.cla()
    #plt.close()


