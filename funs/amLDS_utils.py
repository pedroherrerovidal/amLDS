import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

# amLDS random parameter initialization functions
def A_init(zdim,on_mu=0.4, on_std=.1, off_mu=0, off_std=.2):
    M = np.random.normal(off_mu, off_std, (zdim, zdim))
    np.fill_diagonal(M, np.random.normal(on_mu, on_std, (zdim, )))
    _, v, _ = np.linalg.svd(M)
    # make sure its full rank and eigenvalues are smaller than one
    if str(np.linalg.matrix_rank(M))==str(M.shape[0]) and sum(v >= 1) ==0:
        return M # check full-rank
    else:
        A_init(zdim,on_mu, on_std, off_mu, off_std)

def Q_init(zdim, noise=.25):
    return np.abs( np.eye(zdim)*np.random.normal(noise, .0025, zdim) )

def x0_init(zdim, noise=.35):
    return np.random.normal(0, noise, (zdim,))

def R_init(xdim, noise=.5):
    return np.abs( np.diag(np.random.normal(0, noise, (xdim,))) ) 

def C_init(xdim, zdim):
    return np.random.normal(0, .25, (xdim, zdim)) # 

def C_inits(C_gold, alpha=.1):
    u, s, v = np.linalg.svd(C_gold, full_matrices=True)
    smat = np.zeros((u.shape[0], s.shape[0]))
    smat[:s.shape[0], :s.shape[0]] = np.diag(s)

    a = np.linalg.norm(v)
    x_temp = (v/a) + alpha * np.random.normal(0, 1, (v.shape))
    x_prime = (a + np.random.normal(0, a/30)) * (x_temp/np.linalg.norm(x_temp))
    return u @ (smat @ x_prime)

def b_init(zdim, timepoints):
#     return np.zeros((zdim, timepoints))
    return np.random.normal(0, .45, (zdim, timepoints))

def b_init2D(Fs = 41, f = .5, sample = 41, a1 = 3, a2= 3):
    x  = np.arange(sample)
    y1 = np.sin(2*np.pi*f*x/Fs)*a1
    y2 = np.cos(2*np.pi*f*x/Fs)*a2
    return np.stack((y1-y1[0] , y2-y2[0] ))

def bs_init(stimIDs, angles =170):
    b_ = {}
    Nstim = len(stimIDs)
    thetas = np.linspace(0, angles, Nstim) 

    for ii in range(Nstim):
        b = b_init2D(a1=np.random.normal(1.1, .02), a2=np.random.normal(1, .02))
        theta = np.radians(thetas[ii])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        b_[stimIDs[ii]] = (R @ b)
    return b_
 
def b_init3D(Fs=41, f=.5, sample=41, a1=np.random.normal(1.1, .02), a2=np.random.normal(1, .02),
             a3=np.random.normal(1, .02)):
    x  = np.arange(sample)
    y1 = np.sin(2*np.pi*f*x/Fs)*a1
    y2 = np.cos(2*np.pi*f*x/Fs)*a2
    y3 = np.cos(2*np.pi*f*x/Fs)*a3
    return np.stack((y1-y1[0] , y2-y2[0] , y3-y3[0]))
    
def bs_init3D(stimIDs, angles =170, mu1=1.1,mu2=1,mu3=1):
    b_ = {}
    Nstim = len(stimIDs)
    thetas = np.linspace(0, angles, Nstim)

    for ii in range(Nstim):
        b = b_init3D(a1=np.random.normal(mu1, .02), a2=np.random.normal(mu2, .02),a3=np.random.normal(mu3, .02))
        theta = np.radians(thetas[ii])
        c, s = np.cos(theta), np.sin(theta)
#         R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        R = np.array(( (1, 0, 0), (0, c, -s), (0, s, c) ))
        b_[stimIDs[ii]] = (R @ b)
    return b_

# visualization functions
def plot_LatentsObserveds(ZZ, sbj, stimIDs, XX, sbjt, stit, trialN =5, cmap=cmap):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), facecolor='w')
    if len(stimIDs) > 5:
        stimIDs = [stimIDs[i] for i in np.linspace(0, len(stimIDs)-1, 5).astype(int)]
    for stim in range(len(stimIDs)):      
        ax1.plot(np.mean(ZZ[sbj][stimIDs[stim]],axis=0)[:, 0], 
                 np.mean(ZZ[sbj][stimIDs[stim]],axis=0)[:, 1], '-', c=cmap[stim], lw=3)
    
    for i in np.random.choice(trialN,5, replace=False):
        idx = 0
        for stim in stimIDs:
            ax1.plot(np.array(ZZ[sbj][stim])[i,:, 0], 
                     np.array(ZZ[sbj][stim])[i,:, 1], '--', c=cmap[idx],alpha=.1)
            idx+=1
    ax1.set_xlabel('LD1'); ax1.set_ylabel('LD2'); ax1.legend(stimIDs);
    ax1.set_title('Latent trajectories')
    
    res = sns.heatmap(XX[sbjt][stit][0].T, cmap='Blues', vmax=3, vmin=-3, ax=ax2) # Reds_r
    ax2.set_yticks([], []); ax2.set_xticks([], []); ax2.set_xlabel('time'); ax2.set_ylabel('Neurons');
    ax2.set_title('Observed measurements')

def plot_LatentsLatents(ZZ, sbj, stimIDs, ZZ2, trialN =5, cmap=cmap):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), facecolor='w')
    if len(stimIDs) > 5:
        stimIDs = [stimIDs[i] for i in np.linspace(0, len(stimIDs)-1, 5).astype(int)]
    for stim in range(len(stimIDs)):      
        ax1.plot(np.mean(ZZ[sbj][stimIDs[stim]],axis=0)[:, 0], 
                 np.mean(ZZ[sbj][stimIDs[stim]],axis=0)[:, 1], '-', c=cmap[stim], lw=3)
    
    for i in np.random.choice(trialN,5, replace=False):
        idx = 0
        for stim in stimIDs:
            ax1.plot(np.array(ZZ[sbj][stim])[i,:, 0], 
                     np.array(ZZ[sbj][stim])[i,:, 1], '--', c=cmap[idx],alpha=.1)
            idx+=1
    ax1.set_xlabel('LD1'); ax1.set_ylabel('LD2'); ax1.legend(stimIDs);
    ax1.set_title('Latent trajectories')
    
    for stim in range(len(stimIDs)):      
        ax2.plot(np.mean(ZZ2[sbj][stimIDs[stim]],axis=0)[:, 0], 
                 np.mean(ZZ2[sbj][stimIDs[stim]],axis=0)[:, 1], '-', c=cmap[stim], lw=3)
    for i in np.random.choice(trialN,5, replace=False):
        idx = 0
        for stim in stimIDs:
            ax2.plot(np.array(ZZ2[sbj][stim])[i,:, 0], 
                     np.array(ZZ2[sbj][stim])[i,:, 1], '--', c=cmap[idx],alpha=.2)
            idx+=1
    ax2.set_xlabel('LD1'); ax2.set_ylabel('LD2'); 
    ax2.set_title('Infered latents')
    
def plot_estimatedLatentDimensionality(LDS_err, LDS_LL, DIMS, sbjIDs):
    nLLdd = -np.array([ [LDS_LL[i][j] for j in sbjIDs] for i in range(len(DIMS)) ])

    fig, ax1 = plt.subplots(figsize=(3, 3.5), facecolor='w')

    ax2 = ax1.twinx()
    ax1.plot(DIMS, np.mean(LDS_err,axis=1), ls='-',  lw=2, c='k')
    ax1.plot(DIMS, np.array(LDS_err), ls='--', lw=1, c='k', alpha=.3)
    ax1.set_xticks([0, 5, 10, 15]);
    ax1.set_xlim(0,15);
    ax2.plot(DIMS, np.mean(nLLdd,axis=1), ls='-',  lw=2, c='darkorange')
    ax2.plot(DIMS, np.array(nLLdd), ls='--', lw=1, c='darkorange', alpha=.3)

    ax1.set_xlabel('# latent dim.')
    ax1.set_ylabel('reconstruction error', color='k')
    ax2.set_ylabel('n.l.l.', color='darkorange');

    
def plot_mixtures(ZZ, stimA, stimB, sbj, zamples, trialN, cmap=cmap):
    plt.figure(figsize=(4,4), facecolor='w')

    plt.plot(np.mean(ZZ[sbj][stimA],axis=0)[:, 0], 
               np.mean(ZZ[sbj][stimA],axis=0)[:, 1], '-', c=cmap[0], lw=4)
    plt.plot(np.mean(ZZ[sbj][stimB],axis=0)[:, 0], 
               np.mean(ZZ[sbj][stimB],axis=0)[:, 1], '-', c=cmap[1], lw=4)
    plt.plot(np.mean(np.array(zamples),axis=0)[:, 0], 
               np.mean(np.array(zamples),axis=0)[:, 1], '-', c='limegreen', lw=4)

    for i in np.random.choice(trialN,5, replace=False):
        plt.plot(np.array(ZZ[sbj][stimA])[i,:, 0], 
                   np.array(ZZ[sbj][stimA])[i,:, 1], '--', c=cmap[0], lw=1,alpha=.3)
        plt.plot(np.array(ZZ[sbj][stimB])[i,:, 0], 
                   np.array(ZZ[sbj][stimB])[i,:, 1], '--', c=cmap[1], lw=1,alpha=.3)
        plt.plot(np.array(np.array(zamples))[i,:, 0], 
                   np.array(np.array(zamples))[i,:, 1], '--', c='limegreen', lw=1,alpha=.3)

    plt.ylabel('LD-2')        
    plt.xlabel('LD-1')
    plt.legend(['stim-A','stim-B','stim-A+B']);

def plot_concentrationPerformance(paramsss, AccCN, sbjIDs):  
    toplot = np.array([[i[sbj] for i in AccCN] for sbj in sbjIDs])

    x = range(len(paramsss))
    y = np.mean(toplot, axis=0)
    yerr = stats.sem(toplot, axis=0)
    
    plt.figure(figsize=(4,4), facecolor='w')
    plt.plot(x, y, 'k', lw=3);
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.2, color='k')
    
    plt.ylabel('Accuracy')        
    plt.xlabel('Concentration')
    plt.xticks(x, [str(i[-1]*100) for i in paramsss]);
    
    