from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import FactorAnalysis
import numpy as np

def FA_procustes(XX, stimss, zdimss, timepointss, trialss, Rs=None, project_only=False):
    fooo = []
    sbjidxx = 0
    zz_FA = {}
    zz_FA_decod = []
    yy_FA_decod = []
    if Rs is None:
        Rs = {}
    for key in list(XX.keys()):
        temp0 = np.concatenate([XX[key][i] for i in list(XX[key].keys())])
        temp1 = temp0.reshape((np.prod(temp0.shape[0:2]), temp0.shape[2]), order='C')

        trialss = int(temp0.shape[0]/stimss)
        
        avg0 = np.mean([XX[key][i] for i in list(XX[key].keys())], axis=1)
        avg1 = avg0.reshape((np.prod(avg0.shape[0:2]), avg0.shape[2]), order='C')

        fa = FactorAnalysis(n_components=zdimss)
        fa.fit(avg1)
        zzz_avg = fa.transform(avg1)
        if project_only == True:
            print('projection only')
            if sbjidxx == 0:
                template = np.copy(zzz_avg)
                zzz = fa.transform(temp1)
            elif sbjidxx != 1:
                zzz = np.dot(zzz, Rs[key])
            
        else:
            if sbjidxx == 0:
                template = np.copy(zzz_avg)
                zzz = fa.transform(temp1)
            elif sbjidxx != 0:
                R, _ = orthogonal_procrustes(zzz_avg, template)
                zzz = fa.transform(temp1)
                zzz = np.dot(zzz, R)
                Rs[key] = R

        zzz1 = np.reshape(zzz, (stimss, trialss*timepointss ,zdimss), order='C')
        zzz2 = np.reshape(zzz, (stimss, trialss,timepointss ,zdimss), order='C')
        zz_FA[key] = zzz2
        sbjidxx +=1
        
        nn1 = np.reshape(zzz2, (stimss* trialss, timepointss, zdimss), order='C' )
        nn2 = np.reshape(nn1,  (stimss* trialss, timepointss* zdimss), order='F' )
        zz_FA_decod.append(nn2)
        yy_FA_decod.append(np.array([np.ones(trialss)*i for i in range(stimss)]).ravel())
    return zz_FA, Rs, np.concatenate(zz_FA_decod), np.concatenate(yy_FA_decod)




