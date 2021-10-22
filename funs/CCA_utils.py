import numpy as np
from sklearn.cross_decomposition import  CCA

def CCA_s(XX, XX_val, keyS, keyT, stimss, zdimss, timepointss, trialss, trialss_val): 
    # source mean stimuli
    temp0S = np.mean([XX[keyS][i] for i in list(XX[keyS].keys())], axis=1)
    temp1S = temp0S.reshape((np.prod(temp0S.shape[0:2]), temp0S.shape[2]), order='C')
    # target mean stimuli
    temp0T = np.mean([XX[keyT][i] for i in list(XX[keyT].keys())], axis=1)
    temp1T = temp0T.reshape((np.prod(temp0T.shape[0:2]), temp0T.shape[2]), order='C')
    # source trial stimuli
    temp0Szz = np.concatenate([XX[keyS][i] for i in list(XX[keyS].keys())])
    temp1Szz = temp0Szz.reshape((np.prod(temp0Szz.shape[0:2]), temp0Szz.shape[2]), order='C')
    # target trial stimuli
    temp0Tzz = np.concatenate([XX[keyT][i] for i in list(XX[keyT].keys())])
    temp1Tzz = temp0Tzz.reshape((np.prod(temp0Tzz.shape[0:2]), temp0Tzz.shape[2]), order='C')
    # source trial stimuli validation
    temp0_valT = np.concatenate([XX_val[keyT][i] for i in list(XX_val[keyT].keys())])
    temp1_valT = temp0_valT.reshape((np.prod(temp0_valT.shape[0:2]), temp0_valT.shape[2]), order='C')

    # number of trials per condition
    trialsS     = int(temp0Szz.shape[0]/stimss)
    trialsT     = int(temp0Tzz.shape[0]/stimss)
    trialsT_val = int(temp0_valT.shape[0]/stimss)
    
    # CCA 
    cca = CCA(n_components=zdimss)
    cca.fit(temp1S, temp1T)
    zzzT, zzzS = cca.transform(temp1Szz, temp1Tzz)
    _, zzz_val = cca.transform(temp1Szz, temp1_valT)
    
    # structure data for decoding
    zzz1 = np.reshape(zzzT, (stimss, trialsS*timepointss ,zdimss), order='C')
    zzz2 = np.reshape(zzz1, (stimss, trialsS,timepointss ,zdimss), order='C')
    nn1  = np.reshape(zzz2, (stimss* trialsS, timepointss, zdimss), order='C' )
    zz_decodT = np.reshape(nn1,  (stimss* trialsS, timepointss* zdimss), order='F' )
    yyT = np.array([np.ones(trialsS)*i for i in range(stimss)]).ravel()
    
    zzz1 = np.reshape(zzzS, (stimss, trialsT*timepointss ,zdimss), order='C')
    zzz2 = np.reshape(zzz1, (stimss, trialsT,timepointss ,zdimss), order='C')
    nn1  = np.reshape(zzz2, (stimss* trialsT, timepointss, zdimss), order='C' )
    zz_decodS = np.reshape(nn1,  (stimss* trialsT, timepointss* zdimss), order='F' )
    yyS = np.array([np.ones(trialsT)*i for i in range(stimss)]).ravel()

    zzz1_val = np.reshape(zzz_val, (stimss, trialsT_val*timepointss ,zdimss), order='C')
    zzz2_val = np.reshape(zzz1_val, (stimss, trialsT_val,timepointss ,zdimss), order='C')
    nn1_val  = np.reshape(zzz2_val, (stimss* trialsT_val, timepointss, zdimss), order='C' )
    zz_decodS_val = np.reshape(nn1_val,  (stimss* trialsT_val, timepointss* zdimss), order='F' )
    yyS_val  = np.array([np.ones(trialsT_val)*i for i in range(stimss)]).ravel()

    return zz_decodT, yyT, zz_decodS, yyS, zz_decodS_val, yyS_val



