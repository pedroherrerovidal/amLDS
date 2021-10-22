import argparse
import pickle
import numpy as np
from funs.amLDS_engine import amLDS as amLDSm
from funs.amLDS_utils  import *
from funs.FAprocrustes_utils  import FA_procustes as FAp
from funs.CCA_utils  import CCA_s as CCAp
from sklearn import linear_model
import copy

def main_ex(nStim=50, nSbj=5, zdim=3, xdim=40, timepoints=41, trialSource=50, trialTargetval=20, Qnoise=.25, Rnoise =.5, Calpha=.1, parm_seed=21, idxx=33,idxx_val = 1033):
    sbjIDs  = ['sbj'+str(i)  for i in range(nSbj)]
    stimIDs = ['stim'+str(i) for i in range(nStim)]
    
    zdim = zdim                           # same for all subjects and stimuli
    xdim = {key: xdim for key in sbjIDs}  # subject specific ; could be different for different subjects
    timepoints = timepoints               # number of time steps ; it could vary on a trial by trial basis
    
    np.random.seed(21) #7
    # simulate true latent (stimulus specific) parameters
    true_A_  = {key: A_init(zdim)  for key in stimIDs}
    true_Q_  = {key: Q_init(zdim,noise=Qnoise)  for key in stimIDs}
    true_Q0_ = {key: Q_init(zdim,noise=Qnoise)  for key in stimIDs}
    true_x0_ = {key: x0_init(zdim) for key in stimIDs}
    # true_b_  = bs_init(stimIDs)
    true_b_  = bs_init3D(stimIDs)

    # simulate true observation (subject specific) parameters
    refC = C_init(xdim[sbjIDs[0]], zdim)
    true_C_  = {key: C_inits(refC,alpha=Calpha)     for key in sbjIDs}
    true_R_  = {key: R_init(xdim[key],noise=Rnoise) for key in sbjIDs}
    
    print('Generating synthetic dataset')
    amLDSsample = amLDSm(zdim=zdim, sbjID=sbjIDs, stimID=stimIDs)
    
    # sample from true latent parameters and structure observed data
    XX = {}
    for sbj in sbjIDs[:-1]:
        CC = {}
        for stim in stimIDs:
            samples = []
            for i in range(trialSource):
                samples.append(amLDSsample.sample(A=true_A_[stim], Q=true_Q_[stim], C=true_C_[sbj], R=true_R_[sbj], 
                                                  x0=true_x0_[stim], b=true_b_[stim], timepoints= timepoints,seed=idxx)[1])
                idxx +=1
            CC[stim] = samples
        XX[sbj] = CC

    # sample from true latent parameters and structure observed data
    CC = {}
    CC_val = {}
    for stim in stimIDs:
        samples = []
        for i in range(250): # should be high enough
            samples.append(amLDSsample.sample(A=true_A_[stim], Q=true_Q_[stim], C=true_C_[sbj], R=true_R_[sbj], 
                                      x0=true_x0_[stim], b=true_b_[stim], timepoints= timepoints, seed=idxx)[1])
            idxx +=1
        samples_val = []
        for i in range(int(trialTargetval)):
            samples_val.append(amLDSsample.sample(A=true_A_[stim], Q=true_Q_[stim], C=true_C_[sbj], R=true_R_[sbj], 
                                        x0=true_x0_[stim],b=true_b_[stim],timepoints=timepoints,seed=idxx_val)[1]) 
            idxx_val +=1
        CC[stim] = samples
        CC_val[stim] = samples_val
        
    # Sort datasets for performance comparison as a function of the amount of data    
    # test set
    XX_val = {}
    XX_val[sbjIDs[-1]] = CC_val
    # target data only
    xx_10 = {}
    xx_20 = {}
    xx_30 = {}
    xx_50 = {}
    xx_70 = {}
    xx_90 = {}
    xx_10[sbjIDs[-1]] = {stim: CC[stim][:1]  for stim in stimIDs}
    xx_20[sbjIDs[-1]] = {stim: CC[stim][:2]  for stim in stimIDs}
    xx_30[sbjIDs[-1]] = {stim: CC[stim][:3]  for stim in stimIDs}
    xx_50[sbjIDs[-1]] = {stim: CC[stim][:5]  for stim in stimIDs}
    xx_70[sbjIDs[-1]] = {stim: CC[stim][:10] for stim in stimIDs}
    xx_90[sbjIDs[-1]] = {stim: CC[stim][:20] for stim in stimIDs}
    # full dataset
    XX_10 = copy.deepcopy(XX)
    XX_20 = copy.deepcopy(XX)
    XX_30 = copy.deepcopy(XX)
    XX_50 = copy.deepcopy(XX)
    XX_70 = copy.deepcopy(XX)
    XX_90 = copy.deepcopy(XX)
    XX_10[sbjIDs[-1]]  = copy.deepcopy(xx_10[sbjIDs[-1]])
    XX_20[sbjIDs[-1]]  = copy.deepcopy(xx_20[sbjIDs[-1]])
    XX_30[sbjIDs[-1]]  = copy.deepcopy(xx_30[sbjIDs[-1]])
    XX_50[sbjIDs[-1]]  = copy.deepcopy(xx_50[sbjIDs[-1]])
    XX_70[sbjIDs[-1]]  = copy.deepcopy(xx_70[sbjIDs[-1]])
    XX_90[sbjIDs[-1]]  = copy.deepcopy(xx_90[sbjIDs[-1]])
    # missing conditions
    x0_1_ = copy.deepcopy(XX)
    x0_10 = copy.deepcopy(XX)
    x0_20 = copy.deepcopy(XX)
    x0_30 = copy.deepcopy(XX)
    x0_50 = copy.deepcopy(XX)
    x0_70 = copy.deepcopy(XX)
    x0_90 = copy.deepcopy(XX)
    x0_1_[sbjIDs[-1]] = {stim: CC[stim][:1]   for stim in [stimIDs[1]]}
    x0_10[sbjIDs[-1]] = {stim: CC[stim][:10]  for stim in [stimIDs[1]]}
    x0_20[sbjIDs[-1]] = {stim: CC[stim][:20]  for stim in [stimIDs[1]]}
    x0_30[sbjIDs[-1]] = {stim: CC[stim][:30]  for stim in [stimIDs[1]]}
    x0_50[sbjIDs[-1]] = {stim: CC[stim][:50]  for stim in [stimIDs[1]]}
    x0_70[sbjIDs[-1]] = {stim: CC[stim][:100] for stim in [stimIDs[1]]}
    x0_90[sbjIDs[-1]] = {stim: CC[stim][:250] for stim in [stimIDs[1]]}
    
    #### Test model performance ####
    # amLDS performance and data demands
    print('Testing amLDS performance and data demands on target animal')
#     XX_list = [XX_10, XX_20, XX_30, XX_50, XX_70, XX_90]
    AccDD = []
#     for fooxx1 in XX_list:
#         TL_DD = amLDSm(zdim=zdim, sbjID=sbjIDs, stimID=stimIDs, timepoints=timepoints)# initialize model class
#         TL_DD.import_data(data=fooxx1)                                                # import data
#         TL_DD.init_params(xdim=xdim[sbjIDs[0]], TL_FAinit=True)                          # initialize parameters
#         TL_DD.LDS_EM_TranferLearning();                                               # EM parameter learning
#         AccDD.append(TL_DD.probability_decoding(XX_val, sbjIDs=[sbjIDs[-1]])[2])      # decode in test set
#         print(AccDD)
        
    # amLDS single animal (no-TL)    
    print('Testing amLDS performance and data demands with one animal')
#     xx_list = [xx_10, xx_20, xx_30, xx_50, xx_70, xx_90]
    AccSA = []
#     for fooxx2 in xx_list:
#         TL_SA = amLDSm(zdim=zdim,sbjID=[sbjIDs[-1]],stimID=stimIDs,timepoints=timepoints)# initialize model class
#         TL_SA.import_data(data=fooxx2)                                                   # import data
#         TL_SA.init_params(xdim=xdim[sbjIDs[0]], FAinit=True)                             # initialize parameters
#         TL_SA.LDS_EM_TranferLearning();                                                  # EM parameter learning
#         AccSA.append(TL_SA.probability_decoding(XX_val, sbjIDs=[sbjIDs[-1]])[2])         # decode in test set
#         print(AccSA)
        
    # amLDS data demands, single stimlulus condition
    print('Testing amLDS data performance and demands on target animal, one stimulus condition for training') 
    x0_list = [x0_1_, x0_10, x0_20, x0_30, x0_50, x0_70, x0_90]
    AccMC = []
    for fooxx3 in x0_list:
        TL_MC = amLDSm(zdim=zdim, sbjID=sbjIDs, stimID=stimIDs, timepoints=timepoints)# initialize model class
        TL_MC.import_data(data=fooxx3)                                                # import data
        TL_MC.init_params(xdim=xdim[sbjIDs[0]], TL_FAinit=True)                       # initialize parameters
        TL_MC.LDS_EM_TranferLearning();                                               # EM parameter learning
        AccMC.append(TL_MC.probability_decoding(XX_val, sbjIDs=[sbjIDs[-1]])[2])      # decode in test set
        print(AccMC)
      
    #### Model comparison ####
    # FA+Procrustes performance and data demands
    print('Testing FA+Procrustes performance and data demands on target animal')
    XX_list = [XX_10, XX_20, XX_30, XX_50, XX_70, XX_90]
    AccDD_FAp = []
    for fooxx4 in XX_list:
        _, Rs, zzS, yyS = FAp(fooxx4, nStim, zdim, timepoints, 3, Rs=None, project_only=False)
        _, _,  zzT, yyT = FAp(XX_val, nStim, zdim, timepoints, 20, Rs=Rs, project_only=True)

        clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        clf.fit(zzS, yyS)
        AccDD_FAp.append(np.mean(clf.predict(zzT) == yyT))
        print(AccDD_FAp)

    # CCA performance and data demands
    print('Testing CCA performance and data demands on target animal')
    XX_list = [XX_10, XX_20, XX_30, XX_50, XX_70, XX_90]
    AccDD_CCA = []
    for fooxx5 in XX_list:
        acc_temp = []
        for source in sbjIDs[:-1]:
            fooxxtemp = {}
            fooxxtemp[sbjIDs[-1]] = copy.deepcopy(fooxx5[sbjIDs[-1]])
            fooxxtemp[source]     = copy.deepcopy(fooxx5[source])

            zT, yT, zS, yS, zSv, ySv = CCAp(fooxxtemp, XX_val, source, sbjIDs[-1], 
                                            nStim, zdim, timepoints, 7, 7)

            clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
            clf.fit(np.concatenate((zT, zS)), np.concatenate((yT, yS)))
            acc_temp.append(np.mean((clf.predict(zSv) == ySv)))
        AccDD_CCA.append(acc_temp)
        print( np.mean(AccDD_CCA, axis=1) )
    AccDD_CCA = np.mean(AccDD_CCA, axis=1)   
    
    return AccDD, AccSA, AccMC, AccDD_FAp, AccDD_CCA
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='amLDS_performance_dataDemands_modelComparison')
    parser.add_argument("--nStim",  type=int,   default=50,  help='number of stimulus classes')
    parser.add_argument("--nSbj",   type=int,   default=5,   help='number of subjects')
    parser.add_argument("--zdim",   type=int,   default=3,   help='number of latent dimensions')
    parser.add_argument("--xdim",   type=int,   default=40,  help='number of observed dimensions')
    parser.add_argument("--time",   type=int,   default=41,  help='timebins, sequence length')
    parser.add_argument("--trS",    type=int,   default=50,  help='number of source trials per stimulus')
    parser.add_argument("--trT",    type=int,   default=20,  help='number of target test trials')
    parser.add_argument("--Qnoise", type=float, default=.55, help='latent dimension noise')
    parser.add_argument("--Rnoise", type=float, default=.5,  help='measurement noise')
    parser.add_argument("--Calpha", type=float, default=.1,  help='C missalignment')
    parser.add_argument("--pSeed",  type=int,   default=10,  help='seed for parameter initiation')
    parser.add_argument("--idxx",   type=int,   default=1000,help='seed for source trials')
    parser.add_argument("--idxxV",  type=int,   default=12,  help='seed for test trials')
    
    
    args = parser.parse_args()

    nStim  = args.nStim
    nSbj   = args.nSbj
    zdim   = args.zdim
    xdim   = args.xdim
    time   = args.time
    trS    = args.trS
    trT    = args.trT
    Qnoise = args.Qnoise
    Rnoise = args.Rnoise
    Calpha = args.Calpha
    pSeed  = args.pSeed
    idxx   = args.idxx
    idxxV  = args.idxxV
    
    print('Starting amLDS testing')
    
    AccDD, AccSA, AccMC, AccDD_FAp, AccDD_CCA = main_ex(nStim=nStim, nSbj=nSbj, zdim=zdim, xdim=xdim, timepoints=time, 
                                                        trialSource=trS, trialTargetval=trT, 
                                                        Qnoise=Qnoise, Rnoise =Rnoise, Calpha=Calpha, 
                                                        parm_seed=pSeed, idxx=idxx, idxx_val=idxxV)
    
    pickle.dump( AccDD,     open( 'amLDS_acc_'+str(idxx)+'.pickle', 'wb' ) )
    pickle.dump( AccSA,     open( 'amLDS_SA_acc_'+str(idxx)+'.pickle', 'wb' ) )
    pickle.dump( AccMC,     open( 'amLDS_MC_acc_'+str(idxx)+'.pickle', 'wb' ) )
    pickle.dump( AccDD_FAp, open( 'FAp_acc_'+str(idxx)+'.pickle', 'wb' ) )
    pickle.dump( AccDD_CCA, open( 'CCA_acc_'+str(idxx)+'.pickle', 'wb' ) )
    
    print('Outputs in current directory')
    print('Done running!')


    