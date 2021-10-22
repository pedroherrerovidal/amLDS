import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from sklearn.decomposition import FactorAnalysis
from funs.TL_FA import TL_FA
import time

class amLDS:
    def __init__(self, zdim, sbjID, stimID, data=None, timepoints=None):
#         self.xdim = xdim  # list of int ; number of observed dimensions
        self.zdim = zdim      # int ; number of latent dimensions ## same for all subjects and stimuli
        
        self.sbjID  = sbjID   # list of str ; subjects ID
        self.stimID = stimID  # list of str ; stimulus ID
        
        if data is not None:
            self.data = data  # dict of subjects ; nested dict of stimuli ; list trials ; observation [timesteps x xdim]
        if timepoints is not None:
            self.timepoints = timepoints # int ; length of the sequence
            
        self.cmap_stim = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
        self.cmap_sbj  = ['#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        
    def init_params(self, xdim, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None, timepoints=None, FAinit=False, binit=True, TL_FAinit=False, AQ_init=False):
    
        if C is not None:
            self.C_ = C    # dict of [xdim x zdim] ; projection matrix
        elif FAinit != True:
            self.C_ = {key: np.eye(xdim, self.zdim) for key in self.sbjID}

        if R is not None:
            self.R_ = R    # dict of [xdim x xdim] ; observation noise
        elif FAinit != True:
            self.R_ = {key: np.eye(xdim) for key in self.sbjID}

        if A is not None:
            self.A_ = A    # dict of [zdim x zdim] ; latent transfer matrix
        elif AQ_init != True:
            self.A_ = {key: np.eye(self.zdim)*.1 for key in self.stimID}
            
        if Q is not None:
            self.Q_ = Q    # dict of [zdim x zdim] ; latent covariance noise
        elif AQ_init != True:
            self.Q_ = {key: np.eye(self.zdim)*.05 for key in self.stimID}

        if x0 is not None:
            self.x0_ = x0  # dict of [zdim]        ; latent mean prior
        else:
            self.x0_ = {key: np.zeros(self.zdim) for key in self.stimID}
            
        if Q0 is not None:
            self.Q0_ = Q0  # dict of [zdim x zdim] ; latent covariance prior
        else:
            self.Q0_ = {key: np.eye(self.zdim)*.05 for key in self.stimID}

        if FAinit == True:
            print('Factor Analysis parameter initialization')
            try:
                Cs = {}
                Rs = {}
                for key in self.sbjID:
                    temp0 = np.concatenate([self.data[key][i] for i in list(self.data[key].keys())])
#                     temp0 = np.concatenate([self.data[key][i] for i in self.stimID])
                    temp1 = temp0.reshape((np.prod(temp0.shape[0:2]), temp0.shape[2]))
                    fa = FactorAnalysis(n_components=self.zdim)
                    fa.fit(temp1)
                    Cs[key] = fa.components_.T
                    Rs[key] = np.diag(fa.noise_variance_)
                if C is None:
                    print('init C')
                    self.C_ = Cs
                if R is None:
                    print('init R')
                    self.R_ = Rs
            except:
                print('Import data for Factor Analysis parameter initialization')   

        if TL_FAinit == True:
            print('Initializating C and R using population factor analysis')
            try:
                XX_FA = {}
                for key in self.sbjID:
                    temp = []
                    for stim in self.stimID:
                        if stim in list(self.data[key].keys()):
                            xi = np.array(self.data[key][stim])
                            temp.append(xi.reshape(np.prod(xi.shape[:2]), xi.shape[2]))
                    XX_FA[key] = np.concatenate(temp)
                TLfa = TL_FA(xdim=xi.shape[2], zdim=int(self.zdim), sbjID=list(self.sbjID))
                TLfa.import_data(data=XX_FA)
                TLfa.data_stats()
                TLfa.init_params()
                init_C, Rs, _ = TLfa.FA_EM_TransferLearning();
                Cs = {key: init_C for key in list(self.sbjID)}
                if C is None:
                    self.C_ = Cs
                if R is None:
                    self.R_ = Rs
            except:
                print('Import data for Factor Analysis parameter initialization')  
                
        if b is not None:
            self.b_ = b  # dict of [zdim x time] ; latent mean offset
        elif binit == True:
            print('Initializating b using latent estimates')
            bs = {}
            for stim in self.stimID:
                temp = []
                for sbj in self.sbjID:
                    if stim in list(self.data[sbj].keys()):
                        x = np.mean(self.data[sbj][stim],axis=0)
                        x_mu = np.mean(x, axis=0)
                        temp.append(self.FA_latent(self.C_[sbj], self.R_[sbj], x, x_mu))
                temp = np.mean(temp,axis=0)
#                 temp[:,0] = 0
                bs[stim] = temp - temp[:,0].reshape(temp.shape[0],1)
            self.b_ = bs
        else:
            if timepoints is not None:
                self.b_ = {key: np.ones((self.zdim, timepoints)) for key in self.stimID}
            else: 
                self.b_ = {key: np.ones((self.zdim, self.timepoints)) for key in self.stimID}
                
        if AQ_init == True:
            print('Initializating A and Q using latent estimates')
            As = {}
            Qs = {}
            for stim in self.stimID:
                temp = []
                for sbj in self.sbjID:
                    if stim in list(self.data[sbj].keys()):
                        xx = self.data[sbj][stim]
                        for xi in xx:
                            x_mu = np.mean(xi, axis=0)
                            temp.append(self.FA_latent(self.C_[sbj], self.R_[sbj], xi, x_mu).T)

                temp = np.array(temp)
                aa1 = temp[:, :-1, :]
                aa2 = temp[:, 1:, :]
                Z_0 = aa1.reshape((temp.shape[0]* (self.timepoints-1), self.zdim), order='C')
                Z_1 = aa2.reshape((temp.shape[0]* (self.timepoints-1), self.zdim), order='C')
                A = np.linalg.pinv(Z_0.T @ Z_0) @ Z_0.T @ Z_1
                e = Z_1 - ((Z_0 @ np.linalg.pinv(Z_0.T @ Z_0) @ Z_0.T) @ Z_1)
                Q = np.sqrt(np.diag(np.diag((e.T @ e) / (Z_0.shape[0]-Z_0.shape[1]))))

                As[stim] = A
                Qs[stim] = Q
            self.A_ = As
            self.Q_ = Qs
            
    def import_data(self, data):
        self.data = data
        
    def import_validationData(self, data):
        self.data_val = data
  
    def sample(self, A, Q, C, R, x0, b, timepoints, seed=None):
        if seed is not None:
            np.random.seed(seed)
        xdim = C.shape[0]
        
        # output variables preallocation
        Z = np.zeros((timepoints, self.zdim)) *np.nan
        X = np.zeros((timepoints, xdim)) *np.nan
        
        for t in range(timepoints):
            if t ==0:
                zt = np.array(x0)
                Z[t, :] = zt
            else:
                zt = np.dot(A, zt) + np.random.multivariate_normal(np.zeros(self.zdim), Q) + b[:, t]
            xt = np.dot(C, zt) + np.random.multivariate_normal(np.zeros(xdim), R)

            Z[t, :] = zt
            X[t, :] = xt
        return Z, X
 
    def LDS_filter(self, X, A, Q, C, R, x0, Q0, b):
        # X [timepoints, xdim] ; observations
        timepoints, xdim = X.shape

        # output variables preallocation
        filt_mean = np.zeros((timepoints, self.zdim)) *np.nan
        filt_cov  = np.zeros((timepoints, self.zdim, self.zdim)) *np.nan
        pred_cov  = np.zeros((timepoints, self.zdim, self.zdim)) *np.nan

        # LDS filtering
        for t in range(0, timepoints):
            # estimate next state
            if t == 0:
                pred_mean_t = x0.copy()
                pred_cov_t  = Q0.copy()
            else:
                pred_mean_t = np.dot(A, temp_mean_t) + b[:, t]
                pred_cov_t  = np.dot(np.dot(A, temp_cov_t), A.T) + Q

            pred_cov[t, :] = pred_cov_t
            # correct latent prediction with data
            K = np.dot(np.dot(pred_cov_t, C.T), np.linalg.pinv(np.dot(np.dot(C,pred_cov_t),C.T) + R))

            temp_mean_t = pred_mean_t + np.dot(K,(X[t, :] - np.dot(C, pred_mean_t)))
            temp_cov_t  = np.dot((np.eye(pred_cov_t.shape[0]) - np.dot(K, C) ), pred_cov_t)

            filt_mean[t, :] = temp_mean_t.copy()
            filt_cov[t, :]  = temp_cov_t.copy()
            
        return filt_mean, filt_cov, pred_cov

    def LDS_smooth(self, X, A, Q, C, R, x0, Q0, b):
        # depends on LDS_filter

        # X [timepoints, xdim] ; observations
        # validate inputs
        timepoints, xdim = X.shape

        # run the forward path or LDS filtering
        mu_list, v_list, v_pred = self.LDS_filter(X, A, Q, C, R, x0, Q0, b)

        # output preallocation
        smth_mean = np.zeros((timepoints, self.zdim)) *np.nan
        smth_cov  = np.zeros((timepoints, self.zdim, self.zdim)) *np.nan

        ## KF smoothing
        smth_mean[timepoints-1,:] = np.copy(mu_list[timepoints-1,:])
        smth_cov[timepoints-1,:,:]= np.copy(v_list[timepoints-1,:,:])
        Js = []

        for t in np.flip(np.arange(timepoints-1)): 
            pt = np.copy(v_pred[t+1, :,:])
            J  = np.dot(v_list[t,:,:], np.dot(A.T, np.linalg.pinv(pt)))
            Js.append(J)

            smth_mean[t,:] = mu_list[t,:] +np.dot(J,(smth_mean[t+1,:]- np.dot(A, mu_list[t,:])))
            smth_cov[t,:,:]= v_list[t,:,:]+np.dot(J,np.dot((smth_cov[t+1,:,:] - pt),J.T))

        # append last J
        v_predT =  np.copy(np.dot(np.dot(A, v_list[-1, :, :]), A.T) + Q)
        Js = list(reversed(Js))
        Js.append( np.dot(v_list[t,:,:], np.dot(A.T, np.linalg.pinv(v_predT)) )  )
        return smth_mean, smth_cov, Js


    def E_step(self, X, A, Q, C, R, x0, Q0, b):
        timepoints, xdim = X.shape
        # E-step
#         _, filt_cov, _ = self.LDS_filter(X)
        smth_mean, smth_cov, Js = self.LDS_smooth(X, A, Q, C, R, x0, Q0, b)  

        Ezn = []
        Eznznminus   = []
        Eznzn = []
        for t in range(timepoints):
            Ezn.append(smth_mean[t,:])
            Eznzn.append(smth_cov[t, :, :] + np.outer(smth_mean[t,:].T,smth_mean[t,:]))
            # Eznznminus is n-1 dimensional
            if t != 0:
                Eznznminus.append(   np.dot(Js[t-1], smth_cov[t, :, :]) +  # pair_wise_covariances
                                     np.outer(smth_mean[t,:],smth_mean[t-1,:].T) )

        Ezn = np.array(Ezn)                                # x0, Q0, Q
        Eznznminus   = np.array(Eznznminus)
        Eznzn = np.array(Eznzn)
#         smth_covs.append(smth_cov)                          # Q0                         

        Eznzn_F      = np.mean(Eznzn, axis=0)            # C, R
        Eznzn_1      = np.mean(Eznzn[1:,:,:], axis=0)    # m4 Q
        Eznzn_n      = np.mean(Eznzn[:-1,:,:], axis=0)   # m2 A, Q
        Eznznminus_F = np.mean(Eznznminus, axis=0)       # m1 A, Q

        bnbn      = np.inner(b[:,1:], b[:,1:])/(timepoints-1)     # m6 Q 
        bnzn      = np.inner(b[:,1:], Ezn[1:,:].T)/(timepoints-1) # m5
        bnznminus = np.inner(b[:,1:],Ezn[:-1,:].T)/(timepoints-1) # m3 A

        xnxn  = np.dot(X.T, X)/timepoints
        xnezn = np.dot(X.T, Ezn)/timepoints  # the next one .T

        return Ezn, smth_cov, Eznzn_F, Eznzn_1, Eznzn_n, Eznznminus_F, bnbn, bnzn, bnznminus, xnxn, xnezn
#         return filt_cov, smth_mean, smth_cov, Js, Ezn, Eznznminus, Eznzn

    def x0_update(self, Ezns):
        return np.mean(Ezns[:,0,:], axis=0)

    def Q0_update(self, Ezns, smth_covs):
        trN = smth_covs.shape[0]
        p1 = np.mean(smth_covs[:,0,:,:], axis=0)
        p2 = np.mean([np.outer(Ezns[tt,0,:],Ezns[tt,0,:]) for tt in range(trN)],axis=0)
        return np.diag(np.diag(p1+p2)) ### -
    
    def A_update(self, Eznznminus_Fs, bnznminuss, Eznzn_ns):
        p1 = np.mean([Eznznminus_Fs[k,:,:] - bnznminuss[k,:,:] for k in range(bnznminuss.shape[0])],axis=0)
        p2 = np.linalg.pinv(np.mean(Eznzn_ns, axis=0))
        return np.dot(p1,p2)

    def Q_update(self, A, Eznznminus_Fs, Eznzn_ns, bnznminuss, Eznzn_1s, bnzns, bnbns):
        return np.diag(np.diag(sum([Eznzn_1s[k,:,:] 
                                 - np.dot(Eznznminus_Fs[k,:,:], A.T) 
                                 - np.dot(Eznznminus_Fs[k,:,:], A.T).T      
                                 + np.dot(np.dot(A, Eznzn_ns[k,:,:]), A.T) 
                                 - bnzns[k,:,:] 
                                 - bnzns[k,:,:].T 
                                 + np.dot(A, bnznminuss[k,:,:].T) # bnznminuss[k,:,:].T
                                 + np.dot(bnznminuss[k,:,:], A.T) # A.T
                                 + bnbns[k,:,:]
                                 for k in range(Eznzn_1s.shape[0]) ] ) / Eznzn_1s.shape[0])) 
    
    def b_update(self, A, Ezns):
        bup = []
        for k in range(Ezns.shape[0]):
            b_ = np.zeros((A.shape[0], Ezns.shape[1]))
            for t in range(1, Ezns.shape[1]):
                b_[:, t] = Ezns[k, t,:] - np.dot(A, Ezns[k, t-1,:])
            bup.append(b_)
        return np.mean(bup, axis=0)
    
    def C_update(self, xnezns, Eznzn_Fs):
        return np.dot(np.mean(xnezns, axis=0), np.linalg.pinv(np.mean(Eznzn_Fs, axis=0)))
    
    def R_update(self, C, xnxns, xnezns, Eznzn_Fs):
        return np.diag(np.diag(np.mean([xnxns[k,:,:] - np.dot(C, xnezns[k,:,:].T) 
                        - np.dot(xnezns[k,:,:], C.T) + np.dot(np.dot(C, Eznzn_Fs[k,:,:]), C.T) 
                        for k in range(xnxns.shape[0])], axis=0)))

    def LDS_EM_TranferLearning(self, XX=None, max_iter=10, sbjID=None, stimID=None, LL_flag=False, update_A=True, update_Q=True, update_C=True, update_R=True, update_x0=True, update_Q0=True, update_b=True):
        """
        Method that perform the EM algorithm to update the model parameters
        Note that in this exercise we ignore offsets
        @param X: a numpy 2D array whose dimension is [trials, n_example, self.n_dim_obs]
                X: list of trials whose dimensions are [ttimesteps, xdim]
                XX: dict of subjects dict stimuli 
        @param max_iter: an integer indicating how many iterations to run
        """
        if XX == None:
            XX = self.data
        if sbjID == None:
            sbjIDs = self.sbjID
        if stimID == None:
            stimIDs = self.stimID

        # keep track of log posterior (use function calculate_posterior below)
        em_log_posterior = np.zeros(max_iter,)*np.nan

        # output holders
        Cs  = self.C_
        Rs  = self.R_
        As  = self.A_
        Qs  = self.Q_
        x0s = self.x0_
        Q0s = self.Q0_
        bs  = self.b_
        
        # EM learning time
        EM_time = []

        # add initial parameters to structure
        
        print('Learning amLDS parameters via EM')

        # EM
        for step in range(max_iter):
            time1 = time.time()
            # holder for EM step updates
            Ezns          = []
            smth_covs     = [] 
            Eznzn_Fs      = []
            Eznzn_1s      = [] 
            Eznzn_ns      = [] 
            Eznznminus_Fs = []
            bnbns         = []
            bnzns         = []
            bnznminuss    = []
            xnxns         = []
            xnezns        = []

            ## trial type index >> replace ; then sbjIdx and condIdx go 
            trialID_sbj  = []
            trialID_cond = []

            sbjIdx = 0
            for sbj in sbjIDs:
                tC = Cs[sbj]
                tR = Rs[sbj]
                condIdx = 0
                for cond in stimIDs:
                    if cond in list(XX[sbj].keys()):
                        for n in range(len(XX[sbj][cond])):
                            x   = XX[sbj][cond][n]
                            tA  = As[cond]
                            tQ  = Qs[cond]
                            tx0 = x0s[cond]
                            tQ0 = Q0s[cond]
                            tb  = bs[cond]
                            timepoints = x.shape[0]

                            # E-step
                            Ezn,smth_cov,Eznzn_F,Eznzn_1,Eznzn_n,Eznznminus_F,bnbn,bnzn,bnznminus,xnxn,xnezn = self.E_step(x,tA,tQ,tC,tR,tx0,tQ0,tb)

                            Ezns.append( Ezn )
                            smth_covs.append( smth_cov )
                            Eznzn_Fs.append( Eznzn_F )
                            Eznzn_1s.append( Eznzn_1 )
                            Eznzn_ns.append( Eznzn_n )
                            Eznznminus_Fs.append( Eznznminus_F )
                            bnbns.append( bnbn )
                            bnzns.append( bnzn )
                            bnznminuss.append( bnznminus )
                            xnxns.append( xnxn )
                            xnezns.append( xnezn )

                            trialID_sbj.append(sbjIdx)
                            trialID_cond.append(condIdx)
                    condIdx +=1
                sbjIdx +=1

            # transform list into numpy arrays    
            Ezns = np.array(Ezns)
            smth_covs = np.array(smth_covs)
            Eznzn_Fs = np.array(Eznzn_Fs)
            Eznzn_1s = np.array(Eznzn_1s)
            Eznzn_ns = np.array(Eznzn_ns)
            Eznznminus_Fs = np.array(Eznznminus_Fs)
            bnbns = np.array(bnbns)
            bnzns = np.array(bnzns)
            bnznminuss = np.array(bnznminuss)
            xnxns = np.array(xnxns)
            xnezns = np.array(xnezns)

            # estimate posterior log-likelihood
            if LL_flag == True:
                em_log_posterior[step] = self.marginal_loglikelihood(A=As, Q=Qs, C=Cs, R=Rs, x0=x0s, Q0=Q0s, b=bs)
               
            # M-step
            # update observation parameters ; subject specific parameters
            sbjIdx = 0        
            for sbj in sbjIDs:
                sbj_mask = np.array(trialID_sbj) == sbjIdx
                Cs[sbj] = self.C_update(xnezns[sbj_mask,:,:], Eznzn_Fs[sbj_mask,:,:])
                Rs[sbj] = self.R_update(Cs[sbj], xnxns[sbj_mask,:,:], xnezns[sbj_mask,:,:], Eznzn_Fs[sbj_mask,:,:])
                sbjIdx +=1

            # update latent dynamics parameters ; stimulus specific parameters
            condIdx = 0
            for cond in stimIDs:
                stim_mask = np.array(trialID_cond) == condIdx
                x0s[cond] = self.x0_update(Ezns[stim_mask,:,:])
                Q0s[cond] = self.Q0_update(Ezns[stim_mask,:,:], smth_covs[stim_mask,:,:])
                As[cond]  = self.A_update(Eznznminus_Fs[stim_mask,:,:], bnznminuss[stim_mask,:,:], Eznzn_ns[stim_mask,:,:])
                Qs[cond]  = self.Q_update(As[cond], Eznznminus_Fs[stim_mask,:,:], Eznzn_ns[stim_mask,:,:], 
                                          bnznminuss[stim_mask,:,:], Eznzn_1s[stim_mask,:,:], bnzns[stim_mask,:,:],
                                          bnbns[stim_mask,:,:])
                bs[cond]  = self.b_update(As[cond], Ezns[stim_mask,:,:])
                condIdx +=1
                
            EM_time.append( time1 - time.time() )

        if update_A == True:
            self.A_ = As
        if update_Q == True:
            self.Q_ = Qs
        if update_x0 == True:
            self.x0_ = x0s
        if update_Q0 == True:
            self.Q0_ = Q0s
        if update_C == True:
            self.C_ = Cs
        if update_R == True:
            self.R_ = Rs
        if update_b == True:
            self.b_ = bs
        if LL_flag == True:
            self.em_log_posterior = em_log_posterior
        self.EM_time = EM_time 

        return x0s, Q0s, As, Qs, Cs, Rs, bs, em_log_posterior

    def trial_reconstruction_error(self, X, A, Q, C, R, x0, Q0, b, var=None):
        # X [timepoints, xdim] ; observations
        X_rec = np.zeros(X.shape)*np.nan
        X_rec_var = np.zeros(X.shape)*np.nan

        for nn in range(X.shape[1]):
            mask = np.ones(X.shape[1], dtype='bool')
            mask[nn] = False
            X_nn = X[:, mask]

            smth_mean,smth_cov,_ = self.LDS_smooth(X_nn, A, Q, C[mask, :], R[mask, mask], x0, Q0, b) 
            X_rec[:, nn] = (np.dot(smth_mean, C.T)[:, ~mask]).ravel()
            if var is not None:
                for tt in range(X.shape[0]):
                    X_rec_var[tt, nn] = np.diag(np.dot(np.dot(C, smth_cov[tt,:,:]), C.T) + R)[~mask][0]

        err = np.mean((X.ravel()-X_rec.ravel())**2)
        
        return err, X_rec, X_rec_var
    
    def reconstruction_error(self, XX=None, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None, sbjID=None, stimID=None):
        if XX == None:
            XX = self.data
        if sbjID == None:
            sbjIDs = self.sbjID
        if stimID == None:
            stimIDs = self.stimID
        if A == None:
            A = self.A_
        if Q == None:
            Q = self.Q_
        if x0 == None:
            x0 = self.x0_
        if Q0 == None:
            Q0 = self.Q0_
        if C == None:
            C = self.C_
        if R == None:
            R = self.R_
        if b == None:
            b = self.b_
        
        sbj_recErr = {}
        
        for sbj in sbjIDs:
            recErr = []
            for cond in stimIDs:
                if cond in list(XX[sbj].keys()):
                    for n in range(len(XX[sbj][cond])):
                        x = XX[sbj][cond][n]
                        recErr.append( self.trial_reconstruction_error(x, A[cond], Q[cond], C[sbj], R[sbj], 
                                                                       x0[cond], Q0[cond], b[cond])[0] )
            sbj_recErr[sbj] = np.mean(recErr)
        self.sbj_recErr = sbj_recErr
        
        return np.array( [ sbj_recErr[sbj] for sbj in sbjIDs ] )    
    
    def prob_trialParam(self, X, A, Q, C, R, x0, Q0, b):
        # X [timepoints, xdim] ; observations
        timepoints, xdim = X.shape

        # output variables preallocation
        LL = 0

        # LDS filtering
        for t in range(0, timepoints):
            # estimate next state
            if t == 0:
                pred_mean_t = x0.copy()
                pred_cov_t  = Q0.copy()
            else:
                pred_mean_t = np.dot(A, temp_mean_t) + b[:,t]
                pred_cov_t  = np.dot(np.dot(A, temp_cov_t), A.T) + Q
            
            obs_mean_t = np.dot( C, pred_mean_t )
            obs_cov_t  = np.dot(np.dot(C, pred_cov_t), C.T) + R
            
            # compute log likelihood P( x_t | x_{1:t-1}, params)
            LL += np.log( stats.multivariate_normal.pdf(X[t,:], obs_mean_t, obs_cov_t) ) # np.log

            # correct latent prediction with data
            K = np.dot(np.dot(pred_cov_t, C.T), np.linalg.pinv(np.dot(np.dot(C,pred_cov_t),C.T) + R))

            temp_mean_t = pred_mean_t + np.dot(K,(X[t, :] - np.dot(C, pred_mean_t)))
            temp_cov_t  = np.dot((np.eye(pred_cov_t.shape[0]) - np.dot(K, C) ), pred_cov_t) 
        return LL #/ timepoints
    
    def prob_trial(self, X, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None, stimIDs=None):
        if stimIDs == None:
            stimIDs = self.stimID
        if A is None:
            A = self.A_
        if Q is None:
            Q = self.Q_
        if x0 is None:
            x0 = self.x0_
        if Q0 is None:
            Q0 = self.Q0_
#         if C == None:
#             C = self.C_
#         if R == None:
#             R = self.R_
        if b is None:
            b = self.b_
            
        LLs = []
        smth_means = []
        smth_covs  = []
            
        for cond in stimIDs:
            
            LLs.append( self.prob_trialParam(X, A[cond], Q[cond], C, R, x0[cond], Q0[cond], b[cond]) )
            smth_mean, smth_cov,_ = self.LDS_smooth(X, A[cond], Q[cond], C, R, x0[cond], Q0[cond], b[cond])  
            smth_means.append(smth_mean)
            smth_covs.append(smth_cov)
            
        return LLs, np.array(smth_means), np.array(smth_covs)
    
    def probability_decoding(self, XX=None, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None, sbjIDs=None, stimIDs=None):
        if XX is None:
            XX = self.data_val
        if sbjIDs is None:
            sbjIDs = self.sbjID
        if stimIDs is None:
            stimIDs = self.stimID
        if A is None:
            A = self.A_
        if Q is None:
            Q = self.Q_
        if x0 is None:
            x0 = self.x0_
        if Q0 is None:
            Q0 = self.Q0_
        if C is None:
            C = self.C_
        if R is None:
            R = self.R_
        if b is None:
            b = self.b_
            
        print('Decoding stimulus class')
            
        sbj_trialsLL   = {} # probability of class
        sbj_trialsPr   = {} # normalized probability of class
        sbj_trialsTL   = {} # true label
        sbj_trialsPL   = {} # predicted label
        sbj_ProbAcc    = {} # accuracy of probabilistic classification
        sbj_ProbCM     = [] # classification confusion matrix
        sbj_trialsMean = {} # estimated trial mean
        sbj_trialsCovs = {} # estimated trial confidence
        decoding_time  = [] # decoding time demands
        
        for sbj in sbjIDs:
            trialsLL = []
            trialsTL = []
            trialsMean = []
            trialsCovs = []
            TL_idx = 0
            for cond in stimIDs:
                if cond in list(XX[sbj].keys()):
                    for n in range(len(XX[sbj][cond])):
                        x = XX[sbj][cond][n]
                        time1 = time.time()
                        LLs, smth_means, smth_covs = self.prob_trial(x, A, Q, C[sbj], R[sbj], x0, Q0, b, stimIDs)
                                                                                                # XX[sbj].keys() )
                        trialsLL.append( LLs )
                        trialsMean.append( smth_means )
                        trialsCovs.append( smth_covs )
                        trialsTL.append(TL_idx)
                        decoding_time.append(time1 - time.time())
                    TL_idx += 1

            fooTrLL      = np.array(trialsLL).T
            class_ll     = np.exp((fooTrLL - np.max(fooTrLL, axis=0)))
            class_llnorm = (class_ll / np.sum(class_ll, axis=0)).T
            pred_lbl     = np.argmax(class_llnorm, axis=1)
            CMt          = confusion_matrix(np.array(trialsTL) , pred_lbl)
            CMnorm       = CMt / CMt.sum(axis=1, keepdims=True)

            sbj_trialsLL[sbj]   = np.array(trialsLL)
            sbj_trialsPr[sbj]   = np.array(class_llnorm)
            sbj_trialsTL[sbj]   = np.array(trialsTL)
            sbj_trialsPL[sbj]   = pred_lbl
            sbj_ProbAcc[sbj]    = np.mean(sbj_trialsTL[sbj] == sbj_trialsPL[sbj])
            sbj_ProbCM.append( CMnorm )
            sbj_trialsMean[sbj] = trialsMean
            sbj_trialsCovs[sbj] = trialsCovs
                  
        # store variables in object
        self.sbj_trialsLL   = sbj_trialsLL
        self.sbj_trialsPr   = sbj_trialsPr
        self.sbj_trialsTL   = sbj_trialsTL
        self.sbj_trialsPL   = sbj_trialsPL
        self.sbj_ProbAcc    = sbj_ProbAcc
        self.sbj_ProbCM     = sbj_ProbCM
        self.sbj_trialsMean = sbj_trialsMean
        self.sbj_trialsCovs = sbj_trialsCovs
        self.decoding_time  = decoding_time
        
        return sbj_trialsPr, sbj_ProbCM, sbj_ProbAcc, sbj_trialsLL


    def marginal_loglikelihood(self, XX=None, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None, sbjIDs=None, stimIDs=None):
        if XX is None:
            XX = self.data
        if sbjIDs == None:
            sbjIDs = self.sbjID
        if stimIDs == None:
            stimIDs = self.stimID
        if A is None:
            A = self.A_
        if Q is None:
            Q = self.Q_
        if x0 is None:
            x0 = self.x0_
        if Q0 is None:
            Q0 = self.Q0_
        if C is None:
            C = self.C_
        if R is None:
            R = self.R_
        if b is None:
            b = self.b_
            
        sbj_trialsLL   = {} # marginal likelihood per subject
        for sbj in sbjIDs:
            trialsLL = []
            for cond in stimIDs:
                if cond in list(XX[sbj].keys()):
                    for n in range(len(XX[sbj][cond])):
                        x = XX[sbj][cond][n]
                        trialsLL.append( self.prob_trialParam(x,A[cond],Q[cond],C[sbj],R[sbj],x0[cond],Q0[cond],b[cond]) )
            sbj_trialsLL[sbj]   = np.mean(trialsLL)
                  
        return sbj_trialsLL
    
    def latentTrials_forDecoding(self, XX=None, sbjID=None, stimID=None, folds=4, cheat=False):
        if XX == None:
            XX = self.data
        if sbjID == None:
            sbjIDs = self.sbjID
        if stimID == None:
            stimIDs = self.stimID
            
        Z_LDS = {}
        Z_LDS_decod = {}
        y_LDS = {}
        y_fold_LDS = {}

        for sbj in sbjIDs:
            fooz = []
            stimz = []
            fooy = []
            fooyfold = []
            cidx = 0
            C = self.C_[sbj]
            R = self.R_[sbj]

            for stim in stimIDs:
                avgzz = []
                A  = self.A_[stim]
                Q  = self.Q_[stim]
                x0 = self.x0_[stim]
                Q0 = self.Q0_[stim]
                b  = self.b_[stim]

                nTr = len(XX[sbj][stim])

                fooy.append(np.ones((nTr, )) * cidx)
                fooyfold.append(np.random.choice(range(folds), nTr))
                cidx +=1

                for tt in range(nTr):
                    xi = XX[sbj][stim][tt][:,:]
                    if cheat == True:
                        foozz,_,_ = self.LDS_smooth(xi, self.A_[stim], self.Q_[stim], C, R, 
                                                    self.x0_[stim], self.Q0_[stim], self.b_[stim])
                    elif cheat == False:
                        LLs,_,_ = self.prob_trial(xi, self.A_, self.Q_, C, R, self.x0_, self.Q0_, self.b_,XX[sbj].keys())
                        cond = list(XX[sbj].keys())[np.argmax(LLs)]
                        foozz,_,_ = self.LDS_smooth(xi, self.A_[cond], self.Q_[cond], C, R, 
                                                    self.x0_[cond], self.Q0_[cond], self.b_[cond])
#                     foozz,_,_ = self.LDS_smooth(xi, A, Q, C, R, x0, Q0, b)
                    fooz.append(foozz)
                    avgzz.append(foozz)
                stimz.append(avgzz)

            fooz = np.array(fooz)
            Z_LDS[sbj] = np.array(stimz)               # stim x trials x time x latent_dim
            Z_LDS_decod[sbj] = np.reshape(fooz, (fooz.shape[0], fooz.shape[1]* fooz.shape[2]))
            y_LDS[sbj] = np.concatenate(fooy)          # trials
            y_fold_LDS[sbj] = np.concatenate(fooyfold) # trials
        
        self.Z_LDS = Z_LDS
        self.Z_LDS_decod = Z_LDS_decod
        self.y_LDS = y_LDS
        self.y_fold_LDS = y_fold_LDS
        
        return Z_LDS, Z_LDS_decod, y_LDS, y_fold_LDS
    
    def linearSVM_withinSubject(self, folds=4):
        ''' Depends on latentTrials_forDecoding()'''
        
        LDS_withinAcc      = np.zeros((len(self.sbjID), len(self.stimID), len(self.stimID), folds))*np.nan
        LDS_withinAcc_rand = np.zeros((len(self.sbjID), len(self.stimID), len(self.stimID), folds))*np.nan

        for sbj in range(len(self.sbjID)):
            temp_z = self.Z_LDS_decod[self.sbjID[sbj]]
            temp_y = self.y_LDS[self.sbjID[sbj]]
            temp_foldy = self.y_fold_LDS[self.sbjID[sbj]]
            temp_randy = np.array(temp_y)
            np.random.shuffle(temp_randy)
#             clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#             scores = cross_val_score(clf, temp_z, temp_y, cv=5)
            
            for fold in range(folds):
                # true labels
                clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
                clf.fit(temp_z[temp_foldy!=fold], temp_y[temp_foldy!=fold])
                predy = clf.predict(temp_z[temp_foldy==fold])
                yt = temp_y[temp_foldy==fold]
                LDS_withinAcc[sbj, :, :, fold] = confusion_matrix(predy, yt)

                # random labels
                clfr = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
                clfr.fit(temp_z[temp_foldy!=fold], temp_randy[temp_foldy!=fold])
                predy = clfr.predict(temp_z[temp_foldy==fold])
                yt = temp_y[temp_foldy==fold]
                LDS_withinAcc_rand[sbj, :, :, fold] = confusion_matrix(yt, predy)

        LDS_withinAcc  = np.mean(LDS_withinAcc, axis=3)
        self.LDS_withinAcc_NormCM = LDS_withinAcc / LDS_withinAcc.sum(axis=2, keepdims=True)
        
        LDS_withinAcc_rand  = np.mean(LDS_withinAcc_rand, axis=3)
        self.LDS_withinAcc_randNormCM = LDS_withinAcc_rand / LDS_withinAcc_rand.sum(axis=2, keepdims=True)
        
        self.LDS_withinAcc = np.mean([np.diag(self.LDS_withinAcc_NormCM[i, :, :]) for i in range(len(self.sbjID))], axis=1)
        self.LDS_withinAcc_rand = np.mean([np.diag(self.LDS_withinAcc_randNormCM[i, :, :]) for i in range(len(self.sbjID))], axis=1)
        
        return self.LDS_withinAcc_NormCM, self.LDS_withinAcc_randNormCM
    
    def linearSVM_acrossSubject(self):
        LDS_TLacc      = np.zeros((len(self.sbjID), len(self.stimID), len(self.stimID)))*np.nan
        LDS_TLacc_rand = np.zeros((len(self.sbjID), len(self.stimID), len(self.stimID)))*np.nan
        
        for sbj in range(len(self.sbjID)):
            target = self.sbjID[sbj]
            source = [ i for i in self.sbjID if i !=  target]

            temp_z_test = self.Z_LDS_decod[target] 
            temp_y_test = self.y_LDS[target]

            temp_z_train = np.concatenate([ self.Z_LDS_decod[i] for i in source ])
            temp_y_train = np.concatenate([ self.y_LDS[i] for i in source ])
            temp_randy_train = np.array(temp_y_train)
            np.random.shuffle(temp_randy_train)

            clf = linear_model.SGDClassifier(max_iter=100, tol=1e-3)
            clf.fit(temp_z_train, temp_y_train)
            predy = clf.predict(temp_z_test)
            LDS_TLacc[sbj, :, :] = confusion_matrix(predy, temp_y_test)

            clfr = linear_model.SGDClassifier(max_iter=100, tol=1e-3)
            clfr.fit(temp_z_train, temp_randy_train)
            predy = clfr.predict(temp_z_test)
            LDS_TLacc_rand[sbj, :, :] = confusion_matrix( temp_y_test , predy ) # true , pred

        self.LDS_TLacc_NormCM = LDS_TLacc / LDS_TLacc.sum(axis=2, keepdims=True)
        self.LDS_TLacc_randNormCM = LDS_TLacc_rand / LDS_TLacc_rand.sum(axis=2, keepdims=True)
        
        self.LDS_TLAcc = np.mean([np.diag(self.LDS_TLacc_NormCM[i, :, :]) for i in range(len(self.sbjID))], axis=1)
        self.LDS_TLAcc_rand = np.mean([np.diag(self.LDS_TLacc_randNormCM[i, :, :]) for i in range(len(self.sbjID))], axis=1)
        
        return self.LDS_TLacc_NormCM, self.LDS_TLacc_randNormCM
    
    def getLatents(self, XX=None, sbj=None, stim=None):
        if XX == None:
            XX = self.data
        if sbj == None:
            sbj = self.sbjID[0]
        if stim == None:
            stim = self.stimID[0]
            
        nTr = len(XX[sbj][stim])
        latents = []
        
        for tt in range(nTr):
            xi = XX[sbj][stim][tt][:,:]
            foozz,_,_ = self.LDS_smooth(xi, self.A_[stim], self.Q_[stim], 
                                        self.C_[sbj], self.R_[sbj], self.x0_[stim], self.Q0_[stim], self.b_[stim])
            latents.append(foozz)  
        return latents # trials x time x latent_dim
    
    def getObserveds(self, ZZ, sbj=None, stim=None, C=None, R=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if sbj == None:
            sbj = self.sbjID[0]
        if stim == None:
            stim = self.stimID[0]
        if C is None:
            C = self.C_[sbj]
        if R is None:
            R = self.R_[sbj]
            
        xdim = C.shape[0]
        nTr = len(ZZ[sbj][stim])
        
        # output variables preallocation
        XX = []
        for tt in range(nTr):
            zi = ZZ[sbj][stim][tt]
            timepoints = zi.shape[0]
            X = np.zeros((timepoints, xdim)) *np.nan
            for t in range(timepoints):
                xt = np.dot(C, zi[t, :]) + np.random.multivariate_normal(np.zeros(xdim), R)
                X[t, :] = xt
            XX.append(X)
        return XX

    def FA_latent(self, C, R, X, X_mu):
        return C.T @ np.linalg.pinv(C @ C.T + R) @ (X.T - np.tile(X_mu, (X.shape[0], 1)).T)
    
    def visualize_performance(self):
        plt.figure(figsize=(1.8,4), facecolor='w')
        cc = 'purple'

        y    = np.array( [np.mean(self.LDS_TLAcc),  np.mean(self.LDS_TLAcc_rand)] )
        yerr = np.array( [np.std(self.LDS_TLAcc)*2, np.std(self.LDS_TLAcc_rand)*2] )
      
        y_bound    = np.array( [np.mean(self.LDS_withinAcc),  np.mean(self.LDS_withinAcc)] )
        yerr_bound = np.array( [np.std(self.LDS_withinAcc)*2, np.std(self.LDS_withinAcc)*2] )
        
        plt.plot([0, 1], y, c= cc, marker ='o', markersize=10)
        plt.errorbar([0, 1], y, yerr = yerr, c= cc, marker ='o', fmt='none')
        plt.hlines(1/len(self.stimID), -.2, 1.2, linestyle= '--', color=cc)
        plt.fill_between([-.2, 1.2],  y_bound-yerr_bound, y_bound+yerr_bound, alpha=.3, color=cc)
        plt.hlines(y_bound, -.2, 1.2, linestyle= '-', color=cc)
        
        plt.ylim(0,1)
        plt.xlim(-.2, 1.2)
        plt.ylabel('Accuracy')
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.xticks([0,1], ['TL-LDS', 'TL_LDS_Shuffle'])
        plt.gca().spines["bottom"].set_position(('outward', 10))
        plt.gca().spines["left"].set_position(('outward', 10))
        plt.gca().spines["bottom"].set_linewidth(2.5)
        plt.gca().spines["left"].set_linewidth(2.5)
        plt.gca().spines['bottom'].set_bounds(0,1);
    
    def visualize_performanceCM(self, CM, title='', annott=True):
        fig, ax = plt.subplots(1, len(self.sbjID), figsize=(17,4), facecolor='w')

        for sbj in range(len(self.sbjID)):
            sns.heatmap(CM[sbj], annot=annott, ax=ax[sbj], cbar=False, vmin=0, vmax=1, 
                        xticklabels = self.stimID, yticklabels = self.stimID, cmap='binary')
            ax[sbj].set_title(title+'  '+self.sbjID[sbj])

    def visualize_params(self, param, IDs=None, minv=-.2, maxv=.8):
        params = getattr(self, str(param+'_'))
        if IDs == None:
            if str(param) in ['R', 'C']:
                IDs = self.sbjID
            elif str(param) in ['Q', 'Q0', 'A', 'x0', 'b']:
                IDs = self.stimID
        
        if str(param) == str('x0'):
            plt.figure(figsize=(5,5), facecolor='w')
            cmapidx=0
            for ID in IDs:
                plt.plot(params[ID], lw=2, color=self.cmap_stim[cmapidx])
                cmapidx+=1
            plt.legend(IDs); plt.xlabel('LD #'); plt.ylabel(str(param));
            
        elif str(param) == str('R'):
            plt.figure(figsize=(5,5), facecolor='w')
            cmapidx=0
            for ID in IDs:
                plt.plot(np.diag(params[ID]), lw=2, color=self.cmap_sbj[cmapidx])
                cmapidx+=1
            plt.legend(IDs); plt.xlabel('Observed dimension #'); plt.ylabel(str(param));
        
        elif str(param) == str('b'):
            plt.figure(figsize=(5,5), facecolor='w')
            IDs = [IDs[i] for i in np.linspace(0, len(IDs)-1, 5).astype(int)]
            cmapidx=0
            for ID in IDs:      
                plt.plot(self.b_[ID][0, :], self.b_[ID][1, :], '-', c=self.cmap_stim[cmapidx], lw=3)
                cmapidx+=1
            plt.legend(IDs); plt.xlabel('LD 1'); plt.ylabel('LD 2');
        
        else:
            fig, ax = plt.subplots(1, len(IDs), figsize=(20,5), facecolor='w')
            for ID in range(len(IDs)):
                sns.heatmap(params[IDs[ID]], annot=True, ax=ax[ID],cbar=False, vmin=minv, vmax=maxv)
                ax[ID].set_title(str(param)+' '+str(IDs[ID]))

    def visualize_latent(self, sbjTP=0, stimTP=0, sbjIDs=None, stimIDs=None, trials=5):
        fig, ax = plt.subplots(1, self.zdim, figsize=(17,4), facecolor='w')

        for dim in range(self.zdim):
            if stimIDs is not None:
                cmap = self.cmap_stim
                for stim in range(len(stimIDs)):
                    y = np.mean(self.Z_LDS[self.sbjID[sbjTP]][stim, :, :, dim], axis=0)
                    x = range(len(y))
                    ax[dim].plot(x, y, '-', c=cmap[stim], lw=3)
                ax[dim].set_title('LDS '+self.sbjID[sbjTP]+'_dim'+str(dim))
            elif sbjIDs is not None:
                cmap = self.cmap_sbj
                for sbj in range(len(sbjIDs)):
                    y = np.mean(self.Z_LDS[sbjIDs[sbj]][stimTP, :, :, dim], axis=0)
                    x = range(len(y))
                    ax[dim].plot(x, y, '-', c=cmap[sbj], lw=3)
                ax[dim].set_title('LDS '+self.stimID[stimTP]+'_dim'+str(dim))

        for dim in range(self.zdim):
            if stimIDs is not None:
                cmap = self.cmap_stim
                for stim in range(len(stimIDs)):
                    fooz = self.Z_LDS[self.sbjID[sbjTP]]
                    x = range(fooz.shape[2])
                    trial_idxs = np.random.choice(fooz.shape[1], trials, replace=False)
                    ax[dim].plot(x, fooz[stim, trial_idxs, :, dim].T, '--', c=cmap[stim], alpha=.4)
            elif sbjIDs is not None:
                cmap = self.cmap_sbj
                for sbj in range(len(sbjIDs)):
                    fooz = self.Z_LDS[sbjIDs[sbj]]
                    x = range(fooz.shape[2])
                    trial_idxs = np.random.choice(fooz.shape[1], trials, replace=False)
                    ax[dim].plot(x, fooz[stimTP, trial_idxs, :, dim].T, '--', c=cmap[sbj], alpha=.4)
                    
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('LD')
        if sbjIDs is not None:
            ax[0].legend(self.sbjID);
        elif stimIDs is not None:
            ax[0].legend(self.stimID);
            
    def visualize_pairLatentDim(self, dim1=0, dim2=1, sbjIDs=None, stimIDs=None, stim0sbj1 = 0,trials=5, Tmax=50):
        if sbjIDs is None:
            sbjIDs = list(self.sbjID)
        if stimIDs == None:
            stimIDs = list(np.arange(len(self.stimID)))
        else:
            stimIDs = [self.stimID.index(i) for i in stimIDs]
   
        if stim0sbj1 ==0:
            cmap = self.cmap_sbj
            fig, ax = plt.subplots(1, len(stimIDs), figsize=(17,4), facecolor='w')
            ax_idx = 0
            for stim_toPlt in stimIDs:
                for sbj in sbjIDs:
                    x = np.mean(self.Z_LDS[sbj][stim_toPlt, :, :Tmax, dim1], axis=0)
                    y = np.mean(self.Z_LDS[sbj][stim_toPlt, :, :Tmax, dim2], axis=0)

                    ax[ax_idx].plot(x, y, '-', c=cmap[sbjIDs.index(sbj)], lw=3)
                ax[ax_idx].set_title('LDS '+str(self.stimID[stim_toPlt]))
                ax_idx +=1

            ax_idx = 0
            for stim_toPlt in stimIDs:
                for sbj in sbjIDs:
                    trial_idxs = np.random.choice(self.Z_LDS[sbj].shape[1], trials, replace=False)
                    ax[ax_idx].plot(self.Z_LDS[sbj][stim_toPlt, trial_idxs, :Tmax, dim1].T, 
                                    self.Z_LDS[sbj][stim_toPlt, trial_idxs, :Tmax, dim2].T, 
                                    '--', c=cmap[sbjIDs.index(sbj)], alpha=.4)
                ax_idx +=1
            ax[0].legend([self.stimID[i] for i in stimIDs]);
            
        elif stim0sbj1 ==1:
            cmap = self.cmap_stim
            fig, ax = plt.subplots(1, len(sbjIDs), figsize=(17,4), facecolor='w')
            ax_idx = 0
            for sbj in sbjIDs:
                for stim_toPlt in stimIDs: 
                    x = np.mean(self.Z_LDS[sbj][stim_toPlt, :, :Tmax, dim1], axis=0)
                    y = np.mean(self.Z_LDS[sbj][stim_toPlt, :, :Tmax, dim2], axis=0)

                    ax[ax_idx].plot(x, y, '-', c=cmap[stim_toPlt], lw=3)
                ax[ax_idx].set_title('LDS '+sbj)
                ax_idx +=1

            ax_idx = 0
            for sbj in sbjIDs:
                trial_idxs = np.random.choice(self.Z_LDS[sbj].shape[1], trials, replace=False)
                for stim_toPlt in stimIDs:
                    ax[ax_idx].plot(self.Z_LDS[sbj][stim_toPlt, trial_idxs, :Tmax, dim1].T, 
                                    self.Z_LDS[sbj][stim_toPlt, trial_idxs, :Tmax, dim2].T, 
                                    '--', c=cmap[stim_toPlt], alpha=.4)
                ax_idx +=1
            ax[0].legend(stimIDs);

        ax[0].set_xlabel('LD-'+str(dim1))
        ax[0].set_ylabel('LD-'+str(dim2));
        
    
    def visualize_reconstruction(self, XX=None, cond=None, sbj=None, trial=None, neuron=None, A=None, Q=None, C=None, R=None, x0=None, Q0=None, b=None):
        if XX == None:
            XX = self.data
        if trial == None:
            trial = np.random.choice(len(XX[sbj][cond]), 1)[0]
        if neuron == None:
            neuron = np.random.choice(XX[sbj][cond][trial].shape[1], 1)[0]
        if A == None:
            A = self.A_
        if Q == None:
            Q = self.Q_
        if x0 == None:
            x0 = self.x0_
        if Q0 == None:
            Q0 = self.Q0_
        if C == None:
            C = self.C_
        if R == None:
            R = self.R_
        if b == None:
            b = self.b_
            
        x = XX[sbj][cond][trial]
        _,x_rec,x_rec_err= self.trial_reconstruction_error(x,A[cond],Q[cond],C[sbj],R[sbj],x0[cond],Q0[cond],b[cond],var=True)
        
        plt.figure(figsize=(7,3), facecolor='w')
        plt.plot(range(len(x[:, neuron])),     x[:, neuron],     c='k', lw=2)
        plt.plot(range(len(x_rec[:, neuron])), x_rec[:, neuron], c='k', lw=2, ls='--')
        plt.fill_between(range(len(x_rec[:, neuron])), 
                         x_rec[:, neuron]-x_rec_err[:, neuron]*2, 
                         x_rec[:, neuron]+x_rec_err[:, neuron]*2, 
                         alpha=0.2, color='k')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.legend(['data', 'reconstruction'])
        plt.title(sbj+' '+cond+' trial '+str(trial)+' neuron '+str(neuron))
        return x, x_rec, x_rec_err
    
    def visualize_Prstim(self, sbjIDs=None):
        if sbjIDs is None:
            sbjIDs = list(self.sbjID)
            
        fig, ax = plt.subplots(1, len(sbjIDs), facecolor='w', figsize=(20, 8))

        toHeat = [self.sbj_trialsPr[sbjIDs[i]] for i in range(len(sbjIDs))]
        for m in range(len(sbjIDs)):
            sns.heatmap(toHeat[m], cmap='binary', ax=ax[m], cbar=False);
            ax[m].set_title(sbjIDs[m])
        ax[0].set_ylabel('trials')
        ax[0].set_xlabel('stim');
    
    
    