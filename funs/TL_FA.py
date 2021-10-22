import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

class TL_FA:
    def __init__(self, xdim, zdim, sbjID, data=None):
        self.xdim = xdim      # int ; number of observed dimensions ## same for all subjects and stimuli
        self.zdim = zdim      # int ; number of latent dimensions   ## same for all subjects and stimuli
        
        self.sbjID  = sbjID   # list of str ; subjects ID
        
        if data is not None:
            self.data = data  # dict of subjects ; observation [samples x xdim]
            
        self.cmap_sbj  = ['#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        
    def import_data(self, data):
        self.data = data
        
    def data_stats(self, X=None, sbj_list=None):
        if X == None:
            X = self.data
        if sbj_list == None:
            sbj_list = self.sbjID
            
        self.X_mu      = {key: np.mean(X[key].T, axis=1) for key in sbj_list}
        self.X_cov     = {key: np.cov( X[key].T ) for key in sbj_list} 
        self.X_samples = {key: X[key].shape[0] for key in sbj_list}
        
    def init_params(self):
        try:
            self.X_cov
            self.R_ = {key: np.diag(np.diag(self.X_cov[key])) for key in self.sbjID} 
        except:
            self.R_ = {key: np.diag(np.random.uniform(0,1, (self.xdim,))) for key in self.sbjID} 
            
        self.C = abs(np.random.uniform(0,1, (self.xdim, self.zdim))/np.sqrt(self.zdim))
         
    def sample(self, samples, C=None, R=None):
        if C.all() == None:
            C = self.C
        if R.all() == None:
            R = self.R_[self.sbjID[0]]

        Z = np.random.normal(0, 1, (self.zdim, samples))
        w = np.dot(R, np.random.randn(self.xdim, samples))
        X = np.dot(C, Z) +w
        return X, Z, C, R
    
    def reconstruction(self, X, C=None, R=None):
        ''' Performs leave-neuron-out (LNO) error for factor analysis
        @ arguments:
        - C loading factors: 2D numpy array [xDim x zDim]
        - R observation noise: 2D numpy array [xDim x xDim]
        - X data: 2D numpy array [observations x xDim]
        - X_mu data mean: 1D numpy array [xDim]
        @ output:
        - err LNO-CV error: scalar '''
        xDim, zDim = C.shape
        I = np.eye(zDim)

        Xcs = np.zeros((X).shape)*np.nan
    #     Vcs = np.zeros((xDim, 1))*np.nan

        for ii in range(xDim):
            idx = np.ones(xDim, dtype='bool')
            idx[ii] = False

            Rinv   = 1 / np.diag(R)[idx]                                # [ xDim-1 x 1      ]
            CRinv  = (C[idx, :] * (np.tile(Rinv, (zDim, 1)).T)).T       # [   zDim x xDim-1 ]
            CRinvC = np.dot(CRinv, C[idx, :])                           # [   zDim x zDim   ]

            term2  = np.dot(C[ii, :], (I - np.dot(CRinvC , np.linalg.pinv(I + CRinvC)))) # [ zDim ]

            dif    = (X[idx, :].T - np.mean(X[idx, :], 1)).T            # [ xDim-1 x observations ]
            Xcs[ii,:] = np.mean(X, 1)[ii] + np.dot(np.dot(term2, CRinv), dif) # [ observations ] 
    #         Vcs[ii] = C[~idx, :] @ C[~idx, :].T + np.diag(R)[~idx] - term2 @ CRinvC @ C[~idx, :].T

        err = np.mean((Xcs.ravel() - X.ravel()) ** 2)  
        return err, Xcs
    
    def posterior_likelihood(self, X_cov=None, C=None, R=None, sbjID=None, X_samples=None, xdim=None):
        if X_cov == None:
            X_cov = self.X_cov
        if sbjID == None:
            sbjIDs = self.sbjID
        if C == None:
            C = self.C
        if R == None:
            Rs = self.R_
        if X_samples == None:
            X_samples = self.X_samples
        if xdim == None:
            xdim = self.xdim
            
        sbj_postLL = {}
        for sbj in sbjIDs:
            T   = X_samples[sbj]
            # E-step
            delta = np.linalg.pinv(C @ C.T + Rs[sbj])
            sbj_postLL[sbj] = (-T/2*np.trace(delta @ X_cov[sbj]) + 
                                T/2*np.linalg.slogdet(delta)[1] - 
                                T*xdim/2*np.log(2*np.pi) )
        self.sbj_postLL = sbj_postLL
            
        return np.array( [ sbj_postLL[sbj] for sbj in sbjIDs ] )
    
    def reconstruction_error(self, X=None, C=None, R=None, sbjID=None):
        if X == None:
            X = self.data
        if sbjID == None:
            sbjIDs = self.sbjID
        if C == None:
            C = self.C
        if R == None:
            Rs = self.R_
            
        sbj_recErr = {}
        
        for sbj in sbjIDs:
            sbj_recErr[sbj] = self.reconstruction(X[sbj].T, C, Rs[sbj])[0]
        self.sbj_recErr = sbj_recErr
        
        return np.array( [ sbj_recErr[sbj] for sbj in sbjIDs ] ) 
    
    def TL_FA_getR(self, X, C, eps=1e-7):
        R = ( np.cov(X.T) - C @ C.T ) * np.eye(X.shape[1])
        R[R < 0] = eps
        return R
    
    def TL_FA_getR_cov(self, X_cov, C, eps=1e-7):
        R = ( X_cov - C @ C.T ) * np.eye(X_cov.shape[1])
        R[R < 0] = eps
        return R
    
    def infere_latent(self, C, R, X, X_mu):
        return C.T @ np.linalg.pinv(C @ C.T + R) @ (X.T - np.tile(X_mu, (X.shape[0], 1)).T)
    
    def shuffle_along_axis(self, a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)
        
    def FA_EM_TransferLearning(self, C=None, Rs=None, sbj_list=None, eps=1e-7, max_iter=20, update_C=True, update_R=True): 
        if C == None:
            C = self.C
        if Rs == None:
            Rs = self.R_
        if sbj_list == None:
            sbj_list = self.sbjID
            
        idx = 0
        counter = 0 
        LL_prev = 0                   # initiate LL reference
        LL_step = eps+1               # non-stop value for while loop
        LL_cache = []                 # array with LL values
        
        while (LL_step > eps) or (counter < len(sbj_list)):    # EM FA
            sbj = sbj_list[idx]
            T   = self.X_samples[sbj]
            # E-step
            delta = np.linalg.pinv(C @ C.T + Rs[sbj])
            beta = C.T @ delta

            # M-step; parameter update
            C = (self.X_cov[sbj] @ beta.T @ 
                 np.linalg.pinv(np.identity(self.zdim) - beta @ C + beta @ self.X_cov[sbj] @ beta.T))
            Rs[sbj] = np.diag(np.diag(self.X_cov[sbj] - C @ beta @ self.X_cov[sbj]))

            # estimate posterior log-likelihood
            if np.linalg.slogdet(delta)[0] > 0:
                LL = -T/2*np.trace(delta @ self.X_cov[sbj]) + T/2*np.linalg.slogdet(delta)[1] - T*self.xdim/2*np.log(2*np.pi) 
                                                # N*sum(log(diag(chol(MM))))
            elif np.linalg.slogdet(delta)[0] < 0:
    #             print(str(zDim)+'Negative determinant')
                LL = -T/2*np.trace(delta @ self.X_cov[sbj]) + T/2*np.linalg.slogdet(delta)[1] - T*self.xdim/2*np.log(2*np.pi) 
        
            LL_step = abs((LL-LL_prev)/abs(LL))
            LL_prev = LL
            LL_cache.append(LL)
            counter += 1
            idx +=1
            if counter > max_iter:
                break
            if idx == len(sbj_list):
                idx = 0 
        
        # update and store parameters
        if update_C == True:
            self.C = C
        if update_R == True:
            self.R_ = Rs
        
        self.EM_LL = LL_cache
        return C, Rs, LL_cache
    
    def visualize_params(self, R=None, C=None):
        if C is None:
            C = self.C
        if R == None:
            R = self.R_
            
        fig, ax = plt.subplots(1, 2, figsize=(13,4), facecolor='w')
        
        plotR = [np.diag(R[sbj]) for sbj in self.sbjID]
        [ax[0].plot(plotR[i], c=self.cmap_sbj[i]) for i in range(len(plotR))];
        ax[0].legend(self.sbjID)
        ax[0].set_xlabel('OD #')
        ax[0].set_ylabel('observation noise')
        
        sns.heatmap(C, annot=True, ax=ax[1], cbar=False, vmin=-.2, vmax=.8)
        ax[1].set_xlabel('LD #')
        ax[1].set_ylabel('OD #')
        ax[1].set_title('loading matrix TL-FA')
        
        
class TL_FA_decoding:
    def __init__(self, xdim, zdim, sbjID, tr_, Y, nStim=5, t=41, data=None, stimID=['AIR','BZD','ETG','HEX','MVT']):
        self.xdim = xdim      # int ; number of observed dimensions ## same for all subjects and stimuli
        self.zdim = zdim      # int ; number of latent dimensions   ## same for all subjects and stimuli
        
        self.sbjID  = sbjID   # list of str ; subjects ID
        
        self.Y     =Y
        self.tr_   = tr_      # dict of subjects ; number of trials
        self.t     = t        # int ; trial time length
        self.nStim = nStim    # int ; number of stimulus conditions
        self.stimID = stimID  # list of str ; stimulus ID
        
        if data is not None:
            self.data = data  # dict of subjects ; observation [samples x xdim]
        
    def import_data(self, data):
        self.data = data
        
    def linearSVM_decoding(self, X=None, sbjID=None, Y=None, tr_= None, t=None, nStim=None):
        if X == None:
            X = self.data
        if sbjID == None:
            sbjIDs = self.sbjID
        if t == None:
            t = self.t
        if nStim == None:
            nStim = self.nStim
        if tr_ == None:
            tr_ = self.tr_
        if Y == None:
            Y = self.Y
            
        FA_withinAcc   = []
        FA_withinAccCM = np.zeros((len(sbjIDs), nStim, nStim))*np.nan
        FA_acrossAcc   = []
        FA_acrossAccCM = np.zeros((len(sbjIDs), nStim, nStim))*np.nan
        
        # train with more or less data
        TLpop_SS = []
        dataRange = [5, 6, 7, 8, 10, 20, 30, 40, 60, 80, 100, 150, 200, 250]#, 300, 350, 400, 450]
        
        # shuffle neuron labels: bootstrapping
        boots = 10
        TL_PS = np.zeros((len(sbjIDs), boots))
            
        for i in range(len(sbjIDs)):
            test_mouse = sbjIDs[i]
            source_mice = [ i for i in sbjIDs if i !=  test_mouse]
            X_source = {key: X[key] for key in source_mice}
#             X_source = ( [ X[i] for i in source_mice] )
            
            TLfa = TL_FA(xdim=self.xdim, zdim=self.zdim, sbjID=source_mice)
            TLfa.import_data(data=X_source)
            TLfa.data_stats()
            TLfa.init_params()
            C, Rs, _ = TLfa.FA_EM_TransferLearning();
            
            # Project source data into source manifold for classifier training
            Z_respP = []
            cnt = 0
            for mouse in source_mice:
                ZP = TLfa.infere_latent(C, Rs[mouse], X[mouse], np.mean(X[mouse].T, axis=1))
                b = np.reshape(ZP, (self.zdim, t, min(tr_[mouse]) , nStim ), order='F')
                c = np.reshape(b,  (self.zdim, t, min(tr_[mouse]) * nStim ), order='F')
                d = np.reshape(c, ((self.zdim)*t, min(tr_[mouse]) * nStim ), order='C')
                Z_respP.append(np.moveaxis(d, 0, -1))
                cnt+=1
            ZP = np.concatenate(Z_respP)
            YP = np.concatenate( [ Y[i] for i in source_mice] )

            # Structure and project target data into source manifold
            Xt    = X[test_mouse]
            Xt_mu = np.mean(X[test_mouse].T, axis=1)

            # estimate target (test) subject observation noise and project into source manifold
            Rt = TLfa.TL_FA_getR(Xt, C)
            zt = TLfa.infere_latent(C, Rt, Xt, Xt_mu)

            b = np.reshape(zt, (self.zdim, t, min(tr_[test_mouse]) , nStim ), order='F')
            c = np.reshape(b,  (self.zdim, t, min(tr_[test_mouse]) * nStim ), order='F')
            d = np.reshape(c, ((self.zdim)*t, min(tr_[test_mouse]) * nStim ), order='C')
            zt_resp = np.moveaxis(d, 0, -1)

            # Train decoder using source data and predict target
            clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#             scores = cross_val_score(clf, ZP, YP, cv=5)
            clf.fit(ZP, YP)

            predy = clf.predict(zt_resp)
            yt = Y[test_mouse]
            FA_acrossAcc.append( np.sum(np.equal(predy , yt))/len(yt) )
            FA_acrossAccCM[i, :, :] = confusion_matrix(yt, predy)

            # train new classifier
            clfS = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#             clfS.fit(zt_resp, yt)
#             predy = clf.predict(zt_resp)
            FA_withinAcc.append( np.mean(cross_val_score(clfS, zt_resp, yt, cv=5)) )
#             FA_withinAccCM[i, :, :] = confusion_matrix(yt, predy)

#             # train with more or less data
#             TLpop_samplesize = []
#             for ii in dataRange:
#                 idxs = sample(range(len(YP)), ii)
#                 temp_ZP, temp_YP = ZP[idxs, :], YP[idxs] 
#                 clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#                 clf.fit(temp_ZP, temp_YP)

#                 predy = clf.predict(zt_resp)
#                 TLpop_samplesize.append(np.sum(np.equal(predy , yt))/len(yt))
#             TLpop_SS.append(TLpop_samplesize)
            
            # Shuffle, project and predict
            for k in range(boots):
                Xt = TLfa.shuffle_along_axis(X[test_mouse], 1)  
                Xt_mu = np.mean(Xt.T, axis=1); 

                Rt = TLfa.TL_FA_getR(Xt, C)
                zt = TLfa.infere_latent(C, Rt, Xt, Xt_mu)

                b = np.reshape(zt, (self.zdim, t, min(tr_[test_mouse]) , nStim ), order='F')
                c = np.reshape(b,  (self.zdim, t, min(tr_[test_mouse]) * nStim ), order='F')
                d = np.reshape(c, ((self.zdim)*t, min(tr_[test_mouse]) * nStim ), order='C')
                zt_resp = np.moveaxis(d, 0, -1)  

                # Train decoder using source data and predict target
                clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
                scores = cross_val_score(clf, ZP, YP, cv=5)
                clf.fit(ZP, YP)

                predy = clf.predict(zt_resp)
                yt = Y[test_mouse] 
                TL_PS[i, k] = np.sum(np.equal(predy , yt))/len(yt)

        self.FA_acrossAcc_rand = np.mean(TL_PS, axis=1)
        
        self.FA_withinAcc   = FA_withinAcc
        self.FA_acrossAcc   = FA_acrossAcc
        self.FA_acrossAccCM = FA_acrossAccCM / FA_acrossAccCM.sum(axis=2, keepdims=True)

        return FA_withinAcc, FA_acrossAcc, FA_acrossAccCM
    
    
    def visualize_performance(self):
        plt.figure(figsize=(1.8,4), facecolor='w')
        cc = 'darkorange'

        y    = np.array( [np.mean(self.FA_acrossAcc),  np.mean(self.FA_acrossAcc_rand)] )
        yerr = np.array( [np.std(self.FA_acrossAcc)*2, np.std(self.FA_acrossAcc_rand)*2] )
      
        y_bound    = np.array( [np.mean(self.FA_withinAcc),  np.mean(self.FA_withinAcc)] )
        yerr_bound = np.array( [np.std(self.FA_withinAcc)*2, np.std(self.FA_withinAcc)*2] )
        
        plt.plot([0, 1], y, c= cc, marker ='o', markersize=10)
        plt.errorbar([0, 1], y, yerr = yerr, c= cc, marker ='o', fmt='none')
        plt.hlines(1/self.nStim, -.2, 1.2, linestyle= '--', color=cc)
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
    
    def visualize_performanceCM(self, CM, title=''):
        fig, ax = plt.subplots(1, len(self.sbjID), figsize=(17,4), facecolor='w')

        for sbj in range(len(self.sbjID)):
            sns.heatmap(CM[sbj], annot=True, ax=ax[sbj], cbar=False, vmin=0, vmax=1, 
                        xticklabels = self.stimID, yticklabels = self.stimID)
            ax[sbj].set_title(title+'  '+self.sbjID[sbj])
            
            
            