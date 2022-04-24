import numpy as np
import pandas as pd

from pyitlib import discrete_random_variable as drv

def SU(data, i, j):
    
    '''
    Symmetric Uncertainty as defined in the paper
    
    SU(X,Y) = 2*IG(X,Y)/(H(X)+H(Y))
    
    where 
    
    - IG(.,.) is the Information Gain 
    - H(.) is the Shannon entropy
    
    '''
    su = 2*drv.information_mutual(data[:,i], data[:,j])/(drv.entropy(data[:,i])+drv.entropy(data[:,j]))
    return su

def SU_M(data):
    
    '''
    
    It returns the matrix of the Symmetric Uncertainties
    
    '''
    
    SU_matrix = [[0 for i in range(np.shape(data)[1])] for j in range(np.shape(data)[1])]
    for i in np.arange(np.shape(data)[1]):
        for j in np.arange(i, np.shape(data)[1]):
            if not i == j:
                SU_matrix[i][j] = SU(data, i, j)
                SU_matrix[j][i] = SU_matrix[i][j]
    SU_matrix = np.asarray(SU_matrix)
    SU_matrix = pd.DataFrame(SU_matrix)
    return SU_matrix
    
def AR(data, f, Cj):
    
    ''' 
    
    Average Redundancy for a given feature f in the cluster C
    
    '''
    
    ar = 0
    for j in range(len(Cj)):
        ar = ar + SU(data, f , Cj[j])
    ar = ar/len(Cj)
    return ar 

def FSFC(data, k):
    
    ''' 
    
    k = number of nearest neighbors
    data is the original data input 
    
    it computes the FSFC based on the paper
    
    'A new unsupervised feature selection algorithm using similarity‐based feature clustering' 
    Xiaoyan Zhu, Yu Wang, Yingbin Li, Yonghui Tan, Guangtao Wang, Qinbao Song
    
    '''
    
    FS_set = set(np.arange(np.shape(data)[1]))
   
    # initialization of SU matrix and parameter
    
    m = 0
    SU_matrix = SU_M(data) 
    
    # create the k-nearest neighbor for each variable 
    
    knn = [0 for i in range(np.shape(data)[1])] 
    for i in np.arange(np.shape(data)[1]):
        for less_than_k in np.arange(k):
            knn[i] = set(SU_matrix.sort_values(i, ascending = False).index[:k])
    
    # k nearest neighbor density
    dknn = SU_matrix.apply(lambda x: x.sort_values(ascending = False).values).iloc[0:k].sum().values
    
    # initialization of FS and FC
    
    FS = pd.DataFrame(dknn).sort_values(0, ascending = False)
    FC = []
    
    FC.append(FS.index[0])    
    max_FC = 0
    for fs in FS_set.difference(set(FC)):
        for fc in FC:
            if ((fc not in knn[fs]) and (fs not in knn[fc])):
                FC.append(fs)
                max_FC = max(max_FC, SU_matrix[fs][fc])
                m = m + 1
                break
    
    # initialize centers 
    
    C = [[] for i in range(len(FC))]
    
    for p in range(len(FC)):
        C[p].append(FC[p])

    for fi in FS_set.difference(set(FC)):
        max_ = SU_matrix.loc[fi][FC[0]]
        
        argmax_ = 0
        f_max = FC[argmax_]
        
        for i in range(len(FC)):
            fc = FC[i]
            if SU_matrix.loc[fi][fc] > max_:
                max_ = SU_matrix.loc[fi][fc]
                argmax_ = i
                f_max = FC[argmax_]
        if SU_matrix.loc[fi][f_max] > max_FC:
            C[argmax_].append(fi)
        else:
            FC.append(fi)
            C.append([fi])
            m = m + 1
    
    F_selected = []
    for i in range(len(C)):
        ar_max = 0
        for j in range(len(C[i])):
            if max(ar_max, AR(data, C[i][j], C[i])) > ar_max:
                f = C[i][j]
                
        F_selected.append(f)
    
    return set(F_selected)

