#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

def estimate_then_test(X_train,y_train,X_test,y_test,sigma):
    
    '''
    Input:
    X_train, y_train : data to estimate
    X_test, y_test : calculate the error
    
    sigma : estimation
    
    '''
    iter = 36
    
    #run Frank-Wofle
    f, B, alpha, L_rec, L_final = NPMLE_FW(X_train,y_train,iter,sigma)
    
    cluster_test = np.zeros((len(y_test),1),dtype = int)
    N = len(B[0])
    
    error = 0
    for i in range(len(X_test[:,0])):
        prob = np.zeros(N)
        for j in range(N):
            prob[j] = alpha[j] * np.exp(-0.5*(y_test[i] - np.dot(X_test[i],B[:,j]))**2 /(sigma**2))
        cluster_test[i] = np.argmax(prob)
        #print(cluster_test[i])
        error = error + (y_test[i] - np.dot(X_test[i],B[:,cluster_test[i]]))**2
    return error/len(y_test) 

def cross_validation(X,y,sigma,k):
    '''
    k - fold cross-validation
    
    
    '''
    n = len(X[:,0])
    m = int(n/k)
    
    
    # permutate X and y
    X_rdn = np.array(X[np.random.permutation(np.random.permutation(n)),:])
    y_rdn = np.array(y[np.random.permutation(np.random.permutation(n))])
    
    error = 0
    
    for idex in range(k):
        print(idex)
        if idex == 0:
            error = error + estimate_then_test(X_rdn[(idex+1)*m+1:,:], \
                                           y_rdn[(idex+1)*m+1:],\
                                           X_rdn[:(idex+1)*m+1,:],y_rdn[:(idex+1)*m+1],sigma)
        elif idex == k-1:
            error = error + estimate_then_test(X_rdn[:idex*m,:], \
                                           y_rdn[:idex*m],\
                                           X_rdn[idex*m:,:],y_rdn[idex*m:],sigma)
        else:
            error = error + estimate_then_test(np.concatenate((X_rdn[:idex*m,:],X_rdn[(idex+1)*m+1:,:])), \
                                           np.concatenate((y_rdn[:idex*m],y_rdn[(idex+1)*m+1:])),\
                                           X_rdn[idex*m:(idex+1)*m+1,:],y_rdn[idex*m :(idex+1)*m+1],sigma)
    return error/k
   