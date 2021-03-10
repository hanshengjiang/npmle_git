#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""
'''
Cross-Validation procedure for choosing sigma
'''

from package_import import *
from alg1_lib import *

def cross_validation(X,y,sigma,k,BL,BR):
    '''
    k - fold cross-validation for one sigma value

    '''
    n = len(X[:,0])
    m = int(n/k)
    
    
    # permutate X and y
    X_rdn = np.array(X[np.random.permutation(np.random.permutation(n)),:])
    y_rdn = np.array(y[np.random.permutation(np.random.permutation(n))])
    
    error = 0
    
    for idex in range(k):
        # print(idex)
        if idex == 0:
            error = error + estimate_then_test(X_rdn[(idex+1)*m+1:,:], \
                                           y_rdn[(idex+1)*m+1:],\
                                           X_rdn[:(idex+1)*m+1,:],y_rdn[:(idex+1)*m+1],sigma,BL,BR)
        elif idex == k-1:
            error = error + estimate_then_test(X_rdn[:idex*m,:], \
                                           y_rdn[:idex*m],\
                                           X_rdn[idex*m:,:],y_rdn[idex*m:],sigma,BL,BR)
        else:
            error = error + estimate_then_test(np.concatenate((X_rdn[:idex*m,:],X_rdn[(idex+1)*m+1:,:])), \
                                           np.concatenate((y_rdn[:idex*m],y_rdn[(idex+1)*m+1:])),\
                                           X_rdn[idex*m:(idex+1)*m+1,:],y_rdn[idex*m :(idex+1)*m+1],sigma,BL,BR)
    return error/k


def estimate_then_test(X_train,y_train,X_test,y_test,sigma, BL,BR):
    
    '''
    Input:
    X_train, y_train : data to estimate
    X_test, y_test : calculate the error
    
    sigma : estimation
    
    '''
    iter = 36
    
    #run Frank-Wofle
    f, B, alpha, L_rec, L_final = NPMLE_FW(X_train,y_train,iter,sigma,BL,BR)
    
    cluster_test = np.zeros((len(y_test),1),dtype = int)
    N = len(B[0])
    
    error = 0
    
    y_predict = np.zeros(len(y_test))
    for i in range(len(X_test[:,0])):
        prob = np.zeros(N)
        for j in range(N):
            prob[j] = alpha[j] * np.exp(-0.5*(y_test[i] - np.dot(X_test[i],B[:,j]))**2 /(sigma**2))
        prob_sum = np.sum(prob)
        prob = prob/prob_sum
        for j in range(N):
            y_predict[i] = y_predict[i] + prob[j] * np.dot(X_test[i],B[:,j])
            
        #-----------another kind of CV criterion------#
#        cluster_test[i] = np.argmax(prob)
#        y_predict[i] = np.dot(X_test[i],B[:,cluster_test[i]])
        #---------------------------------------------#
        error = error + (y_test[i] - y_predict[i])**2
    
    beta0 = np.reshape(np.dot( np.matmul(linalg.inv(np.matmul(X_train.T,X_train)),X_train.T),y_train),(p,1)) 
    y_ols = np.matmul(X_test,beta0)
    print("difference of y_ols and y_predict",np.linalg.norm(y_ols.ravel()-y_predict.ravel(),ord = 1))
    return error/len(y_test) 







