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
from itertools import repeat
import multiprocessing 
from multiprocessing import Pool
from regression_func_lib import *

def cross_validation_parallel(X,y,sigma_list,k,BL,BR,func=lin_func):
    '''
    k - fold cross-validation for one sigma value
    
    Based on total likelihood value
    
    Input
    -----
        X: design points
        y: data
        sigma_list: a list of candidate sigma values
        k: number of folds
        [BL,BR]: parameter range
  
    Output
    ------
        CV criterion for the given list of sigma
    '''
    n = len(X[:,0])
    m = int(n/k)
    
    rdn_index = np.random.permutation(n)
    # permutate X and y
    X_rdn = np.array(X[rdn_index,:])
    y_rdn = np.array(y[rdn_index])
    
    CV_result = np.zeros((len(sigma_list),2))
    CV_result[:,0] = sigma_list
    
    with Pool(8) as cv_pool:
        CV_result[:,1] = cv_pool.starmap(cross_validation, \
            zip(repeat(X_rdn), repeat(y_rdn), sigma_list, repeat(k), repeat(BL), repeat(BR), repeat(func)))
    return CV_result

def cross_validation(X,y,sigma,k,BL,BR,func=lin_func):
    '''
    k - fold cross-validation for one sigma value
    
    Based on total likelihood value
    '''
    n = len(X[:,0])
    m = int(n/k)

    # no need to permuate here 
    X_rdn = X
    y_rdn = y

    log_L = 0

    for idex in range(k):
        if idex == 0:
            log_L = log_L + estimate_then_testlikelihood(X_rdn[(idex+1)*m:,:], \
                                           y_rdn[(idex+1)*m:],\
                                           X_rdn[:(idex+1)*m,:],y_rdn[:(idex+1)*m],sigma,BL,BR,func)
        elif idex == k-1:
            log_L = log_L + estimate_then_testlikelihood(X_rdn[:idex*m,:], \
                                           y_rdn[:idex*m],\
                                           X_rdn[idex*m:,:],y_rdn[idex*m:],sigma,BL,BR,func)
        else:
            log_L = log_L + estimate_then_testlikelihood(np.concatenate((X_rdn[:idex*m,:],X_rdn[(idex+1)*m:,:])), \
                                           np.concatenate((y_rdn[:idex*m],y_rdn[(idex+1)*m:])),\
                            X_rdn[idex*m:(idex+1)*m,:],y_rdn[idex*m :(idex+1)*m],sigma,BL,BR,func)
    # return negative log likelihood
    return -log_L/k


def estimate_then_testlikelihood(X_train,y_train,X_test,y_test,sigma, BL,BR,func=lin_func):
    
    '''
    Input:
    X_train, y_train : data to estimate
    X_test, y_test : calculate the error
    
    sigma: 
    
    '''
    iter = 50
    
    #run Frank-Wofle
    f, B, alpha, L_rec, L_final = NPMLE_FW(X_train,y_train,iter,sigma,BL,BR, func)
    
    cluster_test = np.zeros((len(y_test),1),dtype = int)
    N = len(B[0])
    
    log_L = 0
    
    y_predict = np.zeros(len(y_test))
    for i in range(len(X_test[:,0])):
        y_L = 0
        for j in range(N):
            y_L = y_L + alpha[j] * 1/(np.sqrt(2*np.pi)*sigma)* \
            np.exp(-0.5*(y_test[i] - func(X_test[i],B[:,j]))**2 /(sigma**2))
        log_L = log_L + np.log(y_L)
    return log_L




















#------------------------------------------------#
# Another cross validation procedure
# Based on posterior mean 
# \hat{y}_i = \sum_{l} \hat{\PP}_{l}^{-c}(x_i,y_i) x_i^\T \hat{\beta}_l^{-c}.
# see reference: section 4.2 in Huang, Mian et al. (2013)
#------------------------------------------------#

def cross_validation_posterror(X,y,sigma,k,BL,BR):
    '''
    k - fold cross-validation for one sigma value

    '''
    n = len(X[:,0])
    m = int(n/k)
    
    rdn_index = np.random.permutation(n)
    
    # permutate X and y
    X_rdn = np.array(X[rdn_index,:])
    y_rdn = np.array(y[rdn_index])
    
    error = 0
    
    for idex in range(k):
        if idex == 0:
            error = error + estimate_then_test(X_rdn[(idex+1)*m:,:], \
                                           y_rdn[(idex+1)*m:],\
                                           X_rdn[:(idex+1)*m,:],y_rdn[:(idex+1)*m],sigma,BL,BR)
        elif idex == k-1:
            error = error + estimate_then_test(X_rdn[:idex*m,:], \
                                           y_rdn[:idex*m],\
                                           X_rdn[idex*m:,:],y_rdn[idex*m:],sigma,BL,BR)
        else:
            error = error + estimate_then_test(np.concatenate((X_rdn[:idex*m,:],X_rdn[(idex+1)*m:,:])), \
                                           np.concatenate((y_rdn[:idex*m],y_rdn[(idex+1)*m:])),\
                                           X_rdn[idex*m:(idex+1)*m,:],y_rdn[idex*m :(idex+1)*m],sigma,BL,BR)
    return error/k


def estimate_then_test(X_train,y_train,X_test,y_test,sigma, BL,BR):
    
    '''
    Input:
    X_train, y_train : data to estimate
    X_test, y_test : calculate the error
    
    sigma : estimation
    
    '''
    iter = 50
    
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
    
    beta0 = np.reshape(np.dot( np.matmul(linalg.inv(np.matmul(X_train.T,X_train)),X_train.T),y_train),(len(X_test[0]),1)) 
    y_ols = np.matmul(X_test,beta0)
    print("L1 distance of y_ols and y_predict",np.linalg.norm(y_ols.ravel()-y_predict.ravel(),ord = 1))
    return error







