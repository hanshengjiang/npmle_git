#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""
from package_import import *
import sys


def EMA_sigma(X,y,k,iter,BL,BR,sigma):
    '''
    Use EM algorithm to fit (fixed component number) mixture of linear regression  
    Use known fixed sigma value
    ---------------------------------------------
    Input
    X: n * p, covariate matrix
    y: n * 1, response variable
    k: number of components
    iter: number of iterations
    [BL,BR]^p: randomization range of regression parameters
    sigma: std of noise
    ---------------------------------------------
    Output
    f: n * k, atomic likelihood vectors in active set
    B: p * k, coefficients corresponding to vectors in active set
    alpha: k*1, mixing proportions of vectors in active set
    sigma_array: k*1, sigma value of each component
    L_rec: neg-log likelihood over iterations
    L_temp: final neg-log likelihood
    ---------------------------------------------
    
    '''  
    n = len(X)
    p = len(X[0])
    wden = np.zeros((n,k)) # unweighted posterior probability 
    w = np.zeros((n,k)) # posterior probability
    sigma_array = np.zeros((k,))
    z = np.zeros((n,)) #latent variable, class of each data point
    f = np.zeros((n,))
    L_rec = []
    
    #intialization
    B = np.zeros((p,k))
    alpha = np.zeros((k,1))
    for j in range(k):
        B[:,j] = np.random.uniform(BL,BR,(p,))
        alpha[j][0] = 1/k  
        sigma_array[j] = sigma # use know sigma
    
    for r in range(iter):
        
        # C step
        for i in range(n):
            wden_temp = 0
            for j in range(k):
                wden_temp = max(wden_temp, (y[i]- np.dot(B[:,j],X[i]))**2 )
            for j in range(k):
                if ((y[i]- np.dot(B[:,j],X[i]))**2-wden_temp) < -20:
                    wden[i][j] = 0
                else:
                    wden[i][j] = (alpha[j]*np.exp(-0.5*((y[i]- np.dot(B[:,j],X[i]))**2 - wden_temp)/(sigma_array[j]**2))/np.sqrt(2*np.pi)/sigma).ravel()
            f[i] = np.sum(wden[i])*np.exp(-wden_temp)
        #record negative log likelihood
        L_temp = np.sum(np.log(1/f))
        L_rec.append(L_temp)
        
        for i in range(n):
            temp = np.sum(wden[i])
            for j in range(k):
                w[i][j] = wden[i][j]/temp
        
        
        #M step: update B, alpha
        for j in range(k):
            alpha[j] = np.sum(w[:,j])/n
            w_diag = np.diag(w[:,j])
            
            #update B
            temp1 = np.linalg.pinv(np.matmul(np.matmul(X.T, w_diag),X))
            temp2 = np.matmul(np.matmul(X.T,w_diag),y)
            B[:,j] = np.dot(temp1, temp2).ravel()
            
            #update sigma
            #sigma_temp = 0
            #for i in range(n):
                #sigma_temp = sigma_temp + w[i][j]*(y[i] - np.dot(B[:,j],X[i]))**2
            #sigma_array[j] = np.sqrt(sigma_temp/np.sum(w[:,j]))
    return f, B, alpha, sigma_array, L_rec, temp


def EMA(X,y,k,iter,BL,BR,sigmaL,sigmaR):
    '''
    Use EM algorithm to fit (fixed component number) mixture of linear regression  
    without knowing sigma value
    ---------------------------------------------
    Input
    X: n * p, covariate matrix
    y: n * 1, response variable
    k: number of components
    iter: number of iterations
    [BL,BR]^p: randomization range of parameters, used only in initialization
    [sigmaL,sigmaR]: randomization range of sigma, used only initialization
    ---------------------------------------------
    Output
    f: n * k, atomic likelihood vectors in active set
    B: p * k, coefficients corresponding to vectors in active set
    alpha: k*1, mixing proportions of vectors in active set
    sigma: std of noise
    L_rec: neg-log likelihood over iterations
    L_temp: final neg-log likelihood
    ---------------------------------------------
    
    '''  
    
    n = len(X)
    p = len(X[0])
    wden = np.zeros((n,k)) # unweighted posterior probability 
    w = np.zeros((n,k)) # posterior probability
    sigma_array = np.zeros((k,))
    z = np.zeros((n,)) #latent variable, class of each data point
    f = np.zeros((n,))
    L_rec = []
    
    #intialization
    B = np.zeros((p,k))
    alpha = np.zeros((k,1))
    for j in range(k):
        B[:,j] = np.random.uniform(BL,BR,(p,))
        alpha[j][0] = 1/k  
        sigma_array[j] = np.random.uniform(sigmaL,sigmaR)
    
    for r in range(iter):
        
        for i in range(n):
            for j in range(k):
                if (y[i]- np.dot(B[:,j],X[i]))**2 >100:
                    wden[i][j] = 0
                else:
                    wden[i][j] = (alpha[j]*np.exp(-0.5*((y[i]- np.dot(B[:,j],X[i]))**2\
                        )/(sigma_array[j]**2))/np.sqrt(2*np.pi)/sigma_array[j]).ravel()
            f[i] = np.sum(wden[i])
            
        #record negative log likelihood
        L_temp = np.sum(np.log(1/f))
        L_rec.append(L_temp)
        if r> 0:
            print("recent log likelihood difference (should <=0)" L_rec[-1] - L_rec[-2])
        
        # normalize
        for i in range(n):
            temp = np.sum(wden[i])
            for j in range(k):
                w[i][j] = wden[i][j]/temp
        
        
        #M step: update B, alpha, and sigma
        for j in range(k):
            alpha[j] = np.sum(w[:,j])/n
            w_diag = np.diag(w[:,j])
            
            #update B
            temp1 = np.linalg.pinv(np.matmul(np.matmul(X.T, w_diag),X))
            temp2 = np.matmul(np.matmul(X.T,w_diag),y)
            B[:,j] = np.dot(temp1, temp2).ravel()
            
            # update sigma
            sigma_temp = 0
            for i in range(n):
                sigma_temp = sigma_temp + w[i][j]*(y[i] - np.dot(B[:,j],X[i]))**2
            sigma_array[j] = np.sqrt(sigma_temp/np.sum(w[:,j]))
    return f, B, alpha, sigma_array, L_rec, temp

    


#----------------------------------------------#
# Alternatives: A variant of standard EM algorithm
# Classification EM
# See reference 
# "A classification EM algorithm for clustering and two stochastic versions"
#----------------------------------------------#
def CEMA_sigma(X,y,k,iter,BL,BR,sigma):
    '''
    Use classification EM algorithm to fit (fixed component number) mixture of linear regression  
    Use known fixed sigma value
    ---------------------------------------------
    Input
    X: n * p, covariate matrix
    y: n * 1, response variable
    k: number of components
    iter: number of iterations
    [BL,BR]^p: randomization range of regression parameters
    sigma: std of noise
    ---------------------------------------------
    Output
    f: n * k, atomic likelihood vectors in active set
    B: p * k, coefficients corresponding to vectors in active set
    alpha: k*1, mixing proportions of vectors in active set
    sigma_array: k*1, sigma value of each component
    L_rec: neg-log likelihood over iterations
    L_temp: final neg-log likelihood
    ---------------------------------------------
    
    '''  
    n = len(X)
    p = len(X[0])
    wden = np.zeros((n,k)) # unweighted posterior probability 
    w = np.zeros((n,k)) # posterior probability
    sigma_array = np.zeros((k,))
    z = np.zeros((n,)) #latent variable, class of each data point
    f = np.zeros((n,))
    L_rec = []
    
    #intialization
    B = np.zeros((p,k))
    alpha = np.zeros((k,1))
    for j in range(k):
        B[:,j] = np.random.uniform(BL,BR,(p,))
        alpha[j][0] = 1/k  
        sigma_array[j] = sigma # use know sigma
    
    for r in range(iter):
        
        # Expectation step
        for i in range(n):
            for j in range(k):
                wden[i][j] = (alpha[j][0]*np.exp(-0.5*(y[i]- np.dot(B[:,j],X[i]))**2/(sigma_array[j]**2))/np.sqrt(2*np.pi)/sigma).ravel()
        for i in range(n):
            temp = np.sum(wden[i])
            for j in range(k):
                w[i][j] = wden[i][j]/temp
            z[i] = np.argmax(np.reshape(w[i],(k,)))
            
        # record neg-log likelihood
        for i in range(n):
            f[i] = np.sum(wden[i])  
        L_temp = np.sum(np.log(1/f))
        # print(L_temp)
        L_rec.append(L_temp)
        
        #M step: update B, alpha, (and sigma)
        for j in range(k):
            alpha[j] = sum(z == j)/n
            X_temp = np.zeros((sum(z == j),p))
            y_temp = np.zeros((sum(z == j),1))
            W_temp = np.zeros((sum(z == j), sum(z == j)))
            count = 0
            for i in range(n):
                if z[i] == j:
                    X_temp[count] = X[i].ravel()
                    y_temp[count] = y[i]
                    W_temp[count][count] = w[i][j]
                    count = count + 1
            temp1 = np.copy(np.linalg.pinv(np.matmul(np.matmul(X_temp.T, W_temp),X_temp))) #used generalized inverse
            temp2 = np.copy(np.matmul(np.matmul(X_temp.T, W_temp),y_temp))
            B[:,j] = (np.matmul(temp1, temp2)).ravel()
    return f, B, alpha, sigma_array, L_rec, temp


def CEMA(X,y,k,iter,BL,BR,sigmaL,sigmaR):
    '''
    Use classification EM algorithm to fit (fixed component number) mixture of linear regression  
    without knowing sigma value
    ---------------------------------------------
    Input
    X: n * p, covariate matrix
    y: n * 1, response variable
    k: number of components
    iter: number of iterations
    [BL,BR]^p: randomization range of parameters, used only in initialization
    [sigmaL,sigmaR]: randomization range of sigma, used only initialization
    ---------------------------------------------
    Output
    f: n * k, atomic likelihood vectors in active set
    B: p * k, coefficients corresponding to vectors in active set
    alpha: k*1, mixing proportions of vectors in active set
    sigma: std of noise
    L_rec: neg-log likelihood over iterations
    L_temp: final neg-log likelihood
    ---------------------------------------------
    
    '''  
    
    n = len(X)
    p = len(X[0])
    wden = np.zeros((n,k)) # unweighted posterior probability 
    w = np.zeros((n,k)) # posterior probability
    sigma_array = np.zeros((k,))
    z = np.zeros((n,)) #latent variable, class of each data point
    f = np.zeros((n,))
    L_rec = []
    
    #intialization
    B = np.zeros((p,k))
    alpha = np.zeros((k,1))
    for j in range(k):
        B[:,j] = np.random.uniform(BL,BR,(p,))
        alpha[j][0] = 1/k  
        sigma_array[j] = np.random.uniform(sigmaL,sigmaR)
    
    for r in range(iter):
        
        # Expectation step
        for i in range(n):
            for j in range(k):
                wden[i][j] = (alpha[j][0]*np.exp(-0.5*(y[i]- np.dot(B[:,j],X[i]))**2/\
                    (sigma_array[j]**2))/np.sqrt(2*np.pi)/sigma_array[j]).ravel()
        for i in range(n):
            temp = np.sum(wden[i])
            for j in range(k):
                w[i][j] = wden[i][j]/temp
            z[i] = np.argmax(np.reshape(w[i],(k,)))
            
        # record neg-log likelihood
        for i in range(n):
            f[i] = np.sum(wden[i])  
        L_temp = np.sum(np.log(1/f))
        # print(L_temp)
        L_rec.append(L_temp)
        
        #M step: update B, alpha, (and sigma)
        for j in range(k):
            alpha[j] = sum(z == j)/n # this is classification EM
            X_temp = np.zeros((sum(z == j),p))
            y_temp = np.zeros((sum(z == j),1))
            W_temp = np.zeros((sum(z == j), sum(z == j)))
            count = 0
            for i in range(n):
                if z[i] == j:
                    X_temp[count] = X[i].ravel()
                    y_temp[count] = y[i]
                    W_temp[count][count] = w[i][j]
                    count = count + 1
            temp1 = np.copy(np.linalg.pinv(np.matmul(np.matmul(X_temp.T, W_temp),X_temp))) #used generalized inverse
            temp2 = np.copy(np.matmul(np.matmul(X_temp.T, W_temp),y_temp))
            B[:,j] = (np.matmul(temp1, temp2)).ravel()
            
            if count == 0:
                sys.exit("Opps, no data is assgined to component {}".format(beta))
            else: 
                sigma_temp = 0
                w_sum = 0
                for i in range(n):
                    if z[i] == j:
                        sigma_temp = sigma_temp + w[i][j]*(y[i] - np.dot(B[:,j],X[i]))**2
                        w_sum = w_sum + w[i][j]
                sigma_array[j] = np.sqrt(sigma_temp/w_sum)
    return f, B, alpha, sigma_array, L_rec, temp

    