#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

from package_import import *
from regression_func_lib import *

def lmo(beta,X,y,sigma,f,func = lin_func):
    #return the objective funciton of the linear oprimization problem
    n = len(X)
    p = len(X[0])
    obj = 0
    for i in range(n):
        obj = obj - np.exp(-0.5*(y[i] - func(X[i],beta))**2 /(sigma**2))/f[i]
    return obj  


#===============================================================================================
def sollmo(X,y,sigma,f,BL,BR,func = lin_func):
    '''solve linear minimization oracle
    this is an nonconvex problem with respect to beta, the result is only approximate (heuristic)
    return a new supporting vector g and corresponding beta
    
    '''
    n = len(X)
    p = len(X[0])
    
    #nonconvex algorithms are sensitive to initialization!!!!
    #initialize beta0 with OLS solution or 0 or random
    #beta0 = np.reshape(np.dot( np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1)) 
    #beta0 = np.zeros((p,1)) 
    #beta0 = np.reshape(np.random.uniform(BL,BR,p),(p,1))
    
    #minimize exponential sum approximately
    #nonconvex problem
    
    #OptResult = minimize(lmo, beta0, args = (X,y,sigma,f),jac = jacobian, method = 'BFGS')
    #OptResult = minimize(lmo, beta0, args = (X,y,sigma,f),method = 'Powell')
   
    #beta_sol = OptResult.x
    
    num_rdn = 5
    opt_fun = np.zeros(num_rdn)
    opt_x = np.zeros((num_rdn,p))
    for rdn in range(num_rdn):
        beta0 = np.reshape(np.random.uniform(BL,BR,p),(p,1))
        OptResult = minimize(lmo, beta0, args = (X,y,sigma,f,func),
                             method = 'Powell'
                            #,options={'disp':False, 'xtol':0.01, 'ftol':0.001}  #use options to improve running speed
                            )
        if OptResult.success == False:
            opt_fun[rdn] = np.inf
            opt_x[rdn] = beta0.ravel()
        else:
            opt_fun[rdn] = OptResult.fun
            opt_x[rdn] = OptResult.x
    print(opt_x)
    min_rdn = np.argmin(opt_fun)
    #print(min_rdn)
    beta_sol = np.reshape(opt_x[min_rdn],(p,1))
    
    g = np.zeros((n,1))
    for i in range(n):
        g[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta_sol))**2 /(sigma**2))
    return g,beta_sol
#===========================================================================================================


def FW_FC(f,alpha,P,n):
    #solve the fully corective step using classic FW
    #warm start with f from last iteration
    #P each column of P is a candidate component
    #return new f, and f as a convex combination of columns of P with coefficients alpha
    iter = 5000
    
    k = len(P[0])
    alpha = np.append(alpha,0)
    alpha = np.reshape(alpha,(k,1))
    for t in range(1,iter):
        g = 1/f
        g = np.reshape(g,(1,n))
        s = np.argmax(np.matmul(g,P))
        gamma = 2/(t+2)
        f = (1-gamma)*f +gamma*np.reshape(P[:,s],(n,1))
        temp = np.zeros((k,1))
        temp[s] = 1
        alpha = (1-gamma)*np.reshape(alpha,(k,1))+gamma*temp
    return f,alpha
#===========================================================================================================
    
def NPMLE_FW(X,y,iter,sigma,BL,BR, func = lin_func):
    '''
    Use FW algorithm to solve NPMLE problem of MLR  
    sigma is estimated before
    
    Input
    X: covariate matrix
    y: response variable
    iter: number of iterations
    sigma: std of noise (estimated)
    BL,BR
    
    Output
    f: n * J, atomic likelihood vectors in active set
    B: p * J, coefficients corresponding to vectors in active set
    alpha: J * 1, mixing proportions of vectors in active set
    L_rec: neg-log likelihood over iterations / C_L
    temp: final neg-log likelihood
    
    '''  
    n = len(X)
    
    p = func.p
    
    L_rec = []
    curvature_rec = []
    #------------initialize beta0 and f-------------#
    beta0 = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1)) #beta0 is OLS solution
    # caution ols might not be the best choice for initialization
    # in case of nonlinear regrression func
    
    #beta0 = np.zeros((p,1))
    f = np.zeros((n,1))
    for i in range(n):
        f[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta0))**2 /(sigma**2))
    # print("min f:", np.min(f))
    #-----------------------------------------------#
    
    # initialize P,B
    # P active set: (n,k)
    # B beta's corresponding to columns of P :(p,k)
    P = np.zeros((n,1))
    P[:,0] = f.ravel()
    B = np.zeros((p,1))
    B[:,0] = beta0.ravel()
    
    # intialize coefficients of convex combinations
    alpha = np.array([1]) 
    
    dual_gap_rec = np.zeros(iter)
    
    for t in range(1,iter):
        #solve LMO
        g, beta_sol = sollmo(X,y,sigma,f,BL,BR,func)
        
        #check stopping criterion
        dual_gap_rec[t] = np.dot(g.T,1/f) -np.dot(f.T,1/f)
        print("dual_gap", t,":",dual_gap_rec[t])
        
        if t%10 == 0:
            print("beta_sol",beta_sol)
        
        g = np.reshape(g,(n,1))
        beta_sol = np.reshape(beta_sol,(p,1))
        P = np.append(P,g,axis = 1)
        B = np.append(B,beta_sol,axis = 1)
        
        f_old = np.copy(f)
        
        #fully corrective step wrt current active set P
        f, alpha = FW_FC(f,alpha,P,n)
        
        # linear search (ls) version of f
        f_ls = t/(t+2)*f_old + 2/(t+2)*g
        # calculate the "curvature constant"
        curvature_temp = np.sum(np.log(f_old)) -  np.sum(np.log(f_ls)) + 2/(t+2)* np.dot(g.T,1/f_old) - 2/(t+2)*np.dot(f_old.T,1/f_old)
        curvature_rec.append(float(curvature_temp) * (t+2)**2/2)
        print("C_L", t, ":",curvature_rec[-1])
        
        #prune P by deleting columns corresponding to very small alpha
        P_prune = np.zeros((n,1))
        B_prune = np.zeros((p,1))
        alpha_prune = np.zeros((1,))
        flag = 0
        for i in range(len(P[0])):
            if alpha[i] > 0.01:
                if flag == 0:
                    P_prune[:,0] = P[:,i].ravel()
                    B_prune[:,0] = B[:,i].ravel()
                    alpha_prune[0] = alpha[i]
                    flag = 1
                else:
                    P_prune = np.append(P_prune,np.reshape(P[:,i],(n,1)), axis = 1)
                    alpha_prune = np.append(alpha_prune, alpha[i])
                    B_prune = np.append(B_prune,np.reshape(B[:,i],(p,1)),axis = 1)
        
        P = P_prune
        B = B_prune
        alpha = np.reshape(alpha_prune/np.sum(alpha_prune), (len(P[0]),1))
        
        #record the change of neg-log likelihood function
        temp = np.sum(np.log(1/f))
        L_rec.append(temp)
        
        
        
        # early stopping
        gap_thresh = n*0.001
        if (t>50) and (dual_gap_rec[t] < gap_thresh) and (dual_gap_rec[t-1] < gap_thresh) and (dual_gap_rec[t-2] < gap_thresh):
            print("stop at iteration", t)
            # return f, B, alpha, L_rec, temp
            return f, B, alpha, curvature_rec, temp
    # return f, B, alpha, L_rec, temp 
    return f, B, alpha, curvature_rec, temp 



#===========================================================================================================
def jacobian(beta,X,y,sigma,f):
    #return gradient of the linear objective function
    n = len(X)
    p = len(X[0])
    jac = np.zeros((p,1))
    for i in range(n):
        temp = (y[i] - np.dot(X[i],beta))**2 /(sigma**2)
        jac = jac - np.exp(-0.5*temp)*temp/f[i]/(sigma**2) * np.reshape(X[i],(p,1))
        jac = np.reshape(jac,(p,1))
    jac = np.reshape(jac, (p,))
    return jac

def hessian(beta,X,y,sigma,f):
    #return Hessian of the linear objetive function
    n = len(X)
    p = len(X[0])
    hess = np.zeros((p,p))
    for i in range(n):
        temp = (y[i] - np.dot(X[i],beta))**2 /(sigma**2)
        x1 = np.reshape(X[i],(p,1))
        x2 = np.reshape(X[i],(1,p))
        temp2 = np.matmul(x1,x2)
        hess = hess - (np.exp(-0.5*temp)*temp**2/(sigma** 4) -np.exp(-0.5*temp)/(sigma**2))/f[i]*temp2
    return hess

#===========================================================================================================

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool

def sollmo_parallel(X,y,sigma,f,BL,BR,func = lin_func):
    '''solve linear minimization oracle
    this is an nonconvex problem with respect to beta, the result is approximal
    return a new supporting vector g and corresponding beta
    
    '''
    n = len(X)
    p = len(X[0])
    
    # uses a very large number of random starts for solving one subproblem
    num_pool = 10
    # actual number of random starts is 5 * num_pool 
    
    with Pool(num_pool) as sollmo_pool:
        pool_result = sollmo_pool.starmap(sollmo, \
                zip(repeat(X), repeat(y), np.repeat(sigma,num_pool), repeat(f), repeat(BL),repeat(BR),repeat(func)))
    
    pool_opt = np.zeros(len(pool_result))
    for i in range(len(pool_result)):
        [g, beta] = pool_result[i]
        pool_opt[i] = -np.dot(g.T,1/f)
    
    min_rdn = np.argmin(pool_opt)
    g, beta_sol = pool_result[min_rdn]

    g = np.reshape(g, (n,1))
    beta_sol = np.reshape(beta, (p,1))
    return g,beta_sol

#===========================================================================================================
    
def NPMLE_FW_parallel(X,y,iter,sigma,BL,BR, func = lin_func):
    '''
    Use FW algorithm to solve NPMLE problem of MLR  
    sigma is estimated before
    
    Input
    X: covariate matrix
    y: response variable
    iter: number of iterations
    sigma: std of noise (estimated)
    BL,BR
    
    Output
    f: n * J, atomic likelihood vectors in active set
    B: p * J, coefficients corresponding to vectors in active set
    alpha: J * 1, mixing proportions of vectors in active set
    L_rec: neg-log likelihood over iterations
    temp: final neg-log likelihood
    
    '''  
    

    n = len(X)
    p = len(X[0])
    
    L_rec = []
    curvature_rec = []
    #------------initialize beta0 and f-------------#
    beta0 = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1)) #beta0 is OLS solution
    # caution ols might not be the best choice for initialization
    # in case of nonlinear regrression func
    
    #beta0 = np.zeros((p,1))
    f = np.zeros((n,1))
    for i in range(n):
        f[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta0))**2 /(sigma**2))
    # print("min f:", np.min(f))
    #-----------------------------------------------#
    
    # initialize P,B
    # P active set: (n,k)
    # B beta's corresponding to columns of P :(p,k)
    P = np.zeros((n,1))
    P[:,0] = f.ravel()
    B = np.zeros((p,1))
    B[:,0] = beta0.ravel()
    
    # intialize coefficients of convex combinations
    alpha = np.array([1]) 
    
    dual_gap_rec = np.zeros(iter)
    
    for t in range(1,iter):
        #solve LMO
        g, beta_sol = sollmo_parallel(X,y,sigma,f,BL,BR,func)
        
        #check stopping criterion
        dual_gap_rec[t] = np.dot(g.T,1/f) -np.dot(f.T,1/f)
        print("dual_gap", t,":",dual_gap_rec[t])
        
        if t%10 == 0:
            print("beta_sol",beta_sol)
        
        g = np.reshape(g,(n,1))
        beta_sol = np.reshape(beta_sol,(p,1))
        P = np.append(P,g,axis = 1)
        B = np.append(B,beta_sol,axis = 1)
        
        f_old = np.copy(f)
        #fully corrective step wrt current active set P
        f, alpha = FW_FC(f,alpha,P,n)
        
        # calculate the "curvature constant"
        curvature_temp = np.sum(np.log(1/f)) -  np.sum(np.log(1/f_old)) + np.dot(f.T,1/f_old) -np.dot(f_old.T,1/f_old)
        curvature_rec.append(float(curvature_temp) * (t+2)**2/2)
        print("C_L", t, ":",curvature_rec[-1])
        
        #prune P by deleting columns corresponding to very small alpha
        P_prune = np.zeros((n,1))
        B_prune = np.zeros((p,1))
        alpha_prune = np.zeros((1,))
        flag = 0
        for i in range(len(P[0])):
            if alpha[i] > 0.01:
                if flag == 0:
                    P_prune[:,0] = P[:,i].ravel()
                    B_prune[:,0] = B[:,i].ravel()
                    alpha_prune[0] = alpha[i]
                    flag = 1
                else:
                    P_prune = np.append(P_prune,np.reshape(P[:,i],(n,1)), axis = 1)
                    alpha_prune = np.append(alpha_prune, alpha[i])
                    B_prune = np.append(B_prune,np.reshape(B[:,i],(p,1)),axis = 1)
        
        P = P_prune
        B = B_prune
        alpha = np.reshape(alpha_prune/np.sum(alpha_prune), (len(P[0]),1))
        
        #record the change of neg-log likelihood function
        temp = np.sum(np.log(1/f))
        L_rec.append(temp)
        
        
        
        # early stopping
#        gap_thresh = n*0.001
#        if (t>50) and (dual_gap_rec[t] < gap_thresh) and (dual_gap_rec[t-1] < gap_thresh) and (dual_gap_rec[t-2] < gap_thresh):
#            print("stop at iteration", t)
#            # return f, B, alpha, L_rec, temp
#            return f, B, alpha, curvature_rec, temp
    # return f, B, alpha, L_rec, temp 
    return f, B, alpha, curvature_rec, temp 