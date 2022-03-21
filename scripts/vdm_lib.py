#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""


from package_import import *
from regression_func_lib import *

#===========================================================================================================

def beta_round(beta, BL,BR, epsilon):
    '''
    Approximate beta with epsilon net of [BL,BR]^p
    
    Parameters
    ----------
    beta : (p,1) numpy array
        vector to be approximated
    BL : float
        lower bound of area
    BR : float
        upper bound of area
    epsilon : float
        discretization accuracy

    Returns
    -------
    beta_rounded

    '''
    
    beta = beta.ravel()
    p = len(beta)
    beta_rounded = np.zeros(p)
    
    for i in range(p):
        idx = round((beta[i] - BL)/epsilon)
        idx = min(idx, int((BR - BL)/epsilon))
        idx = max(idx, 0)
        beta_rounded[i] = BL + idx * epsilon
        
    return np.reshape(beta_rounded, (p,1))


#===========================================================================================================
def idx_to_beta(idx, BL,BR, epsilon):
    '''
    Parameters
    ----------
    idx : numpy array
        coordinate over the epsilon net
    BL : float
         lower bound of area
    BR : float
         upper bound of area
    epsilon : float
         discretization accuracy

    Returns
    -------
    beta

    '''
    p = len(idx)
    beta = np.zeros(p)
    for i in range(p):
        beta[i] = BL + epsilon * idx[i]
    
    return np.reshape(beta, (p,1))

#===========================================================================================================

# def beta_find(X,y,sigma,f,BL,BR,epsilon,func = lin_func, option = 'max'):
#     '''

#     '''
#     n = len(X)
#     p = len(X[0])
    
#     M = int((BR-BL)/epsilon)
    
    
#     value_map = np.zeros(np.repeat(M+1,p)) 
#     # g_map =  np.zeros(np.concatenate(np.repeat(M+1,p), [n])) 
    
#     for idx_iter in itertools.product([np.arange(0,M+1,1)] * p):
        
#         idx = idx_iter[0]
        
#         beta = idx_to_beta(idx, BL, BR, epsilon)

#         g = np.zeros((n,1))
#         for i in range(n):
#             g[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta))**2 /(sigma**2))
        
#         # record value
#         # g_map[idx,:] = g.ravel()
#         value_map[idx] =  np.dot(g.T,1/f)
#     if option == 'max':
#         idx_ext = np.argmax(value_map)
#         beta_ext = idx_to_beta(idx_ext, BL, BR, epsilon)
        
#         g_ext = np.zeros((n,1))
#         for i in range(n):
#             g_ext[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta_ext))**2 /(sigma**2))
#     if option == 'min':
#         idx_ext = np.argmin(value_map)
#         beta_ext = idx_to_beta(idx_ext, BL, BR, epsilon)
        
#         g_ext = np.zeros((n,1))
#         for i in range(n):
#             g_ext[i] = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(-0.5*(y[i] - func(X[i],beta_ext))**2 /(sigma**2))
            
#     return g_ext,beta_ext





#===========================================================================================================
def VDM(X,y,iter,sigma,BL,BR,epsilon, func = lin_func):
    '''
    Use VDM (Vertex Direction Method) algorithm to solve NPMLE problem of MLR  
    sigma is estimated before
    
    Input
    X: covariate matrix
    y: response variable
    iter: number of iterations
    sigma: std of noise (estimated)
    BL,BR
    epsilon: discretization accuracy 
    
    Output
    f: n * J, atomic likelihood vectors in active set
    B: p * J, coefficients corresponding to vectors in active set
    alpha: J * 1, mixing proportions of vectors in active set
    L_rec: neg-log likelihood over iterations / C_L
    temp: final neg-log likelihood
    
    '''  
    n = len(X)
    p = len(X[0]) # dimension of features
    
    L_rec = []
    curvature_rec = []
    
    # Discretize over [BL,BR]^p with accuracy epsilon
    
    #------------initialize beta0 and f-------------#
    beta0 = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1)) #beta0 is OLS solution
    # caution ols might not be the best choice for initialization
    # in case of nonlinear regrression func
    
    
    # approximate beta0 with 
    beta0 = beta_round(beta0, BL, BR, epsilon)
    
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
        
        option = 'max'
        
        # first create and store g_map
        
        #solve LMO
        g, beta_sol = beta_find(X,y,sigma,f,BL,BR,epsilon,func, option )
        
        
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
