#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""
'''
Test Algorithm 1 in simulated settings and make visualization plots

'''
from package_import import *
from alg1_lib import *
from plotting_lib import *
import os

def generate_test_data(n,iter, b1, b2, b3,pi1,pi2,sigma_list,func=lin_func):
    '''
    generate testing data
    
    output
    ------
    X (n,p)
    y (n,1)
    C (n,1): labels of each datapoints
    
    '''
    p = len(b1)

    #parameters for generating synthetic data
    sigma1 = sigma_list[0]  # variance of 1st component
    sigma2 = sigma_list[1]      #variance of 2nd component
    sigma3 = sigma_list[2]
    
    #sigma_est is what we use for Frank-Wofle method
    
    # synthesize two component data
    b1 = np.reshape(b1,(p,1))
    b2 = np.reshape(b2,(p,1))
    b3 = np.reshape(b3,(p,1))

    X = np.zeros((n,2))
    y = np.zeros((n,1))
    
    # C denots the true class of each data point
    C = np.zeros((n,1)) 
    
    # np.random.seed(26)
    for i in range(n):
        X[i] = np.reshape([1,np.random.uniform(-1,3)],(1,2))
        z = np.random.uniform(0,1)
        if z < pi1:
            y[i] = func(X[i],b1) + np.random.normal(0,sigma1)
            C[i] = 1
        elif z < pi1 + pi2 :
            y[i] = func(X[i],b2) + np.random.normal(0,sigma2)
            C[i] = 2
        else:
            y[i] = func(X[i],b3) + np.random.normal(0,sigma3)
            C[i] = 3
    return X,y,C

def test(X,y,C, n,iter, b1, b2, b3,pi1,pi2,sigma_est,BL,BR,fname,func=lin_func):
    
    '''
    
    
    Input
    X
    y
    C
    sigma: ground truth sigma
    sigma_est: sigma value selected 
    
    '''

    #set parameters
    p =2    #number of components (currently we only consider 2 component)
    #sigma = 0.8 # standard deviation
    threprob = 0.01
    
#    fname = str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) +'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
#    fname = fname.replace('.','dot')

    #run Frank-Wofle
    f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma_est,BL,BR,func)
    
    
    beta_ave = np.matmul(B,alpha)
    print("averge beta is ", beta_ave)
    #print("training error is ",train_error(X,y,B,alpha) )

    beta_ols = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1))
    print("beta_ols",beta_ols)
    #index = np.argwhere(alpha.ravel() == np.amax(alpha.ravel()))
    
    #plot
    print("final neg log likelihood is ", L_final)
    print("number of components is", len(alpha))
    print("only components with probability at least ", threprob, " are shown below:")
    
    
    if not os.path.exists('./../data/{}'.format(fname)):
        os.mkdir('./../data/{}'.format(fname))
        
    pd.DataFrame(B).to_csv('./../data/{}/B_NPMLE.csv'.format(fname), index = False, header = False)
    pd.DataFrame(alpha).to_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), index = False, header = False)
    pd.DataFrame(L_rec).to_csv('./../data/{}/L_rec_NPMLE.csv'.format(fname), index = False, header = False)
    
    B = pd.read_csv('./../data/{}/B_NPMLE.csv'.format(fname), header = None).values
    alpha = pd.read_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), header = None).values
    L_rec = pd.read_csv('./../data/{}/L_rec_NPMLE.csv'.format(fname), header = None).values
    #------------make plots-----------------#
    plot_func_name = func.__name__[:-4] + 'plot'
    eval(plot_func_name + '(X,y,C,b1,b2,b3,pi1,pi2,sigma_est,B,alpha,L_rec,fname,threprob,func)')
    #---------------------------------------#






















#========================================================================
#def generate_and_test(n,iter, b1, b2, b3,pi1,pi2,sigma,sigma_est,BL,BR):
#    
#    '''
#    test function with synthetic data
#    
#    Input
#    n : number of samples
#    iter : number of iterations in Frank-Wofle method
#    
#    '''
#
#    
#    #set parameters
#    p =2    #number of components (currently we only consider 2 component)
#    #sigma = 0.8 # standard deviation
#    threprob = 0.01
#    #pi1 = 0.5 # first componrnt has probability pi
#    #pi2 = 0.25
#    
#    fname = str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) +'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
#    
#    #parameters for generating synthetic data
#    sigma1 = sigma  # variance of 1st component
#    sigma2 = sigma      #variance of 2nd component
#    sigma3 = sigma
#    
#    #sigma_est is what we use for Frank-Wofle method
#    
#    # synthesize two component data
#    b1 = np.reshape(b1,(2,1))
#    b2 = np.reshape(b2,(2,1))
#    b3 = np.reshape(b3,(2,1))
#
#    X = np.zeros((n,2))
#    y = np.zeros((n,1))
#    
#    # C denots the true class of each data point
#    C = np.zeros((n,1)) 
#    
#    # np.random.seed(26)
#    for i in range(n):
#        X[i] = np.reshape([1,np.random.uniform(-1,3)],(1,2))
#        z = np.random.uniform(0,1)
#        if z < pi1:
#            y[i] = np.dot(X[i],b1) + np.random.normal(0,sigma1)
#            C[i] = 1
#        elif z < pi1 + pi2 :
#            y[i] = np.dot(X[i],b2) + np.random.normal(0,sigma2)
#            C[i] = 2
#        else:
#            y[i] = np.dot(X[i],b3) + np.random.normal(0,sigma3)
#            C[i] = 3
#            
#            
#    #test what sigma we use
#    #sigma_est = np.sqrt(pi* sigma1**2 + (1-pi)*sigma2**2)
#    
#    #run Frank-Wofle
#    f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma_est,BL,BR)
#    
#    
#    beta_ave = np.matmul(B,alpha)
#    print("averge beta is ", beta_ave)
#    #print("training error is ",train_error(X,y,B,alpha) )
#
#    beta_ols = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1))
#    print("beta_ols",beta_ols)
#    #index = np.argwhere(alpha.ravel() == np.amax(alpha.ravel()))
#    
#    #plot
#    print("final neg log likelihood is ", L_final)
#    print("number of components is", len(alpha))
#    print("only components with probability at least ", threprob, " are shown below:")
#    
#    #**********************************************************
#    fig_raw = plt.figure(figsize = (8,8))
#    plt.scatter(X[:,1],y,color = 'black',marker = 'o',label = 'Noisy data', facecolors = 'None');
#    ax = plt.gca()
#    ax.set_xlim([-2,4])
#    ax.set_ylim([-3,8])
#    ax.set_xlabel(r'$x$')
#    ax.set_ylabel(r'$y$')
#    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
#    plt.savefig('./../pics/%s_noisy.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#    # plt.show();
#    
#    #**************************************************
#    fig0 = plt.figure(figsize = (8,8))
#    line_styles = ['-','-','-','-']
#    # line_styles = ['-','--',':','-.']
#    for i in range(len(y)):
#        if C[i] == 1:
#            plt.scatter(X[i][1],y[i],color = 'red',marker = 'o',label = 'Class 1', facecolors = 'None');
#        elif C[i] == 2:
#            plt.scatter(X[i][1],y[i],color = 'blue',marker = 'o',label = 'Class 2', facecolors = 'None');
#        else:
#            plt.scatter(X[i][1],y[i],color = 'green',marker = 'o',label = 'Class 3', facecolors = 'None');
#            
#    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
#    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
#    i = 0
#    plt.plot(t,b1[0]+b1[1]*t, color = 'red',linewidth = pi1*8,linestyle = line_styles[0])
#    plt.plot(t,b2[0]+b2[1]*t, color = 'blue',linewidth = pi2*8,linestyle = line_styles[1] )
#    if pi1 + pi2 < 1:
#        plt.plot(t,b3[0]+b3[1]*t, color = 'green',linewidth = (1-pi2-pi2)*8 , linestyle = line_styles[2])
#           
#            
#    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
#    if pi1 + pi2 <1:
#        custom_lines = [(Line2D([], [], color='red', marker='o',markerfacecolor = 'None', linestyle=line_styles[0],linewidth = 8*pi1),Line2D([], [], color='red')),
#                        (Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle=line_styles[1],linewidth = 8*pi2),Line2D([], [], color='blue')),
#                         (Line2D([], [], color='green', marker='o',markerfacecolor = 'None', linestyle=line_styles[2],linewidth = 8*(1-pi1-pi2)),Line2D([], [], color='green'))
#                        #Line2D([0], [0], color= 'red'# ),
#                        #Line2D([0], [0], color='black')
#                        #,Line2D([0], [0], color='green')#
#                        ]
#        lgd = plt.legend(custom_lines, [r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
#                                  r'$y = %.2f %+.2f x$ with probability $%.2f$' %(b2[0], b2[1], pi2),
#                                  r'$y = %.2f %+.2f x$ with probability $%.2f$' %(b3[0], b3[1], 1-pi1-pi2),
#                                   #'NPMLE component'#, 'OLS'#
#                                 ],loc = 2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
#    else:
#        custom_lines = [(Line2D([], [], color='red', marker='o',markerfacecolor = 'None', linestyle=line_styles[0],linewidth = 8*pi1),Line2D([], [], color='red')),
#                        (Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle=line_styles[1],linewidth = 8*pi2),Line2D([], [], color='blue')),
#                    
#                        #Line2D([0], [0], color= 'red'# ),
#                        #Line2D([0], [0], color='black')
#                        #,Line2D([0], [0], color='green')#
#                        ]
#        lgd = plt.legend(custom_lines, [r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
#                                  r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b2[0], b2[1], pi2),
#                                
#                                   #'NPMLE component'#, 'OLS'#
#                                 ],loc=2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
#    ax = plt.gca()
#    ax.set_xlim([-2,4])
#    ax.set_ylim([-3,8])
#    ax.set_xlabel(r'$x$')
#    ax.set_ylabel(r'$y$')
#    plt.savefig('./../pics/%s_true.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#    # plt.show();
#    
#    #*******************************************
#    
#    fig1 = plt.figure(figsize = (8,8))
#
#    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
#    
#    
#    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
#    
#    N = len(alpha)
#    
#    RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
#    (43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
#    (194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
#    ( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]
#    
#    component_plot = []
#    component_color = []
#    
#    temp = 0
#    index_sort = np.argsort(-np.reshape(alpha,(len(alpha),)))
#    count = 0 
#    for i in index_sort:
#        b = B[:,i]
#        if alpha[i] >threprob:
#            component_plot.append(i)
#            component_color.append(temp)
#            plt.plot(t,b[0]+b[1]*t, color = tuple( np.array(RGB_tuples[temp])/255),\
#                     linestyle = line_styles[int(count%4)],linewidth = alpha[i][0]*8 ,\
#                     label = r'$y = %.2f %+.2f x$ with probability $%.2f$'%(b[0], b[1], alpha[i]))
#            temp = temp + 1
#            print("coefficients", b, "with probability", alpha[i])
#            count = count + 1
#    
#    # ONLY clustering based on plotted components, i.e. components with high probability (>threprob)
#    C_cluster = np.zeros((n,1))
#    for i in range(len(y)):
#        prob = np.zeros((N,1))
#        for j in component_plot:
#            prob[j] = alpha[j] * np.exp(-0.5*(y[i] - np.dot(X[i],B[:,j]))**2 /(sigma**2))
#        C_cluster[i] = np.argmax(prob)
#        plt.scatter(X[i][1],y[i],color = tuple(np.array(RGB_tuples[component_color[component_plot.index(C_cluster[i])]])/255) ,marker = 'o', facecolors = 'None'); 
#            
#    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
#    
#    #custom_lines = [Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle='None'),
#                    #Line2D([0], [0], color= 'red'# ),
#                    #Line2D([0], [0], color='black')
#                    #,Line2D([0], [0], color='green')#
#                    #,]
#    #plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
#                            #  , 'NPMLE component'#, 'OLS'#
#                            # ],loc=0);
#    ax = plt.gca()
#    ax.set_xlim([-2,4])
#    ax.set_ylim([-3,8])
#    ax.set_xlabel(r'$x$')
#    ax.set_ylabel(r'$y$')
#    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
#    plt.savefig('./../pics/%s_fitted.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#    # plt.show();
    
#-----------------------------------#
#
#  plot density function            #
#
#-----------------------------------#
  
#-----------------------------------------------------------------   
#    fig2 = plt.figure(figsize = (20,3))
#    x_list = [-3,-1,1,3,5] #List of x values
#    i = 0
#    for i in range(len(x_list)):
#        x = x_list[i]
#        y = np.linspace(- 15, 15, 100)
#           
#        #calculate difference of square root of density functions
#        #dist_fit = lambda y: (np.sqrt(0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x), 0, sigma)+0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x),0, sigma)) \
#        #- np.sqrt(sum(alpha[i]*scipy.stats.norm.pdf( y - (B[0,i]+B[1,i]*x), 0, sigma) for i in range(len(alpha)))))**2
#        
#        #print("Fix x = %.1f, squared Hellinger distance for NPMLE is %.5f" % (x, quad(dist_fit, -np.inf, np.inf)[0]))
#
#        
#        plt.subplot(1,len(x_list),i+1)
#        plt.plot(y,pi1*scipy.stats.norm.pdf(y - (b1[0]+b1[1]*x), 0, sigma)+pi2*scipy.stats.norm.pdf(y-(b2[0]+b2[1]*x),0, sigma)+\
#        (1-pi1-pi2)*scipy.stats.norm.pdf(y-(b3[0]+b3[1]*x),0, sigma),'red',label = 'True distribution',linestyle =line_styles[0])
#        plt.plot(y, sum(alpha[i]*scipy.stats.norm.pdf( y-(B[0,i]+B[1,i]*x), 0, sigma_est) \
#        for i in range(len(alpha))),'black',label = 'NPMLE distribution',linestyle = line_styles[1])
#        plt.title(r'$x = %.1f$'%x)
#        
#        plt.xlabel(r'$y$')
#        if i == 0:
#            plt.ylabel(r'$\rm{pdf}$')
#        
#        
#           
#    custom_lines = [
#                Line2D([0], [0], color= 'red', linestyle = '-'),
#                Line2D([0], [0], color='black',linestyle = '--'),
#                Line2D([0], [0], color='green')#
#        ]
#    ax = plt.gca()
#    lgd = ax.legend(custom_lines, ['True mixture', 'Fitted mixture'#, 'OLS'#
#                             ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.savefig('./../pics/%s_density.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#    plt.show();
#---------------------------------------------------------------------------------
#    
#    #MLE 
#    fig3 = plt.figure(figsize = (6,5))
#    plt.plot(L_rec);plt.title("neg-log likelihood over iterations");
#    
#    #mixing weights
#    fig4 = plt.figure(figsize = (6,5))
#    plt.plot(-np.sort(-alpha.ravel()),marker = 'o', linestyle = '--')
#    plt.title("mixing weights in descending order");
#    ax = plt.gca()
#    ax.set_xlabel(r"index of mixing components $\beta$'s")
#    ax.set_ylabel(r'mixing weights')
#    plt.savefig('./../pics/%s_alpha.png'%fname, dpi = 300, bbox_inches='tight')
#
