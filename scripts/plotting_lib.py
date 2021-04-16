#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:31:27 2021

@author: hanshengjiang
"""

'''
Plotting library that generates nice scatter and line plots


IMPORTANT
---------
Some plot legends does not automatically changes atogether with r(x,beta)
therefore need to be manually recalibrated
whenever funcrions in regression_func_lib changes

'''

from package_import import *
from regression_func_lib import *
from matplotlib import cm

def lin_plot(X,y,C,b1,b2,b3,pi1,pi2,sigma,B,alpha,L_rec,fname,threprob,func = lin_func):
    
    n = len(X)
    
#**********************************************************
    fig_raw = plt.figure(figsize = (8,8))
    plt.scatter(X[:,1],y,color = 'black',marker = 'o',label = 'Noisy data', facecolors = 'None');
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,8])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    plt.savefig('./../pics/%s_noisy.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();

#**************************************************
    
    fig0 = plt.figure(figsize = (8,8))
    line_styles = ['-','-','-','-']
    # line_styles = ['-','--',':','-.']
    for i in range(len(y)):
        if C[i] == 1:
            plt.scatter(X[i][1],y[i],color = 'red',marker = 'o',label = 'Class 1', facecolors = 'None');
        elif C[i] == 2:
            plt.scatter(X[i][1],y[i],color = 'blue',marker = 'o',label = 'Class 2', facecolors = 'None');
        else:
            plt.scatter(X[i][1],y[i],color = 'green',marker = 'o',label = 'Class 3', facecolors = 'None');
            
    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    i = 0
    plt.plot(t,b1[0]+b1[1]*t, color = 'red',linewidth = pi1*8,linestyle = line_styles[0])
    plt.plot(t,b2[0]+b2[1]*t, color = 'blue',linewidth = pi2*8,linestyle = line_styles[1] )
    if pi1 + pi2 < 1:
        plt.plot(t,b3[0]+b3[1]*t, color = 'green',linewidth = (1-pi2-pi2)*8 , linestyle = line_styles[2])
           
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    if pi1 + pi2 <1:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle=line_styles[0],linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue',markerfacecolor = 'None', linestyle=line_styles[1],linewidth = 8*pi2),Line2D([], [], color='blue')),
                         (Line2D([], [], color='green',markerfacecolor = 'None', linestyle=line_styles[2],linewidth = 8*(1-pi1-pi2)),Line2D([], [], color='green'))
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f  %+.2f x$ with probability $%.2f$'%(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b2[0], b2[1], pi2),
                                  r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b3[0], b3[1], 1-pi1-pi2),
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc = 2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    else:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle=line_styles[0],linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue',markerfacecolor = 'None', linestyle=line_styles[1],linewidth = 8*pi2),Line2D([], [], color='blue')),
                    #marker='o'
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f  %+.2f x$ with probability $%.2f$' %(b2[0], b2[1], pi2),
                                
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc=2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,8])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    
    plt.savefig('./../pics/%s_true.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #*******************************************
    line_styles = ['-','-','-','-']
    # line_styles = ['-','--',':','-.']
    
    fig1 = plt.figure(figsize = (8,8))

    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    
    
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    
    N = len(alpha)
    
    RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
    (43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
    (194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
    ( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]
    
    component_plot = []
    component_color = []
    
    temp = 0
    index_sort = np.argsort(-np.reshape(alpha,(len(alpha),)))
    count = 0 
    for i in index_sort:
        b = B[:,i]
        # select component with mixing probability above certain thresholds
        if alpha[i] >threprob:
            component_plot.append(i)
            component_color.append(temp)
            plt.plot(t,b[0]+b[1]*t, color = tuple( np.array(RGB_tuples[temp])/255),\
                     linestyle = line_styles[int(count%4)],linewidth = alpha[i][0]*8 ,\
                     label = r'$y = %.2f %+.2f x$ with probability $%.2f$'%(b[0], b[1], alpha[i]))
            temp = temp + 1
            print("coefficients", b, "with probability", alpha[i])
            count = count + 1
    
    # ONLY clustering based on plotted components, i.e. components with high probability (>threprob)
    C_cluster = np.zeros((n,1))
    for i in range(len(y)):
        prob = np.zeros((N,1))
        for j in component_plot:
            prob[j] = alpha[j] * np.exp(-0.5*(y[i] - np.dot(X[i],B[:,j]))**2 /(sigma**2))
        C_cluster[i] = np.argmax(prob)
        plt.scatter(X[i][1],y[i],color = tuple(np.array(RGB_tuples[component_color[component_plot.index(C_cluster[i])]])/255) ,marker = 'o', facecolors = 'None'); 
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    
    #custom_lines = [Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle='None'),
                    #Line2D([0], [0], color= 'red'# ),
                    #Line2D([0], [0], color='black')
                    #,Line2D([0], [0], color='green')#
                    #,]
    #plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                            #  , 'NPMLE component'#, 'OLS'#
                            # ],loc=0);
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,8])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad=0.);
    plt.savefig('./../pics/%s_fitted.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show();
      
    #MLE 
    fig3 = plt.figure(figsize = (6,5))
    ax = plt.gca()
    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r'$\log C_L$')
    plt.plot(np.log(np.array(L_rec)));
    plt.savefig('./../pics/%s_C_L.png'%fname, dpi = 300, bbox_inches='tight')
    
    #mixing weights
    fig4 = plt.figure(figsize = (6,5))
    plt.plot(-np.sort(-alpha.ravel()),marker = 'o', linestyle = '--')
    plt.title("mixing weights in descending order");
    ax = plt.gca()
    ax.set_xlabel(r"Index of mixing components $\beta$'s")
    ax.set_ylabel(r'Mixing weights')
    plt.savefig('./../pics/%s_alpha.png'%fname, dpi = 300, bbox_inches='tight')

#-------------------------------------------------------------------#
#
#
#
#
#
#-------------------------------------------------------------------#
   


def poly_plot(X,y,C,b1,b2,b3,pi1,pi2,sigma,B,alpha,L_rec,fname,threprob,func):
    
    n = len(X)
    
    fig_raw = plt.figure(figsize = (8,8))
    plt.scatter(X[:,1],y,color = 'black',marker = 'o',label = 'Noisy data', facecolors = 'None');
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,12])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    plt.savefig('./../pics/%s_noisy.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();
    
    
    fig0 = plt.figure(figsize = (8,8))
    for i in range(len(y)):
        if C[i] == 1:
            plt.scatter(X[i][1],y[i],color = 'red',marker = 'o',label = 'Class 1', facecolors = 'None');
        elif C[i] == 2:
            plt.scatter(X[i][1],y[i],color = 'blue',marker = 'o',label = 'Class 2', facecolors = 'None');
        else:
            plt.scatter(X[i][1],y[i],color = 'green',marker = 'o',label = 'Class 3', facecolors = 'None');
            
    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    i = 0
    plt.plot(t,func([1,t],b1), color = 'red',linewidth = pi1*8 )
    plt.plot(t,func([1,t],b2), color = 'blue',linewidth = pi2*8 )
    if pi1 + pi2 < 1:
        plt.plot(t,func([1,t],b3), color = 'green',linewidth = (1-pi2-pi2)*8 )
           
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    if pi1 + pi2 <1:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                         (Line2D([], [], color='green', markerfacecolor = 'None', linestyle='None',linewidth = 8*(1-pi1-pi2)),Line2D([], [], color='green'))
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2fx + ( 1%+.2fx)^2$ with probability $%.2f$'%(b1[0],b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2fx + (1%+.2fx)^2$ with probability $%.2f$'%(b2[0], b2[1], pi2),
                                  r'$y = %.2fx + ( 1%+.2fx)^2$ with probability $%.2f$'%(b3[0], b1[1], 1- pi1 - pi2)
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc = 2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    else:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [],  color='blue',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                    
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2fx + ( 1%+.2fx)^2$ with probability $%.2f$'%(b1[0],b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2fx + ( 1%+.2fx)^2$ with probability $%.2f$'%(b2[0], b2[1], pi2)
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc=2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,12])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig('./../pics/%s_true.png'%fname, dpi = 300,\
               bbox_extra_artists=(lgd,), \
              bbox_inches='tight')
    

    
    
    fig1 = plt.figure(figsize = (8,8))

    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    
    
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    
    N = len(alpha)
    
    RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
    (43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
    (194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
    ( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]
    
    component_plot = []
    component_color = []
    
    temp = 0
    index_sort = np.argsort(-np.reshape(alpha,(len(alpha),)))
    for i in index_sort:
        b = B[:,i]
        if alpha[i] >threprob:
            component_plot.append(i)
            component_color.append(temp)
            plt.plot(t,b[0]*t+(1+b[1]*t)**2, color = tuple( np.array(RGB_tuples[temp])/255)\
                     ,linewidth = alpha[i][0]*8 ,\
                     label = r'$y = %.2fx + ( 1%+.2fx)^2$ with probability $%.2f$'%(b[0],b[1], alpha[i]))
            temp = temp + 1
            print("coefficients", b, "with probability", alpha[i])
    
    # we ONLY do clustering based on plotted components, i.e. components with high probability (>threprob)
    C_cluster = np.zeros((n,1))
    for i in range(len(y)):
        prob = np.zeros((N,1))
        for j in component_plot:
            prob[j] = alpha[j] * np.exp(-0.5*(y[i] - func(X[i],B[:,j]))**2 /(sigma**2))
        C_cluster[i] = np.argmax(prob)
        plt.scatter(X[i][1],y[i],color = tuple(np.array(RGB_tuples[component_color[component_plot.index(C_cluster[i])]])/255) ,marker = 'o', facecolors = 'None'); 
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    
    #custom_lines = [Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle='None'),
                    #Line2D([0], [0], color= 'red'# ),
                    #Line2D([0], [0], color='black')
                    #,Line2D([0], [0], color='green')#
                    #,]
    #plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                            #  , 'NPMLE component'#, 'OLS'#
                            # ],loc=0);
    plt.legend(bbox_to_anchor=(0., -0.11), loc=2, borderaxespad=0.)
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-3,12])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad=0.);
    #lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    plt.savefig('./../pics/%s_fitted.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')

    #MLE 
    fig3 = plt.figure(figsize = (6,5))
    ax = plt.gca()
    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r'$\log C_L$')
    plt.plot(np.log(np.array(L_rec)));
    plt.savefig('./../pics/%s_C_L.png'%fname, dpi = 300, bbox_inches='tight')
    
    #mixing weights
    fig4 = plt.figure(figsize = (6,5))
    plt.plot(-np.sort(-alpha.ravel()),marker = 'o', linestyle = '--')
    plt.title("mixing weights in descending order");
    ax = plt.gca()
    ax.set_xlabel(r"index of mixing components $\beta$'s")
    ax.set_ylabel(r'mixing weights')
    plt.savefig('./../pics/%s_alpha.png'%fname, dpi = 300, bbox_inches='tight')

#-------------------------------------------------------------------#
#
#
#
#
#
#-------------------------------------------------------------------#
    
    
    
def exp_plot(X,y,C,b1,b2,b3,pi1,pi2,sigma,B,alpha,L_rec,fname,threprob,func):
    
    n = len(X)
    
    fig_raw = plt.figure(figsize = (8,8))
    plt.scatter(X[:,1],y,color = 'black',marker = 'o',label = 'Noisy data', facecolors = 'None');
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    plt.savefig('./../pics/%s_noisy.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();
    
    
    fig0 = plt.figure(figsize = (8,8))
    for i in range(len(y)):
        if C[i] == 1:
            plt.scatter(X[i][1],y[i],color = 'red',marker = 'o',label = 'Class 1', facecolors = 'None');
        elif C[i] == 2:
            plt.scatter(X[i][1],y[i],color = 'blue',marker = 'o',label = 'Class 2', facecolors = 'None');
        else:
            plt.scatter(X[i][1],y[i],color = 'green',marker = 'o',label = 'Class 3', facecolors = 'None');
            
    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    i = 0
    plt.plot(t,func([1,t],b1), color = 'red',linewidth = pi1*8 )
    plt.plot(t,func([1,t],b2), color = 'blue',linewidth = pi2*8 )
    if pi1 + pi2 < 1:
        plt.plot(t,func([1,t],b3), color = 'green',linewidth = (1-pi2-pi2)*8 )
           
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    if pi1 + pi2 <1:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue', markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                         (Line2D([], [], color='green',markerfacecolor = 'None', linestyle='None',linewidth = 8*(1-pi1-pi2)),Line2D([], [], color='green'))
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b2[0], b2[1], pi2),
                                  r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b3[0], b3[1], 1-pi1-pi2),
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc = 2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    else:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                    
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b2[0], b2[1], pi2)
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc=2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig('./../pics/%s_true.png'%fname, dpi = 300, \
                 bbox_extra_artists=(lgd,), \
                bbox_inches='tight')
    #plt.show();

    
    
    fig1 = plt.figure(figsize = (8,8))

    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    
    
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    
    N = len(alpha)
    
    RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
    (43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
    (194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
    ( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]
    
    component_plot = []
    component_color = []
    
    temp = 0
    index_sort = np.argsort(-np.reshape(alpha,(len(alpha),)))
    for i in index_sort:
        b = B[:,i]
        if alpha[i] >threprob:
            component_plot.append(i)
            component_color.append(temp)
            plt.plot(t,func([1,t],b), color = tuple( np.array(RGB_tuples[temp])/255)\
                     ,linewidth = alpha[i][0]*8 ,\
                     label = r'$y = %.2f + {\rm exp}(%+.2f x)}$ with probility $%.2f$' %(b[0], b[1], alpha[i]))
            temp = temp + 1
            print("coefficients", b, "with probability", alpha[i])
    
    # we ONLY do clustering based on plotted components, i.e. components with high probability (>threprob)
    C_cluster = np.zeros((n,1))
    for i in range(len(y)):
        prob = np.zeros((N,1))
        for j in component_plot:
            prob[j] = alpha[j] * np.exp(-0.5*(y[i] - func(X[i],B[:,j]))**2 /(sigma**2))
        C_cluster[i] = np.argmax(prob)
        plt.scatter(X[i][1],y[i],color = tuple(np.array(RGB_tuples[component_color[component_plot.index(C_cluster[i])]])/255) ,marker = 'o', facecolors = 'None'); 
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    
    #custom_lines = [Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle='None'),
                    #Line2D([0], [0], color= 'red'# ),
                    #Line2D([0], [0], color='black')
                    #,Line2D([0], [0], color='green')#
                    #,]
    #plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                            #  , 'NPMLE component'#, 'OLS'#
                            # ],loc=0);
    plt.legend(bbox_to_anchor=(0., -0.11), loc=2, borderaxespad=0.)
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1.0),borderaxespad=0.);
    plt.savefig('./../pics/%s_fitted.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();

    
    #MLE 
    fig3 = plt.figure(figsize = (6,5))
    ax = plt.gca()
    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r'$\log C_L$')
    plt.plot(np.log(np.array(L_rec)));
    plt.savefig('./../pics/%s_C_L.png'%fname, dpi = 300, bbox_inches='tight')
    
    fig4 = plt.figure(figsize = (6,5))
    plt.plot(-np.sort(-alpha.ravel()),marker = 'o', linestyle = '--')
    plt.title("mixing weights in descending order");
    ax = plt.gca()
    ax.set_xlabel(r"index of mixing components $\beta$'s")
    ax.set_ylabel(r'mixing weights')
    plt.savefig('./../pics/%s_alpha.png'%fname, dpi = 300, bbox_inches='tight')
    
#-------------------------------------------------------------------#
#
#
#
#
#
#-------------------------------------------------------------------#
    
    
    
def sin_plot(X,y,C,b1,b2,b3,pi1,pi2,sigma,B,alpha,L_rec,fname,threprob,func):
    
    n = len(X)
    
    fig_raw = plt.figure(figsize = (8,8))
    plt.scatter(X[:,1],y,color = 'black',marker = 'o',label = 'Noisy data', facecolors = 'None');
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    plt.savefig('./../pics/%s_noisy.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();
    
    
    fig0 = plt.figure(figsize = (8,8))
    for i in range(len(y)):
        if C[i] == 1:
            plt.scatter(X[i][1],y[i],color = 'red',marker = 'o',label = 'Class 1', facecolors = 'None');
        elif C[i] == 2:
            plt.scatter(X[i][1],y[i],color = 'blue',marker = 'o',label = 'Class 2', facecolors = 'None');
        else:
            plt.scatter(X[i][1],y[i],color = 'green',marker = 'o',label = 'Class 3', facecolors = 'None');
            
    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    i = 0
    plt.plot(t,func([1,t],b1), color = 'red',linewidth = pi1*8 )
    plt.plot(t,func([1,t],b2), color = 'blue',linewidth = pi2*8 )
    if pi1 + pi2 < 1:
        plt.plot(t,func([1,t],b3), color = 'green',linewidth = (1-pi2-pi2)*8 )
           
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    if pi1 + pi2 <1:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue', markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                         (Line2D([], [], color='green',markerfacecolor = 'None', linestyle='None',linewidth = 8*(1-pi1-pi2)),Line2D([], [], color='green'))
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b2[0], b2[1], pi2),
                                  r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b3[0], b3[1], 1-pi1-pi2),
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc = 2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    else:
        custom_lines = [(Line2D([], [], color='red',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi1),Line2D([], [], color='red')),
                        (Line2D([], [], color='blue',markerfacecolor = 'None', linestyle='None',linewidth = 8*pi2),Line2D([], [], color='blue')),
                    
                        #Line2D([0], [0], color= 'red'# ),
                        #Line2D([0], [0], color='black')
                        #,Line2D([0], [0], color='green')#
                        ]
        lgd = plt.legend(custom_lines, [r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b1[0], b1[1], pi1), #,'True mixture'# 
                                  r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b2[0], b2[1], pi2)
                                   #'NPMLE component'#, 'OLS'#
                                 ],loc=2,bbox_to_anchor=(0., -0.11),borderaxespad=0.);
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig('./../pics/%s_true.png'%fname, dpi = 300, \
                 bbox_extra_artists=(lgd,), \
                bbox_inches='tight')
    #plt.show();

    
    
    fig1 = plt.figure(figsize = (8,8))

    t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
    
    
    #plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
    
    N = len(alpha)
    
    RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
    (43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
    (194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
    ( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]
    
    component_plot = []
    component_color = []
    
    temp = 0
    index_sort = np.argsort(-np.reshape(alpha,(len(alpha),)))
    for i in index_sort:
        b = B[:,i]
        if alpha[i] >threprob:
            component_plot.append(i)
            component_color.append(temp)
            plt.plot(t,func([1,t],b), color = tuple( np.array(RGB_tuples[temp])/255)\
                     ,linewidth = alpha[i][0]*8 ,\
                     label = r'$y = %.2f + {\rm sin}(%+.2f x)}$ with probility $%.2f$' %(b[0], b[1], alpha[i]))
            temp = temp + 1
            print("coefficients", b, "with probability", alpha[i])
    
    # we ONLY do clustering based on plotted components, i.e. components with high probability (>threprob)
    C_cluster = np.zeros((n,1))
    for i in range(len(y)):
        prob = np.zeros((N,1))
        for j in component_plot:
            prob[j] = alpha[j] * np.exp(-0.5*(y[i] - func(X[i],B[:,j]))**2 /(sigma**2))
        C_cluster[i] = np.argmax(prob)
        plt.scatter(X[i][1],y[i],color = tuple(np.array(RGB_tuples[component_color[component_plot.index(C_cluster[i])]])/255) ,marker = 'o', facecolors = 'None'); 
            
    #plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
    
    #custom_lines = [Line2D([], [], color='blue', marker='o',markerfacecolor = 'None', linestyle='None'),
                    #Line2D([0], [0], color= 'red'# ),
                    #Line2D([0], [0], color='black')
                    #,Line2D([0], [0], color='green')#
                    #,]
    #plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                            #  , 'NPMLE component'#, 'OLS'#
                            # ],loc=0);
    plt.legend(bbox_to_anchor=(0., -0.11), loc=2, borderaxespad=0.)
    ax = plt.gca()
    ax.set_xlim([-2,4])
    ax.set_ylim([-4,6])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1.0),borderaxespad=0.);
    plt.savefig('./../pics/%s_fitted.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show();

    
    #MLE 
    fig3 = plt.figure(figsize = (6,5))
    ax = plt.gca()
    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r'$\log C_L$')
    plt.plot(np.log(np.array(L_rec)));
    plt.savefig('./../pics/%s_C_L.png'%fname, dpi = 300, bbox_inches='tight')
    
    fig4 = plt.figure(figsize = (6,5))
    plt.plot(-np.sort(-alpha.ravel()),marker = 'o', linestyle = '--')
    plt.title("mixing weights in descending order");
    ax = plt.gca()
    ax.set_xlabel(r"index of mixing components $\beta$'s")
    ax.set_ylabel(r'mixing weights')
    plt.savefig('./../pics/%s_alpha.png'%fname, dpi = 300, bbox_inches='tight')
    
    
#-----------------------------------------------------------------   
import joypy  
def density_ridgeline_plot(x_list,sigma,B,alpha,fname, min_,max_,func = lin_func, approach = 'True'):
    
    
    '''
    Input
    -----
    sigma (k,) np array
    B (p,k)
    alpha (k,)
    fname string
    
    Output
    ------
    Ridgeline plots
    
    '''
    line_styles = ['-','-','-','-']
    # line_styles = ['-','--',':','-.']
    fig = plt.figure(figsize = (16,5))
    #List of x values
    i = 0
    
    # set up a uniform range of y for all x
    
    
    if func.__name__ == 'lin_func':
        y = np.linspace(min_ -1, max_ + 1, 100)
    else:
        y = np.linspace(min_ -3, max_ + 3, 100)
    #-------------------------------------------
    
    density_array = np.zeros((len(y),len(x_list)))

    for i in range(len(x_list)):
        x = x_list[i]
        fy = sum(alpha[j]*scipy.stats.norm.pdf(y-func([1,x],B[:,j]), 0, sigma[j]) \
        for j in range(len(alpha)))
        density_array[:,i] = np.array(fy).ravel()
            
    
    #----------Ridgeplot-------------------#
    df = pd.DataFrame(density_array)
    df.columns = x_list
    
    # choose a few index to be included in the labels
    sparse_index = np.append(np.arange(0,100,25),99) # gap 20 can be adjusted 
    y_labeled = np.around(y[sparse_index], decimals = 1)
    # x_label = [x if int(10*x)%20 ==0 else None for x in x_list ]
 
    fig1, ax1 = joypy.joyplot(df, kind="values", overlap = 0.5 \
                          ,x_range = list(range(100)), ylabels = False\
                          ,background='k', linecolor="w",grid=False,  linewidth=1.5,fill=False \
                          )
    ax1[-1].set_xticks(sparse_index)
    ax1[-1].set_xticklabels(y_labeled)
    
    plt.savefig("./../pics/ridgeline_{}_{}".format(fname, approach), \
                dpi = 300, bbox_inches='tight')
    #---------------------------------------------------------------------------------
      
    
    