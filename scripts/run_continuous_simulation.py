#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys
from itertools import repeat
import multiprocessing 
from multiprocessing import Pool

if __name__ == "__main__":
    # default
    if len(sys.argv) < 5:
        sigma = 0.5
        n = 500
        config = '2'
        run_cv = '0.51'
        # or -----------
        # config = '1'
        # run_cv = '1.04'
        #---------------
        cv_granularity = 0.01
        
    # otherwise take argyments from command line
    else:
        #sys_argv[0] is the name of the .py file
        sigma = float(sys.argv[1]) 
        n = int(sys.argv[2]) # number of data points
        config = sys.argv[3]
        run_cv = sys.argv[4]
        cv_granularity = float(sys.argv[5]) 

# preset configurations      
if config == '1':
    #----------- configuration 1-----#
    meanb1 = [1,0.5]
    covb1 = np.array([[0.5,0.2],[0.2,0.3]])
    meanb2 = [2,3]
    covb2 = np.array([[0.5,0.2],[0.2,0.3]])
    pi1 = 1.0
    df_ = 3
    
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-1.5,0,1.5] #x_list for later density plots
    
    continuous_type = 'continuous_multivariate_t'
    fname = continuous_type + str(meanb1[0]) + '_'+ str(meanb1[1])+'_'+ str(meanb2[0]) \
    +'_' +str(meanb2[1])+'_'+str(int(100*pi1)) +'percent'
    
if config == '2':
    #----------- configuration 2-----#
    c1 = [0,0]
    r1 = 2
    c2 = [0,0]
    r2 = 1
    pi1 = 0.5
    
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-1.5,0,1.5] #x_list for later density plots
    
    continuous_type = 'continuous_uniform_circle'
    fname = continuous_type + str(c1[0]) + '_'+ str(c1[1])+'_'+ str(c2[0]) \
    +'_' +str(c2[1])+'_'+str(int(100*pi1)) +'percent'
    
iter = 100 # iterations of NPMLE_FW
fname = fname.replace('.','dot')  

#-----------------------------------------------------------#
# generate simulated dataset
if config == '1':
    np.random.seed(626)
    X,y,Bn = generate_multivariate_t_test_data\
    (n,iter, meanb1,covb1,meanb2,covb2, pi1, sigma,df_, func)
elif config == '2':
    np.random.seed(626)
    X,y,Bn = generate_circle_test_data(n,iter, c1,r1,c2,r2, pi1, sigma,func)
#-----------------------------------------------------------#


#-----------------------------------------------------------#
# storage dataset
if not os.path.exists('./../data/{}'.format(fname)):
    os.mkdir('./../data/{}'.format(fname))
pd.DataFrame(X).to_csv('./../data/{}/X.csv'.format(fname), index = False, header = False)
pd.DataFrame(y).to_csv('./../data/{}/y.csv'.format(fname), index = False, header = False)
pd.DataFrame(Bn).to_csv('./../data/{}/Bn.csv'.format(fname), index = False, header = False)
pd.DataFrame(np.reshape(np.array(sigma),(1,1))).to_csv('./../data/{}/sigma_true.csv'.format(fname), index = False, header = False)
#-----------------------------------------------------------#    


if run_cv == 'yes':
    #------------------------run CV-------------#
    #define a range of candidate sigma values
    sigma_max = min(0.6, np.sqrt(stats.variance(np.reshape(y, (len(y),)))))
    sigma_min = 0.25
    # NOTE: we started with a bigger range of sigma and narrowed down to this
    # NOTE: running with smaller range so that we can test very small granularity within less time

    cv_sigma_list = np.arange(sigma_min, sigma_max, cv_granularity)
    
    kfold = 5 # number of fold in CV procedure
    CV_result = cross_validation_parallel(X,y,cv_sigma_list,kfold,BL,BR)
    pd.DataFrame(CV_result).to_csv("./../data/{}/CV_result.csv".format(fname), index = False, header = False)
   
    # CV_result = pd.read_csv('./../data/{}/CV_result.csv'.format(fname), header = None).values
    #
    #--------------------------------------------#
    # choose sigma according to CV_result
    idx_min = np.argmin(CV_result[:,1])
    CV_min = CV_result[idx_min,1]
    
    epsilon = 0.0 # epsilon = 0.0 then normal selection
    # positive epsilon allows smaller sigma
    
    idx_approx_set = np.argwhere(CV_result[:,1] <= CV_min + epsilon * CV_min)
    
    sigma_cv = cv_sigma_list[np.min(idx_approx_set.ravel())]
    #--------------------------------------------#
    
else:
    #--------------------------------------------#
    #otherwise take sigma value from command line
    sigma_cv = float(run_cv)
    #--------------------------------------------#

print("sigma:{},sigma_cv:{}".format(sigma,sigma_cv))


#-------------------NPMLE-sigma----------------#
np.random.seed(626)
# needs sigma as input
f1, B1, alpha1, L_rec1, L_final1 = NPMLE_FW(X,y,iter,sigma,BL,BR,func)
pd.DataFrame(np.repeat(sigma,len(alpha1))).\
        to_csv("./../data/{}/sigma_NPMLEsigma.csv".format(fname), index = False, header = False)
pd.DataFrame(B1).to_csv('./../data/{}/B_NPMLEsigma.csv'.format(fname), index = False, header = False)
pd.DataFrame(alpha1).to_csv('./../data/{}/alpha_NPMLEsigma.csv'.format(fname), index = False, header = False)
pd.DataFrame(L_rec1).to_csv('./../data/{}/L_rec_NPMLEsigma.csv'.format(fname), index = False, header = False) 
    
#------------------------------------------------------------#    
#    
#-------------------NPMLE-CV ----------------#
np.random.seed(626)
f2, B2, alpha2, L_rec2, L_final2 = NPMLE_FW(X,y,iter,sigma_cv,BL,BR,func)
pd.DataFrame(np.repeat(sigma_cv,len(alpha2))).\
        to_csv("./../data/{}/sigma_CV.csv".format(fname), index = False, header = False)
pd.DataFrame(B2).to_csv('./../data/{}/B_NPMLE.csv'.format(fname), index = False, header = False)
pd.DataFrame(alpha2).to_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), index = False, header = False)
pd.DataFrame(L_rec2).to_csv('./../data/{}/L_rec_NPMLE.csv'.format(fname), index = False, header = False)

# alternatively read from previous runs
#B2 = pd.read_csv('./../data/{}/B_NPMLE.csv'.format(fname), header = None).values
#alpha2 = pd.read_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), header = None).values
#L_rec2 = pd.read_csv('./../data/{}/L_rec_NPMLE.csv'.format(fname), header = None).values
# 
#------------------------------------------------------------#    
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
    
    

line_styles = ['-','-','-','-']
# line_styles = ['-','--',':','-.']
    
fig1 = plt.figure(figsize = (8,8))

t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)


#plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')

N = len(alpha2)
threprob = 0.02

RGB_tuples = [(240,163,255),(0,117,220),(153,63,0),(76,0,92),(0,92,49),
(43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),
(194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
( 224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(25,25,25),(255,80,5)]

component_plot = []
component_color = []

temp = 0
index_sort = np.argsort(-np.reshape(alpha2,(len(alpha2),)))
count = 0 
for i in index_sort:
    b = B2[:,i]
    # select component with mixing probability above certain thresholds
    if alpha2[i] >threprob:
        component_plot.append(i)
        component_color.append(temp)
        plt.plot(t,b[0]+b[1]*t, color = tuple( np.array(RGB_tuples[temp])/255),\
                 linestyle = line_styles[int(count%4)],linewidth = alpha2[i][0]*8 ,\
                 label = r'$y = %.2f %+.2f x$ with probability $%.2f$'%(b[0], b[1], alpha2[i]))
        temp = temp + 1
        print("coefficients", b, "with probability", alpha2[i])
        count = count + 1

# ONLY clustering based on plotted components, i.e. components with high probability (>threprob)
C_cluster = np.zeros((n,1))
for i in range(len(y)):
    prob = np.zeros((N,1))
    for j in component_plot:
        prob[j] = alpha2[j] * np.exp(-0.5*(y[i] - np.dot(X[i],B2[:,j]))**2 /(sigma**2))
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


#-----------------------------------#
#                                   # 
#                                   #
#    empirical bayes                #
#                                   #
#                                   #
#-----------------------------------#

# read data files
X = pd.read_csv('./../data/{}/X.csv'.format(fname), header = None).values
y = pd.read_csv('./../data/{}/y.csv'.format(fname), header = None).values
Bn = pd.read_csv('./../data/{}/Bn.csv'.format(fname), header = None).values

if not os.path.exists('./../pics/EB_test'):
    os.mkdir('./../pics/EB_test')
    
#np.random.seed(626)
#X,y,Bn = generate_circle_test_data(10000,iter, c1,r1,c2,r2, pi1, sigma,func)

# true
fig_t = plt.figure(figsize = (8,8))
ax_t = fig_t.add_subplot(1,1,1)
ax_t.scatter(Bn[:,0],Bn[:,1], marker = '.')
ax_t.set_xlabel(r"$\beta_1$")
ax_t.set_ylabel(r"$\beta_2$")
fig_t.savefig('./../pics/EB_test/%s_truth.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#--------------------------------------#


# oracle bayes

def func_xy_beta_circ(x,y,c,r,sigma):
    # b * density function
    def func_circ_0(angle):
        return (float(c[0]) + r*np.cos(angle)) * 1/(np.sqrt(2*np.pi) *sigma) *np.exp(-0.5*(y- float(c[0])-r*np.cos(angle)\
                          -float(c[1])*x - r*np.sin(angle)*x)**2/sigma**2)/(2*np.pi)
    def func_circ_1(angle):
        return (float(c[1]) + r*np.sin(angle)) * 1/(np.sqrt(2*np.pi) *sigma) *np.exp(-0.5*(y- float(c[0])-r*np.cos(angle)\
                          -float(c[1])*x - r*np.sin(angle)*x)**2/sigma**2)/(2*np.pi)
    return func_circ_0, func_circ_1

def func_xy_beta_circ_int(x,y,c,r,sigma):
    # mean of b
    b = np.zeros(2)
    
    b[0] = integrate.quad(func_xy_beta_circ(x,y,c,r,sigma)[0],0.0,2 * np.pi,epsrel = 1e-3)[0]
    b[1] = integrate.quad(func_xy_beta_circ(x,y,c,r,sigma)[1],0.0,2 * np.pi,epsrel = 1e-3)[0]
    return b

# computing oracle bayes
p = 2
Bn_ob = np.zeros((n,p))
for i in range(n):
    if pi1 == 1:
        # only consider pi1 =1 case for simplicity
        b_temp = func_xy_beta_circ_int(X[i,1],y[i],c1,r1,sigma)
        Bn_ob[i,:] = b_temp/func_xy_circ_int(X[i,1],y[i],c1,r1,sigma)
fig_ob = plt.figure(figsize = (8,8))
ax_ob = fig_ob.add_subplot(1,1,1)
ax_ob.scatter(Bn_ob[:,0],Bn_ob[:,1], marker = '.')
ax_ob.set_xlabel(r"$\beta_1$")
ax_ob.set_ylabel(r"$\beta_2$")
fig_ob.savefig('./../pics/EB_test/%s_OB.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')


#--------------------------------------#
    

# empirical bayes
#-----------------

Bn_eb = np.zeros((n,p))
for i in range(n):
    # B2 is estimation from NPMLE-CV
    # read from storage earlier
    k = len(B2[0])
    
    prob_ = np.zeros(k)
    temp_ = np.zeros(k)
    
    for j in range(k):
        temp_[j] = -0.5*(y[i] - np.dot(X[i],B2[:,j]))**2/(sigma_cv**2)
    max_ = max(temp_)
    b = np.zeros(p)
    for j in range(k):
        if temp_[j] - max_ > -10:
            b = b + B2[:,j]*np.exp(temp_[j])/np.sum(np.exp(temp_))
    Bn_eb[i,:] = b
    
fig_eb = plt.figure(figsize = (8,8))
ax_eb = fig_eb.add_subplot(1,1,1)
ax_eb.scatter(Bn_eb[:,0],Bn_eb[:,1], marker = '.')
ax_eb.set_xlabel(r"$\beta_1$")
ax_eb.set_ylabel(r"$\beta_2$")
fig_eb.savefig('./../pics/EB_test/%s_EB.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')




#-----------------------------------#
#                                   # 
#                                   #
#    plot density function          #
#                                   #
#                                   #
#-----------------------------------#
    
    
# -------------------------------------------#
# source: https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
# pdf of multivarite_t
from  scipy.special import gammaln


def pdf(x, mean, shape, df):
    return np.exp(logpdf(x, mean, shape, df))


def logpdf(x, mean, shape, df):
    dim = mean.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E
#-----------------------------#
    
# density function under continuous measure
#def func_xy(x,y,meanb,covb,df_):
#    func = lambda b,k: 1/(np.sqrt(2*np.pi) *sigma) *np.exp(-0.5*(y-b-k*x)**2/sigma**2)\
#    *pdf(np.array([b,k], meanb, covb, df_ ))
#    return func


#--------------------------------------#
def func_xy(x,y,meanb,covb,sigma,df_):
    # density function of G^* under continuous measure
    def func(b,k):
        return 1/(np.sqrt(2*np.pi) *sigma) *np.exp(-0.5*(y-b-k*x)**2/sigma**2)\
    *pdf(np.array([b,k]), np.array(meanb), np.array(covb), df_ )
    return func
def func_xy_int(x,y,meanb,covb,sigma,df_):
    # density function of y under continuous measure
    return integrate.dblquad(func_xy(x,y,meanb,covb,sigma,df_),\
                             -np.inf,np.inf,lambda b:-np.inf,lambda b :np.inf, epsabs = 1e-3)[0]

#--------------------------------------#
def func_xy_circ(x,y,c,r,sigma):
    # density function of G^* under continuous measure
    def func_circ(angle):
#        b = c.ravel()[0] + r * np.cos(angle)
#        k = c.ravel()[1] + r * np.sin(angle)
        return 1/(np.sqrt(2*np.pi) *sigma) *np.exp(-0.5*(y- float(c[0])-r*np.cos(angle)\
                          -float(c[1])*x - r*np.sin(angle)*x)**2/sigma**2)/(2*np.pi)
    return func_circ
def func_xy_circ_int(x,y,c,r,sigma):
    # density function of y under continuous measure
    return integrate.quad(func_xy_circ(x,y,c,r,sigma),0.0,2 * np.pi,epsrel = 1e-3)[0]


   
#-----------------------------------------------------------------   
line_styles = ['-','-','-','-']
# line_styles = ['-','--',':','-.']
fig = plt.figure(figsize = (16,5))
#List of x values
i = 0
for i in range(len(x_list)):
    x = x_list[i]
    if config == '1':
        min_ = min(func([1,x],meanb1), func([1,x],meanb2))
        max_ = max(func([1,x],meanb1), func([1,x],meanb2))
    if config == '2':
        min_ = min(func([1,x],c1), func([1,x],c2))
        max_ = max(func([1,x],c1), func([1,x],c2))
    if func.__name__ == 'lin_func':
        y = np.linspace(min_ -3, max_ + 3, 100)
    else:
        y = np.linspace(min_ -6, max_ + 6, 100)
       
    
    plt.subplot(1,len(x_list),i+1)

    fy1 = sum(alpha1[i]*scipy.stats.norm.pdf( y-func([1,x],B1[:,i]), 0, sigma) \
    for i in range(len(alpha1)))
    plt.plot(y,fy1.ravel() ,color = 'tab:green',label = r'NPMLE-$\sigma$',linestyle = line_styles[0])
    
    fy2 = sum(alpha2[i]*scipy.stats.norm.pdf( y-func([1,x],B2[:,i]), 0, sigma_cv) \
    for i in range(len(alpha2)))
    plt.plot(y,fy2.ravel() ,color = 'tab:orange',label = 'NPMLE-CV',linestyle = line_styles[1])
    
    #-------------------- ditribution of ground truth needs integral
    # CAUTION: this step may take very long time to run on a laptop
    #----------------------------------------------------------------
    # fy_true = np.zeros(len(y))
#    for j in range(len(y)):
#        if pi1 < 1:
#            fy_true[j] = pi1* integrate.dblquad(func_xy(x,y[j],meanb1,covb1,df_),-np.inf,np.inf,lambda b:-np.inf,lambda b :np.inf)[0]\
#            +(1-pi1)*integrate.dblquad(func_xy(x,y[j],meanb2,covb2,df_),-np.inf,np.inf,lambda b: -np.inf,lambda b :np.inf)[0]
#        else:
#            fy_true[j] = integrate.dblquad(func_xy(x,y[j],meanb1,covb1,df_),-np.inf,np.inf,lambda b:-np.inf,lambda b :np.inf)[0]
    
    if pi1 < 1:
        if config == '1':
            with Pool(8) as integral_pool:
                fy_true = pi1* np.array(integral_pool.starmap(func_xy_int, \
                        zip(repeat(x), y, repeat(np.array(meanb1)), repeat(np.array(covb1)),repeat(sigma_cv),repeat(df_) )))\
                                                     + (1-pi1)*np.array(integral_pool.starmap(func_xy, \
                        zip(repeat(x), y, repeat(np.array(meanb2)), repeat(np.array(covb2)),repeat(sigma_cv),repeat(df_) )))
        elif config == '2':
            with Pool(8) as integral_pool:
                fy_true = pi1* np.array(integral_pool.starmap(func_xy_circ_int,\
                            zip(repeat(x), y, repeat(c1), repeat(r1), repeat(sigma_cv))))\
                + (1-pi1) *np.array( integral_pool.starmap(func_xy_circ_int,\
                            zip(repeat(x), y, repeat(c2), repeat(r2), repeat(sigma_cv))))

    elif pi1 == 1:
        if config == '1':
            with Pool(8) as integral_pool:
                fy_true = np.array(integral_pool.starmap(func_xy_int, \
                    zip(repeat(x), y, repeat(meanb1), repeat(covb1),repeat(sigma_cv),repeat(df_) )))
        elif config == '2':
            with Pool(8) as integral_pool:
                fy_true = np.array(integral_pool.starmap(func_xy_circ_int,\
                            zip(repeat(x), y, repeat(c1), repeat(r1), repeat(sigma_cv))))
                
    fy_true = np.array(fy_true)
    plt.plot(y,fy_true.ravel(),color = 'tab:blue',label = 'Truth',linestyle =line_styles[0])
    
#    plt.plot(y, sum(alpha3[i]*scipy.stats.norm.pdf( y-(B3[0,i]+B3[1,i]*x), 0, sigma) \
#    for i in range(len(alpha3))),color = 'tab:purple',label = 'EM_sigma',linestyle = line_styles[2])
        

    plt.title(r'$x = %.1f$'%x)
    plt.xlabel(r'$y$')
    if i == 0:
        plt.ylabel(r'$\rm{pdf}$')

# legend      
custom_lines = [
            Line2D([0], [0], color= 'tab:blue', linestyle = line_styles[0]),
          Line2D([0], [0], color= 'tab:green', linestyle = line_styles[0]),
          Line2D([0], [0], color= 'tab:orange', linestyle = line_styles[1])
          # Line2D([0], [0], color= 'tab:purple', linestyle = line_styles[2]),
    ]
ax = plt.gca()
lgd = ax.legend(custom_lines, ['Truth',r'NPMLE-$\sigma$','NPMLE-CV',
                                #'EM_sigma'
                         ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('./../pics/%s_multi_density.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#plt.show();
#---------------------------------------------------------------------------------
  
   








