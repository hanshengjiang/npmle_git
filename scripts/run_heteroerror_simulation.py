#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:47:31 2021

@author: hanshengjiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hanshengjiang

"""

'''

1. simulations of discrete mixtures 
2. density plots that compare EM, CGM, with or without knowing sigma

'''


from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys


if __name__ == "__main__":
    # default
    if len(sys.argv) < 6:
        sigma_list = [0.5,0.8,1] # hetero errors
        n = 500
        config = '1'
        run_cv = '0.54'
        cv_granuality = 0.01
    # otherwise take argyments from command line
    else:
        #sys_argv[0] is the name of the .py file
        sigma_list = []
        sigma_list.append(float(sys.argv[1]))
        sigma_list.append(float(sys.argv[2]))
        sigma_list.append(float(sys.argv[3]))
        n = int(sys.argv[4]) # number of data points
        config = sys.argv[5]
        run_cv = sys.argv[6]
        cv_granuality = 0.04
# preset configurations      
if config == '1':
    #----------- configuration 1-----#
    b1 = (1,1)
    b2 = (4,-1)
    b3 = (-1,0.5)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[1,4],[1,-1]]
    alpha_true = [0.5,0.5]
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-3,0,5] #x_list for later density plots
elif config == '2':
    #----------- configuration 2-----#
    b1 = (0,1)
    b2 = (2,1)
    b3 = (0,1)
    pi1 = 0.3
    pi2 = 0.7
    B_true = [[0,1],[2,1]]
    alpha_true = [0.3,0.7]
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-3,0,5]
elif config == '3':
    #----------- configuration 3-----#
    b1 = (3,-1)
    b2 = (1,1.5)
    b3 = (-1,0.5)
    pi1 = 0.3
    pi2 = 0.3
    B_true = [[4,1,-1],[-1,1.5,0.5]]
    alpha_true = [0.3,0.3,0.4]
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-3,0,5] 
else:
    sys.exit("Wrong configuration number!")



iter = 100 # iterations of NPMLE_FW


fname = 'hetero_' + str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
+'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
fname = fname.replace('.','dot')


#-----------------------------------------------------------#
# generate simulated dataset
np.random.seed(626)
#----------------------------

#-----------------------------
X,y,C = generate_test_data(n,iter, b1, b2, b3,pi1,pi2,sigma_list,func)
#-----------------------------------------------------------#

#-------------------------------------------------------------#
# EM algorithms need to specify the number of vomponents
num_component = int(float(config)/3) + 2
#-------------------------------------------------------------#

#-----------------------------------------------------------#
# storage dataset
if not os.path.exists('./../data/{}'.format(fname)):
    os.mkdir('./../data/{}'.format(fname))
pd.DataFrame(X).to_csv('./../data/{}/X.csv'.format(fname), index = False, header = False)
pd.DataFrame(y).to_csv('./../data/{}/y.csv'.format(fname), index = False, header = False)
pd.DataFrame(np.reshape(np.array(num_component), (1,1))).to_csv('./../data/{}/num_component.csv'.format(fname), index = False, header = False)
pd.DataFrame(alpha_true).to_csv('./../data/{}/alpha_true.csv'.format(fname), index = False, header = False)
pd.DataFrame(B_true).to_csv('./../data/{}/B_true.csv'.format(fname), index = False, header = False)
pd.DataFrame(sigma_list).to_csv('./../data/{}/sigma_true.csv'.format(fname), index = False, header = False)
#-----------------------------------------------------------#    



if run_cv == 'yes':
    #------------------------run CV-------------#
    #define a range of candidate sigma values
    sigma_max = np.sqrt(stats.variance(np.reshape(y, (len(y),))))
    sigma_min = 0.1
    cv_sigma_list = np.arange(sigma_min, sigma_max, cv_granuality)
    
    kfold = 5 # number of fold in CV procedure
    CV_result = cross_validation_parallel(X,y,cv_sigma_list,kfold,BL,BR)
    pd.DataFrame(CV_result).to_csv("./../data/{}/CV_result.csv".format(fname), index = False)
    idx_min = np.argmin(CV_result[:,1])
    sigma_cv = sigma_list[idx_min]
    pd.DataFrame(sigma_cv).to_csv("./../data/{}/sigma_EM.csv".format(fname), index = False, header = False)
else:
    #--------------------------------------------#
    #otherwise take sigma value from command line
    sigma_cv = float(run_cv)
    #--------------------------------------------#

print("sigma_cv:{}".format(sigma_cv))


#------------------------------------------------------------#  
# paramters of b's and pi's are only for plotting purposes
test(X, y,C, n,iter, b1, b2, b3,pi1,pi2,sigma_cv,-10,10,fname,func)
#------------------------------------------------------------#    

#--------------------density plots--------------------------#
  
#-------------------1: CGM with known sigma----------------#
# needs sigma as input
# uses the smallest heterosedastic error
f1, B1, alpha1, L_rec1, L_final1 = NPMLE_FW(X,y,iter,sigma_list[0],BL,BR,func)
    
    
#------------------------------------------------------------#    
#    
#-------------------2: CGM without knowing sigma----------------#

f2, B2, alpha2, L_rec2, L_final2 = NPMLE_FW(X,y,iter,sigma_cv,BL,BR,func)
    

#-------------------------   Run EM    ----------------------#
import os
import sys
os.system("Rscript " + "run_regmixEM.R " + fname + ' hetero_error')

#-------------------Read EM estimated results ----------------#
B4 = pd.read_csv('./../data/{}/B_EM.csv'.format(fname), header = None).values
alpha4 = pd.read_csv('./../data/{}/alpha_EM.csv'.format(fname), header = None).values
sigma = pd.read_csv('./../data/{}/sigma_EM.csv'.format(fname), header = None).values
sigma_array4 = np.repeat(sigma, num_component)
#------------------------------------------------------------# 





#-----------------------------------#
#
#  plot density function            #
#
#-----------------------------------#
  
#-----------------------------------------------------------------   
line_styles = ['-','-','-','-']
# line_styles = ['-','--',':','-.']
fig = plt.figure(figsize = (16,5))
#List of x values
i = 0
for i in range(len(x_list)):
    x = x_list[i]
    min_ = min(func([1,x],b1), func([1,x],b2))
    max_ = max(func([1,x],b1), func([1,x],b2))
    y = np.linspace(min_ -3, max_ + 3, 100)
       
    #calculate difference of square root of density functions
    #dist_fit = lambda y: (np.sqrt(0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x), 0, sigma)+0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x),0, sigma)) \
    #- np.sqrt(sum(alpha[i]*scipy.stats.norm.pdf( y - (B[0,i]+B[1,i]*x), 0, sigma) for i in range(len(alpha)))))**2
    
    #print("Fix x = %.1f, squapink Hellinger distance for NPMLE is %.5f" % (x, quad(dist_fit, -np.inf, np.inf)[0]))

    plt.subplot(1,len(x_list),i+1)
    
    if func.__name__ == 'lin_func':
        fy4 = sum(alpha4[i]*scipy.stats.norm.pdf( y-func([1,x],B4[:,i]), 0, sigma_array4[i]) \
               for i in range(len(alpha4)))
        plt.plot(y, fy4.ravel(),color = 'tab:pink',label = 'EM-true',linestyle = line_styles[3])
        
    fy1 = sum(alpha1[i]*scipy.stats.norm.pdf( y-func([1,x],B1[:,i]), 0, sigma) \
    for i in range(len(alpha1)))
    plt.plot(y,fy1.ravel() ,color = 'tab:green',label = r'NPMLE-$\sigma$',linestyle = line_styles[0])
    
    fy2 = sum(alpha2[i]*scipy.stats.norm.pdf( y-func([1,x],B2[:,i]), 0, sigma_cv) \
    for i in range(len(alpha2)))
    plt.plot(y,fy2.ravel() ,color = 'tab:orange',label = 'NPMLE-CV',linestyle = line_styles[1])
        
    fy_true = pi1*scipy.stats.norm.pdf(y - func([1,x],b1), 0, sigma_list[0])+pi2*scipy.stats.norm.pdf(y-func([1,x],b2),0, sigma_list[1])+\
    (1-pi1-pi2)*scipy.stats.norm.pdf(y-func([1,x],b3),0, sigma_list[2])
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
          # Line2D([0], [0], color= 'tab:green', linestyle = line_styles[0]),
          Line2D([0], [0], color= 'tab:orange', linestyle = line_styles[1])
          # Line2D([0], [0], color= 'tab:purple', linestyle = line_styles[2]),
    ]
if func.__name__ == 'lin_func':
    custom_lines.append( Line2D([0], [0], color= 'tab:pink', linestyle = line_styles[3]))

ax = plt.gca()
lgd = ax.legend(custom_lines, ['Truth', # 'NPMLE_sigma',
                               r'NPMLE-$\sigma$', #'EM_sigma'
                               'NPMLE-CV',
                               'EM-true'
                         ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('./../pics/%s_multi_density.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#plt.show();
#---------------------------------------------------------------------------------
  





