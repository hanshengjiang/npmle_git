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
import sys


if __name__ == "__main__":
    # default
    if len(sys.argv) < 5:
        sigma = 0.5
        n = 500
        config = '3'
        run_cv = 'yes' # should be a value from cross-validation
    else:
        #sys_argv[0] is the name of the .py file
        sigma = float(sys.argv[1]) 
        n = int(sys.argv[2]) # number of data points
        config = sys.argv[3]
        run_cv = sys.argv[4]
if config == '1':
    #----------- configuration 1-----#
    b1 = (1,1)
    b2 = (4,-1)
    b3 = (-1,0.5)
    pi1 = 0.5
    pi2 = 0.5
elif config == '2':
    #----------- configuration 2-----#
    b1 = (0,1)
    b2 = (2,1)
    b3 = (0,1)
    pi1 = 0.3
    pi2 = 0.7
elif config == '3':
    #----------- configuration 3-----#
    b1 = (3,-1)
    b2 = (1,1.5)
    b3 = (-1,0.5)
    pi1 = 0.3
    pi2 = 0.3
else:
    sys.exit("Wrong configuration number!")

# range of regression coefficients
BL = -10
BR = 10
iter = 50

fname = str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
+'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'

np.random.seed(626)
# generate simulated dataset
X,y,C = generate_test_data(n,iter, b1, b2, b3,pi1,pi2,sigma)

if run_cv == 'yes':
    #------------------------run CV-------------#
    #define a range of candidate sigma values
    sigma_max = np.sqrt(stats.variance(np.reshape(y, (len(y),))))
    sigma_min = 0.1
    sigma_list = np.arange(sigma_min, sigma_max, 0.02)
    
    kfold = 5 # number of fold in CV procedure
    CV_result = cross_validation_parallel(X,y,sigma_list,kfold,BL,BR)
    pd.DataFrame(CV_result).to_csv("./../data/CV_result_{}.csv".format(fname), index = False)
    
    idx_min = np.argmin(CV_result[:,1])
    sigma_cv = sigma_list[idx_min]
else:
    #--------------------------------------------#
    #otherwise take sigma value from command line
    sigma_cv = float(run_cv)
    #--------------------------------------------#

print("sigma:{},sigma_cv:{}".format(sigma,sigma_cv))

iter = 50 

# paramters of b's and pi's are only for plotting purposes
test(X, y,C, n,iter, b1, b2, b3,pi1,pi2,sigma,sigma_cv,-10,10)

#------------------------------------------------------------#    
#    
#-------------------1: CGM with known sigma----------------#
# needs sigma as input
f1, B1, alpha1, L_rec1, L_final1 = NPMLE_FW(X,y,iter,sigma,BL,BR)
    
    
#------------------------------------------------------------#    
#    
#-------------------2: CGM without knowing sigma----------------#

f2, B2, alpha2, L_rec2, L_final2 = NPMLE_FW(X,y,iter,sigma_cv,BL,BR)
    
#-------------------------------------------------------------#
# EM algorithms need to specify the number of 
k = int(float(config)/3) + 2
#-------------------------------------------------------------#

iter_EM = 100 # EM needs more iterations
#------------------------------------------------------------#    
#    
#-------------------3: EMA with known sigma------------------#
# needs sigma as input
f3, B3, alpha3, sigma_array3, L_rec3, L_final3 = EMA_sigma(X,y,k,iter_EM,BL,BR,sigma)


#------------------------------------------------------------#    
#    
#-------------------4: EMA_sigma without knowing sigma----------------#

# need specification on the range of sigma
sigmaL =  0.1# = sigma_min
sigmaR = np.sqrt(stats.variance(np.reshape(y, (len(y),)))) # = sigma_max
# sigma_min sigma_max are also used in the cross-validation procedure

f4, B4, alpha4, sigma_array4, L_rec4, L_final4 = EMA(X,y,k,iter_EM,BL,BR,sigmaL,sigmaR)

#-----------------------------------#
#
#  plot density function            #
#
#-----------------------------------#
  
#-----------------------------------------------------------------   
line_styles = ['-','-','-','-']
# line_styles = ['-','--',':','-.']
fig = plt.figure(figsize = (16,5))
x_list = [-3,0,5] #List of x values
i = 0
for i in range(len(x_list)):
    x = x_list[i]
    min_ = min(b1[0]+b1[1]*x, b2[0]+b2[1]*x)
    max_ = max(b1[0]+b1[1]*x, b2[0]+b2[1]*x)
    y = np.linspace(min_ -2, max_ + 2, 100)
       
    #calculate difference of square root of density functions
    #dist_fit = lambda y: (np.sqrt(0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x), 0, sigma)+0.5*scipy.stats.norm.pdf(y-(b1[0]+b1[1]*x),0, sigma)) \
    #- np.sqrt(sum(alpha[i]*scipy.stats.norm.pdf( y - (B[0,i]+B[1,i]*x), 0, sigma) for i in range(len(alpha)))))**2
    
    #print("Fix x = %.1f, squared Hellinger distance for NPMLE is %.5f" % (x, quad(dist_fit, -np.inf, np.inf)[0]))

    
    plt.subplot(1,len(x_list),i+1)
    
    plt.plot(y, sum(alpha4[i]*scipy.stats.norm.pdf( y-(B4[0,i]+B4[1,i]*x), 0, sigma_array4[i]) \
    for i in range(len(alpha4))),color = 'tab:red',label = 'EM',linestyle = line_styles[3])
    
    plt.plot(y,pi1*scipy.stats.norm.pdf(y - (b1[0]+b1[1]*x), 0, sigma)+pi2*scipy.stats.norm.pdf(y-(b2[0]+b2[1]*x),0, sigma)+\
    (1-pi1-pi2)*scipy.stats.norm.pdf(y-(b3[0]+b3[1]*x),0, sigma),color = 'tab:blue',label = 'Truth',linestyle =line_styles[0])

    plt.plot(y, sum(alpha2[i]*scipy.stats.norm.pdf( y-(B2[0,i]+B2[1,i]*x), 0, sigma_cv) \
    for i in range(len(alpha2))),color = 'tab:green',label = 'NPMLE',linestyle = line_styles[1])
        
    plt.plot(y, sum(alpha1[i]*scipy.stats.norm.pdf( y-(B1[0,i]+B1[1,i]*x), 0, sigma) \
    for i in range(len(alpha1))),color = 'tab:orange',label = 'NPMLE_sigma',linestyle = line_styles[0])
    
    
#    plt.plot(y, sum(alpha3[i]*scipy.stats.norm.pdf( y-(B3[0,i]+B3[1,i]*x), 0, sigma) \
#    for i in range(len(alpha3))),color = 'tab:purple',label = 'EM_sigma',linestyle = line_styles[2])
        

    plt.title(r'$x = %.1f$'%x)
    plt.xlabel(r'$y$')
    if i == 0:
        plt.ylabel(r'$\rm{pdf}$')

# legend      
custom_lines = [
            Line2D([0], [0], color= 'tab:blue', linestyle = line_styles[0]),
          Line2D([0], [0], color= 'tab:orange', linestyle = line_styles[0]),
          Line2D([0], [0], color= 'tab:green', linestyle = line_styles[1]),
          Line2D([0], [0], color= 'tab:red', linestyle = line_styles[3]),
          # Line2D([0], [0], color= 'tab:purple', linestyle = line_styles[2]),
    ]
ax = plt.gca()
lgd = ax.legend(custom_lines, ['Truth','NPMLE_sigma','NPMLE', 'EM',
                                #'EM_sigma'
                         ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('./../pics/%s_multi_density.png'%fname, dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#plt.show();
#---------------------------------------------------------------------------------
  





