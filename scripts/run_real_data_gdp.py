#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""
from package_import import *
from alg1_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *

import sys


if __name__ == "__main__":
    # default
    if len(sys.argv) < 2:
        run_cv = 0.30 # chosen by cross-validation procedure
        cv_granuality = 0.01
    # otherwise take argyments from command line
    else:
        #sys_.rgv[0] is the name of the .py file
        run_cv = sys.argv[1]
        cv_granuality = float(sys.argv[2])

# read data
df = pd.read_csv('./../real_data/co2-emissions-vs-gdp_simplified.csv')


dfs = (df.iloc[:,[2,6,7]]).dropna()
dataf = dfs.iloc[:,[1,2]].values.astype(np.float)
# 5 population in billions; 6: CO2 in millions ; GDP percapita in 1000USD 
region_list = dfs.iloc[:,0].values
# dataf[:,1:] = dataf[:,1:].astype(np.float)
n = np.shape(dataf)[0]
ones = np.ones((n,1))
X = np.concatenate((ones, np.reshape(dataf[:,0],(n,1))/10000 ), axis = 1) #GDP per capita (1000 USD)
y = np.reshape(dataf[:,1]/10,(n,1)) #CO2 per capita (10 ton)


if run_cv == 'yes':
    #------------------------run CV-------------#
    #define a range of candidate sigma values
    # sigma_max = np.sqrt(stats.variance(np.reshape(y, (len(y),))))
    sigma_max = 0.5
    sigma_min = 0.05
    sigma_list = np.arange(sigma_min, sigma_max, cv_granuality)
    
    kfold = 10 # number of fold in CV procedure
    CV_result = cross_validation_parallel(X,y,sigma_list,kfold,-10,10)
    pd.DataFrame(CV_result).to_csv("./../real_data/CV_result_gdp.csv", index = False)
    
    idx_min = np.argmin(CV_result[:,1])
    sigma_cv = sigma_list[idx_min]
else:
    #--------------------------------------------#
    #otherwise take sigma value from command line
    sigma_cv = float(run_cv)
    #--------------------------------------------#

print("sigma_cv:{}".format(sigma_cv))




iter = 200
threprob = 1e-2

#Use Algorithm 1
np.random.seed(26)
f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma_cv, -10, 10)
print("number of components", len(alpha))
##########IMPORTANT subproblem initializes with beta = 0

#plot
print("final neg log likelihood is ", L_final)
print("number of components is", len(alpha))
print("only components with probability at least ", threprob, " are shown below:")

fig1, ax = plt.subplots(figsize = (10,7.5))
plt.scatter(X[:,1],y,color = 'blue',marker = 'o', facecolors = 'None',linewidths = 0.5);
#label = 'Noisy data',

# for i, txt in enumerate(region_list):
#     if X[i,1] > 1:
#         ax.annotate(txt, (X[i,1],y[i])) # replace marker by text

nation_list = [ 'USA','CAN','KAZ','DNK','GBR','BHR','SGP','NOR']
# nation_list = region_list
corr = -0.1
for i, txt in enumerate(region_list):
    if txt in nation_list and i%1 == 0:
        ax.annotate(txt, (X[i,1] + corr, y[i] + corr),size = 12) # replace marker by text
    
t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-2)
i = 0
index_sorted = np.argsort(-np.reshape(alpha,(len(alpha),)))
for i in index_sorted:
    b = B[:,i]
    if alpha[i] >threprob:
        plt.plot(t,b[0]+b[1]*t, color = str((1-alpha[i][0])/100),linewidth = alpha[i][0]*8 ,\
                 label = '$y = %.4f + %.4f x$ with probability $%.2f$' %(b[0], b[1], alpha[i]) )
        print("coefficients", b, "with probability", alpha[i])
# custom_lines = [Line2D([], [], color='gray', marker='o',markerfacecolor = 'None', linestyle='None'),
#                 Line2D([0], [0], color='black')
#                 ,]
# ax.legend(custom_lines, ['Noisy data'
#                           , 'NPMLE component'
#                          ],loc=9);
ax = plt.gca()
ax.set_xlabel(r'$x$ ($\rm{GDP}$)')
ax.set_ylabel(r'$y$ ($\rm{CO_2}$)')
lgd = ax.legend(loc=9, bbox_to_anchor=(1.43, 1),borderaxespad=0.) 
plt.savefig('./../pics/co2_gdp.png', dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')