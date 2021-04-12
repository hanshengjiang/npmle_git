#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:12:07 2021

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
        run_cv = 'yes'
        cv_granuality = 0.01
    # otherwise take argyments from command line
    else:
        #sys.argv[0] is the name of the .py file
        run_cv = sys.argv[1]
        cv_granuality = float(sys.argv[2])

#read tonedata into file
with open('./../real_data/tonedata.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[1:] # remove row containing column name
dataf = data.astype(np.float)
n = np.shape(dataf)[0]
ones = np.ones((n,1))
X = np.concatenate((ones, np.reshape(dataf[:,0],(n,1))), axis = 1)
y = np.reshape(dataf[:,1],(n,1))


if run_cv == 'yes':
    #------------------------run CV-------------#
    #define a range of candidate sigma values
    # sigma_max = np.sqrt(stats.variance(np.reshape(y, (len(y),))))
    sigma_max = 0.5
    sigma_min = 0.05
    sigma_list = np.arange(sigma_min, sigma_max, cv_granuality)
    
    kfold = 10 # number of fold in CV procedure
    CV_result = cross_validation_parallel(X,y,sigma_list,kfold,-10,10)
    pd.DataFrame(CV_result).to_csv("./../real_data/CV_result_tone.csv", index = False)
    
    idx_min = np.argmin(CV_result[:,1])
    sigma_cv = sigma_list[idx_min]
else:
    #--------------------------------------------#
    #otherwise take sigma value from command line
    sigma_cv = float(run_cv)
    #--------------------------------------------#

print("sigma_cv:{}".format(sigma_cv))



iter = 200
threprob = 0.02

#Use Frank-Wofle with an estimated sigma
sigma = 0.05
f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma_cv,-10,10)


#beta_ols = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1))

#plot
print("final neg log likelihood is ", L_final)
print("number of components is", len(alpha))
print("only components with probability at least ", threprob, " are shown below:")

fig1 = plt.figure(figsize = (10,10))
# label = 'Original data',
plt.scatter(X[:,1],y,color = 'blue',marker = 'o',linewidths=0.5, facecolors = 'None');
t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
#plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
i = 0

index_sorted = np.argsort(-np.reshape(alpha,(len(alpha),)))
for i in index_sorted:
    b = B[:,i]
    if alpha[i] >threprob:
        plt.plot(t,b[0]+b[1]*t, color = str((1-alpha[i][0])/100),linewidth = alpha[i][0]*8 ,\
                 label = r'$y = %.4f + %.4f x$ with probability $%.2f$' %(b[0], b[1], alpha[i]))
        print("coefficients", b, "with probability", alpha[i])
#plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
#plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                         # , 'NPMLE component'#, 'OLS'#
                         #],loc=2);
ax = plt.gca()
lgd = ax.legend(loc=9, bbox_to_anchor=(1.43, 1),borderaxespad=0.) 
ax.set_xlabel(r'$x$ (stretching ratio)')
ax.set_ylabel(r'$y$ (adjusted ratio)')
plt.savefig('./../pics/tone.png', dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show();

