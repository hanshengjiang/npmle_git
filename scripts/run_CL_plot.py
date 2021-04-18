#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:10:44 2021

@author: hanshengjiang

"""

'''
Plot CL for general settings
'''

from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys

from tqdm import tqdm


n_repeat = 100
iter = 50

#---------------------------------------------#
n = 500
sigma = 0.5
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
#---------------------------------------------#

fname = func.__name__[:-4] + str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
+'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
fname = fname.replace('.','dot')

#---------------------------------------------#

np.random.seed(26)

CL_record = np.zeros((n_repeat, iter-1))

for i in tqdm(range(n_repeat)):
    sigma_list = [sigma,sigma,sigma] # homo error
    X,y,C = generate_test_data(n,iter, b1, b2, b3,pi1,pi2,sigma_list,func)
    
    f, B, alpha, L_rec, L_final = NPMLE_FW_parallel(X,y,iter,sigma_cv,BL,BR,func)
 
    CL_record[i,:] = L_rec
    
fig = plt.figure(figsize = (6,5))
ax = plt.gca()
ax.set_xlabel(r"Iteration")
ax.set_ylabel(r'$C_L$')
for i in range(n_repeat):
    plt.plot(np.arange(1,iter), np.array(CL_record[i,:]), linewidth = 0.3);
plt.savefig('./../pics/%s_C_L_multi.png'%fname, dpi = 300, bbox_inches='tight')

# save to file
pd.DataFrame(CL_record).to_csv('./../data/{}/CL_record.csv'.format(fname), index = False, header = False)














