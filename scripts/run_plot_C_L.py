#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:34:40 2021

@author: hanshengjiang
"""

from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys
from plotting_lib import *

'''
Ridgeline plots
'''

# ----------------------------
# python run_ridgeline_plot.py 1 homo
# ----------------------------



if __name__ == "__main__":
    # default
    if len(sys.argv) < 3:
        config = 'cont-2'
        error_type = 'homo' # error type can be hetero for config = 1,2,3
    # otherwise take argyments from command line
    else:
        #sys_argv[0] is the name of the .py file
        config = sys.argv[1]
        error_type = sys.argv[2]
        
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
elif config == '2':
    #----------- configuration 2-----#
    b1 = (0,1)
    b2 = (2,1)
    b3 = (0,1)
    pi1 = 0.3
    pi2 = 0.7
    B_true = [[0,2],[1,1]]
    alpha_true = [0.3,0.7]
    func = lin_func
    BL = -10
    BR = 10
elif config == '3':
    #----------- configuration 3-----#
    b1 = (3,-1)
    b2 = (1,1.5)
    b3 = (-1,0.5)
    pi1 = 0.3
    pi2 = 0.3
    B_true = [[3,1,-1],[-1,1.5,0.5]]
    alpha_true = [0.3,0.3,0.4]
    func = lin_func
    BL = -10
    BR = 10
elif config == '3-negative':
    #----------- configuration 3-negative -----#
    
    # redefine sigma_list because this is a negative case
    # sigma_list = [0.3,0.5,0.1]
    
    b1 = (3,-1)
    b2 = (1,1.5)
    b3 = (-1,0.5)
    pi1 = 0.3
    pi2 = 0.3
    B_true = [[3,1,-1],[-1,1.5,0.5]]
    alpha_true = [0.3,0.3,0.4]
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-1.5,0,1.5]
elif config == '4':
    #----------- configuration 4-----# 
    b1 = (-4,1)
    b2 = (1, -1)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-4,1],[1,-1]]
    alpha_true = [0.5,0.5]
    func = poly_func
    BL = -10
    BR = 10
elif config == '5':
    #----------- configuration 5-----#

    b1 = (-0.5,-1)
    b2 = (-1.5,-1.5)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-0.5,-1.5],[-1,-1.5]]
    alpha_true = [0.5,0.5]
    func = exp_func
    BL = -10 #BL 
    BR = 10
elif config == '6':
    #----------- configuration 6-----#
    b1 = (-0.5,1)
    b2 = (-1.5,1.5)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-0.5,-1.5],[1,1.5]]
    alpha_true = [0.5,0.5]
    func = sin_func
    BL = -10
    BR = 10
elif config == 'cont-1':
    #----------- continuous case-----#
    meanb1 = [1,0.5]
    covb1 = np.array([[0.5,0.2],[0.2,0.3]])
    meanb2 = [2,3]
    covb2 = np.array([[0.5,0.2],[0.2,0.3]])
    pi1 = 1.0
    df_ = 3
    
    func = lin_func
    BL = -10
    BR = 10
    
    sigma = 0.5
    
    continuous_type = 'continuous_multivariate_t'
    fname = continuous_type + str(meanb1[0]) + '_'+ str(meanb1[1])+'_'+ str(meanb2[0]) \
    +'_' +str(meanb2[1])+'_'+str(int(100*pi1)) +'percent'

elif config == 'cont-2':
    #----------- configuration -----#
    c1 = [0,0]
    r1 = 2
    c2 = [0,0]
    r2 = 1
    pi1  = 0.5
    
    func = lin_func
    BL = -10
    BR = 10
    x_list = [-1.5,0,1.5] #x_list for later density plots
    
    sigma = 0.5
    continuous_type = 'continuous_uniform_circle'
    fname = continuous_type + str(c1[0]) + '_'+ str(c1[1])+'_'+ str(c2[0]) \
    +'_' +str(c2[1])+'_'+str(int(100*pi1)) +'percent'
    
else:
    sys.exit("Wrong configuration number!")

if config[:4] != 'cont' :
    fname = func.__name__[:-4] + str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
    +'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
    fname = fname.replace('.','dot')

if config[1:] == '-negative':
    fname = fname + '_negative_case'

if error_type == 'hetero':
    fname = 'hetero_'+fname
    
# read data 
L_rec = pd.read_csv('./../data/{}/L_rec_NPMLE.csv'.format(fname), header = None).values

print(fname)
fig3 = plt.figure(figsize = (6,5))
ax = plt.gca()
ax.set_xlabel(r"Iteration")
ax.set_ylabel(r'$\log C_L$')
plt.plot(np.log(np.array(L_rec)+1e-10));
plt.savefig('./../pics/%s_C_L.png'%fname, dpi = 300, bbox_inches='tight')
  
    
