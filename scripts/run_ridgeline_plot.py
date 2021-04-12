#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:32:07 2021

@author: hanshengjiang
"""
from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys

'''
Ridgeline plots
'''

# ----------------------------
# python run_ridgeline_plot.py 1 homo
# ----------------------------


if __name__ == "__main__":
    # default
    if len(sys.argv) < 3:
        config = '1'
        error_type = 'homo'
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
    x_list = [0,1,2] #x_list for later density plots
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
    x_list = [0,1,2]
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
    x_list = [0,1,2]
elif config == '4':
    #----------- configuration 4-----# 
    b1 = (-4,1)
    b2 = (1, -1)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[0.5,1],[2,2.5]]
    alpha_true = [0.5,0.5]
    func = poly_func
    BL = -10
    BR = 10
    x_list = [0,1,2]
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
    x_list = [0,1,2]
elif config == '6':
    #----------- configuration 5-----#
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
    x_list = [0,1,2]
else:
    sys.exit("Wrong configuration number!")

fname = func.__name__[:-4] + str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
+'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
fname = fname.replace('.','dot')

if error_type == 'hetero':
    fname = 'hetero_'+fname
#------------------------------------------------------------------------

x_list_dense = np.arange(-1,3,0.1)

#----------------------True-------------------------------------------#
B1 = pd.read_csv('./../data/{}/B_true.csv'.format(fname), header = None).values
alpha1 = pd.read_csv('./../data/{}/alpha_true.csv'.format(fname), header = None).values
sigma = pd.read_csv('./../data/{}/sigma_true.csv'.format(fname), header = None).values.ravel()

df_NPMLE = density_ridgeline_plot(x_list_dense,sigma,\
                                    B1,alpha1,fname,func = lin_func, approach = 'true')  


#----------------------NPMLE-CV --------------------------------------#
B2 = pd.read_csv('./../data/{}/B_NPMLE.csv'.format(fname), header = None).values
alpha2 = pd.read_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), header = None).values
sigma_cv = pd.read_csv('./../data/{}/sigma_CV.csv'.format(fname), header = None).values.ravel()

df_NPMLE = density_ridgeline_plot(x_list_dense,sigma_cv,\
                                    B2,alpha2,fname,func = lin_func, approach = 'NPMLE-CV')  


#-----------------------   EM-true  -------------------------------------#
B4 = pd.read_csv('./../data/{}/B_EM.csv'.format(fname), header = None).values
alpha4 = pd.read_csv('./../data/{}/alpha_EM.csv'.format(fname), header = None).values
sigma_EM = pd.read_csv('./../data/{}/sigma_EM.csv'.format(fname), header = None).values.ravel()

df_NPMLE = density_ridgeline_plot(x_list_dense,sigma_EM,\
                                    B4,alpha4,fname,func = lin_func, approach = 'EM-true')  

#-------------------------------------------------------------------#
