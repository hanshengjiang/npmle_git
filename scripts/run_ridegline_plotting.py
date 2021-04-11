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


#-------------------Ridgeline plot ----------------#
B2 = pd.read_csv('./../data/{}/B_NPMLE.csv'.format(fname), header = None).values
alpha2 = pd.read_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), header = None).values
x_list_dense = np.arange(-1,3,0.1)

df_fitted, df_true = density_ridgeline_plot(x_list_dense,b1,b2,b3,pi1,pi2,sigma,sigma_cv,B2,alpha2,fname,func = lin_func)  
#------------------------------------------------------------#
