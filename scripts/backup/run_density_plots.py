#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:14:40 2021

@author: hanshengjiang
"""

from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
import sys


if __name__ == "__main__":
    sigma = sys.argv[1] #sys_argv[0] is the name of the .py file
    sigma_cv = sys.argv[2]
    
n = 500
#----------- configuration 1-----#
#b1 = (1,1)
#b2 = (4,-1)
#b3 = (-1,0.5)
#pi1 = 0.5
#pi2 = 0.5
##----------- configuration 2-----#
#b1 = (0,1)
#b2 = (1,1)
#b3 = (0,1)
#pi1 = 0.3
#pi2 = 0.7
##----------- configuration 3-----#
#b1 = (3,-1)
#b2 = (1,1.5)
#b3 = (-1,0.5)
#pi1 = 0.3
#pi2 = 0.3


fname = str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
+'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'

# generate test data


#------------------------------------------------------------#    
#    
#-------------------1: CGM with known sigma----------------#
    
f1, B1, alpha1, L_rec1, L_final1 = NPMLE_FW(X,y,iter,sigma)
    
    
#------------------------------------------------------------#    
#    
#-------------------2: CGM without knowing sigma----------------#



f2, B2, alpha2, L_rec2, L_final2 = NPMLE_FW(X,y,iter,sigma_cv)
    











