#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:14:40 2021

@author: hanshengjiang
"""
'''
compare EM, CGM, with or without knowing sigma
'''
from package_import import *
from simulation_lib import *
import sys


if __name__ == "__main__":
    sigma = sys.argv[1] #sys_argv[0] is the name of the .py file
    
print(sigma)
X,y = generate_test_data(n,iter, b1, b2, b3,pi1,pi2,sigma)

#define a range of candidate sigma values
sigma_max = np.sqrt(stats.variance(np.reshape(y, (len(y),))))
sigma_min = 0.1
sigma_list = np.arange(sigma_min, sigma_max, 0.02)

