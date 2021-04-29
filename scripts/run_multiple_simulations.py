#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 01:11:55 2021

@author: hanshengjiang
"""

import sys
import os


if __name__ == "__main__":
    filename = sys.argv[1] #sys_argv[0] is the name of the .py file
    exp_type = sys.argv[2] #experiment type

if exp_type == 'discrete':
    # directly run with reported sigma value 
    # (reported sigma's were previously chosen by cross-validation)
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + '0.47'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '2' + ' ' + '0.47'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + '0.48'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'discrete_cv':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '2' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'hetero_discrete':
    #  directly run with sigma selected by CV 
    params_ = ' '+'0.3' + ' '+'0.5' + ' '+'0.7' + ' ' + \
    '500' + ' ' + '1' + ' ' + '0.27'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.3' + ' ' + '0.5' + ' '+'0.7' + ' ' +\
    '500' + ' ' + '2' + ' ' + '0.29'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.3' + ' ' + '0.5' + ' '+'0.7' + ' ' +\
    '500' + ' ' + '3' + ' ' + '0.31'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'hetero_discrete_cv':
    # re-run cross-validation
    params_ = ' '+'0.3' + ' '+'0.5' + ' '+'0.7' + ' ' + \
    '500' + ' ' + '1' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.3' + ' ' + '0.5' + ' '+'0.7' + ' ' +\
    '500' + ' ' + '2' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.3' + ' ' + '0.5' + ' '+'0.7' + ' ' +\
    '500' + ' ' + '3' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
        
elif exp_type == 'poly_cv':
    #  directly run with sigma selected by CV 
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '4' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')

elif exp_type == 'poly':
    # re-run cross-validation
   
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '4' + ' ' + '0.48'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
      
elif exp_type == 'exp_cv':
    #  directly run with sigma selected by CV 
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '5' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'exp':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '5' + ' ' + '0.52'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'sin_cv':
    #  directly run with sigma selected by CV 
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '6' + ' ' + 'yes'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')

elif exp_type == 'sin':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '6' + ' ' + '0.5'+ ' ' + '0.01' + ' '
    os.system("python " + filename + params_ + '&')
   