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
    exp_type = sys.argv[2]

if exp_type == 'discrete':
    # directly run with reported sigma value 
    # (reported sigma were previously chosen by cross-validation)
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + '0.47'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '2' + ' ' + '0.47'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + '0.48'
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'discrete_cv':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '2' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')


elif exp_type == 'hetero_discrete':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' '+'0.8' + ' '+'1.0' + ' ' + \
    '500' + ' ' + '1' + ' ' + '0.54'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '0.8' + ' '+'1.0' + ' ' +\
    '500' + ' ' + '2' + ' ' + '0.54'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '0.8' + ' '+'1.0' + ' ' +\
    '500' + ' ' + '3' + ' ' + '0.58'
    os.system("python " + filename + params_ + '&')
    
elif exp_type == 'hetero_discrete_cv':
    # re-run cross-validation
    params_ = ' '+'0.5' + ' '+'0.8' + ' '+'1.0' + ' ' + '500' + ' ' + '1' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '0.8' + ' '+'1.0' + ' ' +'500' + ' ' + '2' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')
    params_ = ' '+'0.5' + ' ' + '0.8' + ' '+'1.0' + ' ' +'500' + ' ' + '3' + ' ' + 'yes'
    os.system("python " + filename + params_ + '&')