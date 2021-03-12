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
    # params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + 'yes'
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '1' + ' ' + '0.4'
    os.system("python " + filename + params_ + '&')
    # params_ = ' '+'0.3' + ' ' + '300' + ' ' + '2' + ' ' + 'yes'
    params_ = ' '+'0.3' + ' ' + '300' + ' ' + '2' + ' ' + '0.2'
    os.system("python " + filename + params_ + '&')
    # params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + 'yes'
    params_ = ' '+'0.5' + ' ' + '500' + ' ' + '3' + ' ' + '0.4'
    os.system("python " + filename + params_ + '&')