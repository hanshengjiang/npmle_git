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
    input_param = sys.argv[2]
    
os.system("python " + filename + input_param)