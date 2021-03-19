#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:45:09 2021

@author: hanshengjiang
"""

from package_import import *

def lin_func(x,beta):
    return np.dot(x,beta)


def poly_func(x,beta):
    return (beta[0]**2)*x[0] + 0.5*beta[0]*beta[1]*x[1]+0.5*beta[1]*x[1]**2+beta[0]*beta[1]


def exp_func(x,beta):
    # two dimensional models
    return beta[0] + np.exp(-beta[1]*x[1])