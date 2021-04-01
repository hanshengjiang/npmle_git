#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:45:09 2021

@author: hanshengjiang
"""

from package_import import *

def lin_func(x,beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    return np.dot(x,beta)


def poly_func(x,beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    return (beta[0]+beta[1]*x)**2


def exp_func(x,beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    # two dimensional models
    return beta[0] + np.exp(beta[1]*x[1])

def sin_func(x,beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    # two dimensional models
    return beta[0] + np.sin(beta[1]*x[1])