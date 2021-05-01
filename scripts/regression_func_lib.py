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
    return beta[0]*x[1] + (1+ beta[1]*x[1])**2


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

def sinusoid_func(x, beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    # beta is three-dimensional 
    return beta[1] * np.cos(beta[0]*x[1]) + beta[2] * np.sin(beta[0]*x[1])

def piecelin_func(x,beta):
    x = np.array(x).ravel()
    beta = np.array(beta).ravel()
    return beta[0] + beta[1]*max(x[1] - beta[2], 0)