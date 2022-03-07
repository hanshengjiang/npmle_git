#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:55:44 2021

@author: hanshengjiang
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from IPython.display import display
import scipy.linalg
from itertools import combinations
from numpy import linalg
from numpy.linalg import matrix_rank
from scipy.sparse.linalg import svds, eigs
import random
import time
from scipy.optimize import minimize
import math
import matplotlib.mlab as mlab
from scipy import integrate
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import scipy.stats
import csv
import statistics as stats
import pandas as pd
import colorsys
from scipy.integrate import quad
from scipy.integrate import dblquad
from sympy import sin, cos, symbols
import pandas as pd
import itertools


# change plot fonts
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "cm",
     "font.size": 20}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]