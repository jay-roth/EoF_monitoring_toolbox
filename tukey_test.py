# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:43:24 2023

@author: Jason.Roth
"""

import scipy as sp
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

g0 = [24.5, 23.5, 26.4, 27.1, 29.9]
g1 = [28.4, 34.2, 29.5, 32.2, 30.1]
g2 = [26.1, 28.3, 24.3, 26.2, 27.8]


tr = sp.stats.tukey_hsd(g0,g1,g2)
tr.pvalue