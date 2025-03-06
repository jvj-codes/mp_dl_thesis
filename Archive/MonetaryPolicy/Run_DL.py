# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:53 2025

@author: JVJ
"""

#%load_ext autoreload
#%autoreload 2

from rl_project.MonetaryPolicy import MonetaryPolicyModel
import numpy as np
import sympy
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.grid": True,
    "grid.color": "black",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 14                    # Set font size
})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from rl_project.EconDLSolvers import choose_gpu

# Hyperparameters
device = choose_gpu()
model_DL = {}


K_time = 3.0
model_DL['DeepVPD'] = MonetaryPolicyModel.MonetaryPolicyModelClass(algoname='DeepVPD', device=device,train={'K_time': K_time})
model_DL['DeepVPD'].solve(do_print=True)
