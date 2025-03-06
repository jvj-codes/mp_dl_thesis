# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:53 2025

@author: JVJ
"""

#%load_ext autoreload
#%autoreload 2

from WealthChannelsModel import WealthBehaviorModelClass
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
from EconDLSolvers import choose_gpu

# Hyperparameters
device = choose_gpu()
model_DL = {}

# DeepSimulate
K_time = 1.0
model_DL['DeepSimulate'] = WealthBehaviorModelClass(algoname='DeepSimulate', device=device,train={'K_time': K_time})
model_DL['DeepSimulate'].solve(do_print=True)

# DeepVPD
K_time = 3.0
model_DL['DeepVPD'] = WealthBehaviorModelClass(algoname='DeepVPD', device=device,train={'K_time': K_time})
model_DL['DeepVPD'].solve(do_print=True)


figs, axes = plt.subplots(2, 2, figsize=(16, 9))

for model_key, label in model_DL.items():
    axes[0,0].plot(np.mean(model_DL[model_key].sim.states[:, :, 0].cpu().numpy(), axis=1), label=model_key, lw =2)
    axes[0,1].plot(np.mean(model_DL[model_key].sim.outcomes[:, :, 0].cpu().numpy(), axis=1), lw = 2)
    axes[1,0].plot(np.mean(model_DL[model_key].sim.outcomes[:, :, 1].cpu().numpy(), axis=1), lw = 2)
    axes[1,1].plot(np.mean(model_DL[model_key].sim.outcomes[:, :, 2].cpu().numpy(), axis=1), lw = 2)

axes[0,0].legend()
axes[0,0].set_title('Money Holdings')
axes[0,1].set_title('Consumption')
axes[1,0].set_title('House Holdings')
axes[1,1].set_title('Labor')