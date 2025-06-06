# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:53 2025

@author: JVJ
"""

#%load_ext autoreload
#%autoreload 2

from WealthIncomeMPModel import WealthIncomeModelClass
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
K_time = 20 #20.0
model_DL['DeepSimulate'] = WealthIncomeModelClass(algoname='DeepSimulate', par={'Nstates_fixed': 1}, device=device,train={'K_time': K_time})
model_DL['DeepSimulate'].solve(do_print=True)

# DeepFOC
K_time = 20 #20.0
model_DL['DeepFOC'] = WealthIncomeModelClass(algoname='DeepFOC', par={'Nstates_fixed': 1}, device=device,train={'K_time': K_time})
model_DL['DeepFOC'].solve(do_print=True)

model_DL['DeepSimulate'].train.K_time += 20


model_DL['DeepSimulate'].compute_euler_errors()

# 3. Inspect results
euler_errors = model_DL['DeepSimulate'].sim.euler_error.cpu().numpy()
print(euler_errors.shape)
print(np.sum(euler_errors > 0))
euler_errors = model_DL['DeepSimulate'].sim.euler_error.cpu().numpy().flatten()
# # DeepVPD
#K_time = 1.0
#model_DL['DeepVPD'] = WealthBehaviorModelClass(algoname='DeepVPD', device=device,train={'K_time': K_time})
#model_DL['DeepVPD'].solve(do_print=True)


figs, axes = plt.subplots(2, 3, figsize=(16, 9))

for model_key, label in model_DL.items():
    axes[0,0].plot(np.mean(model_DL[model_key].sim.actions[:, :, 0].cpu().numpy(), axis=1),label=model_key, lw = 2) # labor share
    axes[0,1].plot(np.mean(model_DL[model_key].sim.actions[:, :, 1].cpu().numpy(), axis=1), lw = 2) # debt share
    axes[1,0].plot(np.mean(model_DL[model_key].sim.actions[:, :, 2].cpu().numpy(), axis=1), lw = 2) # equity share

figs.suptitle("Actions")
axes[0,0].legend()
axes[0,0].set_title('Labor Supply')
axes[0,1].set_title('Illiquid Share')
axes[1,0].set_title('Liquid Share')


figs, axes = plt.subplots(3, 3, figsize=(16, 9))

for model_key, label in model_DL.items():
    axes[0,0].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  0].cpu().numpy(), axis=1), label=model_key, lw =2)
    axes[0,1].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  1].cpu().numpy(), axis=1), lw = 2) # labor share
    axes[1,0].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  2].cpu().numpy(), axis=1), lw = 2) # debt share
    axes[1,1].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  3].cpu().numpy(), axis=1), lw = 2) # debt share
    axes[1,2].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  4].cpu().numpy(), axis=1), lw = 2) # debt share
    axes[2,2].plot(np.mean(model_DL[model_key].sim.outcomes[:, :,  4].cpu().numpy(), axis=1), lw = 2) # debt share
    
figs.suptitle("Outcomes")

axes[0,0].legend()
axes[0,0].set_title('Consumption')
axes[0,1].set_title('Labor Supply')
axes[1,0].set_title('Ressources')
axes[1,1].set_title('Bonds')
axes[1,2].set_title('Illiquid Assets')
axes[2,2].set_title('New debt')


figs, axes = plt.subplots(3, 3, figsize=(16, 9))

for model_key, label in model_DL.items():
    axes[0,0].plot(np.mean(model_DL[model_key].sim.states[:, :, 0].cpu().numpy(), axis=1),label=model_key, lw = 2) # labor share
    axes[0,1].plot(np.mean(model_DL[model_key].sim.states[:, :, 1].cpu().numpy(), axis=1), lw = 2) # debt share
    axes[1,0].plot(np.mean(model_DL[model_key].sim.states[:, :, 2].cpu().numpy(), axis=1), lw = 2) # equity share
    axes[1,1].plot(np.mean(model_DL[model_key].sim.states[:, :, 3].cpu().numpy(), axis=1), lw = 2) # bond share
    axes[1,2].plot(np.mean(model_DL[model_key].sim.states[:, :, 4].cpu().numpy(), axis=1), lw = 2) # house share
    axes[2,1].plot(np.mean(model_DL[model_key].sim.states[:, :, 5].cpu().numpy(), axis=1), lw = 2) # bond share
    axes[2,2].plot(np.mean(model_DL[model_key].sim.states[:, :, 6].cpu().numpy(), axis=1), lw = 2) # bond share

figs.suptitle("States patient beta = 0.99")
axes[0,0].legend()
axes[0,0].set_title('Illiquid Assets Previous')
axes[0,1].set_title('Wage')
axes[1,0].set_title('Money Holdings')
axes[1,1].set_title('Inflation')
axes[1,2].set_title('Nominal Interest Rate')
axes[2,1].set_title('Return on Illiquid Assets')
axes[2,2].set_title('Debt')

