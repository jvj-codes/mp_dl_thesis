# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:01:27 2025

@author: nuffz
"""

import numpy as np
import matplotlib.pyplot as plt

# Opret et fast grid
x_grid = np.linspace(0, 1, 15)
y_grid = np.linspace(0, 1, 15)
Xg, Yg = np.meshgrid(x_grid, y_grid)

# Generer simulerede datapunkter (Beta-distribution)
n_sim = 225
x_sim = np.random.beta(2, 5, size=n_sim)
y_sim = np.random.beta(2, 5, size=n_sim)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(Xg.flatten(), Yg.flatten(), marker='x', alpha=0.6, label='Fixed grid')
plt.scatter(x_sim, y_sim, marker='x', alpha=0.6, label='Simulated data')
#plt.title('Illustration of grid vs. simulerede punkter')
plt.xlabel('State dimension 1')
plt.ylabel('State dimension 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
