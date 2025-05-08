# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:12:58 2025

@author: JVJ
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


##### Money Holdings, INDKP201  ####
# ------------------------------------------------------------------
# 1.  Raw quartiles (20–24-year-olds) 
# ------------------------------------------------------------------
q1 = np.array([
    91200, 90500, 90400, 88900, 89300, 89900, 93000, 96000,
    94800, 87500, 84300, 80200, 78300, 77300, 77500, 78000,
    78900, 79800, 81500, 82100, 83700, 89800, 85000, 84500
])

median = np.array([
    126500, 125800, 125700, 124600, 127100, 128200, 130900, 134000,
    133500, 130700, 126800, 120600, 117000, 116000, 115600, 116100,
    117600, 118900, 122400, 123500, 126200, 135400, 128100, 129300
])

q3 = np.array([
    175500, 174500, 174100, 172000, 175800, 176400, 180200, 184500,
    184300, 178500, 174800, 166700, 161300, 160700, 161300, 162000,
    163900, 165600, 170500, 171700, 177800, 186000, 175300, 178300
])

# ------------------------------------------------------------------
# 2.  Pool the quartiles across all years (simple average)
# ------------------------------------------------------------------
avg_q1     = q1.mean()
avg_median = median.mean()
avg_q3     = q3.mean()

# ------------------------------------------------------------------
# 3.  Fit the two-parameter log-normal
#     σ = (ln Q3 − ln Q1) / (2 * 0.674)
#     μ = ln(median)
# ------------------------------------------------------------------
sigma = (np.log(avg_q3) - np.log(avg_q1)) / (2 * 0.674)
mu    = np.log(avg_median)

print(f"Pooled parameters →  μ = {mu:.4f},  σ = {sigma:.4f}")
# >>> μ = 11.7744,  σ = 0.5262   (rounded)

    

##### Debt, FORMUE111  ####
#age group 18-24 in the data range 2014-2023
#using all passives 2014 definition

# averaged inputs
avg_Q3        =  6_300          # kr.
avg_mean_all  = 46_154.667      # kr.
p0            = 0.50            # assume half the households have zero debt

# derived targets
mean_pos = avg_mean_all / (1 - p0)        # = 92 309 kr.
q_pos    = (0.75 - p0) / (1 - p0)         # = 0.50
z_q      = 0.0                            # Φ⁻¹(0.50)

# parameters
mu    = np.log(avg_Q3)                    # 8.7483
sigma = np.sqrt(2 * (np.log(mean_pos) - mu))   # 2.3172

print(f"μ = {mu:.4f},   σ = {sigma:.4f}")

#### WAGE, LONS60 ###
## AGE 18-19 data 2002-2023

q1 = np.array([
    108.39, 109.53, 110.26, 113.94, 117.40, 122.16, 125.78, 130.81,
    129.79, 133.17, 137.63, 138.69, 140.53, 142.14, 142.54, 146.03,
    149.89, 153.41, 157.34, 161.10, 165.72, 171.13
])

median = np.array([
    118.85, 120.48, 121.59, 125.72, 128.77, 133.63, 137.53, 140.92,
    140.16, 143.63, 148.24, 150.87, 153.36, 154.80, 154.95, 158.45,
    162.66, 166.01, 170.22, 175.78, 180.66, 186.20
])

q3 = np.array([
    137.25, 134.48, 136.50, 141.99, 146.49, 152.21, 157.15, 158.36,
    157.29, 159.61, 165.86, 166.85, 171.93, 171.28, 170.47, 174.60,
    179.38, 182.16, 187.69, 195.39, 200.79, 207.18
])


# ------------------------------------------------------------------
# 2.  Calculate μ and σ for each year
#     σ = (ln Q3 − ln Q1) / (2 * 0.67448975)
#     μ = ln(Median)
# ------------------------------------------------------------------
avg_q1     = q1.mean()
avg_median = median.mean()
avg_q3     = q3.mean()

sigma_pool = (np.log(avg_q3) - np.log(avg_q1)) / (2 * 0.67448975)
mu_pool    = np.log(avg_median)

print("\nPooled μ, σ (2002-2023):")
print(f"μ = {mu_pool:.4f},  σ = {sigma_pool:.4f}")

import torch

# your pooled log‐normal parameters
mu_w,   sigma_w = 11.7362, 0.5226   # for wages:   ln w ∼ N(mu_w,   sigma_w^2)
mu_m,   sigma_m =  5.0026, 0.1446   # for money:   ln m ∼ N(mu_m,   sigma_m^2)

# draw an example cross‐section of N agents
N = 10000
eps_w = torch.randn(N)
eps_m = torch.randn(N)
w0    = torch.exp(mu_w   + sigma_w   * eps_w)   # initial wages
m0    = torch.exp(mu_m   + sigma_m   * eps_m)   # initial money holdings

# compute the median‐wage scale factor
scale = torch.exp(torch.tensor(mu_w))   # = median(w0)

# rescale so that median(w0_new)=1
w0_new = w0 / scale       # ∼ lognormal(0, sigma_w^2)
m0_new = m0 / scale       # ∼ lognormal(mu_m-mu_w, sigma_m^2)





