# -*- coding: utf-8 -*-
"""
Combined script: fit pooled log‑normals for money, debt, wage, assets; then rescale so wage mean=1.
Created on Tue May  6 10:12:58 2025
@author: JVJ
"""

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------------
# 1) Money holdings (B.1 Indestående i pengeinstitutter), 18+ pooled 2014–2023
# ------------------------------------------------------------------
# DST Table B.1 for all individuals aged 18+: lower quartile, median, upper quartile (fixed prices)
q1_m = np.array([12738, 13522, 14209, 15035, 16436, 16981, 22396, 22342, 20146, 20836])
median_m = np.array([38451, 40644, 43216, 45417, 49888, 53597, 70702, 70387, 62433, 63519])
q3_m = np.array([140686, 143533, 150129, 154327, 165067, 177628, 208308, 196418, 183954, 191880])
# mean of money holdings
mean_m = np.array([159836, 162803, 168729, 172449, 182334, 192330, 204949, 198614, 191309, 201218])

# compute log-normal parameters from pooled percentiles
z = norm.ppf(0.75)
avg_q1_m, avg_med_m, avg_q3_m = q1_m.mean(), median_m.mean(), q3_m.mean()
sigma_m = (np.log(avg_q3_m) - np.log(avg_q1_m)) / (2*z)
mu_m    = np.log(avg_med_m)

# ------------------------------------------------------------------
# 2) Debt (FORMUE111), age 18-24, pooled
# ------------------------------------------------------------------
avg_Q3_d     = 6300            # 75th percentile (kr.)
avg_mean_all = 46154.667       # overall mean (kr.)
p0 = 0.50                      # share zero debt
mean_pos = avg_mean_all/(1-p0) # mean of positives
mu_d    = np.log(avg_Q3_d)
sigma_d = np.sqrt(2*(np.log(mean_pos) - mu_d))

# ------------------------------------------------------------------
# 3) Wage (LONS60), age 18-19, years pooled
# ------------------------------------------------------------------
q1_w = np.array([
    108.39, 109.53, 110.26, 113.94, 117.40, 122.16, 125.78, 130.81,
    129.79, 133.17, 137.63, 138.69, 140.53, 142.14, 142.54, 146.03,
    149.89, 153.41, 157.34, 161.10, 165.72, 171.13
])
median_w = np.array([
    118.85, 120.48, 121.59, 125.72, 128.77, 133.63, 137.53, 140.92,
    140.16, 143.63, 148.24, 150.87, 153.36, 154.80, 154.95, 158.45,
    162.66, 166.01, 170.22, 175.78, 180.66, 186.20
])
q3_w = np.array([
    137.25, 134.48, 136.50, 141.99, 146.49, 152.21, 157.15, 158.36,
    157.29, 159.61, 165.86, 166.85, 171.93, 171.28, 170.47, 174.60,
    179.38, 182.16, 187.69, 195.39, 200.79, 207.18
])
avg_q1_w, avg_med_w, avg_q3_w = q1_w.mean(), median_w.mean(), q3_w.mean()
sigma_w = (np.log(avg_q3_w) - np.log(avg_q1_w)) / (2*z)
mu_w    = np.log(avg_med_w)

# ------------------------------------------------------------------
# 4) Assets (Aktiver I alt), age 18-24, years pooled 2014–2023
# ------------------------------------------------------------------
q1_a = np.array([ 8861, 9358, 9827, 10591, 11735, 12517, 18543, 19146, 16962, 17232 ])
median_a = np.array([ 29688, 31821, 34212, 36659, 39510, 42722, 58009, 61444, 54121, 54665 ])
q3_a = np.array([ 91783, 97809, 102897, 108048, 111965, 118436, 147280, 155134, 137955, 140711 ])
avg_q1_a, avg_med_a, avg_q3_a = q1_a.mean(), median_a.mean(), q3_a.mean()
sigma_a = (np.log(avg_q3_a) - np.log(avg_q1_a)) / (2*z)
mu_a    = np.log(avg_med_a)

# ------------------------------------------------------------------
# 5) Rescale so that initial mean wage = 1
# ------------------------------------------------------------------
# log-normal mean: E[X]=exp(mu+0.5*sigma^2)
mean_w = np.exp(mu_w + 0.5*sigma_w**2)
shift  = np.log(mean_w)

# rescaled parameters (wage units)
mu_w0,   sigma_w0   = mu_w   - shift, sigma_w
mu_m0,   sigma_m0   = mu_m   - shift, sigma_m
mu_d0,   sigma_d0   = mu_d   - shift, sigma_d
mu_a0,   sigma_a0   = mu_a   - shift, sigma_a

# ------------------------------------------------------------------
# 6) Output
# ------------------------------------------------------------------
print(f"Rescaled wage:   μ_w0 = {mu_w0:.4f}, σ_w0 = {sigma_w0:.4f}")
print(f"Rescaled money:  μ_m0 = {mu_m0:.4f}, σ_m0 = {sigma_m0:.4f}")
print(f"Rescaled debt:   μ_d0 = {mu_d0:.4f}, σ_d0 = {sigma_d0:.4f}")
print(f"Rescaled assets: μ_a0 = {mu_a0:.4f}, σ_a0 = {sigma_a0:.4f}")
