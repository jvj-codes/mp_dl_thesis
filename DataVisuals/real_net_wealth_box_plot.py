# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:16:46 2025

@author: JVJ
"""

import numpy as np
import matplotlib.pyplot as plt

# 1) Data definitions
years = np.arange(2014, 2024)
age_bins = ["18–34", "35–49", "50–64", "65–74", "75–84", "85+"]

median_vals = np.array([
    [ 197421,  228536,  247487,  269796,  276653,  307105,  377404,  408268,  353349,  351181],
    [1590990, 1737010, 1871367, 2007512, 2029843, 2258977, 2459682, 2668349, 2243693, 2249821],
    [4007647, 4132346, 4342172, 4518962, 4548048, 4925685, 5259657, 5555674, 4703578, 4956737],
    [2879371, 3046878, 3225870, 3394118, 3461521, 3679140, 3864996, 4073378, 3587374, 3811680],
    [1803295, 1890181, 1969877, 2091843, 2192046, 2331957, 2462680, 2636172, 2444641, 2626651],
    [1351529, 1385930, 1429698, 1480021, 1502407, 1559649, 1613962, 1683702, 1570770, 1667075],
], float)

q1_vals = np.array([
    [ -18856,  -4163,   2518,  11723,  14291,  20722,  43953,  42639,  36123,  37297],
    [ 425299,  484167,  556629,  625291,  646927,  772384,  889179,  924698,  687961,  662002],
    [1568405, 1641625, 1787734, 1907368, 1948778, 2205552, 2426271, 2567449, 2057140, 2175563],
    [ 914393,  997743, 1101276, 1195433, 1250690, 1385247, 1516433, 1641241, 1437222, 1569782],
    [ 413665,  442871,  473532,  520719,  560465,  624904,  695726,  770854,  725941,  799175],
    [ 269949,  279149,  290186,  308285,  316289,  344106,  374487,  403065,  387279,  411987],
], float)

q3_vals = np.array([
    [  686051,  761663,  798992,  842918,  857164,  915879, 1036782, 1130782, 1018870, 1026816],
    [ 3484099, 3728130, 3894673, 4081435, 4109172, 4417436, 4716617, 5079748, 4400996, 4532789],
    [ 8097660, 8286534, 8500597, 8707066, 8679371, 9185054, 9600071, 9995177, 8548956, 9083680],
    [ 6373071, 6639997, 6886092, 7128501, 7146345, 7433547, 7624379, 7870813, 6915571, 7304362],
    [ 4373954, 4572820, 4755825, 5004776, 5150784, 5415917, 5638434, 5945669, 5408106, 5807797],
    [ 3400158, 3529110, 3631385, 3741386, 3823859, 3939578, 4046111, 4240071, 3863990, 4134716],
], float)

# compute whiskers
iqr = q3_vals - q1_vals
lw  = q1_vals - 1.5 * iqr
hi  = q3_vals + 1.5 * iqr

# macro series
inflation = np.array([0.6, 0.5, 0.3, 1.1, 0.8, 0.8, 0.4, 1.9, 7.7, 3.3])
deposit   = np.array([-0.0294, -0.6966, -0.6516, -0.65, -0.65, -0.6798, -0.6336, -0.547, -0.0067, 2.9444])

# determine common y-axis limits for net-wealth
net_min = lw.min()
net_max = hi.max()

# 2) Create 2x3 grid with shared y-axis for net-wealth
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for idx, ax in enumerate(axes):
    # build each box
    boxes = []
    for j, yr in enumerate(years):
        boxes.append({
            'med':    median_vals[idx, j],
            'q1':     q1_vals[idx, j],
            'q3':     q3_vals[idx, j],
            'whislo': lw[idx, j],
            'whishi': hi[idx, j],
            'fliers': []
        })
    ax.bxp(boxes, positions=years, widths=0.6, showfliers=False)
    ax.set_title(f"Age {age_bins[idx]}")
    ax.set_xlabel("Year")
    if idx % 3 == 0:
        ax.set_ylabel("Real Net-Wealth (DKK)")
    ax.set_ylim(net_min * 1.05, net_max * 1.05)  # uniform scale

    # right-hand axis for macro
    ax2 = ax.twinx()
    ax2.plot(years, inflation, marker='o', color='tab:red',   label='Inflation (%)')
    ax2.plot(years, deposit,   marker='s', color='tab:blue',  label='Deposit rate (%)')
    if idx == 0:
        h, l = ax2.get_legend_handles_labels()
        ax2.legend(h, l, loc='upper left', frameon=False)
    # set independent limits
    low = min(inflation.min(), deposit.min())
    high= max(inflation.max(), deposit.max())
    ax2.set_ylim(low * 1.2, high * 1.2)
    ax2.set_ylabel("Rates (%)")

#plt.suptitle("Real Net-wealth, Inflation & Deposit Rates")
plt.tight_layout(rect=[0,0,1,0.90])
plt.show()
