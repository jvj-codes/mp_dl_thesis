import numpy as np
import matplotlib.pyplot as plt

# --- 1) Data definitions ---

years = np.arange(2014, 2024)
age_bins = ["18–34", "35–49", "50–64", "65–74", "75–84", "85+"]

median_vals = np.array([
    [168.154, 195442, 212707, 234193, 242042, 270790, 334392, 372935, 350926, 351181],
    [1355137,1485465,1608381,1742597,1775895,1991848,2179360,2437413,2228313,2249821],
    [3413540,3533919,3731962,3922630,3979054,4343214,4660228,5074850,4671334,4956737],
    [2452524,2605642,2772535,2946223,3028460,3244074,3424512,3720840,3562783,3811680],
    [1535970,1616453,1693048,1815799,1917806,2056199,2182015,2408020,2427882,2626651],
    [1151175,1185226,1228782,1284715,1314445,1375218,1430024,1537983,1560002,1667075],
], float)

q1_vals = np.array([
    [ -16061,  -3560,   2164,  10176,  12503,  18272,  38943,  38949,  35875,  37297],
    [ 362253, 414052, 478405, 542777, 565992, 681048, 787843, 844668, 683245, 662002],
    [1335899,1403892,1536501,1655668,1704971,1944741,2149755,2345245,2043038,2175563],
    [ 778840, 853254, 946513,1037681,1094220,1221439,1343609,1499197,1427369,1569782],
    [ 352342, 378736, 406986, 452004, 490347, 551008, 616436, 704139, 720965, 799175],
    [ 229931, 238724, 249406, 267603, 276719, 303415, 331808, 368181, 384624, 411987],
], float)

q3_vals = np.array([
    [1167461,1292907,1358001,1443527,1466448,1575636,1725654,1942618,1886970,1911798],
    [4055151,4283642,4486994,4730302,4797358,5216888,5581757,6141575,5730448,6044597],
    [8107275,8319499,8566071,8850738,8873339,9348583,9722369,10348140,9695509,10377039],
    [4484786,4753611,4990418,5287979,5439803,5756225,5981190,6445724,6252680,6643431],
    [3286796,3424387,3556639,3732152,3841851,4007250,4139008,4489445,4477085,4869696],
    [1334610,1391112,1437102,1491095,1543076,1613030,1676686,1803525,1784198,1916229],
], float)

# whiskers
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
        ax.set_ylabel("Nominal Net-Wealth (DKK)")
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