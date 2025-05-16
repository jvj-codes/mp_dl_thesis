import numpy as np
import matplotlib.pyplot as plt

# 1) Data
years = np.arange(2014, 2024)
age_bins = ["18–34","35–49","50–64","65–74","75+"]

median_vals = np.array([
    [699700,705700,712700,727700,750400,768600,797900,845700,866500,903400],
    [1083500,1102700,1120700,1143200,1170200,1191900,1259300,1293900,1299300,1347500],
    [977800,1001900,1027500,1057900,1088800,1117100,1183700,1225400,1231800,1293400],
    [402900,413700,426300,438700,447800,465100,480800,497800,506800,538800],
    [387400,394800,402800,411200,420200,431200,444900,456500,465000,489200],
], float)

q1_vals = np.array([
    [447200,448000,447100,453400,467900,481000,494300,532200,552200,571400],
    [800400,810500,815300,827900,845000,861400,889400,920500,931400,963100],
    [709700,723300,737600,760700,775300,795800,821500,852900,862900,904600],
    [298400,306500,316500,326900,335500,355200,370400,382700,388500,414600],
    [302900,307100,312500,318200,324400,332200,342600,351600,356600,377100],
], float)

q3_vals = np.array([
    [961800,972700,985400,1007200,1037200,1059100,1113700,1165800,1185500,1237700],
    [1409200,1437300,1465000,1496800,1531700,1564000,1655500,1698400,1706800,1776400],
    [1312100,1347200,1380500,1419000,1460600,1504400,1598700,1657300,1663900,1753100],
    [569200,586100,603600,619500,630300,654200,675600,712400,720600,774000],
    [488300,500700,513800,525800,538000,557000,575000,593500,597800,627700],
], float)

iqr = q3_vals - q1_vals
lw  = q1_vals - 1.5 * iqr
hi  = q3_vals + 1.5 * iqr

inflation = np.array([0.6,0.5,0.3,1.1,0.8,0.8,0.4,1.9,7.7,3.3])
deposit   = np.array([-0.0294,-0.6966,-0.6516,-0.65,-0.65,-0.6798,-0.6336,-0.547,-0.0067,2.9444])

# 2) Shared nominal-income limits
y_lo, y_hi = lw.min()*0.95, hi.max()*1.05

# 3) Fixed right-axis limits and ticks
rate_min = min(inflation.min(), deposit.min()) * 1.1
rate_max = max(inflation.max(),   deposit.max())   * 1.1
rate_ticks = np.linspace(rate_min, rate_max, 6)

# 4) Plot
fig, axes = plt.subplots(2,3, figsize=(16,8), sharey=True)
axes = axes.flatten()

for idx, ax in enumerate(axes[:5]):
    # boxplot
    boxes = []
    for j, yr in enumerate(years):
        boxes.append({
            'med':    median_vals[idx,j],
            'q1':     q1_vals[idx,j],
            'q3':     q3_vals[idx,j],
            'whislo': lw[idx,j],
            'whishi': hi[idx,j],
            'fliers': []
        })
    ax.bxp(boxes, positions=years, widths=0.6, showfliers=False)
    ax.set_title(f"Age {age_bins[idx]}")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylim(y_lo, y_hi)
    if idx % 3 == 0:
        ax.set_ylabel("Nominal Income (DKK)")
    ax.set_xlabel("Year")

    # right-hand axis
    ax2 = ax.twinx()
    ax2.plot(years, inflation, marker='o', color='tab:red',   label='Inflation (%)',   linewidth=2)
    ax2.plot(years, deposit,   marker='s', color='tab:blue',  label='Deposit rate (%)', linewidth=2)
    ax2.set_ylim(rate_min, rate_max)
    ax2.set_yticks(rate_ticks)
    ax2.set_ylabel("Rates (%)")
    if idx == 0:
        h,l = ax2.get_legend_handles_labels()
        ax2.legend(h,l,loc='upper left',frameon=False)

# blank extra panel
axes[5].axis('off')

plt.tight_layout()
plt.show()
