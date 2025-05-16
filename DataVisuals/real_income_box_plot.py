import numpy as np
import matplotlib.pyplot as plt

# 1) Data
years = np.arange(2014, 2024)
age_bins = ["18–34","35–49","50–64","65–74","75+"]

median_vals = np.array([
    [823900, 827100, 833200, 841200, 860400, 874700, 904100, 941100, 895100, 903400],
    [1275700,1292500,1310200,1321500,1341600,1356100,1427000,1439600,1342200,1347500],
    [1151200,1174300,1201200,1222800,1248300,1271200,1341300,1363400,1272400,1293400],
    [ 474400, 485000, 498300, 507000, 513300, 529300, 544800, 553800, 523500, 538800],
    [ 456100, 462900, 470900, 475300, 481800, 490700, 504200, 507800, 480400, 489200],
], float)

q1_vals = np.array([
    [526600,524900,522700,524000,536400,547300,560200,592100,570400,571400],
    [942300,950000,953100,956900,968700,980100,1007800,1024100,962200,963100],
    [835500,847900,862200,879200,888800,905600,930900,948800,891400,904600],
    [351400,359200,370000,377800,384700,404100,419700,425800,401400,414600],
    [356700,360100,365300,367900,371800,378100,388200,391100,368300,377100],
], float)

q3_vals = np.array([
    [1132300,1140100,1152000,1164100,1189200,1205000,1261900,1296900,1224500,1237700],
    [1659200,1684600,1712700,1729900,1756100,1779700,1876000,1889500,1763200,1776400],
    [1544800,1579000,1614000,1640100,1674500,1712000,1811600,1843700,1718900,1753100],
    [670300,686900,705600,716100,722500,744400,765600,792500,744400,774000],
    [575000,586800,600700,607800,616800,633700,651600,660300,617600,627700],
], float)

# whiskers
iqr = q3_vals - q1_vals
lw  = q1_vals - 1.5 * iqr
hi  = q3_vals + 1.5 * iqr

# macro
inflation = np.array([0.6,0.5,0.3,1.1,0.8,0.8,0.4,1.9,7.7,3.3])
deposit   = np.array([-0.0294,-0.6966,-0.6516,-0.65,-0.65,-0.6798,-0.6336,-0.547,-0.0067,2.9444])

# 2) Compute and fix LHS limits
net_min, net_max = lw.min(), hi.max()
y_lo = net_min * 0.95
y_hi = net_max * 1.05

# 3) Plot
fig, axes = plt.subplots(2, 3, figsize=(16,8), sharey=True)
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
        ax.set_ylabel("Real Income (DKK)")
    ax.set_xlabel("Year")

    # right-axis with explicit full range
    ax2 = ax.twinx()
    ax2.plot(years, inflation, marker='o', color='tab:red', label='Inflation (%)')
    ax2.plot(years, deposit,   marker='s', color='tab:blue',label='Deposit rate (%)')
    # explicitly set ticks so nothing is cut off
    all_rates = np.concatenate([inflation, deposit])
    ax2.set_yticks(np.linspace(all_rates.min(), all_rates.max(), 6))
    ax2.set_ylabel("Rates (%)")
    if idx == 0:
        h,l = ax2.get_legend_handles_labels()
        ax2.legend(h,l,loc='upper left',frameon=False)

# hide 6th
axes[5].axis('off')

plt.tight_layout()
plt.show()
