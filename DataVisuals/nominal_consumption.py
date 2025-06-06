import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1) Build the DataFrame (age groups ↔ years)
data = {
    "Below 30 years":   [219962, 207217, 203565, 210303, 213993, 207531, 206749, 239745, 242252],
    "30–44":            [346150, 339116, 352492, 359070, 365991, 342659, 376211, 397466, 405160],
    "45–59":            [357770, 352887, 365179, 397173, 394493, 384984, 405594, 439810, 457459],
    "60–74":            [321192, 302153, 303934, 319290, 313968, 327997, 337954, 356138, 389240],
    "75 and above":     [200558, 207335, 221112, 231014, 242708, 251134, 259200, 257745, 277125],
}
years = ["2015","2016","2017","2018","2019","2020","2021","2022","2023"]

df = pd.DataFrame(data, index=years)

# 2) Plot
fig, ax = plt.subplots(figsize=(8,5))

for year in df.index:
    ax.plot(df.columns, df.loc[year], marker='o', label=year)

# 3) Format y-axis in thousands
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x/1000:.0f}k"))

# 4) Labels, legend, grid
ax.set_xlabel("Age Group")
ax.set_ylabel("Consumption (DKK, thousands)")
#ax.set_title("Life-Cycle Consumption Patterns (2015–2023)")
ax.grid(True, linestyle=":", alpha=0.6)

ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.legend(title="Year", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.savefig("consumption_lifecycle_DST.png", dpi=300)
plt.show()
