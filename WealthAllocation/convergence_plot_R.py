# -*- coding: utf-8 -*-
"""
Visualise convergence logs (side-by-side panels) with a configurable legend.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 0.  user option: legend placement
# --------------------------------------------------
LEGEND_POS = "below"     # "inside"  ▸ bottom-right of left panel
                          # "below"   ▸ centred under both panels

# --------------------------------------------------
# 1.  Load and clean
# --------------------------------------------------
data_path = Path.cwd().parent / "data" / "Log.xlsx"
if not data_path.exists():
    raise FileNotFoundError(f"Could not find Log.xlsx at {data_path}")

dfs = []
xls = pd.ExcelFile(data_path)

for sheet in xls.sheet_names:
    df = xls.parse(sheet).dropna().iloc[:, :2]         # reward, time
    df.columns = ["reward", "time"]

    df = (df.astype(str)
            .apply(lambda col: col.str.replace(",", ".", regex=False))
            .astype(float)
          )

    model = "DeepVPD" if "vpd" in sheet.lower() else "DeepSimulate"
    beta  = float(sheet.split("=")[-1].strip())

    df["model"] = model
    df["beta"]  = beta
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# 2.  Plot
# --------------------------------------------------
sns.set(style="whitegrid", font_scale=1.1)
betas = sorted(df_all["beta"].unique())      # e.g. [0.95, 0.99]

fig, ax = plt.subplots(
    1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True
)

palette = sns.color_palette("deep", n_colors=df_all["model"].nunique())

for i, beta in enumerate(betas):
    sns.lineplot(
        ax=ax[i],
        data=df_all[df_all["beta"] == beta],
        x="time", y="reward",
        hue="model", style="model",
        markers=True, dashes=False,
        palette=palette,
        legend=(i == 0)                     # keep handles only on first axis
    )
    ax[i].set_title(fr"$\beta = {beta}$")
    ax[i].set_xlabel("Time (minutes)")
    ax[i].set_ylim(-50, 0)
    ax[i].grid(True)

ax[0].set_ylabel("Reward (R)")

# --------------------------------------------------
# 3.  Legend
# --------------------------------------------------
handles, labels = ax[0].get_legend_handles_labels()

if LEGEND_POS.lower() == "inside":
    ax[0].legend(
        handles, labels,
        loc="lower right",
        frameon=False
    )
elif LEGEND_POS.lower() == "below":
    # remove the small legend added by seaborn
    ax[0].get_legend().remove()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08)
    )
    fig.subplots_adjust(bottom=0.23)   # room for the global legend
else:
    raise ValueError("LEGEND_POS must be 'inside' or 'below'")

# --------------------------------------------------
# 4.  Save & show
# --------------------------------------------------
out_file = Path.cwd() / "convergence_side.pdf"
fig.savefig(out_file, bbox_inches="tight")
print(f"Figure written to {out_file}")

plt.show()
