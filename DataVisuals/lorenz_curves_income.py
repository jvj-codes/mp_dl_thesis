import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# 1) Data
income_deciles = np.array([
    [ 62629,  65028,  64971,  68760,  68868,  71062,  64729,  33815,  44078,  58517,  61405,  72143,  75689,  70459,  77728,  75853,  78128,  78721,  83792,  88188,  95473,  88404,  96295],
    [ 98552, 103999, 107121, 111091, 114266, 118226, 121499, 123441, 126331, 133314, 135247, 138299, 140042, 142148, 143630, 144500, 146503, 151039, 154985, 160361, 166971, 170941, 176981],
    [114430, 120236, 123723, 128763, 132607, 137056, 140882, 144035, 147131, 155258, 157641, 160792, 163578, 166795, 169095, 171292, 174947, 179714, 184053, 190111, 196935, 201579, 208472],
    [129885, 135871, 139750, 146523, 150805, 155709, 159688, 163606, 167022, 176861, 179613, 182871, 185928, 190010, 192715, 195643, 200087, 205721, 210708, 218820, 227383, 231643, 238441],
    [144273, 150513, 154880, 163447, 168260, 173475, 177868, 182478, 186859, 198711, 202048, 205633, 209382, 214412, 217580, 221437, 226804, 233247, 238796, 249210, 259036, 262566, 269697],
    [158392, 165005, 169917, 180018, 185548, 191175, 196045, 201136, 206688, 220835, 224771, 228906, 233521, 239499, 243288, 248126, 254422, 261607, 267917, 280604, 291453, 294157, 301829],
    [173884, 181053, 186526, 197955, 204440, 210551, 215945, 221427, 228017, 245061, 249659, 254534, 260358, 267373, 271974, 277775, 285080, 293222, 300200, 315119, 327257, 329162, 337862],
    [193021, 200865, 207108, 219993, 227752, 234736, 240638, 246484, 253698, 274629, 280171, 285933, 293589, 301774, 307454, 314207, 322842, 331975, 339935, 357273, 371556, 372479, 383209],
    [221638, 230367, 237604, 252665, 262588, 270978, 277551, 283514, 290811, 318054, 325162, 332513, 342856, 352557, 360069, 368072, 378649, 388934, 398818, 419353, 437776, 437373, 452178],
    [338942, 347627, 356782, 383943, 416199, 447312, 466484, 446011, 437019, 500012, 516944, 533189, 560971, 573866, 607665, 615554, 641188, 648231, 688377, 719999, 779827, 769052, 834074],
], float)

years = np.arange(2001, 2024)

def lorenz_from_deciles(deciles):
    pop_share = np.concatenate([[0], np.linspace(0.1,1.0,10)])
    mass      = deciles * 0.1
    cum_inc   = np.concatenate([[0], np.cumsum(mass)])
    return pop_share, cum_inc / cum_inc[-1]

# 2) Colormap: blue → red
cmap = mpl.cm.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=years.min(), vmax=years.max())

# 3) Plot
fig, ax = plt.subplots(figsize=(8,6))

for yr in years:
    p, w = lorenz_from_deciles(income_deciles[:, yr-2001])
    ax.plot(p, w, color=cmap(norm(yr)), lw=1.8, alpha=0.9)

# Equality line
ax.plot([0,1],[0,1], "k--", lw=1, label="Line of equality")

# Legend proxies
early = Line2D([0],[0], color=cmap(norm(years[0])), lw=2, label=f"{years[0]} (earliest)")
late  = Line2D([0],[0], color=cmap(norm(years[-1])), lw=2, label=f"{years[-1]} (most recent)")

ax.legend(handles=[early, late, Line2D([0],[0], linestyle='--', color='k')], 
          labels=[f"{years[0]} (earliest)", f"{years[-1]} (most recent)", "Equality"],
          loc="lower right")

# 4) Colorbar anchored to this Axes
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=years[::3], pad=0.02)
cbar.set_label("Year")

#ax.set_title("Lorenz Curves of Income (2001→2023)")
ax.set_xlabel("Cumulative population share")
ax.set_ylabel("Cumulative income share")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
