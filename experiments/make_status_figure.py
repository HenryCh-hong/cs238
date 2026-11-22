"""Main status-update figure: cumulative net utility minus free-only, lambda = 1 (from the replay log)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs" / "default_log.csv.gz"  # git-ignored; written by experiments.run_replay --tag default
OUT = ROOT / "results" / "figures" / "main_cumulative_lam1.pdf"
lg = pd.read_csv(LOG, parse_dates=["date"])
w = lg.pivot(index="date", columns="policy", values="net_utility").sort_index()
SER = [("optimal_voi", "Optimal VOI (= myopic, fixed order)", "#2a78d6", "-"),
       ("always_buy", "Always buy", "#eda100", (0, (6, 3))),
       ("random", "Random", "#e87ba4", (0, (1.5, 2.5)))]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9ced6", "axes.labelcolor": "#47505e", "xtick.color": "#47505e",
                     "ytick.color": "#47505e", "axes.grid": True, "grid.color": "#e6e9ee", "grid.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(7.2, 1.02))
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-30"), color="#1d5fd1", alpha=0.07, lw=0)
ax.text(pd.Timestamp("2020-02-20"), 150, "Mar–Apr 2020", ha="right", va="bottom", fontsize=7, color="#737c8b")
NUDGE = {"optimal_voi": 3, "always_buy": -3, "random": 0}  # keep the two close end labels apart
for key, label, col, ls in SER:
    cum = ((w[key] - w["free_only"]) * 1e4).cumsum()
    ax.plot(cum.index, cum.values, color=col, ls=ls, lw=1.5, label=label)
    ax.annotate(f"{cum.iloc[-1]:,.0f}".replace("-", "−"), (cum.index[-1], cum.iloc[-1]), xytext=(4, NUDGE[key]),
                textcoords="offset points", va="center", fontsize=7, color="#47505e")
ax.axhline(0, color="#737c8b", lw=0.8)
ax.set_ylim(-13000, 1200)
ax.set_ylabel("cumulative bp")
ax.set_xlim(w.index[0], w.index[-1] + pd.Timedelta(days=330))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "−")))
ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print(f"saved {OUT.relative_to(ROOT)}")
