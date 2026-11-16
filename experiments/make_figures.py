"""Figures from saved experiment tables/logs (cheap; no model fitting).

    python -m experiments.make_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import resolve  # noqa: E402

FIG = resolve("results/figures")
TAB = resolve("results/tables")
LOGS = resolve("results/logs")

# Validated categorical palette (light mode), slots assigned to entities in a fixed order.
POLICY_COLORS = {
    "optimal_voi": "#2a78d6",
    "myopic_voi": "#eb6834",
    "fixed_order": "#1baf7a",
    "always_buy": "#eda100",
    "random": "#e87ba4",
    "free_only": "#008300",
}
POLICY_LABELS = {
    "optimal_voi": "Optimal 2-step VOI", "myopic_voi": "Myopic VOI", "fixed_order": "Fixed order (A→B)",
    "always_buy": "Always buy", "random": "Random", "free_only": "Free only",
}
# Coinciding series stay identifiable: distinct dash + marker per policy; drawn so optimal is on top.
POLICY_STYLE = {
    "always_buy": ("-", "D"), "random": ("-.", "v"), "fixed_order": (":", "^"),
    "myopic_voi": ("--", "s"), "optimal_voi": ("-", "o"),
}
DRAW_ORDER = ("always_buy", "random", "fixed_order", "myopic_voi", "optimal_voi")
ARM_COLORS = {"A": "#2a78d6", "B": "#eb6834", "AB": "#1baf7a"}
ARM_LABELS = {"A": "Buy RF (A) only", "B": "Buy GB (B) only", "AB": "Buy both"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8,
    })


def fig_cumulative() -> list[str]:
    made = []
    logs = sorted(LOGS.glob("exp1_lam*_log.csv.gz"))
    if not logs:
        return made
    fig, axes = plt.subplots(1, len(logs), figsize=(5.2 * len(logs), 3.6), squeeze=False)
    for ax, path in zip(axes[0], logs):
        lg = pd.read_csv(path, parse_dates=["date"])
        wide = lg.pivot(index="date", columns="policy", values="net_utility").sort_index()
        base = wide["free_only"]
        for pol in DRAW_ORDER:
            if pol in wide:
                ax.plot(wide.index, ((wide[pol] - base) * 1e4).cumsum(), color=POLICY_COLORS[pol],
                        linestyle=POLICY_STYLE[pol][0], label=POLICY_LABELS[pol])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.text(0.01, 0.02, "fixed-order and myopic VOI overlap optimal VOI where hidden", transform=ax.transAxes,
                fontsize=7, color=INK2)
        lam = path.name.split("_")[1].replace("lam", "")
        ax.set_title(f"λ = {lam}: cumulative net utility minus free-only")
        ax.set_ylabel("bp of capital (cumulative)")
    axes[0][0].legend(loc="best")
    fig.tight_layout()
    out = FIG / "fig1_cumulative_vs_free_only.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    made.append(out.name)
    return made


def fig_cost() -> list[str]:
    path = TAB / "exp3_cost_sensitivity.csv"
    if not path.exists():
        return []
    t = pd.read_csv(path)
    lams = sorted(t["lam"].unique())
    fig, axes = plt.subplots(2, len(lams), figsize=(5.2 * len(lams), 6.2), squeeze=False, sharex=True)
    for j, lam in enumerate(lams):
        s = t[t["lam"] == lam]
        for pol in DRAW_ORDER:
            g = s[s["policy"] == pol].sort_values("cost_bp")
            if g.empty:
                continue
            ls, mk = POLICY_STYLE[pol]
            kw = dict(color=POLICY_COLORS[pol], linestyle=ls, marker=mk, markersize=4.5)
            axes[0][j].plot(g["cost_bp"], g["mean_purchases"], label=POLICY_LABELS[pol], **kw)
            axes[1][j].plot(g["cost_bp"], g["mean_diff_bp"], **kw)
            if pol == "optimal_voi":  # one shaded band; every policy's CI is in exp3_cost_sensitivity.csv
                axes[1][j].fill_between(g["cost_bp"], g["ci_lo_bp"], g["ci_hi_bp"], color=POLICY_COLORS[pol],
                                        alpha=0.15, linewidth=0, label="95% CI (optimal VOI)")
        axes[0][j].set_title(f"λ = {lam:g}: forecasts bought per date")
        axes[0][j].set_ylabel("mean purchases per date")
        axes[1][j].set_title(f"λ = {lam:g}: net utility minus free-only")
        axes[1][j].set_ylabel("bp per date")
        axes[1][j].axhline(0, color=INK2, linewidth=0.8)
        axes[1][j].set_xlabel("query cost per forecast (bp, symlog scale)")
        for ax in axes[:, j]:
            ax.set_xscale("symlog", linthresh=0.01)
    axes[0][0].legend(loc="best")
    axes[1][0].legend(loc="lower left")
    fig.tight_layout()
    out = FIG / "fig2_cost_sensitivity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return [out.name]


def fig_calibration() -> list[str]:
    path = TAB / "exp2_voi_calibration.csv"
    if not path.exists():
        return []
    t = pd.read_csv(path)
    t = t[t["bin"].astype(str) != "all"]
    t = t[t["arm"].isin(ARM_COLORS)]
    lams = sorted(t["lam"].unique())
    fig, axes = plt.subplots(1, len(lams), figsize=(5.2 * len(lams), 4.0), squeeze=False)
    for ax, lam in zip(axes[0], lams):
        s = t[t["lam"] == lam]
        # x range follows the predictions; the diagonal is drawn over that range only
        xlo, xhi = 0.0, float(s["pred_mean_bp"].max())
        pad = 0.08 * (xhi - xlo or 1.0)
        width = xhi - xlo or 1.0
        ax.plot([xlo - pad, xhi + pad], [xlo - pad, xhi + pad], color=INK2, linewidth=0.8, label="realized = predicted")
        for k, arm in enumerate(("A", "B", "AB")):
            g = s[s["arm"] == arm]
            if g.empty:
                continue
            x = g["pred_mean_bp"].to_numpy(dtype=float) + 0.012 * width * (k - 1)  # small absolute dodge
            y = g["real_mean_bp"].to_numpy(dtype=float)
            yerr = np.vstack([y - g["real_ci_lo_bp"].to_numpy(dtype=float), g["real_ci_hi_bp"].to_numpy(dtype=float) - y])
            ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=5, color=ARM_COLORS[arm], ecolor=ARM_COLORS[arm],
                        elinewidth=1, capsize=0, label=ARM_LABELS[arm])
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.set_title(f"λ = {lam:g}: VOI calibration (quintiles of predicted VOI)")
        ax.set_xlabel("predicted gross VOI (bp, bin mean)")
        ax.set_ylabel("realized paired improvement (bp, mean ± 95% CI)")
    axes[0][0].legend(loc="best")
    fig.tight_layout()
    out = FIG / "fig3_voi_calibration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return [out.name]


def fig_calsize() -> list[str]:
    path = TAB / "exp4_calibration_size.csv"
    if not path.exists():
        return []
    t = pd.read_csv(path)
    lams = sorted(t["lam"].unique())
    fig, axes = plt.subplots(1, len(lams), figsize=(5.2 * len(lams), 3.6), squeeze=False)
    for ax, lam in zip(axes[0], lams):
        s = t[t["lam"] == lam]
        for pol in ("myopic_voi", "optimal_voi"):
            g = s[s["policy"] == pol].sort_values("calibration_days")
            ls, mk = POLICY_STYLE[pol]
            ax.plot(g["calibration_days"], g["mean_diff_bp"], marker=mk, markersize=4.5, linestyle=ls,
                    color=POLICY_COLORS[pol], label=POLICY_LABELS[pol])
            if pol == "optimal_voi":  # one band; both policies' CIs are in exp4_calibration_size.csv
                ax.fill_between(g["calibration_days"], g["ci_lo_bp"], g["ci_hi_bp"], color=POLICY_COLORS[pol],
                                alpha=0.15, linewidth=0, label="95% CI (optimal VOI)")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xscale("log")
        ticks = sorted(s["calibration_days"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(v):,}" for v in ticks])
        ax.minorticks_off()
        ax.set_title(f"λ = {lam:g}: effect of calibration sample size")
        ax.set_xlabel("calibration rows used to fit the belief (log scale)")
        ax.set_ylabel("net utility minus free-only (bp/date)")
    axes[0][0].legend(loc="best")
    fig.tight_layout()
    out = FIG / "fig4_calibration_size.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return [out.name]


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    made = fig_cumulative() + fig_cost() + fig_calibration() + fig_calsize()
    print("figures:", ", ".join(made) if made else "none (no experiment outputs found)")


if __name__ == "__main__":
    main()
