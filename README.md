# 2¢ for Your Two Cents — Bayesian Valuation of Unobserved Information

Stanford AA228/CS238 final project. An agent sees a free linear forecast of the next-day
return of a broad-market ETF (SPY), and may pay for two more forecasts (random forest,
gradient boosting) before choosing a position q ∈ {-1, 0, 1}. It values each unseen
forecast by its expected value of information (VOI) under a Gaussian belief, and plans
purchases myopically or with an exact two-step Bellman recursion.

Work in progress — see `docs/milestone_report.md`.
