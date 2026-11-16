# Mathematical assumptions and precise definitions

This file pins down every modelling choice the proposal left open, and lists
where the implementation departs from `proposal/proposal_revised.tex`.

## 1. Timing, returns and data

* **Decision time.** Close of trading day t. All features, the free forecast and the
  purchasable forecasts for date t use only data dated ≤ t.
* **Return.** r_{t+1} = AdjClose_{t+1} / AdjClose_t − 1 (simple total return,
  dividends included through Yahoo's adjusted close). Positions are held close-to-close.
* **Execution idealization.** The position is assumed to trade at the close of t at that
  close's price, with information from the same close (a market-on-close order placed a
  moment before the close would see essentially the same information). The transaction
  cost τ|q − q_prev| stands in for spread and slippage.
* **Adjusted prices.** Yahoo back-adjusts history for dividends, so adjusted *levels*
  embed later information. Only ratios of adjusted closes over windows ending at t are
  used; such ratios equal total-return ratios and do not depend on dividends after t.
* **Training cutoff.** A forecast model used at origin t is fit only on pairs
  (x_s, r_{s+1}) with s + 1 ≤ t (the label was realized by the decision time).
  Hyperparameters are fixed a priori (no tuning), so model selection never sees the
  evaluation period.

## 2. Utility and the stop value

* u(q, r) = q r − λ q² r² − τ |q − q_prev|, q ∈ {−1, 0, 1}; all money in fractions of
  capital (1 bp = 1e-4). Query costs c_j are in the same units and are subtracted once
  per purchase.
* Under any belief b, E_b[u(q, r)] = q m − λ q² s₂ − τ|q − q_prev| with m = E_b[r],
  s₂ = E_b[r²] = m² + Var_b(r). So G(b) = max_q E_b[u(q, r)] depends on b only through
  (m, Var_b r, q_prev).
* **Ties.** Positions: smallest |q − q_prev|, then smallest |q|, then lowest q.
  Acquisition: stopping wins ties against querying (a query must beat stopping by more
  than `PLAN_TOL`); source A wins ties against source B.

## 3. Belief model ("Bayesian" in the plug-in sense)

* X = (r, Y₀, Y_A, Y_B) ~ N(μ, Σ). μ and Σ are **estimated** on a calibration period of
  out-of-sample forecasts and subsequently realized returns, then **plugged in**. Belief
  updating within a date is exact Bayesian conditioning under this plug-in law;
  uncertainty about (μ, Σ) itself is not integrated over. That estimation error is a
  subject of study (proposal E2 / sprint Experiment 4), not something the agent models.
* Covariance estimator: Ledoit–Wolf shrinkage of the **correlation** matrix toward the
  identity on standardized data, then rescaled. Shrinking the raw covariance would be
  dominated by the return's variance (forecasts have ~10× smaller standard deviations).
  Correlation eigenvalues are floored at 1e-8 to keep Σ positive definite. A plain sample
  covariance is available as an option.
* After observing a set O, r | y_O ~ N(m_O, v_O); v_O does not depend on the observed
  values. Querying an unobserved source j moves the mean by m' − m ~ N(0, Δ_j) with
  Δ_j = Cov(r, Y_j | O)² / Var(Y_j | O), and v' = v − Δ_j (preposterior analysis).
* Calibration never uses held-out information: at decision date t the model is fitted only
  on archive rows whose return was realized at or before t (`BeliefSchedule`). Modes:
  `fixed` (fit once on the calibration period) and `expanding` (refit every 21 evaluation
  dates on all realized history).

## 4. Value of information and planning

* One-step net VOI: VOI_j(b) = E_{Y_j|b}[G(b^{j,Y_j})] − G(b) − c_j. Gross VOI excludes c_j
  and is ≥ 0 by Jensen's inequality (G is convex in the belief, and the belief is a
  martingale).
* For a Gaussian belief, E[G(b')] is computed in **closed form**: G(·, v') is piecewise
  quadratic in the posterior mean with breakpoints that depend only on v', so the
  expectation over m' ~ N(m, Δ) is a finite sum of truncated-normal moments.
* **Per-date acquisition episode** with budget B (per date) and unused sources U:
  V(b, B, U) = max{ G(b), max_{j ∈ U, c_j ≤ B} [ −c_j + E_{Y_j|b} V(b^{j,Y_j}, B − c_j, U∖{j}) ] }.
  With two sources the horizon is at most two queries. The inner stage is closed form;
  the outer expectation over the first observation uses deterministic quadrature whose
  error is measured against adaptive quadrature split at the kinks (see `notes_D.md`).
* **Budget.** The proposal's recursion V(b, U) has no budget; the sprint spec adds B. B ≥
  c_A + c_B is non-binding (the default); smaller B can forbid the second purchase.
* **One-step vs full-policy values.** VOI_j is the value of buying j and then stopping.
  The planner's value V(b, B, U) − G(b) is the value of the full adaptive acquisition
  policy. Experiment 2 reports the two separately.
* **Expected vs realized.** VOI is an expectation under the belief. On one date the
  realized benefit u(q_j, r) − u(q₀, r) can be negative; only averages over many dates
  can be compared with predictions, and only under a correctly specified belief.

## 5. Linking days

Each date is one acquisition episode. Days are linked only through q_prev in the
transaction cost; every policy carries its own position path. Within a date the planner
maximizes that date's expected utility minus query costs: it does **not** value the effect
of today's position on tomorrow's transaction cost. This matches the proposal's
"one-day POMDP episode" framing, which treats cross-day linkage as a stretch goal.

## 6. Policies (all choose the terminal position by maximizing posterior expected utility)

| Policy | Acquisition rule |
|---|---|
| free_only | never buys |
| always_buy | buys every affordable source in the order A, B |
| fixed_order | order A then B, with optimal stopping along that order (the proposal's "fixed order with optimal stopping") |
| random | at each step, with probability p (default 0.5) buys a uniformly chosen affordable remaining source, else stops; seeded per (seed, policy, date) |
| myopic_voi | buys argmax_j VOI_j while it exceeds `PLAN_TOL` |
| optimal_voi | follows the exact two-step Bellman recursion, re-solving after each observation |

## 7. Departures from the proposal

1. **Primary experiment.** The proposal makes synthetic E1–E3 the main study and the
   historical replay a stretch goal (its pilot power analysis needs ~1.2×10⁵ days to
   detect a mean benefit at ρ = 0.05 from realized returns). The sprint spec makes the
   SPY replay the primary milestone. Replay results are therefore expected to be noisy, so
   they are reported with block-bootstrap intervals and never read as proof of
   equivalence.
2. **Budget** B added to the Bellman recursion (non-binding by default).
3. **Sources.** K = 2 purchasable forecasts (random forest, gradient boosting), with a
   ridge-regression free forecast, all trained walk-forward on SPY features.
4. **Context.** The default belief is static (one Gaussian per calibration window).
   A volatility-scaled variant (Section 3 of `notes_C.md`) is available as a simple
   context-dependent extension.
