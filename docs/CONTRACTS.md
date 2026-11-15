# Module contracts (read before editing any module)

Repo root: this directory. Run everything from here; imports are `from src.<module> import ...`
(`pytest.ini` sets `pythonpath = .`). Python >= 3.11, stack: numpy, pandas, scipy,
scikit-learn, matplotlib, pytest. No other dependencies without a written reason.

## Compute rule (hard constraint)

The development Mac is shared by several sessions. **Never run model training, full
historical replays or experiment sweeps locally.** Locally allowed: unit tests and tiny
smoke runs (< ~10 s CPU each), always single-threaded and niced:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 -m pytest tests/test_x.py

No `n_jobs=-1` locally. Full jobs run on GitHub Actions (`.github/workflows/`), where
`n_jobs=-1` is fine.

## Notation and units

* Sources: `"y0"` free linear forecast (always observed first), `"A"` random forest,
  `"B"` gradient boosting. Return variable `"r"`. Constants in `src/interfaces.py`.
* Money is in fractions of capital (1 bp = 1e-4): returns, utilities, costs, tau.
* Utility `u(q,r) = q r - lam q^2 r^2 - tau |q - q_prev|`, q in {-1,0,1} (`src/utility.py`).
* Stop value `G(b) = max_q E_b[u(q,r)]`; depends on b only through `(E_b r, E_b r^2)`.
* Net one-step VOI: `VOI_j(b) = E_{Y_j|b}[G(b^{j,Y_j})] - G(b) - c_j`.
* Bellman: `V(b,B,U) = max{ G(b), max_{j in U, c_j <= B} [ -c_j + E_{Y_j|b} V(b^{j,Y_j}, B-c_j, U\{j}) ] }`.
* Budget is per decision date (one date = one acquisition episode). Each source can be
  bought at most once per date. Budget comparison uses `BUDGET_ATOL`.
* Tie rules (deterministic): positions -> `utility.choose_position` (smallest |q-q_prev|,
  then smallest |q|, then lowest q). Acquisition: STOP wins ties against querying
  (query only if its value exceeds stop by more than `PLAN_TOL`, defined in
  `src/planner.py`); between sources, order `("A","B")` wins ties. Myopic buys only if
  `VOI_j > PLAN_TOL`.

## Fixed modules (owned by the lead; do not change signatures)

* `src/utility.py`: `UtilityParams(lam, tau)`, `POSITIONS`, `trading_utility`,
  `transaction_cost`, `realized_utility`, `expected_utility`, `stop_value_array`,
  `choose_position(mean, second_moment, q_prev, p) -> (G, q)`.
* `src/interfaces.py`: `SOURCES`, `FREE_SOURCE`, `STOP`, `BUDGET_ATOL`, `Belief` protocol
  (`observed`, `return_moments()`, `predictive_branches(source, n_nodes)`,
  `condition(source, value)`), optional fast path
  `expected_stop_value_after(source, q_prev, params) -> float` (exact E[G(b')]),
  `AcquisitionSetup(costs, budget, params)`, `PlanResult`.
* `src/belief_model.py`: `GaussianJointModel(mean, cov, names=("r","y0","A","B"))`,
  `GaussianBelief` (`mean`, `var`, `predictive(j)`, `gain(j)` = Delta_j,
  `condition`, `predictive_branches`, `expected_stop_value_after`, `is_degenerate`),
  `fit_joint_gaussian(data, names, method="ledoit_wolf"|"sample")`.
  Workstream C may add functions/classes and fix bugs, but must keep these names.

## Workstream-owned modules

### A — `src/toy_simulator.py` (+ `tests/test_toy_simulator.py`)
* `DiscreteJointModel(r_values, pmf, signal_values)`: exact joint pmf over
  (r, y0, A, B) on finite alphabets.
* `DiscreteBelief`: implements the `Belief` protocol exactly (+ `expected_stop_value_after`).
* `exact_two_step_value(belief, setup, q_prev)`: independent reference DP (does not import
  `src.planner`).
* `brute_force_optimal(belief, setup, q_prev)`: enumerates every deterministic adaptive
  acquisition-and-position policy tree and returns the best expected utility (and tree).
* `scenarios()`: named deterministic test instances (zero cost, uninformative, duplicate,
  independent informative, myopic-stops-but-two-step-buys, insufficient budget, no
  sources, ties).

### B — data (`src/data_pipeline.py`, `src/forecast_models.py`, `src/forecast_archive.py`,
`experiments/build_archive.py`, `.github/workflows/pipeline.yml`, `tests/test_data_*.py`)
* Decision at the close of trading day t; target `r_{t+1} = AdjClose_{t+1}/AdjClose_t - 1`.
* Features at t use prices/volume dated <= t only; only ratio-type price features (adjusted
  close levels embed future dividend adjustments; ratios do not).
* A model used at origin t is fit only on pairs `(x_s, r_{s+1})` with `s+1 <= t`.
* Archive CSV (`data/processed/forecast_archive.csv`), one row per origin date, columns at
  least: `date, free_forecast, rf_forecast, gb_forecast, realized_return, return_date,
  feature_timestamp, forecast_origin, train_start, train_end, train_label_end, fit_id,
  n_train`. Plus a JSON manifest (config, data hash, package versions, row counts).

### C — `src/belief_model.py` hardening (+ `tests/test_belief_model.py`)

### D — `src/voi.py`, `src/planner.py`, `src/policies.py` (+ `tests/test_voi.py`,
`tests/test_planner.py`, `tests/test_policies.py`)
* `voi.expected_stop_value_gaussian(m, delta, v, q_prev, params) -> float`: closed form of
  `E_{X~N(m,delta)}[G(X, v - delta)]` (G with posterior variance `v - delta`). Must handle
  `delta = 0` and `delta = v`. Vectorized twin `..._array(m: ndarray, ...)`.
* `voi.one_step_voi(belief, source, cost, q_prev, params, n_nodes=None) -> float` (net),
  `voi.gross_voi(...)`.
* `planner.solve(belief, remaining, budget, setup, q_prev, n_nodes=None) -> PlanResult`:
  exact Bellman recursion over the `Belief` protocol.
* `planner.solve_fixed_order(belief, order, budget, setup, q_prev, n_nodes=None)`:
  optimal stopping along a fixed order.
* `policies`: shared `Policy` interface; `FreeOnly`, `AlwaysBuy`, `FixedOrder`,
  `RandomAcquisition`, `MyopicVOI`, `OptimalVOI`; `ForecastVault` (holds hidden forecasts,
  reveals a value only on `query`, charges cost exactly once, refuses repeats/unaffordable);
  `run_episode(policy, belief, vault, setup, q_prev, rng) -> EpisodeResult`.

### Integration — `src/backtest.py`, `experiments/*.py`, results in `results/`.
