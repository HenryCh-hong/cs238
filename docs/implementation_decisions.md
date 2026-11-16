# Implementation decisions (cross-cutting index)

Precise definitions live in `mathematical_assumptions.md`; per-workstream detail, measured
errors and audit findings live in `notes_A.md` … `notes_D.md`. This file lists the
decisions that matter when reading the results, and where each is enforced.

## Architecture

| Layer | Module | Role |
|---|---|---|
| Utility | `src/utility.py` | u(q,r) = q r − λ q² r² − τ\|q − q_prev\|; `choose_position` with the tie rule |
| Contracts | `src/interfaces.py` | `Belief` protocol, `AcquisitionSetup`, `PlanResult`, constants |
| Testbed (A) | `src/toy_simulator.py` | finite joint pmf, exact belief, independent Bellman recursion, brute-force tree enumeration |
| Data (B) | `src/data_pipeline.py`, `src/forecast_models.py`, `src/forecast_archive.py` | SPY download and validation, causal features, walk-forward ridge/RF/GB archive |
| Belief (C) | `src/belief_model.py` | joint Gaussian over (r, y0, A, B), Ledoit–Wolf on correlations, exact conditioning |
| VOI/planning (D) | `src/voi.py`, `src/planner.py`, `src/policies.py` | closed-form E[G], two-step Bellman planner, six policies, `ForecastVault`, `run_episode` |
| Integration | `src/backtest.py`, `src/evaluation.py`, `src/synthetic_archive.py` | replay, leakage-safe belief schedule, accounting checks, block bootstrap, VOI calibration |
| Experiments | `experiments/*.py` | archive build, replay CLI, Experiments 0–4, figures |

The planner only sees the `Belief` protocol, so **the same recursion** is validated
against brute force on discrete beliefs (Milestone A) and then run on historical data
with Gaussian beliefs (Milestone B).

## Decisions that affect results

1. **Timing and leakage** (B, integration). Decision at the close of t; target
   r_{t+1} from adjusted closes; features are ratios over windows ending at t; a model used
   at origin t trains only on labels realized by t (`train_label_end ≤ date`). The archive
   validator (`backtest.validate_archive`) rejects any row that breaks this. The belief
   schedule fits only on rows whose return date is ≤ the decision date. An end-to-end test
   perturbs all data after a cutoff and checks that earlier decisions are bit-identical.
2. **Hidden forecasts** (D). Purchasable forecasts sit in a `ForecastVault` and are revealed
   only by a paid `query`; the replay logs `revealed_A/B` only for purchased sources.
3. **Belief estimation** (C). Plug-in Gaussian; Ledoit–Wolf shrinkage of the correlation
   matrix (not the raw covariance, which r's variance would dominate). Exact duplicate
   sources are collapsed before shrinkage. Conditioning works on the correlation scale of
   the observed block, so it does not depend on units (an audit found and fixed a
   unit-dependent cut-off).
4. **Planner numerics** (D). The last stage is closed form (truncated-normal pieces). The
   outer expectation uses Gauss–Legendre on panels split at the kinks of the integrand;
   `n_quad` is the number of points per panel. Equal-weight quantile nodes converge only
   about like 1/n on this kinked integrand (8 % of the VOI at n = 201 on test instance A3),
   so the planner does not use them. `planner.evaluate_policy` still uses them; it is used
   only in tests, never in the replay.
5. **Ties.** Positions: smallest |q − q_prev|, then smallest |q|, then lowest q.
   Acquisition: STOP wins unless a query beats it by more than `PLAN_TOL = 1e-12`; A beats B.
6. **Budget.** Per date, non-binding by default (budget 1.0 ≫ c_A + c_B). Fixed-order
   skips an unaffordable source rather than stopping, because the budget only falls.
7. **Per-policy state.** Each policy carries its own q_prev path. The random policy draws
   from an rng seeded by (seed, policy index, date ordinal), so runs are reproducible and do
   not depend on which other policies are present.
8. **Accounting** (integration). `backtest._check_log` asserts at runtime that
   net = gross − transaction cost − acquisition cost to 1e-15, that acquisition cost equals
   the sum of the bought sources' costs, that the budget holds, that q ∈ {−1, 0, 1}, and
   that there are at most two purchases per date. A violation raises; it is never logged
   and ignored.
9. **Statistics.** Paired differences against `free_only` with circular moving-block
   bootstrap CIs (block 21 trading days, capped at n/2). With the expected effect sizes
   (the proposal's power analysis), replay CIs are expected to be wide; a CI that contains
   0 is reported as inconclusive, never as "no effect".
10. **VOI calibration** (Experiment 2). For each held-out date, from the free-only
    policy's own position path, the predicted one-step gross VOI of A, B and A+B is compared
    with the realized paired improvement u(q_j, r) − u(q_0, r). The two have equal means
    only if the belief is correct. A synthetic well-specified test checks the pipeline
    (|mean residual| ≤ 3 SE for A, B and AB).
11. **Synthetic data** is produced only by `src/synthetic_archive.py`, flagged
    `synthetic=True`, and run under tags prefixed `synthetic_` or `exp0_`. It is never
    presented as SPY data.
12. **Compute location.** Archive build, full replay and sweeps run on GitHub Actions
    (`.github/workflows/experiments.yml`, `pipeline.yml`). The development machine runs
    only unit tests and tiny smoke runs, niced and single-threaded.

## Deviations from the proposal

See `mathematical_assumptions.md` §7: the replay is the primary experiment (the proposal
called it a stretch goal); a per-date budget was added; K = 2 purchasable sources; the
default belief is static (fit once on the calibration period), with an expanding-refit
mode and a volatility-scaled helper available.
