# Milestone report — implementation sprint

Status as of 2026-11-16. Every number below was produced by running the code in this
repository; nothing is estimated or simulated unless labelled SYNTHETIC. Real-data
results come from GitHub Actions run
[43812639591](https://github.com/HenryCh-hong/cs238/actions/runs/43812639591) at commit
`c12c64a` (tables in `results/tables/`, figures in `results/figures/`, per-date logs in
the run's `replay-results` artifact, copied to the git-ignored `results/logs/`). The two
belief-variant diagnostics in `results/diagnostics/` were run on the laptop afterwards with
unchanged replay code. Units: bp = 1e-4 of capital, per evaluation date.

## A. Implementation summary

| Component | Module | What exists and is tested |
|---|---|---|
| Utility and stop value | `src/utility.py` | u = q r − λ q² r² − τ\|q − q_prev\|, position tie rule |
| Exact testbed | `src/toy_simulator.py` | finite joint pmf, exact Bayes, independent Bellman recursion, brute-force enumeration of every adaptive policy tree |
| SPY data | `src/data_pipeline.py` | Yahoo download (cached + sha256 manifest), validation (dates, NaN, OHLC, splits/dividends), causal ratio-type features |
| Forecasts | `src/forecast_models.py`, `src/forecast_archive.py` | ridge (free y0), random forest (A), gradient boosting (B); walk-forward expanding window, refit every 21 days, hard timing assertions |
| Belief | `src/belief_model.py` | joint Gaussian over (r, y0, A, B), Ledoit–Wolf on correlations, exact conditioning, preposterior gains |
| VOI and planning | `src/voi.py`, `src/planner.py` | closed-form E[G] (truncated normal), exact two-step Bellman planner with budget, kink-split quadrature |
| Policies | `src/policies.py` | free-only, always-buy, fixed order (A→B with optimal stopping), random, myopic VOI, optimal VOI; `ForecastVault`; `run_episode` |
| Replay | `src/backtest.py` | chronological replay of all six policies, leakage-safe belief schedule, run-time accounting checks |
| Evaluation | `src/evaluation.py` | paired differences, circular block bootstrap, VOI calibration counterfactuals |
| Experiments | `experiments/` | archive build, replay CLI, Experiments 0–4, figures; `.github/workflows/` runs them on CI |

## B. Milestone status

* **Milestone A — PASS.** `solve` equals the brute-force optimum (enumeration of every
  adaptive policy tree) to 1e-12 on all 12 named scenarios and 60 seeded random instances
  (`tests/test_planner_vs_bruteforce.py`, 73 tests). The testbed's own recursion equals
  brute force on 252 further cases (`tests/test_toy_simulator.py`).
* **Milestone B — PASS.** A complete chronological replay of SPY over 2,984 held-out dates
  (2015-01-02 … 2026-11-12) ran for all six policies under one protocol, with no
  exceptions, all run-time accounting checks passing, and seeds, configuration, data hash
  and git commit recorded (`results/tables/default_manifest.json`).

## C. Validation matrix

Two sources of evidence: the automated test suite (872 tests; 872 passed locally and on
GitHub Actions), and an independent post-run audit by four separate agents that re-derived
results from the raw logs and archive **without** using the project's summary code
(scripts kept outside the repository).

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| A | Exact small-instance tests | `test_toy_simulator.py` 603 tests (scenarios: uninformative, duplicate, XOR, budget, ties, …) | PASS |
| A | Posteriors match references | `test_belief_model.py` (Schur complement, least squares, order independence; 51 tests); toy posterior vs direct Bayes; audit re-derived real-data posteriors to 1e-12 | PASS |
| A | Bellman recursion vs exhaustive enumeration | 73 planner-vs-brute-force tests at 1e-12; 252 testbed-vs-brute-force tests | PASS |
| A | Utility and cost accounting | unit tests; audit on all 29 real-data logs: max \|net − (gross − tc − acq)\| = 1.0e-16 | PASS |
| B | Time ordering | audit of all 6,994 archive rows: 0 violations of date = origin, feature_ts ≤ date, train_end < date, train_label_end ≤ date, return_date > date | PASS |
| B | Forecasts out of sample | refit at f trains on rows [0, f) only; tests perturb/remove future rows and check forecasts unchanged; refit schedule (334 blocks) reproduced exactly | PASS |
| B | No future labels/features | feature tests (future price perturbation, future dividend back-adjustment, truncation invariance); no forecast has \|corr\| > 0.2 with its target (max 0.15) | PASS |
| B | Training cutoffs | train_label_end = return date of the last training row ≤ origin on 6,994/6,994 rows | PASS |
| B | Missing data and alignment | validator tests (duplicates, NaN, weekends, splits); realized_return equals next-day adjusted-close return from an independent Yahoo download (corr 0.9999999995, max abs diff 1.9e-6, consistent with re-adjustment rounding) | PASS |
| C | No premature access to hidden forecasts | `ForecastVault` tests; on real logs revealed_A/B are NaN unless bought, and equal the archive value when bought | PASS |
| C | Each source at most once | tests; real logs: no duplicates, ≤ 2 purchases | PASS |
| C | Budget enforced | binding-budget tests (unit + synthetic replay). Real runs use a non-binding budget | PASS |
| C | Every query pays its cost | real logs: max \|acq − Σ costs\| = 1.1e-19 | PASS |
| C | Stopping ends acquisition | `test_episode_terminates_and_charges_each_cost_once` | PASS |
| C | Positions maximize posterior expected utility | 0 mismatches in 519,216 logged decisions (tie rule included) | PASS |
| D | Complete chronological replay | 2,984 dates × 6 policies, same date set | PASS |
| D | Same protocol for all policies | one belief per date shared; separate q_prev paths; checked in logs | PASS |
| D | Reproducible results | three CI replays with the same configuration (`default`, `exp1_lam1`, `exp3_lam1_c0.1`) are identical in every column except timing; `test_replay_is_reproducible` | PASS |
| D | Seeds and configuration recorded | 30 manifests with seed, config, config hash, data sha256, git commit | PASS |
| D | No exceptions or silent NaNs | NaN only in not-applicable columns (`predicted_value` for policies without one, `revealed_*` when not bought, `voi_next_1` without a second step) | PASS |
| E | Net utility reconciles | max error 1.0e-16 on all logs | PASS |
| E | Acquisition frequencies valid | 0 ≤ purchases ≤ 2; frequencies in [0, 1] | PASS |
| E | Free-only buys nothing | 0 purchases on every date | PASS |
| E | Always-buy follows its rule | buys A then B on all 2,984 dates; buys only A when budget = c_A (test) | PASS |
| E | Planner checked against the belief objective | dense independent double-quadrature reference agrees to ~2.5e-9; closed form vs Monte Carlo \|z\| ≤ 2.3; 60 real-data episodes re-run independently match the log | PASS |
| E | Numerical accuracy supported by convergence checks | convergence tables in `notes_D.md` §4. Exception: `planner.evaluate_policy` (test helper, not used in the replay) has quadrature error ≈ 8 % of the VOI at n = 201 | PASS (with noted exception) |
| E | VOI estimator unbiased when the belief is correct | SYNTHETIC: 6,000-date test (\|mean residual\| ≤ 3 SE); audit simulation at the fitted real parameters: gap −0.39 bp, SE 0.60 | PASS |

## D. Experimental results

Default configuration: λ ∈ {1, 5}, τ = 1 bp, c_A = c_B = 0.1 bp, non-binding budget;
belief fitted once on 4,010 archive rows (1999-01-26 … 2014-12-31). Intervals are 95 %
circular moving-block bootstrap intervals (block 21 trading days, 2,000 resamples) of the
paired per-date difference against free-only. With a few bp of daily effect and returns of
~110 bp daily sd, intervals are wide, and an interval containing 0 means "inconclusive", not
"no difference".

### Experiment 1 — policy comparison (SPY, 2,984 dates)

| λ | Policy | Net bp/date | Acq. cost | Buys A | Buys B | Long / flat / short | ms/decision | vs free-only [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | free-only | 4.42 | 0.00 | 0 % | 0 % | 74 / 26 / 1 % | 2.6 | — |
| 1 | always-buy | 0.62 | 0.20 | 100 % | 100 % | 78 / 20 / 2 % | 4.2 | −3.80 [−6.65, −1.24] |
| 1 | fixed order | 1.06 | 0.10 | 96 % | 2 % | 78 / 20 / 2 % | 25.8 | −3.36 [−5.95, −0.98] |
| 1 | random | 3.06 | 0.08 | 38 % | 37 % | 75 / 24 / 1 % | 3.4 | −1.35 [−2.75, −0.005]* |
| 1 | myopic VOI | 1.06 | 0.10 | 96 % | 2 % | 78 / 20 / 2 % | 7.6 | −3.36 [−5.95, −0.98] |
| 1 | optimal VOI | 1.06 | 0.10 | 96 % | 2 % | 78 / 20 / 2 % | 45.1 | −3.36 [−5.95, −0.98] |
| 5 | free-only | −0.44 | 0.00 | 0 % | 0 % | 2 / 98 / 0 % | 2.6 | — |
| 5 | always-buy | −1.28 | 0.20 | 100 % | 100 % | 4 / 96 / 1 % | 4.2 | −0.84 [−2.34, +0.24] |
| 5 | fixed order | −0.72 | 0.03 | 33 % | 0 % | 4 / 96 / 0 % | 23.8 | −0.28 [−1.53, +0.68] |
| 5 | random | −0.71 | 0.08 | 38 % | 37 % | 2 / 98 / 0 % | 3.4 | −0.27 [−0.95, +0.27] |
| 5 | myopic VOI | −0.72 | 0.03 | 32 % | 0 % | 4 / 96 / 0 % | 5.8 | −0.28 [−1.53, +0.68] |
| 5 | optimal VOI | −0.72 | 0.03 | 33 % | 0 % | 4 / 96 / 0 % | 50.3 | −0.28 [−1.53, +0.68] |

\* The random-vs-free-only upper endpoint is −0.005 bp with the reported bootstrap seed, but
it crosses 0 for other seeds (−0.056 … +0.058); treat it as borderline.

* At λ = 1, buying forecasts **lowered** net utility relative to free-only for every buying
  policy; for the VOI policies the loss is −3.36 bp/date and the interval excludes 0. At λ = 5
  results are inconclusive.
* Optimal two-step VOI, myopic VOI and fixed order chose the same position on all 2,984
  dates; their purchases differ on 1 date (λ = 1) and 8 dates (λ = 5). Given (y0, A),
  forecast B adds almost no information under the fitted belief, so look-ahead has nothing
  to exploit here.
* Overhead: optimal VOI costs 45–50 ms per decision against 6–8 ms for myopic VOI.

### Experiment 2 — VOI calibration (λ = 1, free-only position path, 2,984 dates)

| Arm | Predicted gross VOI (mean) | Realized paired improvement [95 % CI] | Position changed |
|---|---|---|---|
| buy A only | +0.93 bp | −2.49 [−4.40, −0.86] | 12.3 % |
| buy B only | +0.16 bp | −0.74 [−2.56, +0.96] | 5.4 % |
| buy both | +0.95 bp | −2.89 [−5.03, −1.04] | 12.7 % |
| full optimal policy (net, adaptive) | +0.84 bp | −2.70 [−4.64, −0.98] | buys on 96 % of dates |

One-step values (buy j, then stop) and the full adaptive policy's value are reported
separately. Within quintiles of predicted VOI the realized means do not increase with the
prediction (slope for A: 1.5, CI [−2.8, 6.1]; `results/figures/fig3_voi_calibration.png`). At
λ = 5 the predicted VOI of A is +0.12 bp against realized −0.87 [−2.72, +0.34]
(inconclusive).

### Experiment 3 — cost sensitivity (`results/figures/fig2_cost_sensitivity.png`)

Mean purchases per date for the VOI policies, by query cost (bp):

| Cost (bp) | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|---|---|
| λ = 1 | 1.90 | 1.12 | 1.05 | 0.98 | 0.89 | 0.46 | 0 | 0 |
| λ = 5 | 1.02 | 1.00 | 0.79 | 0.33 | 0.09 | 0.01 | 0 | 0 |

The VOI policies stop buying at 3 bp and above, which equals free-only. Always-buy and random
pay the cost regardless (always-buy at 10 bp: −23.6 bp/date at λ = 1). Wherever the VOI
policies do buy at λ = 1, they are below free-only (e.g. −1.85 [−3.47, −0.30] at 1 bp).

### Experiment 4 — calibration sample size (`results/figures/fig4_calibration_size.png`)

Belief fitted on the last n calibration rows before 2015:

| n | Implied partial corr(r, A \| y0) | Optimal VOI buys A | λ = 1: vs free-only [95 % CI] |
|---|---|---|---|
| 250 | 0.008 | 0.4 % | +0.36 [−0.92, +1.85] |
| 500 | 0.016 | 0 % | 0.00 |
| 1,000 | −0.006 | 0.9 % | −0.20 [−0.99, +0.66] |
| 2,000 | 0.033 | 98 % | −2.16 [−4.26, −0.40] |
| 4,010 (all) | 0.030 | 96 % | −3.36 [−5.95, −0.98] |

Short recent windows make A look uninformative, and the planner then does not buy.
Windows reaching back to 2007–2008 make A look informative, and buying it loses.

### Experiment 0 — SYNTHETIC well-specified sanity check

On a SYNTHETIC archive drawn from a Gaussian law with corr(r, y0) = 0.05, corr(r, A) =
0.15, corr(r, B) = 0.12 (8,000 rows; belief fitted on the first 60 %, 3,200 evaluation
dates, λ = 5, costs 0.2 bp), the same pipeline gives, relative to free-only: always-buy +7.99 [4.87, 11.14], fixed order +8.04 [4.94, 11.23],
optimal and myopic VOI +7.80 [4.70, 10.93] bp/date. When the belief is right
and the signals are real, buying information pays and the code finds it.

### Why buying loses on SPY (post-run diagnosis; evidence first, then interpretation)

Evidence:
1. **No code bug found.** An adversarial audit re-derived the fitted belief (to 3.5e-14), the
   closed-form VOI (vs Monte Carlo), 60 full episodes, the archive alignment and the column
   mapping, and found no error. The same counterfactual code is unbiased in simulation at
   the fitted parameters.
2. **A's skill reversed out of sample.** Partial correlation of the random-forest forecast
   with r_{t+1} given the free forecast: +0.030 (SE 0.016) in 1999–2014, −0.043 (SE 0.018) in
   2015–2026. Sub-periods swing from +0.06 to −0.14. Gradient boosting moved the other way:
   +0.013 in calibration, +0.105 (SE 0.018) out of sample.
3. **The calibration estimate came from turbulent days.** 73 % of the rf–return
   cross-product in calibration comes from the highest-volatility tercile of days and 47 %
   from 2008 alone.
4. **The loss is concentrated.** The optimal-VOI position differs from free-only on 879
   dates (29.5 %), which carry essentially all of the −10,018 bp total. The main channel is
   de-risking before rallies: long→flat switches on 333 dates cost −6,761 bp (worst:
   2020-04-03, SPY +6.7 %). Dropping the 5 % largest-|r| dates moves the mean from −3.36 to
   −1.40 bp. 71 % of the loss falls in the high-volatility tercile, and 9 of 12 years are
   negative.
5. **Refitting the belief helps.** An expanding belief refitted every 21 dates on realized
   history reduces the gap to −1.41 [−4.24, +0.91] (Ledoit–Wolf) or −1.22 [−3.92, +1.08]
   (sample covariance). Both intervals contain 0.

Interpretation: the static plug-in Gaussian belief is misspecified for SPY. Forecast skill
is non-stationary and heteroskedastic, so a correlation estimated on a crisis-dominated
window over-values A and under-values B out of sample. The VOI machinery then faithfully
optimizes the wrong objective. This is the plug-in limitation stated in the proposal
(`mathematical_assumptions.md` §3), now measured.

## E. Compute report

| Step | Where | Wall time | Notes |
|---|---|---|---|
| Unit + integration tests (872) | laptop (Apple M5, 10 cores, 16 GB; niced, 1 thread) | 36 s | Python 3.14 |
| Same 872 tests | GitHub Actions (ubuntu, 4 vCPU) | 74 s | Python 3.12 |
| Download + full walk-forward archive (334 refits × 3 models) | GitHub Actions, 4 processes | 9 min 5 s | |
| Milestone B replay (2,984 dates × 6 policies) | GitHub Actions, 1 process | 4 min 24 s | 2.6–50 ms per decision |
| Experiments 0–4 (31 replays + counterfactuals) | GitHub Actions, 4 processes | 78 min | exp0 3.9, exp1 4.5, exp2 10.3, exp3 41.0, exp4 18.2 min |
| Whole CI job | GitHub Actions | 1 h 34 min | |
| Two belief-variant diagnostic replays (3 policies each) | laptop, 1 process, niced | ~1 min each | |

Peak memory was not measured. The largest in-memory objects are the 17,904-row logs and the
6,994-row archive, so memory is not a constraint. Local compute would have been sufficient,
but the laptop is shared with other sessions, so heavy work ran on GitHub Actions. No paid
infrastructure was used.

## F. Remaining work

Nothing blocks Milestone B. For the final research experiments:

1. **Belief model (main scientific gap).** Make the belief track non-stationary skill:
   expanding/rolling refits (diagnostic above), volatility-scaled beliefs (the
   `ewma_volatility`/`scale_model` helpers exist but are not wired into the replay), and
   parameter uncertainty in (μ, Σ) instead of a plug-in (e.g. a Normal–inverse-Wishart
   posterior predictive, which shrinks VOI when the estimates are weak).
2. **Synthetic study (the proposal's main experiments E1–E3).** Only the sanity run (exp0)
   exists. Regimes where non-myopic planning matters (the A3/A4 instances in the tests) are
   needed, because on SPY the two-step planner coincides with the myopic one.
3. **Power.** Daily effects of a few bp need far more dates than 2,984 to separate
   policies. Report intervals, never equivalence claims.
4. **Reproducibility.** The forecast archive lives in CI artifacts (30-day retention) and is
   git-ignored. Yahoo revisions change the raw file between downloads, and rf/gb forecasts
   move by about 2e-5 to 5e-5 under 1e-6 price noise. Pin `yfinance` and decide whether to
   commit the derived archive.
5. **Minor.** `planner.evaluate_policy` quadrature (test-only); the planner ignores
   tomorrow's transaction cost (one-day episode, as in the proposal); CI actions use the
   deprecated Node 20 runtime.

## G. Reproduction

From a clean environment (see also `README.md`):

```bash
git clone https://github.com/HenryCh-hong/cs238 && cd cs238
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest                                                     # 872 tests
python -m experiments.build_archive --config configs/default.toml --n-jobs 4
python -m experiments.run_replay --config configs/default.toml --tag default
python -m experiments.run_experiments --workers 4
python -m experiments.make_figures
# or on GitHub: gh workflow run experiments.yml --ref main
# belief-variant diagnostic: python -m experiments.run_replay --tag diag_expanding --set belief.mode=expanding
```

Yahoo revises adjusted prices between downloads (CI and the laptop got files with different
sha256), so a rebuilt archive can differ slightly. The manifests record every hash.

## H. Files changed

Created during the sprint (relative to the initial skeleton `d500e4f`):

* `src/`: `utility.py`, `interfaces.py`, `config.py`, `belief_model.py`, `toy_simulator.py`,
  `data_pipeline.py`, `forecast_models.py`, `forecast_archive.py`, `voi.py`, `planner.py`,
  `policies.py`, `backtest.py`, `evaluation.py`, `synthetic_archive.py`
* `tests/`: `conftest.py`, `test_toy_simulator.py`, `test_planner_vs_bruteforce.py`,
  `test_belief_model.py`, `test_data_pipeline.py`, `test_forecast_archive.py`, `test_voi.py`,
  `test_planner.py`, `test_policies.py`, `test_backtest.py`
* `experiments/`: `build_archive.py`, `run_replay.py`, `run_experiments.py`, `make_figures.py`
* `configs/default.toml`, `pytest.ini`, `requirements.txt`, `.gitignore`, `README.md`
* `.github/workflows/`: `pipeline.yml` (tests on push; archive on demand), `experiments.yml`
  (archive → tests → replay → experiments → figures)
* `docs/`: `CONTRACTS.md`, `mathematical_assumptions.md`, `implementation_decisions.md`,
  `milestone_report.md`, `notes_A.md` … `notes_D.md`
* `data/`: `raw/SPY_daily_manifest.json`, `raw/SPY_daily_manifest_ci.json`,
  `processed/forecast_archive_manifest.json`, `processed/forecast_archive_smoke_manifest.json`
  (the CSV data files themselves are git-ignored)
* `results/`: `tables/` (summaries, paired differences, manifests, experiment tables),
  `figures/fig1`–`fig4`, `diagnostics/` (belief-variant summaries)

Existing proposal files and earlier research outputs outside `cs238/` were not modified.
