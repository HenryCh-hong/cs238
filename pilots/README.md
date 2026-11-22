# Proposal-stage pilot scripts

These are the analytic checks and small simulations written in October 2026 for the
project proposal. Until 2026-11-22 they lived outside this repository.

They are a separate, smaller code base from `src/`. They model the same problem (a free
forecast, optional paid forecasts, a position in {-1, 0, 1}) with their own closed-form
VOI code (`voi_core.py`, `pilot_common.py`) and do not import `src/`. Agreement between
the two is therefore an independent check.

## What changed when they were moved here

File paths only. Absolute paths were replaced by paths relative to each script, and the
scratch pickles of `toy_t3_adaptivity.py` and `toy_t4_subset_dp.py` now go to
`pilots/_cache/` (git-ignored). No numerical code was touched: 16 of the 21 scripts are
byte-identical to the originals. Three lines in `reference_outputs/a5_output.txt` and
`reference_outputs/a6_output.txt` that printed an absolute path were shortened.

## Contents

| Script | What it does |
|---|---|
| `voi_core.py` | Exact Gaussian value-of-information core for the two-source problem (used by `a*` and `toy_*`) |
| `pilot_common.py` | Shared Gaussian belief utilities and parameters for `pilot_01`–`pilot_05` |
| `a1_closed_form_and_linear_gaussian_checks.py` | Closed-form expected stop value against quadrature and Monte Carlo; linear-Gaussian claims by simulation |
| `a2_exact_policy_values.py` | Exact expected net utility of each policy under a known law; value of adaptivity |
| `a2b_monte_carlo_policy_check.py` | Independent Monte Carlo check of the exact policy values |
| `a3_a4_counterexample_search.py` | Search for instances where the myopic and optimal policies differ |
| `a5_myopic_regret_grid.py` | Myopic-versus-optimal regret over a parameter grid |
| `a6_correlated_errors.py` | Correlated forecast errors |
| `a7_structural_checks.py` | Structural checks |
| `a8_runtime.py` | Run time per decision |
| `pilot_01_verify_gaussian_update.py` | Linear-Gaussian preposterior claims and the quadrature |
| `pilot_02_b1_plugin_voi_bias.py` | One-step plug-in VOI bias from an estimated covariance; over-buying; Ledoit–Wolf |
| `pilot_03_b1_two_source_policies.py` | Value of the full two-source policy under a fitted belief against the true-law optimum |
| `pilot_04_b2_misspecification.py` | A belief of the wrong family, with no estimation noise |
| `pilot_05_b3_b4_power_and_rao_blackwell.py` | Statistical power of a historical replay; Rao-Blackwellized calibration |
| `toy_t1_discrete_myopic_gap.py` | Smallest discrete instances, exact expectimax |
| `toy_t2_meanvar_artifact.py` | A mean-variance "information value" that is positive when the position can never change |
| `toy_t3_adaptivity.py`, `toy_t3_dependence_check.py`, `toy_t3_summary.py` | Value of adaptivity on a grid, a robustness check, and a summary of the grid |
| `toy_t4_subset_dp.py` | Dynamic program over subsets for K = 2, 3, 4 paid sources |

`results/` holds the JSON results and run logs of `pilot_01`–`pilot_05` from October.
`reference_outputs/` holds the saved console output of the `a*` and `toy_*` scripts.
`figures/` is where the scripts write their plots.

## Running

From the repository root, with the packages in `requirements.txt`:

```bash
python pilots/a1_closed_form_and_linear_gaussian_checks.py
```

`pilot_02` to `pilot_05`, `a3_a4`, `a5`, `toy_t3_adaptivity` and `toy_t4` take minutes and
belong on CI, not on a shared laptop. `toy_t3_summary.py` needs the pickle that
`toy_t3_adaptivity.py` writes.

## Reproduction status

Re-run after the move, on 2026-11-22 (Python 3.14, numpy 2.4.2, scipy 1.17.1):

- `a1`, `a2`, `a7`: console output identical to the saved October output (the saved
  `a1` and `a7` files end with an extra `exit=0` line added by the capture).
- `pilot_01`: the JSON it writes is byte-identical to the October file.

The other scripts have not been re-run since October.

## Parameters differ from the SPY replay

`pilot_common.py` uses a daily return sd of 1 %, lambda = 1, a transaction cost of 2 bp, a
query cost of 1 bp (0.1 bp in the runs tagged `cost1e-5`), a previous position of 0 and no
intercept. The SPY replay in `configs/default.toml` uses a transaction cost of 1 bp, a
query cost of 0.1 bp, and carries the previous position from day to day. A pilot number
should not be set beside a replay number until the pilot is re-run at matched parameters.
