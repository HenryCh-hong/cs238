# Workstream B: historical SPY data, walk-forward forecasts, archive

Owner files: `src/data_pipeline.py`, `src/forecast_models.py`, `src/forecast_archive.py`,
`experiments/build_archive.py`, `.github/workflows/pipeline.yml`,
`tests/test_data_pipeline.py`, `tests/test_forecast_archive.py`, this file.

## 1. Timing convention (what makes the archive causal)

* Decision at the close of trading day `t`. Target `r_{t+1} = AdjClose_{t+1}/AdjClose_t - 1`
  (total return incl. dividends), `return_date = t+1` (next row of the table).
* Features at `t` use rows dated `<= t` only.
* Ratio-type features only. Yahoo back-adjusts every earlier adjusted price when a dividend
  goes ex, so an Adj Close *level* at `t` changes when a future dividend is paid. Every
  feature is a ratio or log ratio over a window ending at `t`, or a same-day ratio. Tests
  verify invariance to a constant rescaling of the whole adjusted series, and to a
  back-adjustment whose ex-date is after `t` (`tests/test_data_pipeline.py`).
* Model at origin `t` is fit only on pairs `(x_s, r_{s+1})` with `s+1 <= t`.

## 2. Features (`src/data_pipeline.build_features`)

| Column | Definition (all at row t) |
|---|---|
| `logret_{k}d`, k in 1,5,10,21,63,126,252 | `log(AdjClose_t / AdjClose_{t-k})` |
| `rv_{k}d`, k in 5,21,63 | std (ddof=1) of the last k daily log returns |
| `vol_ratio_5d_63d` | `rv_5d / rv_63d` |
| `drawdown_252d` | `AdjClose_t / max(AdjClose over last 252 rows incl. t) - 1` (<= 0) |
| `log_price_ma50`, `log_price_ma200` | `log(AdjClose_t / mean of last 50 / 200 AdjClose incl. t)` |
| `log_hl_range` | `log(High_t / Low_t)` (same-day ratio, invariant to a common day adjustment) |
| `volume_z_63d` | `(log V_t - mean) / std` over the last 63 rows incl. t, of `log Volume` |

Warm-up: the first 252 rows are NaN (`logret_252d`). `make_supervised` drops warm-up rows
and the final row (no label). On real SPY this leaves 8254 labelled rows, 1994-01-27 to
2026-11-12.

## 3. Forecast models (`src/forecast_models.py`)

* `free`: `Pipeline(StandardScaler, Ridge(alpha=10))`. The scaler sees training rows only.
* `rf`: `RandomForestRegressor(300 trees, max_depth=6, min_samples_leaf=50, max_features=0.5, random_state=0)`.
* `gb`: `GradientBoostingRegressor(200 trees, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)`.

All hyperparameters come from `configs/default.toml` and are fixed a priori. There is no
tuning on the data, because tuning on realized returns is look-ahead. Each refit clones the
estimators and fits from scratch.

## 4. Walk-forward schedule (`src/forecast_archive.walk_forward`)

* Supervised rows `d_0 < d_1 < ...`. Row `i` has features at `d_i` and label dated `d_{i+1}`.
* First origin: row `i0 = min_train_days = 1260`. Every later labelled row is an origin.
* Refits at `f(i) = i0 + floor((i - i0)/R) * R`, with `R = retrain_every = 21`. The schedule
  is anchored at the first origin, so truncating data at the end never moves a refit.
* A refit at `f` uses rows `0..f-1`, so labels dated up to `d_f`. Origin `i >= f` therefore
  has `train_label_end = d_f <= d_i = forecast_origin`. Forecasts can go stale by up to
  `R-1` days, which is a deliberate simplification.
* `n_train = f` grows by 21 at each refit. `fit_id = fit_<refit date>`.
* Refit blocks run under joblib (`n_jobs`). Results do not depend on `n_jobs` (test).

Hard assertions in code (`_hard_check`, explicit `raise`, so they survive `-O`):
`date == forecast_origin`, `feature_timestamp <= forecast_origin`, `train_end < forecast_origin`,
`train_label_end <= forecast_origin`, `return_date > date`, `n_train >= min_train_days`,
no NaN anywhere. Inputs are also checked: strictly increasing dates, finite features and
targets, and `return_date` equal to the next row's date.

Output columns (`ARCHIVE_COLUMNS`): `date, free_forecast, rf_forecast, gb_forecast,
realized_return, return_date, feature_timestamp, forecast_origin, train_start, train_end,
train_label_end, fit_id, n_train`. CSV dates are ISO `YYYY-MM-DD`.

## 5. Validation (`src/data_pipeline.validate_prices`)

Errors (refuse to build): missing columns, non-DatetimeIndex, duplicated dates, unsorted
dates, NaN values, nonpositive prices or volume, `High < Low`, weekend dates.

Warnings (reported, kept): calendar gaps > 7 days, daily Adj Close returns with |r| > 25%,
Adj/Close ratio steps that are not upward dividend steps (split-like `|log step| > log 1.5`,
or other negative steps above the noise floor).

Adj/Close ratio steps are counted, not warned about. A dividend ex-date multiplies all
earlier adjusted prices by `(1 - d/P)`, so the ratio steps up by about `d/P`. Steps with
`|log step| <= 1e-4` are float rounding in the vendor series. They are counted as
`n_ratio_rounding_steps` when `|log step| > 1e-10` (below that is float64 round-off and is
ignored), and they are not warned about.

### Real SPY data (downloaded once, 2026-11-15, yfinance 1.3.0)

Source: `data/raw/SPY_daily.csv` (not committed; sha256 in `data/raw/SPY_daily_manifest.json`).

* 8507 rows, 1993-01-29 to 2026-11-13. `validate_prices`: **0 errors, 0 warnings**.
* Calendar gaps: max 7 days, on 2001-09-17 (the 9/11 closure). Also 2007-01-03 (5 days, national
  day of mourning) and 2012-10-31 (5 days, Hurricane Sandy).
* Largest daily Adj Close return: +14.5% on 2008-10-13.
* Adjusted/Close ratio: 136 dividend-like steps (one per quarter, all upward, the first on
  1993-03-19). 0 split-like steps. 8265 rounding-level steps, all |step| < 1e-4 (float32-like
  storage of the vendor series).

## 6. Smoke run on real data (plumbing check only)

`python3 -m experiments.build_archive --smoke --n-jobs 1 --no-download`, using the last 1500
labelled rows. First origin is 1000 rows in. Refits every 250. RF has 10 trees.

* Output `data/processed/forecast_archive_smoke.csv`: **500 rows**, 2 refits (`fit_2024-11-14`
  and `fit_2025-11-14`, anchored to the first smoke origin 2024-11-14), origins 2024-11-14 to 2026-11-12.
* Pearson correlation with the realized next-day return, over 500 origins (smoke only, not for
  selection; standard error is about 0.045):
  free +0.094, rf -0.064, gb -0.044.
* The manifest is `forecast_archive_smoke_manifest.json` with `smoke: true`.

The full build (6994 origins, 334 refit blocks) was not run locally, per the compute
rule. It runs in the `archive` job of `pipeline.yml`.

## 7. Manifests and reproducibility

* Raw: `data/raw/SPY_daily_manifest.json` (ticker, requested range, end passed to yfinance
  as end+1 day, rows, first/last date, sha256, UTC download time, yfinance version).
* Archive: `forecast_archive_manifest.json` (or `_smoke_`). It holds the effective config (after
  smoke overrides) and its sha256, the full-config sha256, the raw-data sha256, the archive CSV
  sha256, package versions, validation report, row counts, period boundaries, the smoke flag,
  and descriptive correlations labelled "not for selection".

`data/raw/*.csv` and `data/processed/*.csv` are git-ignored (Yahoo redistribution terms). The
manifests are committed. Yahoo revises history, so a rebuild can differ from a past download.
The manifest hashes make that visible.

## 8. CI (`.github/workflows/pipeline.yml`)

* `tests`: push and pull_request, ubuntu-latest, Python 3.12, `pip install -r requirements.txt`,
  `python -m pytest`. `timeout-minutes: 90`.
* `archive`: workflow_dispatch only. Runs the full build with `--n-jobs -1`, then uploads
  `data/raw` and `data/processed` as artifact `forecast-archive` with 30-day retention.
  `timeout-minutes: 90`. A marked comment shows where the replay and experiment steps go.

Parallelism: `--n-jobs` parallelizes the refit blocks. The random forest inside each block is
single-threaded (`make_models(cfg, n_jobs=1)`), to avoid nested oversubscription.

## 9. Known limitations and open points

* The yfinance and Yahoo data are unofficial and can change between downloads (see section 7).
* `realized_return` is the total return from Adj Close, so the replay uses the same
  dividend treatment as the forecasts.
* `vol_ratio` uses `rv_5d / rv_63d`. The proposal does not fix the feature set, so this is a
  choice made here. Changing any feature changes the archive hash.
* Forecasts are refit every 21 days, not daily. The schedule is a design choice, documented in
  the manifest.
