# 2¢ for Your Two Cents — Bayesian Valuation of Unobserved Information

Stanford AA228/CS238 final project. An agent sees a free linear forecast of the next-day
return of SPY, and may pay for two more forecasts (random forest "A", gradient boosting
"B") before choosing a position q ∈ {−1, 0, 1}. It values each unseen forecast by its
expected value of information (VOI) under a joint Gaussian belief, and decides what to buy
either myopically or with an exact two-step Bellman recursion.

* Results and validation status: `docs/milestone_report.md`
* Precise definitions and departures from the proposal: `docs/mathematical_assumptions.md`
* Design decisions: `docs/implementation_decisions.md`; per-workstream notes: `docs/notes_A.md` … `notes_D.md`
* Module APIs and ownership: `docs/CONTRACTS.md`

## Layout

```
configs/default.toml        every parameter of the default run (seed, dates, models, costs, λ, τ)
src/utility.py              utility, stop value, position tie rule
src/interfaces.py           Belief protocol, AcquisitionSetup, PlanResult
src/toy_simulator.py        exact discrete testbed + brute-force policy enumeration (Milestone A)
src/data_pipeline.py        SPY download / validation / causal features
src/forecast_models.py      ridge (free), random forest (A), gradient boosting (B)
src/forecast_archive.py     walk-forward forecast archive with timing assertions
src/belief_model.py         joint Gaussian belief (Ledoit–Wolf), exact conditioning
src/voi.py, src/planner.py  closed-form VOI, two-step Bellman planner
src/policies.py             six policies, ForecastVault, run_episode
src/backtest.py             historical replay (Milestone B) with accounting checks
src/evaluation.py           block bootstrap, paired differences, VOI calibration
experiments/                build_archive, run_replay, run_experiments, make_figures
tests/                      unit, brute-force and integration tests
```

## Reproduce

Python 3.12, `pip install -r requirements.txt`.

Tests (light; fine on a laptop):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest
```

Full pipeline (heavy; run on GitHub Actions, not on a shared machine):

```bash
gh workflow run experiments.yml --ref main          # builds the archive, then replay + Experiments 0-4 + figures
gh run download <run-id> -n replay-results          # results/ and data/processed/ from that run
```

The same steps by hand:

```bash
python -m experiments.build_archive --config configs/default.toml --n-jobs -1   # downloads SPY, writes data/processed/forecast_archive.csv
python -m experiments.run_replay --config configs/default.toml --tag default    # Milestone B replay
python -m experiments.run_experiments --workers 4                               # Experiments 0-4
python -m experiments.make_figures
```

Small checks: `python -m experiments.build_archive --smoke` (500 origins) and
`python -m experiments.run_replay --synthetic 300 --max-dates 20` (labelled synthetic data).

Raw price data is not committed; the manifests in `data/raw` and `data/processed` record
the sha256 of the files each result was produced from.
