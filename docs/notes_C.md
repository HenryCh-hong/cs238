# Workstream C: Gaussian belief model (notes)

Files owned: `src/belief_model.py`, `tests/test_belief_model.py`, this file.
Public names and signatures from `docs/CONTRACTS.md` are unchanged
(`GaussianJointModel`, `GaussianBelief`, `fit_joint_gaussian`, and the belief methods).

## Model and conditioning

* Joint law `N(mu, Sigma)` over `(r, y0, A, B)`. A belief is the conditional law of the
  unobserved coordinates given the observed ones. It is recomputed from the joint model
  and the full observation dict on every `condition`, so order independence holds by
  construction (A then B = B then A = joint conditioning; tested).
* Gaussian conditioning: `mu_u | o = mu_u + S_uo S_oo^+ (y_o - mu_o)`,
  `Sigma_u | o = S_uu - S_uo S_oo^+ S_ou`. The pseudo-inverse equals the inverse whenever
  `S_oo` is nonsingular, which is always true for fitted models (see floor below).
* Delta_j = Cov(r, Y_j | O)^2 / Var(Y_j | O) (clipped to [0, Var(r|O)], a Cauchy-Schwarz
  bound that only removes rounding excess). Identities tested:
  `Delta_A + Delta_{B|A} = Delta_B + Delta_{A|B} = Var(r|O) - Var(r|O,A,B)`.
* Posterior variance is data independent (tested by varying observed values).

## Decisions and assumptions that make the math precise

1. **Singular observed block.** A `GaussianJointModel` may be PSD but singular (exact
   duplicates given directly). If the observed block is singular (smallest eigenvalue
   <= `SINGULAR_RTOL` times largest), the observations must lie in its range; otherwise
   `condition` raises `ValueError` ("inconsistent"). Previously the pseudo-inverse silently
   accepted inconsistent values, e.g. `B = 5` against a duplicate `A = 0.2`.
2. **Duplicates in data.** Standardized columns with `|corr| > 1 - DUPLICATE_RTOL`
   (`DUPLICATE_RTOL = 1e-10`, up to sign) are collapsed to one representative before
   estimation, then expanded exactly. This is needed because Ledoit-Wolf shrinks toward the
   identity, which turns an exact duplicate into correlation `1 - delta`. Measured on data
   with `B = A` and a strong `r`-`A` correlation: LW gave `gain(B|A,y0)/Var(r) = 3.1e-3`
   before the fix. After the fix it is 2.6e-9 (sample method: 2.8e-9).
3. **Positive definiteness.** The eigenvalue floor (`eig_floor_rel = 1e-8`) is applied to
   the full expanded correlation matrix, after duplicate expansion. Exact duplicates
   therefore get an O(1e-8) perturbation (so the fit is PD), and their conditional gain is
   O(1e-9) of `Var(r)`, which is "~0" in the sense required. Applying the floor only to the
   reduced matrix would have left the expanded model singular.
4. **Constant columns.** A column is constant when its spread is exactly zero or its
   sample sd is at most `CONSTANT_RTOL = 1e-12` times its magnitude. The old test
   `sd <= 0` missed a constant 0.7 column (sd = 1.1e-16 from rounding), which was then
   standardized into noise. Such columns now raise `ValueError` naming the column.
5. **Ledoit-Wolf.** Intensity `beta_bar / ||S - mu I||_F^2` on standardized, centered data.
   Verified equal to `sklearn.covariance.ledoit_wolf_shrinkage` on the standardized data
   (rel 1e-10), and the shrunk correlation equals the normalized `LedoitWolf().covariance_`
   (rel 1e-9). sklearn's intensity on the *raw* data differs (0.0077 against 0.0610 in a
   probe with unequal column scales) because the intensity is invariant only to a common
   rescaling of all columns, not to per-column scaling. Standardizing makes the intensity
   invariant to per-column scale, which is why it is computed on standardized data.
6. **Volatility scaling.** `scale_model(m, s)` gives `N(mu s, Sigma s^2)`. `fit_standardized_model`
   fits `x_t / sigma_t`, row t and entry t referring to the same origin. `ewma_volatility`:
   `sigma[t] = sqrt(sum_{s<=t} lam^{t-s} r_s^2 / sum_{s<=t} lam^{t-s})`, `lam = 0.5^(1/halflife)`,
   with `returns[t]` the return realized at the close of day t. It uses `returns[0..t]` only
   (causality tested by replacing the future). At origin t the target is the next-day
   return, which never enters `sigma[t]`. Squares are not demeaned (zero-mean convention).
7. **Calibration frames.** `fit_from_frame(df, method, last_n, cols)` maps
   `(realized_return, free_forecast, rf_forecast, gb_forecast)` position-wise to
   `("r","y0","A","B")`. It raises on NaN, on too few rows, on a `date` column that is not
   non-decreasing, and on `last_n` larger than the frame. The caller is responsible for
   choosing a calibration window that ends before the decision date (CONTRACTS periods).
8. **Predictive branches (semantics kept).** Equal-weight quantile-midpoint nodes of the
   predictive law of the unobserved forecast. Degenerate sources return one branch at the
   predictive mean. Node count must be >= 1.

## Numerical findings (not bugs; kept as documented limitations)

* **Quantile-midpoint bias.** The preposterior variance of the branch means is
  `gain * (1 - 1.28/n + ...)`: 0.64% low at n=201 (ratio 0.9936), 0.13% low at n=1001,
  0.013% low at n=10001. The n=201 default therefore understates the increment variance
  by about 0.6%. Not corrected, per the instruction to keep the node semantics. The
  closed-form path (`expected_stop_value_after`) does not have this bias.
* **Ledoit-Wolf and exact linear dependence.** Shrinking toward the identity breaks an
  exact linear dependence that is not a duplicate (e.g. `y0 = A + B`). With LW,
  `Var(y0 | A, B)` is about `delta * Var(y0)` (about 5% in the test case), while the sample
  estimator gives a floor-level residual. Both fits are PD and finite. Use
  `method="sample"` if sources are exactly linearly dependent. Exact duplicates are handled
  by collapsing (point 2).
* **Eigenvalue floor.** Affects only models whose correlation has eigenvalues below 1e-8.
  Outside duplicates and exact dependence this does not trigger on the calibration data.

## Tests (tests/test_belief_model.py)

Last run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 -m pytest tests/test_belief_model.py`
gave **50 passed** in about 1.8 s wall time.

Coverage:
* conditional moments vs an independent Schur implementation (explicit `np.linalg.solve`):
  all observed subsets of {y0, A, B} on a fixed model, and 25 random models;
* least-squares regression on 200,000 seeded samples: OLS coefficients within 5 SE of the
  Schur gains; residual variance within 2% of the posterior variance;
* order independence (A-then-B, B-then-A, joint);
* Delta identities with and without y0 observed;
* posterior variance independent of observed values;
* predictive mean and variance vs Schur; branch weights, mean preservation, increment
  variance (rel 1.5/201 at n=201, rel 2e-4 at n=10001), branch variances;
* PSD and near-singular fits: exact duplicate (both methods, PD, gain ~0), B = A + 1e-9
  noise, rank-deficient linear dependence (both methods), direct singular model;
* inconsistent observations on singular blocks raise; non-PSD covariance rejected;
* near-perfect correlation 0.9999 (model and fit), low-variance y0 (scale 1e-9) and global
  scaling by 1e-6 (gains scale as s^2);
* constant columns (including rounding-level constants) raise; NaN and too few rows raise;
* one- and two-acquisition updates;
* Ledoit-Wolf intensity vs sklearn (rel 1e-10) and correlation vs sklearn (rel 1e-9);
* sampling moments of `GaussianJointModel.sample` (5 SE);
* `fit_from_frame` (name mapping, tail window, NaN, short, unsorted, bad columns);
* EWMA vs a direct double sum, causality, two-point closed form, min_periods, validation;
* `scale_model` date-t law and quadratic gain scaling; standardized fit round trip;
* `conditional_summary` vs definitions and the gain identity;
* no hidden access: beliefs store only passed observations; branches depend only on the
  belief; `condition` is the only way values enter; unknown or `r` observations rejected.

Not covered here: `expected_stop_value_after` (needs `src.voi`, owned by workstream D).

## Open items for the lead

* GitHub repository creation was requested in the relayed user request. It is outside this
  workstream and was not done here. The lead should decide when to create it.

## Audit (second engineer, workstream C)

* **Fixed: scale-dependent singularity test in conditioning.** The observed block was
  pseudo-inverted in its raw units, so a source at scale 1e-9 (eigenvalue ratio ~1e-18 < 1e-13)
  was treated as singular and its information was silently dropped (posterior mean off by ~49%).
  Conditioning now works on the correlation scale of the observed block
  (`S_uo D^-1 R_oo^+ D^-1`, `R_oo = D^-1 S_oo D^-1`). Regression test:
  `test_conditioning_is_invariant_to_rescaling_an_observed_source`.
* **Limitation (not fixed, estimator choice): Ledoit-Wolf on near-duplicates.** Only exact
  duplicates (|corr| > 1 - 1e-10) are collapsed. For `B = A + s * noise` with s from 1e-4 to
  0.3 (in units of sd_A), Ledoit-Wolf gives `gain(B | A, y0) / Var(r)` of about 1e-3 to 2.5e-3,
  while the sample estimator gives about 1e-6 to 2e-3 (small at some s and not monotone).
  The LW gain comes from shrinkage toward the identity, not from data. Use `method="sample"`
  when sources are near-duplicates.
* **Quantile-midpoint bias confirmed.** `1 - mean(z_k^2)` over midpoint nodes is 0.637% at
  n = 201, 0.130% at n = 1001 (about 1.28/n). Documented above, not corrected.
