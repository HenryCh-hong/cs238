# Workstream D notes: closed-form VOI, exact two-step planner, six policies

Owned files: `src/voi.py`, `src/planner.py`, `src/policies.py`, `tests/test_voi.py`,
`tests/test_planner.py`, `tests/test_policies.py`, this file. Read-only inputs:
`src/utility.py`, `src/interfaces.py`, `src/belief_model.py`, `src/toy_simulator.py`.

## 1. Model and assumptions that make the math precise

* Utility `u(q, r) = q r - lam q^2 r^2 - tau |q - q_prev|`, `q in {-1, 0, 1}`. Under a belief with
  mean `m` and second moment `s2 = m^2 + v`, `E[u] = q m - lam q^2 s2 - tau|q - q_prev|`, so
  `G(b)` depends on `(m, v)` only.
* Gaussian belief over `(r, y0, A, B)` with `Var(Y_j | Y0) = 1` (WLOG, since `Delta` is scale free).
  Then `Cov(r, Y_j | Y0) = rho_j s` and `Delta_j = rho_j^2 s^2`. Sources are linear-Gaussian, so the
  posterior variance after any set of observations is data independent, and the posterior mean
  moves by `N(0, Delta)` per observation. The preposterior shift of the mean from source `j` given the
  current belief is `N(0, Delta_j)` with `Delta_j = Cov(r, Y_j | O)^2 / Var(Y_j | O)`.
* Regression instances (A2, A3, A4) are built with `y0` independent of `r, A, B` (the belief after
  `y0` observed is then the prior over `(r, A, B)`), reproducing the normalized `(rho, kappa, s, m)`
  parameterization of `research_notes/03_math_checks.md`.
* Budget: per decision date. A source is purchased at most once. Affordability is
  `c_j <= B + BUDGET_ATOL`.
* Ties: STOP wins unless a query is better by more than `PLAN_TOL = 1e-12` (absolute, utility units).
  Between sources the order `("A", "B")` wins unless the later one is better by more than `PLAN_TOL`.
  Myopic buys only if `VOI_j > PLAN_TOL`. `PLAN_TOL` is far above rounding noise (~1e-17) and far below
  every VOI scale in the study (>= 1e-8 in return units).
* Terminal position: `utility.choose_position` on the final belief, for every policy.
* `PlanResult.value` is the value of the chosen action (G when stopping). If a query beats stopping by
  at most `PLAN_TOL` it is not taken, so `value` can be below the best query value by at most `PLAN_TOL`.

## 2. Closed form (`src/voi.py`)

`E_{X ~ N(m, delta)} [G(X, v - delta)]` is a finite sum of truncated-normal moments. `G(., w)` is a max of
three quadratics, so it is piecewise quadratic with breakpoints that depend only on `(w, lam, tau, q_prev)`.
The code is ported from `analysis/voi_core.py` and generalized to any `UtilityParams` and any `q_prev`.

* `delta = 0` returns `G(m, v)` exactly, so `gross_voi = 0` and net `VOI = -c` exactly.
* `delta = v` (full revelation) is handled; `delta > v` raises (beyond `1e-12` relative slack).
* `lam = 0` and `tau = 0` are handled (linear pieces, `A = 0` root formula).
* Gross VOI is nonnegative in exact arithmetic (Jensen plus martingale). Tests check `>= -1e-14`.

Verification (tests/test_voi.py, 54 tests): 40 random `(m, delta, v, lam, tau, q_prev)` cases against
kink-split adaptive quadrature (agreement to 1e-12); seeded Monte Carlo (CRN, 400k draws, within
6 s.e. plus a 1e-12 floor); delta = 0 exact; delta = v; lam = 0 / tau = 0; array twin; generic
predictive-branch path vs closed form.

Caveat found during the work: plain `scipy.integrate.quad` over `[-40, 40]` without splitting at the
breakpoints of `G` gives a reference that is wrong by about 7e-8 on some random cases. The tests
therefore split at the breakpoints. The closed form agrees with the split quadrature to 2e-16.

## 3. Planner (`src/planner.py`)

`solve(belief, remaining, budget, setup, q_prev, n_nodes)` implements

    V(b, B, U) = max{ G(b), max_{j in U, c_j <= B} [ -c_j + E V(b^{j,Y_j}, B - c_j, U \ {j}) ] }

* Generic path (any `Belief`): recursion over `predictive_branches(j, n_nodes)`. At the last stage
  (no affordable continuation) the expectation is `expected_stop_value_after` when available (exact),
  else a sum of `G` over the branches.
* Gaussian fast path (`GaussianBelief`): at most two sources, so the continuation after `j` is
  `max{G(child), -c_k + E G(grandchild)}`. Given `Y_j`, the child mean is `m + sigma_j Z` with
  `sigma_j = Cov(r, Y_j | O) / sqrt(Var(Y_j | O))`, `Z ~ N(0, 1)`, and the child variance and
  `Delta_k` do not depend on `Y_j`. The outer expectation is a composite Gauss-Legendre rule on the
  kinks of the integrand (see section 4).
* `solve_fixed_order(belief, order, budget, setup, q_prev, n_nodes)`: optimal stopping along a fixed
  order. The next query is the first source in `order` that is unobserved and affordable; unaffordable
  sources are skipped, since the budget only falls. The decision is STOP or query that source.
* `evaluate_policy(policy, belief, setup, q_prev, n_nodes)`: expected net utility of a deterministic
  policy, by recursion over `predictive_branches` with the policy re-queried at every node. Where no
  continuation is affordable the last expectation is exact. Stochastic policies are rejected.
  For a Gaussian belief the branches are quantile midpoints, so `evaluate_policy` carries the
  quantile-rule error (section 4). Use n_nodes = 201 or larger in comparisons.

## 4. Outer integrator: measured convergence

Two outer rules, compared against an adaptive reference:

* **GL** (chosen): kink-split composite Gauss-Legendre in `z`, on `[-12, 12]`. Subintervals are split at
  the breakpoints of `G(., v_child)` and at the stop/continue crossings (grid scan of 801 points, then
  vectorized refinement to about 2e-8). Each interval is then split into panels of width at most 1, with
  `n_nodes` Gauss-Legendre points per panel. The integrand is analytic inside each subinterval.
* **QM** (for comparison): equal-weight quantile midpoints, `n` nodes (the `predictive_branches` rule).
  Same exact last stage. This is what the generic recursion computes on a Gaussian belief.
* **REF**: adaptive `scipy.integrate.quad` on the same kink intervals, each further split into panels of
  width at most 0.5, with brentq-located kinks, `epsrel = 1e-13`. Used only for tests and this study.

`n_nodes` in the GL rule is the number of Gauss-Legendre points per panel (not a total count).
`DEFAULT_OUTER_NODES = 51`.

Error in the first-query value of the optimal action `V` (absolute, utility units). The error in the
other first query is in the table below each regime, where noted. Typical VOI is `max_j |value_j - G|`
at the regime.

| regime (s, lam, tau, rho1/rho2, kappa, m, c1/c2) | typical VOI | GL n=4 | GL n=8 | GL n=25 | GL n=51 | QM n=25 | QM n=51 | QM n=201 |
|---|---|---|---|---|---|---|---|---|
| R1 base A2 (0.01, 2, 2e-4, 0.05/0.08, 0.3, 4e-4, 5e-5/1e-4) | 2.9e-4 | 4e-12 | 1e-17 | 1e-17 | 1e-17 | 7.5e-6 | 3.4e-6 | 7.8e-7 |
| R2 lam=1 mid (0.01, 1, 1e-4, 0.10/0.20, -0.4, 1e-3, 1e-5/1e-4) | 1.3e-3 | 4e-19 | 4e-19 | 0 | 9e-19 | 6.2e-6 | 3.1e-6 | 9.1e-7 |
| R3 lam=5 low sig (0.01, 5, 5e-4, 0.02/0.03, 0.8, 3e-3, 1e-4/1e-3) | 1.0e-3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4 zero mean (0.01, 1, 1e-4, 0.10/0.10, 0, 0, 1e-5/1e-5) | 9.2e-4 | 1e-19 | 1e-19 | 1e-19 | 2e-19 | 9.5e-6 | 4.3e-6 | 9.8e-7 |
| R5 A3 designed (1, 0.03, 8e-3, 0.15/0.22, 0.68, 0.575, 2e-5/8e-4) | 5.1e-4 | 1.6e-9 | 0 | 0 | 1e-16 | 1.6e-4 | 1.0e-4 | 3.9e-5 |
| R6 high-signal tail (0.01, 5, 5e-4, 0.40/0.50, 0.9, 1e-2, 1e-3/1e-2) | 9.9e-3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R7 lam=5 neg kappa (0.01, 5, 1e-4, 0.20/0.30, -0.8, 5e-4, 1e-5/1e-5) | 5.7e-3 | 1e-17 | 1e-17 | 1e-17 | 1e-17 | 8.5e-6 | 4.2e-6 | 1.1e-6 |
| R8 A4 designed (1, 0.03, 8e-3, 0.15/0.22, 0.68, 0.575, 5e-6/6.5e-4) | 5.4e-4 | 0 | 0 | 0 | 0 | 1.6e-4 | 1.0e-4 | 4.0e-5 |
| R9 linear lam=0, tau=0 (0.01, 0, 0, 0.10/0.20, 0.2, 5e-4, 1e-5/5e-5) | 1.2e-3 | 4e-17 | 4e-17 | 7e-19 | 7e-19 | 1.3e-5 | 8.6e-6 | 2.0e-6 |

R3 and R6 are "stop" regimes (no query has positive value); the value is G and both rules agree
exactly. The QM first-query errors there are up to 7e-8 (R3, n=8, falling to 3e-9 at n=201) and 4e-6 (R6, n=201) on the non-chosen source.

Non-chosen first-query values converge more slowly. On R5 and R8 (the A3/A4 instances, where the
B-first stage has narrow kink features), the GL error on the non-chosen source is 1.3e-8 at n=8,
4.7e-10 at n=25, 1.4e-10 at n=51 and 2.9e-11 at n=101. This is below 1e-3 of the typical VOI at n >= 8
and well below 1e-3 of it at n = 51 (3e-7 relative). The chosen-source value and `V` are exact to
machine precision from n = 8 on every regime.

**Default:** `DEFAULT_OUTER_NODES = 51`. Solve time per belief (this laptop, Python 3.14, one thread):
about 3-12 ms for n = 51 across the regimes; n = 8 is 3-10 ms. The reference takes 0.3-2.5 s per belief.

**Quantile midpoints are not accurate enough** for this recursion. Their error decays only like `1/n`
on the kinked integrand (audit, section 11: the B-first one-step value on A3 has error 2.6e-4 at n = 201,
4.6e-5 at n = 801; the policy value of `OptimalVOI` on A3 is off by 20% of the VOI at n = 51, 8% at n = 201,
2.6% at n = 801). That is why the Gaussian fast path does not use `predictive_branches`. The generic path
still does, so a generic belief gets `n_nodes` as its outer resolution. This is a concern for
`belief_model.py` (see section 8).

**Complete convergence table (audit, requested n values).** Absolute errors against the independent
references (REF = adaptive quad, exact closed form for one-step values). `GL` = `solve` action values
(`n_nodes` = Gauss-Legendre points per panel); `QM` = one-step `E[G]` on quantile midpoints (`n` branches).
Taken from `src/planner.py` as committed in this audit.

| regime | n | GL err A | GL err B | QM one-step err A | QM one-step err B |
|---|---|---|---|---|---|
| A3 (R5) | 25 | 0 | 7.6e-10 | 2.6e-5 | 6.3e-4 |
| A3 (R5) | 51 | 0 | 1.4e-10 | 9.6e-6 | 6.6e-4 |
| A3 (R5) | 101 | 0 | 2.9e-11 | 1.5e-6 | 3.9e-4 |
| A3 (R5) | 201 | 0 | 2.2e-11 | 2.7e-6 | 2.6e-4 |
| A3 (R5) | 401 | 1e-16 | 1.5e-12 | 4.8e-6 | 9.0e-5 |
| A3 (R5) | 801 | 0 | 3.6e-13 | 5.9e-6 | 4.6e-5 |
| A4 (R8) | 25 | 0 | 4.7e-10 | 2.6e-5 | 6.3e-4 |
| A4 (R8) | 51 | 1e-16 | 9.7e-11 | 9.6e-6 | 6.6e-4 |
| A4 (R8) | 101 | 0 | 1.2e-11 | 1.5e-6 | 3.9e-4 |
| A4 (R8) | 201 | 0 | 6.8e-12 | 2.7e-6 | 2.6e-4 |
| A4 (R8) | 401 | 0 | 1.1e-12 | 4.8e-6 | 9.0e-5 |
| A4 (R8) | 801 | 1e-16 | 3.5e-13 | 5.9e-6 | 4.6e-5 |
| A2 (R1) | 25 | 3e-18 | 1e-17 | 5.1e-6 | 7.6e-6 |
| A2 (R1) | 51 | 3e-18 | 1e-17 | 2.2e-6 | 3.5e-6 |
| A2 (R1) | 101 | 3e-18 | 1e-17 | 1.1e-6 | 1.6e-6 |
| A2 (R1) | 201 | 3e-18 | 1e-17 | 4.9e-7 | 7.8e-7 |
| A2 (R1) | 401 | 2.5e-18 | 9e-18 | 2.3e-7 | 3.7e-7 |
| A2 (R1) | 801 | 3e-18 | 1e-17 | 1.1e-7 | 1.8e-7 |

The GL column is the planner's rule and is at round-off from n = 25 on every instance above. The QM
column decays like `1/n` (each doubling of `n` halves the error, roughly); it is not `1/n^2`.
The earlier statement of `1/n^2` was wrong.

The `evaluate_policy` path on a Gaussian belief keeps the QM rule in its first expectation (see section
8 and 11). The exact last-stage closed form is still used there.

## 5. Reference bug found and fixed (during this workstream)

First version of the reference used `quad` on the kink-split intervals. On the R5 B-first stage this
was wrong by 7e-9 (quad on `[-2.45, 12]` returned an error estimate of 1e-14 for a value that was off;
quad on 0.5-wide panels matched a 400k-point trapezoid to 6e-14). The reference now splits each kink
interval into panels of width at most 0.5 before `quad`. This is the only numerical fix that changed
the tables above. The solve results were not affected, because the GL rule with panels already gave
the right values (the B-first error came from the reference).

## 6. Regression targets reproduced (`tests/test_planner.py`, `tests/test_policies.py`)

| instance | target | this implementation |
|---|---|---|
| A3 (s=1, rho 0.15/0.22, kappa 0.68, lam 0.03, tau 0.008, m 0.575, c 2e-5/8e-4) | V_opt = 0.5275865491277 | 0.5275865491277 (exact to 13 digits), first query A |
| A3 one-step VOIs | -1.300761e-5, -1.025902e-4 | match to 1e-10; myopic stops |
| A4 (same as A3 with c 5e-6/6.5e-4) | V_opt = 0.5276213267930, VOI_A = 1.992390e-6, VOI_B = 4.740979e-5 | 0.5276213267930; optimal first A, myopic argmax B |
| A4 myopic-first branch value | 0.5271286772886 | 0.5271286773860 (difference 1e-10; the notes' value came from a dense trapezoid) |
| A2 (s=0.01, rho 0.05/0.08, kappa 0.3, lam 2, tau 2e-4, m 4e-4, c 5e-5/1e-4) | V_opt = 2.915057e-4 (2-then-1) ; 1-then-2 = 2.715823e-4 | 2.915057061e-4 (action B); fixed (A,B) 2.715823e-4 |

## 7. Policies and episode runner (`src/policies.py`)

* `FreeOnly`: buys nothing.
* `AlwaysBuy`: buys the first affordable remaining source in order A, B; stops when none is affordable.
* `FixedOrder(order)`: optimal stopping along the order (`solve_fixed_order`).
* `RandomAcquisition(p)`: with probability `p` buys a uniformly random affordable remaining source,
  else stops. Randomness only from the episode rng. `deterministic = False`, so `evaluate_policy`
  refuses it.
* `MyopicVOI(n_nodes)`: buys the argmax net one-step VOI if it exceeds `PLAN_TOL`; repeats.
* `OptimalVOI(n_nodes)`: `solve` on the current belief; re-solves after every observation.
* `ForecastVault(values, costs)`: values in a name-mangled private attribute, no public accessor.
  `query(source, budget_left)` returns the value, charges the cost once, and records a log entry. It
  raises `KeyError` for unknown or free sources (`y0` included), `RuntimeError` on a repeat, and
  `RuntimeError` when the cost exceeds the budget left.
* `run_episode(policy, belief, vault, setup, q_prev, rng) -> EpisodeResult`: the only path to a
  hidden value is `vault.query`. Per step it records the belief mean and variance before the step,
  the budget left, the net one-step VOI of every available source, the action, and the predicted value.
  It returns purchases, final mean and variance, `G`, `q`, and the total cost.

Design choices to check:

* `Policy.next_action` takes an extra keyword `rng=None`. The contract signature is kept, and the
  stochastic policy needs the episode rng. Deterministic policies ignore it.
* `FixedOrder` skips unaffordable sources (the budget only falls) rather than stopping at them.
* `run_episode` requires the vault's source set and costs to match the setup.

## 8. Concerns for the lead / other workstreams (contract_concerns)

1. `GaussianBelief.predictive_branches` uses quantile midpoints, which converge only like `1/n` on the
   kinked integrand (corrected from an earlier `1/n^2`; see section 4). For any use of the generic
   recursion or of `predictive_branches` on a Gaussian belief, n = 201 leaves about 8% of the policy value
   on A3 unresolved and about 0.3% of the VOI on A2 (section 11). The planner's Gaussian fast path avoids
   this. `GaussianBelief.expected_stop_value_after` is exact closed form and does NOT inherit the error.
   `evaluate_policy` (first stage) and `voi.gross_voi` on a belief without the fast path do inherit it.
   Recommend: `belief_model.py` documents the rate, or offers a kink-split rule (the planner's
   `_gauss_rule` idea) for `predictive_branches`.
2. `CONTRACTS.md` lists `solve(..., n_nodes=None)` without saying what `n_nodes` means. Here it is the
   Gauss-Legendre order per panel on the Gaussian path and the branch count on the generic path.
3. `src/toy_simulator.py` and `tests/test_planner_vs_bruteforce.py` exist: the brute-force cross-check
   runs against `solve` and passes (section 9).

## 9. Test results (last run)

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 -m pytest tests/test_voi.py tests/test_planner.py tests/test_policies.py tests/test_planner_vs_bruteforce.py`

* `tests/test_voi.py`: 54 passed (about 1.5 s).
* `tests/test_planner.py`: 21 passed (about 10 s; includes the independent dense-reference tests and the
  missing-cost test added in the audit, section 11).
* `tests/test_policies.py`: 18 passed (about 1.5 s).
* `tests/test_planner_vs_bruteforce.py` (Workstream A, not edited): run together; the four files
  report 166 passed in total, so this file contributes 73.

One warning remains: a scipy `IntegrationWarning` from one split-quadrature test case with an
extreme breakpoint configuration. The reference answer agrees with the closed form to its tolerance.

## 10. Runtime

Solve (GL, n = 51, Gaussian belief, two sources, one thread): 3-12 ms per belief on this laptop.
One-step VOI per source: under 1 ms. Full episode with `OptimalVOI` on a two-source instance: under 50 ms.

## 11. Independent audit (second engineer, after implementation)

Checks run against the code as it stood, with the results below. Scripts were scratch files and are not part of the repo.

* **Gaussian two-stage value vs an independent dense reference.** The reference is a plain double
  trapezoid in z (N = 4001 outer, M = 801 inner), using only `GaussianBelief.condition` for child
  moments and the raw utility, with no kink detection and no closed form. Agreement with `solve`:
  A3 1.9e-10, A4 1.9e-9, A2 2.6e-10, R2 1.7e-12, R9 (linear) 5e-11. A random sweep of 12 positive definite
  regimes (s in {0.001, 0.01, 1}, |rho| up to 0.9, |kappa| up to 0.95, lam in {0, 0.03, 1, 5, 50},
  tau in {0, 1e-4, 8e-3, 0.05}, q_prev in {-1, 0, 1}) gave a worst difference of 2.6e-9.
  The dense reference was added as a test (`test_solve_matches_independent_dense_reference`, tolerance 1e-7).
* **Closed form on a wider domain.** 400 random cases with lam in {0, 0.01, 1, 5, 100}, tau in {0, 1e-6, 1e-4,
  5e-3, 0.3}, v in [1e-8, 0.1], delta/v in {1e-9, ..., 1}, and m up to 10 prior standard deviations, against
  a 400001-point trapezoid in z. Worst absolute error 8e-11 (where the reference trapezoid itself is of
  that order), no failures.
* **Mutation tests (scratch copies).** Scaling the Gaussian child-mean sigma by 1.01 is caught by the regression
  targets and by the new dense test. Flipping the sign of the tau term in the closed-form coefficients is caught
  by 12 closed-form tests and the linear-piece test. Dropping the G-breakpoint kinks from the planner's rule
  is caught only by the A4 regression target (at 1e-11); the effect is between 1e-11 and 1e-7.
* **Convergence table.** Completed for n in {25, 51, 101, 201, 401, 801} (section 4). The earlier `1/n^2` claim
  was wrong: the quantile rule converges like `1/n`.
* **Solve time.** About 7 ms per call on this laptop with default n, and 6 ms with n = 8 (checked on A3).

### Findings

1. **Fixed.** `planner.solve` and `planner.solve_fixed_order` raised a bare `KeyError` when a remaining source had
   no cost in the setup. They now raise `ValueError` (`_validate_priced`). Failing test first:
   `test_source_without_cost_is_rejected_with_value_error`.
2. **Fixed (documentation).** The section 4 and section 8 claims of `1/n^2` convergence for quantile midpoints
   were wrong (measured about `1/n`). `tests/test_voi.py` had the same wrong comment; corrected.
3. **Fixed (documentation).** Section 8 said `expected_stop_value_after` inherits the quantile error. It does not:
   it is the exact closed form. Corrected.
4. **Not fixed (accuracy gap, spec-level).** `evaluate_policy` on a Gaussian belief expands the first
   stage over quantile midpoints, so its error decays only like `1/n`. Measured against the planner's
   value `V_opt`: for `OptimalVOI` on A3 the gap is -1.0e-4 (n = 51), -3.9e-5 (n = 201), -1.3e-5 (n = 801),
   which is 20%, 8% and 2.6% of the VOI. On A2 it is -3.4e-6, -7.8e-7, -1.8e-7 (1.2%, 0.27%, 0.06%). The spec
   asks for absolute errors of 1e-3 x VOI, so the default `n_nodes=None` (201) does not meet it. The spec
   describes `evaluate_policy` as recursion over predictive branches, so this is left as is. Fixing it needs
   kink-split nodes over the policy's own decision boundaries (for two sources the tree is depth 2, so only
   the first stage needs quadrature). Recommended as the next step if the policy values in the write-up need
   1e-3 accuracy.
5. **Weak test (tolerance).** `test_optimal_policy_value_dominates_other_policies` uses tol 1e-4 on A3, which is
   about 20% of the VOI at the n it uses, and tol 1e-6 on A2, where the optimal policy is `fixed_BA` and the
   comparison is between the same quantity evaluated twice. The test does not detect suboptimality below
   about 1e-4 on A3. It is kept because it detects gross policy bugs, but it does not verify the optimal
   policy to the accuracy the spec asks for. The dense-reference tests in `test_planner.py` are the tight checks.
6. **Spec gap.** `ForecastVault` enforces only the `budget_left` the caller passes. It has no episode budget of its
   own, so a caller can overspend across calls. `run_episode` passes the correct budget, so the episode path is
   safe. Left as is.
7. **Spec gap.** `FixedOrder` and `solve_fixed_order` skip unaffordable sources (documented choice, section 3).
8. **Fragility, not observed to fail.** `_kinks_z` finds stop/continue crossings by a sign change on an 801-point
   grid. A crossing pair inside one grid cell, or an exact zero at a grid node, is missed. No sweep showed an
   effect above 2.6e-9. Left as is.
9. **Known warning.** One `IntegrationWarning` in `test_closed_form_matches_split_quadrature` (extreme breakpoint
   configuration). The closed form agrees with the reference to its tolerance.

No confirmed error in the closed form, the Gaussian two-stage recursion, the generic recursion, the vault, or
the episode runner. The planner's Gaussian path agrees with an independent computation to about 1e-9.

Test counts after the audit (last run, same command as section 9): 166 passed (voi 54, planner 21,
policies 18, brute-force 73).

