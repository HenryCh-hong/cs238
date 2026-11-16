# Workstream A: exact discrete simulator and brute-force reference

Files: `src/toy_simulator.py`, `tests/test_toy_simulator.py`, `tests/test_planner_vs_bruteforce.py`.

## Purpose

An exact finite testbed for the acquisition-and-position problem. Every posterior weight,
predictive branch and expectation is a finite sum, so the testbed has no quadrature error.
It implements the `Belief` protocol of `src/interfaces.py`, which lets the same planner code run
on the testbed and on the Gaussian belief. Its job is to validate the recursion, not to model SPY.

## Model

* Finite joint pmf `p[r, y0, A, B]` (`DiscreteJointModel`). `r_values` are the return values;
  `signal_values[name]` gives the numeric value each index of `y0`, `A`, `B` denotes. The pmf must be
  nonnegative and sum to 1 within 1e-12. A `y0` alphabet of size 1 is allowed.
* Alphabets hold distinct values (separation > 2e-12 relative). Observed values are matched to the
  alphabet with a relative tolerance of 1e-12; a value that matches no entry raises `ValueError`.
* `DiscreteBelief(model, observations)`: the posterior is the pmf restricted to the observed indices
  and renormalized. Unobserved signal values enter only through marginalization. A zero-probability
  observation raises `ValueError` at construction or in `condition`.
* `predictive_branches(source, n_nodes)` returns `(P(Y=y|b), b^{source,y})` for every positive-probability
  outcome. `n_nodes` is accepted for protocol compatibility and ignored, since the outcome set is finite.
* `expected_stop_value_after` is the exact sum over branches of `G(b')`.

## Value definitions (match CONTRACTS.md)

* `G(b) = max_q E_b[u(q, r)]` with `u(q, r) = q r - lam q^2 r^2 - tau |q - q_prev|`. The stop value uses
  `src.utility.expected_utility` on the first two moments. The terminal position uses
  `src.utility.choose_position`, which resolves ties by smallest `|q - q_prev|`, then smallest `|q|`, then lowest `q`.
* Net one-step VOI: `VOI_j(b) = E[G(b^{j,Y})] - G(b) - c_j` (`myopic_voi`). Gross VOI is the same without `- c_j`.
* Two-step Bellman value: `V(b, B, U) = max{G(b), max_{j in U, c_j <= B} [-c_j + E V(b^{j,Y}, B - c_j, U \ {j})]}`
  (`exact_two_step_value`). `U` is the set of purchasable sources in `setup.costs` that are not yet observed.
* Ties: STOP wins. A query is preferred only if it beats the current best by more than `QUERY_TOL = 1e-12`.
  Sources are considered in the order `("A", "B")`, so "A" wins ties against "B". `QUERY_TOL` equals
  `src.planner.PLAN_TOL`; the planner comparison test checks the two match in behavior.
* Affordability: `c_j <= budget_left + BUDGET_ATOL` (`AcquisitionSetup.affordable`). Each source is bought at most once.
* Queries cost `c_j` and are subtracted from the realized utility. The position utility includes the
  transaction cost relative to `q_prev`.

## Independence of the reference paths

* `exact_two_step_value` does not import `src.planner`. It uses only the belief protocol and `src.utility`
  for the stop value and tie rules.
* `brute_force_optimal` does not use the Bellman recursion or the belief's posterior arrays. It rebuilds the
  positive-probability hidden worlds `(weight, r, unobserved signal values)` from the raw model pmf restricted
  to the observed values. It enumerates every deterministic adaptive policy tree and scores each tree by
  summing `realized_utility - query costs` over those worlds.
* Tree structure: a node is a history `((source, value), ...)`. Its action is `("stop", q)` for
  `q` in `{-1, 0, 1}`, or `("query", j)` for an affordable unqueried source. For each positive-probability
  outcome of a query there is a child node, and children are chosen independently. Zero-probability outcomes are
  not enumerated, because their actions cannot change any expected value. With two binary sources and full support
  the enumeration has 291 trees (3 + 12^2 + 12^2); the count is tested.
* Tree ties: enumeration puts STOP options first and "A" before "B". The first tree within `1e-13` of the best
  value is returned. Position ties are NOT resolved by the CONTRACTS rule in the returned tree: stop options are
  enumerated in the order -1, 0, 1, so for a tie between q=0 and q=1 with q_prev=1 the returned tree stops at 0
  while `utility.choose_position` gives 1. The optimal VALUE is unaffected (both trees score the same). Only
  the value is contractual for brute force; tree tie-breaking is not.
* `evaluate_policy_tree` validates the tree: it rejects an unaffordable or already-queried source, an unknown
  action, an illegal position, and a reachable node with no action (`KeyError`).

## Scenarios (`scenarios()`)

Closed forms use `lam = 1`, `tau = 1e-4` unless stated, `q_prev = 0`, and `d = 0.01` (the return magnitude in
the two-point instances).

| name | instance | documented property |
|---|---|---|
| `zero_cost_information` | informative A, B; costs 0 | V = E[G(b^{AB})] (full information), V > G, first action is a source |
| `uninformative` | A, B independent of r and each other; costs 1e-4 | gross VOI = 0, net VOI = -c exactly, V = G, STOP |
| `uninformative_zero_cost` | same, costs 0 | net VOI = 0 exactly, STOP wins the tie |
| `duplicate` | B == A in every world; equal costs 2e-4 | after A, VOI_B = -c_B; root action "A" (tie resolved to A); never buys both |
| `independent_informative` | A, B conditionally independent given r | both VOIs positive and equal; V >= G + max VOI; action in {A, B} |
| `xor` | r = +d iff A xor B = 1; costs 1e-4; budget 2e-4 | VOI_A = VOI_B = -c (myopic stops); V = (d - lam d^2 - tau) - 2c = 0.0096 > G = 0; action "A" |
| `insufficient_budget` | budget 5e-5 < min cost | V = G exactly, STOP, only the STOP action value |
| `budget_one_source` | budget 1.5e-4, costs 1e-4 | V = G + max(0, VOI_A, VOI_B); no second purchase on any path |
| `budget_equals_cost` | c_A = budget = 1e-4; c_B = 2e-4 | A affordable at equality; B not; action values {STOP, A} |
| `no_remaining_sources` | A, B already observed | V = G, STOP, action values {STOP} |
| `terminal_tie_qprev0`, `terminal_tie_qprev1` | r in {0, 1}, P(r=1) = 0.3, tau = 0 | u(0) = u(1) exactly; position is 0 for q_prev = 0 and 1 for q_prev = 1 |

The xor value `0.0096` is the closed form. After A alone, `E[r] = 0` exactly for each A, so `G(b^A) = 0` and
`VOI_A = -c`. After both, `E[r] = +-d` and `E[r^2] = d^2`, so `G = d - lam d^2 - tau`.

## Random instances

`random_instance(seed, n_a, n_b, n_r)` draws a seeded random pmf (Gamma weights with 25% zeros), random
alphabets, random costs (some zero), a budget chosen from edge values (0, c_A, c_B, c_A + c_B, a
tiny-below-sum value, random), `lam` from {0, 0.5, 1, 3}, `tau` from {0, 1e-4, 1e-3}, `q_prev` in {-1, 0, 1}. The
belief observes `y0` with probability 0.6 and each source with probability 0.15, always at a value drawn
from a random world, so the observation has positive probability.

## Tests

* `tests/test_toy_simulator.py`: model validation; belief posterior against direct masked Bayes; the martingale and
  total-probability identities on predictive branches; Jensen (gross VOI >= 0) on random instances and
  scenarios; the myopic VOI fast-path identity; the tree-count identity (291); every scenario property;
  exact-vs-brute-force on every scenario; 200 seeded binary and 40 seeded ternary random instances to 1e-12;
  tie handling (STOP, A before B, terminal position); the budget boundary with `BUDGET_ATOL`; policy-tree validation.
* `tests/test_planner_vs_bruteforce.py`: `src.planner.solve` against the exact recursion and brute force on all
  scenarios and 60 random instances. It checks value agreement to 1e-12, and checks that any action difference is
  a tie. It also compares `src.voi.one_step_voi` with `myopic_voi`. The file uses `pytest.importorskip`, so it
  skips if `src.planner` or `src.voi` is missing.

## Assumptions that make the math precise

1. Returns are simple returns with the stated utility; query costs are subtracted, not scaled by the position.
2. The planner runs one decision date with a fixed budget. Costs are charged once per purchase, and each source is
   bought at most once per date (CONTRACTS.md).
3. `y0` is observed before the decision and never bought. The planner receives a belief that already has `y0`
   observed; the testbed does not require this but the scenarios follow it.
4. The terminal position is chosen after all purchases and is a deterministic function of the final belief, so the
   brute-force enumeration over trees covers every stop position on every history.
5. Brute-force trees are enumerated over positive-probability outcomes only; nodes with zero probability carry no
   weight, so this does not change any value.

## Contract notes (no shared file was edited)

* `PLAN_TOL` is defined in `src/planner.py` (1e-12). The testbed defines its own `QUERY_TOL = 1e-12`
  so that it does not import the planner. The planner comparison test checks behavioral agreement.
* `planner.solve` is called as `solve(belief, remaining=tuple, budget=..., setup=..., q_prev=...)`, with
  `remaining` in ("A", "B") order. If the signature changes, update the call in
  `tests/test_planner_vs_bruteforce.py`.
