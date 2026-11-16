"""Tests for the exact discrete testbed (src/toy_simulator.py).

Every value is compared against an independent path: closed forms derived by hand in
the scenario docstrings, world enumeration (brute force over policy trees), or direct
Bayes computations on the joint pmf. Run with the single-threaded, niced command in
docs/CONTRACTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import toy_simulator as ts
from src.interfaces import BUDGET_ATOL, STOP, AcquisitionSetup
from src.utility import POSITIONS, choose_position, expected_utility, realized_utility

TOL = 1e-12
SCEN = {s.name: s for s in ts.scenarios()}


# ----------------------------------------------------------------- helpers
def _full_support_model(n_a: int = 2, n_b: int = 2, seed: int = 0) -> ts.DiscreteJointModel:
    rng = np.random.default_rng(seed)
    shape = (2, 1, n_a, n_b)
    pmf = rng.uniform(0.1, 1.0, shape)
    pmf /= pmf.sum()
    return ts.DiscreteJointModel(
        np.array([-0.01, 0.01]), pmf,
        {"y0": np.array([0.0]), "A": np.linspace(-1, 1, n_a), "B": np.linspace(-1, 1, n_b)},
    )


def _full_info_value(belief, setup, q_prev) -> float:
    """E[G(b^{AB})] with both sources unobserved, by iterated predictive branches."""
    params = setup.params
    acc = 0.0
    for wa, ba in belief.predictive_branches("A"):
        for wb, bb in ba.predictive_branches("B"):
            acc += wa * wb * ts.stop_value(bb, q_prev, params)
    return acc


def _direct_r_posterior(model, observations) -> np.ndarray:
    """P(r | observations) by masking the joint pmf (independent of DiscreteBelief)."""
    mask = np.ones(model.pmf.shape, dtype=bool)
    axis = {"y0": 1, "A": 2, "B": 3}
    for name, value in observations.items():
        i = model.signal_index(name, value)
        keep = np.zeros(model.pmf.shape[axis[name]], dtype=bool)
        keep[i] = True
        shape = [1] * 4
        shape[axis[name]] = keep.size
        mask &= keep.reshape(shape)
    masked = np.where(mask, model.pmf, 0.0)
    p_r = masked.sum(axis=(1, 2, 3))
    return p_r / p_r.sum()


def _assert_matches_bruteforce(belief, setup, q_prev) -> float:
    V, action, action_values = ts.exact_two_step_value(belief, setup, q_prev)
    bf, tree = ts.brute_force_optimal(belief, setup, q_prev)
    assert V == pytest.approx(bf, abs=TOL, rel=0)
    assert ts.evaluate_policy_tree(tree, belief, setup, q_prev) == pytest.approx(bf, abs=TOL, rel=0)
    # The brute-force root action must attain the value (robust to exact ties).
    root = tree[()]
    if root[0] == "stop":
        assert ts.stop_value(belief, q_prev, setup.params) == pytest.approx(bf, abs=TOL, rel=0)
    else:
        assert action_values[root[1]] == pytest.approx(bf, abs=TOL, rel=0)
    # The exact recursion's chosen first action attains the value (STOP included).
    assert action_values[action] == pytest.approx(V, abs=TOL, rel=0)
    return V


def _check_scenario_root_matches(scn) -> float:
    return _assert_matches_bruteforce(scn.belief, scn.setup, scn.q_prev)


# ----------------------------------------------------------------- model validation
def test_model_accepts_valid_and_single_point_y0_alphabet():
    m = _full_support_model()
    assert m.pmf.shape == (2, 1, 2, 2)
    b = ts.DiscreteBelief(m, {"y0": 0.0})
    assert b.observed == frozenset({"y0"})


@pytest.mark.parametrize("bad", ["negative", "not_normalized", "shape", "missing_key", "duplicate_alphabet"])
def test_model_rejects_invalid_inputs(bad):
    pmf = np.full((2, 1, 2, 2), 1.0 / 8.0)
    r = np.array([-0.01, 0.01])
    sig = {"y0": np.array([0.0]), "A": np.array([-1.0, 1.0]), "B": np.array([-1.0, 1.0])}
    if bad == "negative":
        pmf[0, 0, 0, 0] = -0.1
        pmf[1, 0, 0, 0] += 0.1 + 0.125
    elif bad == "not_normalized":
        pmf = pmf * (1.0 + 1e-9)
    elif bad == "shape":
        pmf = pmf[:, :, :1, :]
    elif bad == "missing_key":
        sig.pop("B")
    elif bad == "duplicate_alphabet":
        sig["A"] = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        ts.DiscreteJointModel(r, pmf, sig)


def test_belief_rejects_invalid_conditioning():
    m = _full_support_model()
    b = ts.DiscreteBelief(m, {"y0": 0.0})
    with pytest.raises(ValueError):
        b.condition("A", 0.123)  # out of alphabet
    with pytest.raises(KeyError):
        b.condition("y0", 0.0)  # already observed
    with pytest.raises(KeyError):
        ts.DiscreteBelief(m, {"r": 0.0})  # returns are never observed
    with pytest.raises(KeyError):
        b.predictive_branches("y0")
    with pytest.raises(KeyError):
        b.predictive_branches("r")


def test_zero_probability_observations_raise():
    pmf = np.zeros((2, 1, 2, 2))
    pmf[:, :, 0, :] = 0.25  # A = -1 only; A = +1 has zero probability
    m = ts.DiscreteJointModel(np.array([-0.01, 0.01]), pmf,
                              {"y0": np.array([0.0]), "A": np.array([-1.0, 1.0]),
                               "B": np.array([-1.0, 1.0])})
    b = ts.DiscreteBelief(m)
    with pytest.raises(ValueError):
        b.condition("A", 1.0)
    with pytest.raises(ValueError):
        ts.DiscreteBelief(m, {"A": 1.0})
    branches = b.predictive_branches("A")  # zero-probability outcome is not returned
    assert len(branches) == 1
    assert branches[0][0] == pytest.approx(1.0, abs=TOL, rel=0)
    assert branches[0][1].observed == frozenset({"A"})


# ----------------------------------------------------------------- posterior normalization
@pytest.mark.parametrize("seed", range(40))
def test_posterior_matches_direct_bayes_and_normalizes(seed):
    b, setup, q_prev = ts.random_instance(seed)
    p_direct = _direct_r_posterior(b.model, b.observations)
    assert np.allclose(b.r_marginal(), p_direct, atol=TOL, rtol=0)
    assert abs(b.r_marginal().sum() - 1.0) < TOL
    m, s2 = b.return_moments()
    assert m == pytest.approx(float(p_direct @ b.model.r_values), abs=TOL, rel=0)
    assert s2 == pytest.approx(float(p_direct @ b.model.r_values ** 2), abs=TOL, rel=0)


@pytest.mark.parametrize("seed", range(40))
def test_predictive_branches_are_a_martingale(seed):
    b, setup, q_prev = ts.random_instance(seed)
    for src in [s for s in ("A", "B", "y0") if s not in b.observed]:
        branches = b.predictive_branches(src)
        weights = np.array([w for w, _ in branches])
        assert np.all(weights > 0)
        assert abs(weights.sum() - 1.0) < TOL
        # Law of total probability for the full return law, and martingale for the mean.
        p_mix = sum(w * br.r_marginal() for w, br in branches)
        assert np.allclose(p_mix, b.r_marginal(), atol=TOL, rtol=0)
        m0, _ = b.return_moments()
        m_mix = sum(w * br.return_moments()[0] for w, br in branches)
        assert m_mix == pytest.approx(m0, abs=TOL, rel=0)
        for _, br in branches:
            assert br.observed == b.observed | {src}


# ----------------------------------------------------------------- Jensen and VOI identities
@pytest.mark.parametrize("seed", range(200))
def test_gross_voi_is_nonnegative_random(seed):
    b, setup, q_prev = ts.random_instance(seed)
    for src in [s for s in ("A", "B") if s not in b.observed]:
        gross = ts.gross_myopic_voi(b, src, q_prev, setup.params)
        assert gross >= -1e-15


@pytest.mark.parametrize("name", sorted(SCEN))
def test_gross_voi_is_nonnegative_scenarios(name):
    s = SCEN[name]
    for src in [x for x in ("A", "B") if x not in s.belief.observed]:
        assert ts.gross_myopic_voi(s.belief, src, s.q_prev, s.setup.params) >= -1e-15


@pytest.mark.parametrize("seed", range(30))
def test_myopic_voi_matches_expected_stop_fast_path(seed):
    b, setup, q_prev = ts.random_instance(seed)
    G = ts.stop_value(b, q_prev, setup.params)
    for src in [x for x in ("A", "B") if x not in b.observed]:
        c = setup.costs[src]
        net = ts.myopic_voi(b, src, c, q_prev, setup.params)
        fast = b.expected_stop_value_after(src, q_prev, setup.params) - G - c
        assert net == pytest.approx(fast, abs=TOL, rel=0)


def test_enumeration_count_full_support_binary():
    # Binary A and B, all sources affordable, full support:
    # per A-node: 3 stops + (query B: 2 B-nodes x 3 stops) = 12; root: 3 + 12^2 + 12^2 = 291.
    m = _full_support_model(2, 2)
    setup = AcquisitionSetup(costs={"A": 1e-4, "B": 1e-4}, budget=1.0)
    b = ts.DiscreteBelief(m, {"y0": 0.0})
    assert len(ts.enumerate_policy_trees(b, setup)) == 291


# ----------------------------------------------------------------- scenario properties
def test_zero_cost_information_equals_full_information():
    s = SCEN["zero_cost_information"]
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    full = _full_info_value(s.belief, s.setup, s.q_prev)
    assert G == pytest.approx(0.0, abs=TOL)
    assert V == pytest.approx(full, abs=TOL, rel=0)
    assert V > G
    assert V >= G
    assert action in ("A", "B")
    _check_scenario_root_matches(s)


def test_uninformative_net_voi_equals_minus_cost_and_stops():
    s = SCEN["uninformative"]
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    for src in ("A", "B"):
        gross = ts.gross_myopic_voi(s.belief, src, s.q_prev, s.setup.params)
        net = ts.myopic_voi(s.belief, src, s.setup.costs[src], s.q_prev, s.setup.params)
        assert gross == pytest.approx(0.0, abs=TOL)
        assert net == pytest.approx(-s.setup.costs[src], abs=TOL, rel=0)
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert V == pytest.approx(G, abs=TOL, rel=0)
    assert action == STOP
    assert av["A"] == pytest.approx(G - s.setup.costs["A"], abs=TOL, rel=0)
    _check_scenario_root_matches(s)


def test_uninformative_zero_cost_stops_on_tie():
    s = SCEN["uninformative_zero_cost"]
    for src in ("A", "B"):
        assert ts.myopic_voi(s.belief, src, 0.0, s.q_prev, s.setup.params) == pytest.approx(0.0, abs=TOL)
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert action == STOP
    assert V == pytest.approx(ts.stop_value(s.belief, s.q_prev, s.setup.params), abs=TOL, rel=0)
    _check_scenario_root_matches(s)


def test_duplicate_signal_voi_b_is_minus_cost_after_a_and_one_purchase_only():
    s = SCEN["duplicate"]
    params = s.setup.params
    c_b = s.setup.costs["B"]
    for w, after_a in s.belief.predictive_branches("A"):
        assert ts.myopic_voi(after_a, "B", c_b, s.q_prev, params) == pytest.approx(-c_b, abs=TOL, rel=0)
        assert ts.exact_two_step_value(after_a, s.setup, s.q_prev)[1] == STOP
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert action == "A"
    assert av["A"] == pytest.approx(av["B"], abs=TOL, rel=0)  # symmetric tie, A resolved first
    assert V > ts.stop_value(s.belief, s.q_prev, params)
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    assert bf == pytest.approx(V, abs=TOL, rel=0)
    depths = ts.reachable_depth_actions(tree)
    assert 2 not in depths  # never a second purchase on any path
    assert all(a[0] == "stop" for a in depths.get(1, set()))  # after the purchase, stop


def test_independent_informative_both_voi_positive_and_two_step_dominates():
    s = SCEN["independent_informative"]
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    vois = {src: ts.myopic_voi(s.belief, src, s.setup.costs[src], s.q_prev, s.setup.params) for src in ("A", "B")}
    assert vois["A"] > 0 and vois["B"] > 0
    assert vois["A"] == pytest.approx(vois["B"], abs=TOL, rel=0)  # symmetric instance
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert V >= G + max(vois.values()) - TOL
    assert action in ("A", "B")
    _check_scenario_root_matches(s)


def test_xor_myopic_stops_but_two_step_buys_both():
    s = SCEN["xor"]
    params = s.setup.params
    d = 0.01
    c = s.setup.costs["A"]
    G = ts.stop_value(s.belief, s.q_prev, params)
    assert G == pytest.approx(0.0, abs=TOL)
    for src in ("A", "B"):
        assert ts.myopic_voi(s.belief, src, c, s.q_prev, params) == pytest.approx(-c, abs=TOL, rel=0)
        assert ts.myopic_voi(s.belief, src, c, s.q_prev, params) <= 0
    # Closed form: buy both (cost 2c), then stop in the revealed sign: d - lam d^2 - tau |1 - 0|.
    closed = (d - params.lam * d * d - params.tau) - 2 * c
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert V == pytest.approx(closed, abs=TOL, rel=0)
    assert V == pytest.approx(0.0096, abs=1e-12)
    assert V > G
    assert action == "A"
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    assert bf == pytest.approx(closed, abs=TOL, rel=0)
    assert tree[()] == ("query", "A")
    # After either sign of A the tree buys B (A alone reveals nothing about the sign).
    assert tree[(("A", 0.0),)] == ("query", "B")
    assert tree[(("A", 1.0),)] == ("query", "B")


def test_second_purchase_at_float_budget_boundary_uses_atol():
    # Unequal costs whose sum is rounded so that (budget - c_A) < c_B in floating point.
    # The second purchase is affordable only through BUDGET_ATOL, at depth 2 of the recursion.
    cA, cB = 5.056378869683274e-05, 2.6362359173243803e-05
    budget = cA + cB
    assert budget - cA < cB and budget - cA + BUDGET_ATOL >= cB
    s = SCEN["xor"]
    d = 0.01
    setup = AcquisitionSetup(costs={"A": cA, "B": cB}, budget=budget, params=s.setup.params)
    V, action, av = ts.exact_two_step_value(s.belief, setup, s.q_prev)
    closed = (d - setup.params.lam * d * d - setup.params.tau) - cA - cB
    assert V == pytest.approx(closed, abs=TOL, rel=0)
    assert action == "A"
    bf, tree = ts.brute_force_optimal(s.belief, setup, s.q_prev)
    assert bf == pytest.approx(closed, abs=TOL, rel=0)
    assert tree[(("A", 0.0),)] == ("query", "B")
    assert tree[(("A", 1.0),)] == ("query", "B")


def test_xor_budget_one_short_of_both_costs_blocks_acquisition():
    s = SCEN["xor"]
    short = AcquisitionSetup(costs=s.setup.costs, budget=2e-4 - 1e-9, params=s.setup.params)
    V, action, av = ts.exact_two_step_value(s.belief, short, s.q_prev)
    assert V == pytest.approx(ts.stop_value(s.belief, s.q_prev, s.setup.params), abs=TOL, rel=0)
    assert action == STOP
    bf, tree = ts.brute_force_optimal(s.belief, short, s.q_prev)
    assert bf == pytest.approx(V, abs=TOL, rel=0)
    assert all(v[0] == "stop" for v in tree.values())


def test_insufficient_budget_gives_stop_value():
    s = SCEN["insufficient_budget"]
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert V == G  # exact equality: no query is affordable
    assert action == STOP
    assert set(av) == {STOP}
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    assert bf == pytest.approx(G, abs=TOL, rel=0)
    assert len(tree) == 1 and tree[()][0] == "stop"


def test_budget_for_one_source_never_allows_second_purchase():
    s = SCEN["budget_one_source"]
    params = s.setup.params
    G = ts.stop_value(s.belief, s.q_prev, params)
    vois = {src: ts.myopic_voi(s.belief, src, s.setup.costs[src], s.q_prev, params) for src in ("A", "B")}
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    # After the single purchase nothing is affordable, so V = G + max(0, VOI_A, VOI_B).
    assert V == pytest.approx(G + max(0.0, vois["A"], vois["B"]), abs=TOL, rel=0)
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    assert bf == pytest.approx(V, abs=TOL, rel=0)
    assert 2 not in ts.reachable_depth_actions(tree)
    for _, branch in s.belief.predictive_branches("A"):
        assert ts.exact_two_step_value(branch, s.setup, s.q_prev)[1] == STOP


def test_budget_equal_to_cost_is_affordable_and_drops_unaffordable_source():
    s = SCEN["budget_equals_cost"]
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    vA = ts.myopic_voi(s.belief, "A", s.setup.costs["A"], s.q_prev, s.setup.params)
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert set(av) == {STOP, "A"}
    assert action == "A"
    assert V == pytest.approx(G + max(0.0, vA), abs=TOL, rel=0)
    _check_scenario_root_matches(s)


def test_budget_comparison_uses_budget_atol_at_the_boundary():
    s = SCEN["budget_equals_cost"]
    c = s.setup.costs["A"]
    at_edge = AcquisitionSetup(costs=s.setup.costs, budget=c - 0.5 * BUDGET_ATOL, params=s.setup.params)
    just_below = AcquisitionSetup(costs=s.setup.costs, budget=c - 10 * BUDGET_ATOL, params=s.setup.params)
    assert "A" in ts.exact_two_step_value(s.belief, at_edge, s.q_prev)[2]
    assert "A" not in ts.exact_two_step_value(s.belief, just_below, s.q_prev)[2]
    assert ts.brute_force_optimal(s.belief, just_below, s.q_prev)[0] == pytest.approx(
        ts.stop_value(s.belief, s.q_prev, s.setup.params), abs=TOL, rel=0)


def test_no_remaining_sources_is_stop():
    s = SCEN["no_remaining_sources"]
    G = ts.stop_value(s.belief, s.q_prev, s.setup.params)
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert V == G and action == STOP and av == {STOP: G}
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    assert bf == pytest.approx(G, abs=TOL, rel=0)
    assert list(tree) == [()] and tree[()][0] == "stop"


@pytest.mark.parametrize("q_prev", [0.0, 1.0])
def test_terminal_tie_position_follows_contract(q_prev):
    s = SCEN[f"terminal_tie_qprev{int(q_prev)}"]
    params = s.setup.params
    m, s2 = s.belief.return_moments()
    assert m == pytest.approx(0.3, abs=TOL) and s2 == pytest.approx(0.3, abs=TOL)
    vals = {q: expected_utility(q, m, s2, q_prev, params) for q in POSITIONS}
    assert vals[0] == pytest.approx(vals[1], abs=TOL, rel=0)  # exact tie for the stop decision
    # Contract tie rule: smallest |q - q_prev|, then smallest |q|, then lowest q.
    expected_q = 0 if q_prev == 0.0 else 1
    G, q = choose_position(m, s2, q_prev, params)
    assert q == expected_q
    assert G == pytest.approx(0.0, abs=TOL)
    stop_G, stop_q = ts.stop_choice(s.belief, q_prev, params)
    assert stop_q == expected_q
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert action == STOP
    assert V == pytest.approx(0.0, abs=TOL)


# ----------------------------------------------------------------- scenario vs brute force
@pytest.mark.parametrize("name", sorted(SCEN))
def test_exact_equals_bruteforce_on_every_scenario(name):
    _check_scenario_root_matches(SCEN[name])


# ----------------------------------------------------------------- seeded random instances
def _compare_random(seed, n_a, n_b):
    b, setup, q_prev = ts.random_instance(seed, n_a=n_a, n_b=n_b)
    V, action, av = ts.exact_two_step_value(b, setup, q_prev)
    bf, tree = ts.brute_force_optimal(b, setup, q_prev)
    assert abs(V - bf) <= TOL, (seed, V, bf)
    assert ts.evaluate_policy_tree(tree, b, setup, q_prev) == pytest.approx(bf, abs=TOL, rel=0)
    assert V >= ts.stop_value(b, q_prev, setup.params) - TOL


@pytest.mark.parametrize("seed", range(200))
def test_exact_equals_bruteforce_random_binary(seed):
    _compare_random(seed, 2, 2)


@pytest.mark.parametrize("seed", range(40))
def test_exact_equals_bruteforce_random_ternary(seed):
    _compare_random(1000 + seed, 3, 2 if seed % 2 == 0 else 3)


# ----------------------------------------------------------------- tie handling
def test_sources_tie_resolves_to_a_then_stop_ties_win():
    # Identical sources and costs: query A and query B have equal value, so A is chosen.
    s = SCEN["duplicate"]
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert action == "A"
    # A strictly better B must still win over A.
    assert "A" in av and "B" in av


def test_query_only_if_strictly_better_than_stop():
    s = SCEN["uninformative_zero_cost"]
    V, action, av = ts.exact_two_step_value(s.belief, s.setup, s.q_prev)
    assert action == STOP


def test_brute_force_prefers_stop_then_a_on_ties():
    s = SCEN["uninformative_zero_cost"]
    bf, tree = ts.brute_force_optimal(s.belief, s.setup, s.q_prev)
    m, s2 = s.belief.return_moments()
    _, q_star = choose_position(m, s2, s.q_prev, s.setup.params)
    # Stop options are enumerated first, so the optimal root tree stops with the contract q.
    assert tree[()] == ("stop", q_star)
    assert q_star == 1  # E r > 0 here, so the contract position is long


# ----------------------------------------------------------------- policy validation
def test_evaluate_rejects_unaffordable_query_and_missing_nodes():
    s = SCEN["budget_equals_cost"]
    bad_tree = {(): ("query", "B")}  # B costs 2e-4 but budget is 1e-4
    with pytest.raises(ValueError):
        ts.evaluate_policy_tree(bad_tree, s.belief, s.setup, s.q_prev)
    with pytest.raises(KeyError):
        ts.evaluate_policy_tree({(): ("query", "A")}, s.belief, s.setup, s.q_prev)
    # Stop-at-root with q = 0: E[u(0, r)] = -tau |0 - q_prev| (no query cost).
    p = s.belief.r_marginal()
    closed = sum(float(p[i]) * realized_utility(0.0, float(r), s.q_prev, s.setup.params)
                 for i, r in enumerate(s.belief.model.r_values))
    assert ts.evaluate_policy_tree({(): ("stop", 0)}, s.belief, s.setup, s.q_prev) == pytest.approx(
        closed, abs=TOL, rel=0)


def test_setup_rejects_costs_for_non_sources():
    s = SCEN["uninformative"]
    bad = AcquisitionSetup(costs={"y0": 1e-4}, budget=1e-3, params=s.setup.params)
    with pytest.raises(ValueError):
        ts.exact_two_step_value(s.belief, bad, s.q_prev)
