"""Policies, the hidden-value vault, the episode runner, and policy evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from src import planner
from src.belief_model import GaussianJointModel
from src.interfaces import STOP, AcquisitionSetup
from src.policies import (
    AlwaysBuy,
    ForecastVault,
    FixedOrder,
    FreeOnly,
    MyopicVOI,
    OptimalVOI,
    RandomAcquisition,
    run_episode,
)
from src.utility import POSITIONS, UtilityParams, expected_utility


def _belief(s, rho1, rho2, kappa, m):
    cov = np.array(
        [[s * s, 0.0, rho1 * s, rho2 * s], [0.0, 1.0, 0.0, 0.0], [rho1 * s, 0.0, 1.0, kappa], [rho2 * s, 0.0, kappa, 1.0]],
        dtype=float,
    )
    return GaussianJointModel(mean=np.array([m, 0.0, 0.0, 0.0]), cov=cov).belief({"y0": 0.0})


A3 = dict(s=1.0, rho1=0.15, rho2=0.22, kappa=0.68, m=0.575)
A3_SETUP = dict(c1=2e-5, c2=8e-4, lam=0.03, tau=0.008)
A2 = dict(s=0.01, rho1=0.05, rho2=0.08, kappa=0.3, m=4e-4)
A2_SETUP = dict(c1=5e-5, c2=1e-4, lam=2.0, tau=2e-4)


def _setup(d, budget=1.0):
    return AcquisitionSetup(costs={"A": d["c1"], "B": d["c2"]}, budget=budget, params=UtilityParams(lam=d["lam"], tau=d["tau"]))


def _vault(setup, values=None, seed=0):
    rng = np.random.default_rng(seed)
    if values is None:
        values = {"A": float(rng.normal(0.0, 1.0)), "B": float(rng.normal(0.0, 1.0))}
    return ForecastVault(values, dict(setup.costs))


# ---------------------------------------------------------------- vault
def test_vault_hides_values_and_refuses_premature_or_repeated_access():
    setup = _setup(A3_SETUP, budget=1.0)
    vault = ForecastVault({"A": 0.123, "B": -0.456}, {"A": 2e-5, "B": 8e-4})
    # No public accessor for the forecasts.
    public = [n for n in dir(vault) if not n.startswith("_")]
    assert not any(isinstance(getattr(vault, n, None), dict) for n in public)
    assert not any(isinstance(getattr(vault, n, None), dict) for n in public)  # sanity: no public dict
    with pytest.raises(AttributeError):
        _ = vault.values  # noqa: B018
    # Unknown and free sources are refused.
    with pytest.raises(KeyError):
        vault.query("y0", 1.0)
    with pytest.raises(KeyError):
        vault.query("C", 1.0)
    # Unaffordable request is refused and charges nothing.
    with pytest.raises(RuntimeError):
        vault.query("B", 1e-4)
    assert vault.spent == 0.0
    # First purchase: value returned, cost charged once.
    assert vault.query("A", 1.0) == 0.123
    assert vault.spent == pytest.approx(2e-5)
    # Repeat is refused and charges nothing more.
    with pytest.raises(RuntimeError):
        vault.query("A", 1.0)
    assert vault.spent == pytest.approx(2e-5)
    assert [e["source"] for e in vault.log] == ["A"]
    del setup


def test_vault_rejects_inconsistent_construction():
    with pytest.raises(ValueError):
        ForecastVault({"A": 1.0}, {"A": 1.0, "B": 1.0})
    with pytest.raises(ValueError):
        ForecastVault({"y0": 1.0}, {"y0": 0.0})


# ---------------------------------------------------------------- simple policies
def test_free_only_buys_nothing_and_positions_match_choose_position():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    vault = _vault(setup)
    ep = run_episode(FreeOnly(), b, vault, setup, 0, np.random.default_rng(0))
    assert ep.purchases == ()
    assert vault.spent == 0.0
    m, s2 = b.return_moments()
    vals = {q: expected_utility(q, m, s2, 0, setup.params) for q in POSITIONS}
    assert ep.q == max(vals, key=vals.get)
    assert ep.G == pytest.approx(max(vals.values()))


def test_always_buy_follows_its_budget_rule_in_order():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    ep = run_episode(AlwaysBuy(), b, _vault(setup), setup, 0, np.random.default_rng(0))
    assert ep.purchases == ("A", "B")
    assert ep.budget_left == pytest.approx(1.0 - 5e-5 - 1e-4)

    # Budget covers only A: A bought, B skipped.
    tight = _setup(A2_SETUP, budget=6e-5)
    ep2 = run_episode(AlwaysBuy(), b, _vault(tight), tight, 0, np.random.default_rng(0))
    assert ep2.purchases == ("A",)
    assert ep2.budget_left == pytest.approx(1e-5)

    # Budget covers nothing.
    none = _setup(A2_SETUP, budget=1e-6)
    ep3 = run_episode(AlwaysBuy(), b, _vault(none), none, 0, np.random.default_rng(0))
    assert ep3.purchases == ()


def test_fixed_order_stops_optimally_and_respects_order():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    ep = run_episode(FixedOrder(("B", "A")), b, _vault(setup), setup, 0, np.random.default_rng(0))
    # 2-then-1 is the optimal policy for A2; first purchase is B.
    assert ep.purchases[:1] == ("B",)
    f = planner.solve_fixed_order(b, ("B", "A"), 1.0, setup, 0)
    assert f.value == pytest.approx(2.915057e-4, abs=2e-10)


def test_random_policy_is_reproducible_under_seeded_rng():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    runs = []
    for _ in range(2):
        ep = run_episode(RandomAcquisition(0.5), b, _vault(setup, seed=5), setup, 0, np.random.default_rng(42))
        runs.append((ep.purchases, ep.q, ep.G, [s["action"] for s in ep.steps]))
    assert runs[0] == runs[1]


def test_random_policy_extremes_and_validation():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    never = run_episode(RandomAcquisition(0.0), b, _vault(setup), setup, 0, np.random.default_rng(1))
    assert never.purchases == ()
    always = run_episode(RandomAcquisition(1.0), b, _vault(setup), setup, 0, np.random.default_rng(1))
    assert set(always.purchases) == {"A", "B"} and len(always.purchases) == 2
    with pytest.raises(ValueError):
        RandomAcquisition(1.5)
    with pytest.raises(ValueError):
        RandomAcquisition(0.5).next_action(b, ("A", "B"), 1.0, setup, 0, rng=None)


def test_myopic_voi_stops_on_A3_where_optimal_queries():
    b = _belief(**A3)
    setup = _setup(A3_SETUP)
    action, diag = MyopicVOI().next_action(b, ("A", "B"), 1.0, setup, 0)
    assert action == STOP
    assert diag["voi"]["A"] < 0 and diag["voi"]["B"] < 0
    ep = run_episode(MyopicVOI(), b, _vault(setup), setup, 0, np.random.default_rng(0))
    assert ep.purchases == ()
    opt = run_episode(OptimalVOI(), b, _vault(setup), setup, 0, np.random.default_rng(0))
    assert opt.purchases[:1] == ("A",)


def test_myopic_voi_picks_argmax_on_A4():
    b = _belief(**A3)
    setup = _setup(dict(A3_SETUP, c1=5e-6, c2=6.5e-4))
    action, diag = MyopicVOI().next_action(b, ("A", "B"), 1.0, setup, 0)
    assert action == "B"
    assert diag["voi"]["B"] > diag["voi"]["A"] > 0


def test_optimal_voi_reacts_to_observation_and_uses_each_source_once():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    ep = run_episode(OptimalVOI(), b, _vault(setup, seed=3), setup, 0, np.random.default_rng(0))
    assert len(set(ep.purchases)) == len(ep.purchases)
    assert ep.total_cost == pytest.approx(sum(setup.costs[s] for s in ep.purchases))
    assert ep.budget_left == pytest.approx(setup.budget - ep.total_cost)


def test_episode_logs_voi_of_every_available_source_and_budget():
    b = _belief(**A2)
    setup = _setup(A2_SETUP, budget=1.5e-4)
    ep = run_episode(MyopicVOI(), b, _vault(setup), setup, 0, np.random.default_rng(0))
    first = ep.steps[0]
    assert set(first["voi"]) == {"A", "B"}
    assert first["mean_before"] == pytest.approx(b.mean)
    assert first["var_before"] == pytest.approx(b.var)
    assert first["budget_left_before"] == pytest.approx(1.5e-4)
    for step in ep.steps:
        assert step["action"] == STOP or step["cost"] <= step["budget_left_before"] + 1e-15


def test_policy_cannot_choose_unavailable_source():
    class _Bad(FreeOnly):
        name = "bad"

        def next_action(self, belief, remaining, budget_left, setup, q_prev, rng=None):
            return "B", {}

    b = _belief(**A2)
    setup = _setup(A2_SETUP, budget=5e-5)  # B unaffordable
    with pytest.raises(ValueError):
        run_episode(_Bad(), b, _vault(setup), setup, 0, np.random.default_rng(0))


def test_vault_setup_mismatch_is_refused():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    vault = ForecastVault({"A": 0.0}, {"A": setup.costs["A"]})
    with pytest.raises(ValueError):
        run_episode(FreeOnly(), b, vault, setup, 0, np.random.default_rng(0))


def test_episode_terminates_and_charges_each_cost_once():
    b = _belief(**A2)
    setup = _setup(A2_SETUP)
    vault = _vault(setup, seed=9)
    ep = run_episode(AlwaysBuy(), b, vault, setup, 0, np.random.default_rng(0))
    assert ep.total_cost == pytest.approx(vault.spent)
    assert len(vault.log) == len(ep.purchases) == 2


# ---------------------------------------------------------------- policy values
@pytest.mark.parametrize("instance", [A3, A2])
def test_optimal_policy_value_dominates_other_policies(instance):
    setup = _setup(A3_SETUP if instance is A3 else A2_SETUP)
    b = _belief(**instance)
    V_opt = planner.solve(b, ["A", "B"], 1.0, setup, 0).value
    tol = 1e-4 if instance is A3 else 1e-6
    from src.policies import MyopicVOI as _M

    candidates = {
        "free_only": FreeOnly(),
        "always_buy": AlwaysBuy(),
        "fixed_AB": FixedOrder(("A", "B")),
        "fixed_BA": FixedOrder(("B", "A")),
        "myopic": _M(),
    }
    from src.planner import evaluate_policy

    for name, pol in candidates.items():
        v = evaluate_policy(pol, b, setup, 0, n_nodes=201)
        assert V_opt >= v - tol, name
    v_opt_policy = evaluate_policy(OptimalVOI(), b, setup, 0, n_nodes=201)
    assert abs(v_opt_policy - V_opt) < tol


def test_evaluate_policy_matches_known_values_on_A3():
    b = _belief(**A3)
    setup = _setup(A3_SETUP)
    from src.planner import evaluate_policy

    # Free-only and myopic both stop: G = 0.52708125.
    assert evaluate_policy(FreeOnly(), b, setup, 0) == pytest.approx(0.52708125, abs=1e-12)
    assert evaluate_policy(MyopicVOI(), b, setup, 0) == pytest.approx(0.52708125, abs=1e-12)


def test_evaluate_policy_rejects_stochastic_policies():
    from src.planner import evaluate_policy

    b = _belief(**A2)
    with pytest.raises(ValueError):
        evaluate_policy(RandomAcquisition(0.5), b, _setup(A2_SETUP), 0)
