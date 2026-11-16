"""Cross-check src.planner.solve and src.voi.one_step_voi against the exact testbed.

Skips (never errors) while src.planner / src.voi / planner.solve are not yet present.
The lead runs this after Workstream D is integrated.

Assumed planner call (CONTRACTS.md):
    solve(belief, remaining, budget, setup, q_prev, n_nodes=None) -> PlanResult
with ``remaining`` the tuple of unqueried purchasable sources in ("A", "B") order.
"""
from __future__ import annotations

import pytest

from src import toy_simulator as ts
from src.interfaces import STOP

planner = pytest.importorskip("src.planner", reason="src.planner not yet integrated")
solve = getattr(planner, "solve", None)
if solve is None:  # pragma: no cover - depends on integration state
    pytest.skip("src.planner.solve not present", allow_module_level=True)

TOL = 1e-12


def _remaining(belief, setup):
    return tuple(s for s in ("A", "B") if s in setup.costs and s not in belief.observed)


def _check_planner(belief, setup, q_prev):
    remaining = _remaining(belief, setup)
    exact_v, exact_action, exact_av = ts.exact_two_step_value(belief, setup, q_prev)
    bf_v, _ = ts.brute_force_optimal(belief, setup, q_prev)
    res = solve(belief, remaining=remaining, budget=setup.budget, setup=setup, q_prev=q_prev)
    assert res.value == pytest.approx(exact_v, abs=TOL, rel=0)
    assert res.value == pytest.approx(bf_v, abs=TOL, rel=0)
    assert res.stop_value == pytest.approx(ts.stop_value(belief, q_prev, setup.params), abs=TOL, rel=0)
    if res.action != exact_action:
        # Only acceptable when the two first actions are exactly tied (within tolerance).
        assert res.action in exact_av, (res.action, exact_av)
        assert exact_av[res.action] == pytest.approx(exact_v, abs=TOL, rel=0)
    for key, val in exact_av.items():
        if key in res.action_values:
            assert res.action_values[key] == pytest.approx(val, abs=TOL, rel=0)


@pytest.mark.parametrize("name", [s.name for s in ts.scenarios()])
def test_planner_matches_exact_on_scenarios(name):
    s = ts.scenario(name)
    _check_planner(s.belief, s.setup, s.q_prev)


@pytest.mark.parametrize("seed", range(60))
def test_planner_matches_exact_on_random_instances(seed):
    belief, setup, q_prev = ts.random_instance(seed)
    _check_planner(belief, setup, q_prev)


def test_one_step_voi_matches_myopic_voi():
    voi = pytest.importorskip("src.voi", reason="src.voi not yet integrated")
    one_step = getattr(voi, "one_step_voi", None)
    if one_step is None:  # pragma: no cover
        pytest.skip("src.voi.one_step_voi not present")
    for seed in range(40):
        belief, setup, q_prev = ts.random_instance(seed)
        for src in _remaining(belief, setup):
            c = setup.costs[src]
            got = one_step(belief, src, c, q_prev, setup.params)
            want = ts.myopic_voi(belief, src, c, q_prev, setup.params)
            assert got == pytest.approx(want, abs=TOL, rel=0)
