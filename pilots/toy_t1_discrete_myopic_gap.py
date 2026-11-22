"""T1: smallest discrete instances, exact rational expectimax.

Instance A (synergy): r in {-1,+1}, P(r=+1)=1/2. Y1 in {+1,-1}, P(Y1=+1|r)=1/2 (Y1 independent of r).
  Y2 given (r,Y1): P(Y2=+1|r,Y1)=9/10 if r*Y1=+1, else 1/10.
  Pairwise independence: Y1 and Y2 are marginally independent and each is marginally independent of r,
  but the pair (Y1,Y2) is informative about r.
  Costs c1=c2=1/10, utility u = q r - tau|q| with tau=1/10, prior position q_prev=0.

Instance B (control): Y1 as in A but with P(Y1=+1|r=+1)=9/10, P(Y1=+1|r=-1)=1/10; Y2 uninformative.
  c1=c2=1/10, tau=1/10.
"""
from fractions import Fraction as F
from itertools import product

TAU = F(1, 10)
QS = (-1, 0, 1)


def joint_A():
    P = {}
    for r, y1, y2 in product((1, -1), repeat=3):
        p = F(1, 2) * F(1, 2)
        p2 = F(9, 10) if r * y1 == 1 else F(1, 10)
        p *= p2 if y2 == 1 else 1 - p2
        P[(r, y1, y2)] = p
    return P


def joint_B():
    P = {}
    for r, y1, y2 in product((1, -1), repeat=3):
        p = F(1, 2)
        p1 = F(9, 10) if r == 1 else F(1, 10)
        p *= p1 if y1 == 1 else 1 - p1
        p *= F(1, 2)  # Y2 uninformative
        P[(r, y1, y2)] = p
    return P


def posterior(P, obs):
    """obs: dict {1: y1, 2: y2} of observed sources. Returns dict r -> P(r | obs) (unnormalized over matching cells)."""
    w = {1: F(0), -1: F(0)}
    for (r, y1, y2), p in P.items():
        ok = all((y1 if k == 1 else y2) == v for k, v in obs.items())
        if ok:
            w[r] += p
    tot = w[1] + w[-1]
    return {r: w[r] / tot for r in (1, -1)}, tot


def best_action(post):
    """max_q E[q r - tau|q|] under posterior post; returns (value, q)."""
    best = None
    for q in QS:
        val = q * (post[1] - post[-1]) - TAU * abs(q)
        if best is None or val > best[0]:
            best = (val, q)
    return best


def value_stop(P, obs):
    post, _ = posterior(P, obs)
    return best_action(post)[0]


def expectimax(P, obs, costs, bought_order, verbose, depth_label):
    """Full tree: at each node choose stop or buy an unbought source.
    Returns (value, policy-description). Prints nodes if verbose."""
    stop = value_stop(P, obs)
    options = {"stop": stop}
    cont_detail = {}
    for j in (1, 2):
        if j in obs:
            continue
        exp = F(0)
        branches = []
        for y in (1, -1):
            obs2 = dict(obs)
            obs2[j] = y
            _, prob_y = posterior(P, {j: y}) if not obs else (None, None)
            # probability of y given current obs
            pr = sum(p for (r, y1, y2), p in P.items()
                     if all(((y1 if k == 1 else y2) == v) for k, v in obs.items())
                     and (y1 if j == 1 else y2) == y)
            pr_tot = sum(p for (r, y1, y2), p in P.items()
                         if all(((y1 if k == 1 else y2) == v) for k, v in obs.items()))
            pyo = pr / pr_tot
            sub, _ = expectimax(P, obs2, costs, bought_order + [j], verbose, depth_label + f"{j}={y:+d} ")
            exp += pyo * sub
            branches.append((y, pyo, sub))
        val = -costs[j] + exp
        options[f"buy{j}"] = val
        cont_detail[j] = (val, branches)
    best_name = max(options, key=lambda k: (options[k], k == "stop"))
    best_val = options[best_name]
    if verbose:
        print(f"{depth_label}obs={dict(sorted(obs.items()))}  stop={stop}  " +
              "  ".join(f"{k}={float(v):.4f}({v})" for k, v in options.items() if k != 'stop') +
              f"  -> best={best_name} V={best_val}")
    return best_val, best_name


def run_instance(name, P, costs):
    print("=" * 78)
    print(f"Instance {name}: costs c1={costs[1]}, c2={costs[2]}, tau={TAU}")
    # one-step (myopic) VOIs at the prior
    G0 = value_stop(P, {})
    print(f"Stop value G(empty) = {G0}")
    for j in (1, 2):
        vj = F(0)
        for y in (1, -1):
            pyo = sum(p for (r, y1, y2), p in P.items() if (y1 if j == 1 else y2) == y)
            post, _ = posterior(P, {j: y})
            vj += pyo * best_action(post)[0]
        net = vj - G0 - costs[j]
        print(f"  one-step net VOI of source {j}: E[G(post)] - G - c{j} = {vj} - {G0} - {costs[j]} = {net}  (= {float(net):+.6f})")
    print("Full expectimax tree (root first; indentation = depth):")
    V, pol = expectimax(P, {}, costs, [], True, "  ")
    print(f"Optimal value V* = {V} = {float(V):.6f}; optimal first action = {pol}")
    return V


if __name__ == "__main__":
    costs = {1: F(1, 10), 2: F(1, 10)}
    # check marginal independence in instance A
    PA = joint_A()
    assert sum(PA.values()) == 1
    for name_a, pair in [("Y1 vs r", None)]:
        pass
    print("Instance A marginal checks: P(Y1=+1)=", sum(p for (r, y1, y2), p in PA.items() if y1 == 1),
          " P(Y2=+1)=", sum(p for (r, y1, y2), p in PA.items() if y2 == 1),
          " P(Y1=+,Y2=+)=", sum(p for (r, y1, y2), p in PA.items() if y1 == 1 and y2 == 1),
          " P(r=+1,Y1=+)=", sum(p for (r, y1, y2), p in PA.items() if y1 == 1 and r == 1),
          " P(r=+1,Y2=+)=", sum(p for (r, y1, y2), p in PA.items() if y2 == 1 and r == 1))
    VA = run_instance("A (synergy, both one-step VOIs < 0)", PA, costs)
    PB = joint_B()
    VB = run_instance("B (control, myopic = optimal)", PB, costs)
