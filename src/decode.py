"""Turn per-frame probabilities into possessions: Viterbi over {H, A, D}.

Per-frame probabilities flicker, and spellF1 punishes every flicker as a spurious possession. The
decoder finds the single label sequence that best balances agreeing with the probabilities against
the number of label changes, each change paying a fixed cost:

  team_cost     H <-> A while the ball stays alive   (a turnover)
  stop_cost     H/A -> D                             (the ball goes dead)
  restart_cost  D -> H/A                             (play restarts)

alpha tempers the per-frame evidence, because neighbouring frames are nowhere near independent:
with alpha = 1, a classifier that is slightly too confident for half a second would buy a
possession on its own. All four numbers are chosen on the external matches, never on BAR-ATH.

Decoding runs inside each half separately, so nothing can be carried across halftime.
"""

import numpy as np

# Params
H, A, D = 0, 1, 2
EPS = 1e-6


def emissions(p_dead, p_home):
    """(n, 3) log-probabilities of H, A, D from P(dead) and P(home | alive)."""
    p_dead = np.clip(p_dead, EPS, 1 - EPS)
    p_home = np.clip(p_home, EPS, 1 - EPS)
    alive = np.log1p(-p_dead)
    return np.column_stack((alive + np.log(p_home), alive + np.log1p(-p_home), np.log(p_dead)))


def cost_matrix(team_cost, stop_cost, restart_cost):
    cost = np.array([[0.0, team_cost, stop_cost],
                     [team_cost, 0.0, stop_cost],
                     [restart_cost, restart_cost, 0.0]])
    return cost


def viterbi(logp, cost):
    """Best state path for one contiguous block. logp (n, 3), cost (3, 3) >= 0 off-diagonal.

    Plain Python over scalars: three states make every step nine additions, which is faster than
    numpy's per-call overhead on arrays this small, and the result is exactly reproducible.
    """
    n = len(logp)
    back = np.zeros((n, 3), dtype=np.int8)
    lp = logp.tolist()
    c = cost.tolist()
    s0, s1, s2 = lp[0]
    for t in range(1, n):
        e0, e1, e2 = lp[t]
        # Best predecessor of each state: stay, or switch and pay.
        a0, a1, a2 = s0, s1 - c[1][0], s2 - c[2][0]
        if a0 >= a1 and a0 >= a2:
            n0, b0 = a0, 0
        elif a1 >= a2:
            n0, b0 = a1, 1
        else:
            n0, b0 = a2, 2
        a0, a1, a2 = s0 - c[0][1], s1, s2 - c[2][1]
        if a1 >= a0 and a1 >= a2:
            n1, b1 = a1, 1
        elif a0 >= a2:
            n1, b1 = a0, 0
        else:
            n1, b1 = a2, 2
        a0, a1, a2 = s0 - c[0][2], s1 - c[1][2], s2
        if a2 >= a0 and a2 >= a1:
            n2, b2 = a2, 2
        elif a0 >= a1:
            n2, b2 = a0, 0
        else:
            n2, b2 = a1, 1
        back[t, 0], back[t, 1], back[t, 2] = b0, b1, b2
        s0, s1, s2 = n0 + e0, n1 + e1, n2 + e2

    path = np.empty(n, dtype=np.int8)
    last = (s0, s1, s2)
    state = int(np.argmax(last))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def decode(p_dead, p_home, blocks, alpha, team_cost, stop_cost, restart_cost):
    """Label codes (H=0, A=1, D=2) for every frame; -1 outside the blocks."""
    logp = alpha * emissions(p_dead, p_home)
    cost = cost_matrix(team_cost, stop_cost, restart_cost)
    out = np.full(len(p_dead), -1, dtype=np.int8)
    for _, start, stop in blocks:
        out[start:stop] = viterbi(logp[start:stop], cost)
    return out


def to_labels(codes):
    """Codes to the submission alphabet; anything outside a half is X."""
    return np.array(["H", "A", "D", "X"])[np.where(codes < 0, 3, codes)]
