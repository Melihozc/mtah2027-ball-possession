"""Second stage: the first stage's probabilities, read over time.

Stage one decides each frame from features around it. Its mistakes are temporally structured --
a team call that wobbles for a second inside a long possession, a dead-ball probability that rises
a few frames early -- and a second model that sees stage one's output over windows can learn to
correct them. This is stacking ("auto-context"): stage two is trained on stage-one predictions the
stage-one model did not see the labels of, which on the external matches means its
leave-one-match-out predictions.

The table is stage one's two probabilities as logits, smoothed and shifted, plus the handful of
raw features that say where possession can change: touches, control and gap edges.
"""

import numpy as np

from src.context import _shift
from src.features import _smooth

# Params
WINDOWS = (25, 75, 250)
SIDE = 75
EXTREME_WINDOW = 75
CARRIED = ("touch_last", "touch_next", "touch_since", "touch_until", "touch_streak",
           "ctrl_last", "ctrl_next", "ctrl_since", "ctrl_until", "seen_d_diff_last",
           "seen_d_diff_next", "ball_seen", "d_diff", "speed_all_c25", "ball_line_dist",
           "control_c25")
EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).astype(np.float32)


def _rolling_extreme(values, window, fn):
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * pad + 1)
    return fn(view, axis=1).astype(np.float32)


def stack_table(p_dead, p_home, x, names):
    """(X2, names2) for stage two from stage-one probabilities and the stage-one table."""
    out = {}
    for label, p in (("s1_dead", p_dead), ("s1_home", p_home)):
        z = _logit(p)
        out[label] = z
        for window in WINDOWS:
            out["{}_c{}".format(label, window)] = _smooth(z, window)
        centred = _smooth(z, SIDE)
        out[label + "_before"] = _shift(centred, SIDE // 2)
        out[label + "_after"] = _shift(centred, -(SIDE // 2))
        out[label + "_max"] = _rolling_extreme(z, EXTREME_WINDOW, np.max)
        out[label + "_min"] = _rolling_extreme(z, EXTREME_WINDOW, np.min)
    index = {name: k for k, name in enumerate(names)}
    for name in CARRIED:
        out["x_" + name] = x[:, index[name]]
    names2 = sorted(out)
    return np.column_stack([out[name] for name in names2]).astype(np.float32), names2
