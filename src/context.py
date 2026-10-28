"""Temporal context: what the frames around t say about t.

A per-frame classifier sees one instant; possession is a property of an interval. These features
give it the interval -- slow signals averaged over windows before, around and after the frame, and
for the ball, what happened at the edges of the gap a frame sits in. The task is post-match
(issue #15), so looking ahead is allowed and is used as freely as looking back.

The control features are the possession rule in its simplest form: a side controls the ball when
one of its players is within CONTROL_RADIUS and clearly closer than any opponent. Possession then
belongs to the last side that controlled it, so the classifier gets "who controlled it last" and
"who controls it next" for every frame, with how long ago and how far ahead.
"""

import numpy as np

from src.features import _smooth

# Params
SLOW = ("speed_all", "speed_diff", "width_diff", "depth_diff", "adv_diff", "d_diff",
        "ball_speed", "ball_line_dist", "line_home", "line_away", "ball_adv_home")
WINDOWS = (25, 75, 250)      # 1 s, 3 s, 10 s
SIDE_WINDOW = 75             # 3 s for the before/after means
CONTROL_RADIUS = 1.5         # metres
CONTROL_MARGIN = 1.0         # metres closer than the nearest opponent
GAP_CAP = 1500               # frames; beyond 60 s "how long ago" stops carrying information


def _shift(values, frames):
    """values moved forward by `frames` (positive: out[t] = values[t - frames]), NaN-padded."""
    out = np.full_like(values, np.nan)
    if frames > 0:
        out[frames:] = values[:-frames]
    elif frames < 0:
        out[:frames] = values[-frames:]
    else:
        out[:] = values
    return out


def _last_index(mask):
    """For each t, the last index <= t where mask is True, or -1."""
    index = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(index)


def _next_index(mask):
    """For each t, the first index >= t where mask is True, or len(mask)."""
    n = len(mask)
    index = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(index[::-1])[::-1]


def _edge_features(event, values, prefix, n):
    """Value of `values` at the last and next frame where `event` holds, and the distance to it."""
    out = {}
    last, nxt = _last_index(event), _next_index(event)
    padded = np.concatenate((values.astype(np.float32), [np.nan]))
    out[prefix + "_last"] = np.where(last >= 0, padded[np.maximum(last, 0)], np.nan)
    out[prefix + "_next"] = np.where(nxt < n, padded[np.minimum(nxt, n)], np.nan)
    frames = np.arange(n)
    out[prefix + "_since"] = np.where(last >= 0, np.minimum(frames - last, GAP_CAP), GAP_CAP)
    out[prefix + "_until"] = np.where(nxt < n, np.minimum(nxt - frames, GAP_CAP), GAP_CAP)
    return {key: value.astype(np.float32) for key, value in out.items()}


def context_features(feats, trk):
    n = trk.n_frames
    out = dict(feats)
    half_side = SIDE_WINDOW // 2
    for name in SLOW:
        for window in WINDOWS:
            out["{}_c{}".format(name, window)] = _smooth(feats[name], window)
        centred = _smooth(feats[name], SIDE_WINDOW)
        out[name + "_before"] = _shift(centred, half_side)
        out[name + "_after"] = _shift(centred, -half_side)

    seen = trk.ball_seen
    out.update(_edge_features(seen, feats["d_diff"], "seen_d_diff", n))
    out.update(_edge_features(seen, feats["ball_line_dist"], "seen_line", n))

    d_home, d_away = feats["d_home"], feats["d_away"]
    with np.errstate(invalid="ignore"):
        home_ctrl = (d_home < CONTROL_RADIUS) & (d_away - d_home > CONTROL_MARGIN)
        away_ctrl = (d_away < CONTROL_RADIUS) & (d_home - d_away > CONTROL_MARGIN)
    control = np.where(home_ctrl, 1.0, np.where(away_ctrl, -1.0, 0.0)).astype(np.float32)
    out["control"] = np.where(seen, control, np.nan).astype(np.float32)
    out["control_c25"] = _smooth(out["control"], 25)
    out.update(_edge_features(home_ctrl | away_ctrl, control, "ctrl", n))

    # Ball stillness: a ball that sits still for a second on the turf is being placed for a
    # restart, which is the end of most dead-ball stretches.
    mean_x, mean_y = _smooth(trk.bx, 25), _smooth(trk.by, 25)
    spread = np.hypot(trk.bx - mean_x, trk.by - mean_y)
    out["ball_still"] = _smooth(spread, 25)
    return out


def matrix(feats, names=None):
    """(X, names): the features stacked into a float32 matrix, columns in a fixed order."""
    names = sorted(feats) if names is None else names
    return np.column_stack([feats[name] for name in names]).astype(np.float32), names
