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
# Touches. The ATD ball is heavily smoothed (its second differences match a ~17-frame moving mean
# of DFL tracking), so a kick is spread over most of a second: velocities are compared 6 frames
# either side rather than frame to frame.
TOUCH_LAG = 6                # frames
TOUCH_DV = 3.0               # m/s change of ball velocity that counts as a touch
TOUCH_RADIUS = 2.5           # metres from the nearest player
STREAK_CAP = 5               # consecutive same-side touches; the challenge's "two passes" proxy


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


def touch_features(feats, trk):
    """Touches: frames where the ball changes velocity sharply next to a player, and whose.

    A touch is a local maximum of |v_after - v_before| above TOUCH_DV with a player within
    TOUCH_RADIUS; it belongs to the closer side. Possession can only change hands at a touch, so
    the side of the last and next touch, and how many touches in a row one side has made, are the
    possession rule written as features.
    """
    n, lag, fps = trk.n_frames, TOUCH_LAG, trk.fps
    out = {}
    v_before = np.full((n, 2), np.nan, dtype=np.float32)
    v_after = np.full((n, 2), np.nan, dtype=np.float32)
    ball = np.column_stack((trk.bx, trk.by))
    v_before[lag:] = (ball[lag:] - ball[:-lag]) * fps / lag
    v_after[:-lag] = (ball[lag:] - ball[:-lag]) * fps / lag
    dv = np.hypot(*(v_after - v_before).T).astype(np.float32)
    dv[dv > 2 * 40.0] = np.nan  # a jump to a false detection, not a kick
    out["ball_dv"] = dv

    near = np.fmin(feats["d_home"], feats["d_away"])
    filled = np.where(np.isnan(dv), -1.0, dv)
    window = np.lib.stride_tricks.sliding_window_view(
        np.pad(filled, lag // 2, constant_values=-1.0), lag // 2 * 2 + 1)
    with np.errstate(invalid="ignore"):
        peak = (filled >= window.max(axis=1)) & (filled > TOUCH_DV) & (near < TOUCH_RADIUS)
    side = np.where(feats["d_home"] <= feats["d_away"], 1.0, -1.0).astype(np.float32)
    touch = np.where(peak, side, 0.0).astype(np.float32)
    out["touch"] = touch
    for window_frames in (51, 125):
        out["touch_balance_{}".format(window_frames)] = _smooth(touch, window_frames) * window_frames
    out.update(_edge_features(peak, side, "touch", n))

    # Streak of the last touch's side: 1 for a first touch after the other side's, 2 for the
    # next one by the same side, and so on. Carried forward to every frame until the next touch.
    frames = np.flatnonzero(peak)
    streak_at = np.zeros(len(frames), dtype=np.float32)
    for k in range(len(frames)):
        same = k > 0 and side[frames[k]] == side[frames[k - 1]]
        streak_at[k] = min(streak_at[k - 1] + 1, STREAK_CAP) if same else 1
    streak = np.zeros(n, dtype=np.float32)
    last = _last_index(peak)
    has = last >= 0
    position = np.searchsorted(frames, last[has])
    streak[has] = streak_at[position] * side[last[has]]
    out["touch_streak"] = streak
    return out


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
    out.update(touch_features(feats, trk))
    return out


def matrix(feats, names=None):
    """(X, names): the features stacked into a float32 matrix, columns in a fixed order."""
    names = sorted(feats) if names is None else names
    return np.column_stack([feats[name] for name in names]).astype(np.float32), names
