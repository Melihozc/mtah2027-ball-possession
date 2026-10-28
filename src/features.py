"""Per-frame features computed from a Tracking, identically for every source.

Nothing here reads a file or knows which game it is looking at: the same function runs on BAR-ATH
and on the external matches the models are fitted on, so a feature means the same thing in both.
That is the whole reason the module is source-agnostic.

Every feature is a float array of length n_frames, NaN where it is undefined (no ball, nobody
tracked). Speeds come from positions rather than the providers' own speed channels, because those
channels differ in units and smoothing between Metrica and DFL, and a model fitted on one would
silently mis-read the other.
"""

import warnings

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from src.tracking import HOME, AWAY

# Params
SPEED_LAG = 2           # frames either side for central-difference speeds: +/-0.08 s at 25 fps
MAX_PLAYER_SPEED = 12.0  # m/s; above this a "speed" is an identity switch, not a sprint
MAX_BALL_SPEED = 40.0    # m/s; above this the ball track jumped to a false detection
NEAR_RADIUS = 2.0        # metres; "close enough to play the ball"
OUT_MARGIN = 1.0         # metres beyond a line before the ball counts as out
SMOOTH_FRAMES = 25       # 1 s window for the slow collective signals


def _central_speed(x, y, fps, lag=SPEED_LAG, cap=None):
    """|p(t+lag) - p(t-lag)| / (2 lag / fps), NaN where either end is missing or above cap."""
    speed = np.full(x.shape, np.nan, dtype=np.float32)
    dx = x[2 * lag:] - x[:-2 * lag]
    dy = y[2 * lag:] - y[:-2 * lag]
    speed[lag:-lag] = np.hypot(dx, dy) * fps / (2.0 * lag)
    if cap is not None:
        speed[speed > cap] = np.nan
    return speed


def _nanmean(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanstd(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis)


def _smooth(values, frames=SMOOTH_FRAMES):
    """Centred moving mean that ignores NaN, NaN only where the whole window is missing."""
    seen = ~np.isnan(values)
    total = uniform_filter1d(np.where(seen, values, 0.0).astype(np.float64), frames, mode="nearest")
    count = uniform_filter1d(seen.astype(np.float64), frames, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < 1e-9] = np.nan
    return out.astype(np.float32)


def ball_features(trk):
    """Ball position, speed and whether it is beyond a line."""
    length, width = trk.pitch
    seen = trk.ball_seen
    speed = _central_speed(trk.bx, trk.by, trk.fps, cap=MAX_BALL_SPEED)
    # A median over 5 frames removes single-frame jumps without delaying real kicks by more than
    # two frames; NaN propagates, which is what we want around gaps.
    speed = median_filter(speed, size=5, mode="nearest")

    out_x = (trk.bx < -OUT_MARGIN) | (trk.bx > length + OUT_MARGIN)
    out_y = (trk.by < -OUT_MARGIN) | (trk.by > width + OUT_MARGIN)
    return {
        "ball_seen": seen.astype(np.float32),
        "ball_speed": speed,
        "ball_out": np.where(seen, (out_x | out_y), np.nan).astype(np.float32),
        # Distance to the nearest line, negative outside: small near a touchline throw-in or a
        # corner, the geometry of most restarts.
        "ball_line_dist": np.where(seen, np.minimum.reduce([
            trk.bx, length - trk.bx, trk.by, width - trk.by]), np.nan).astype(np.float32),
        # How far forward the ball is for the home side, -length/2 .. +length/2.
        "ball_adv_home": (trk.attack_sign * (trk.bx - length / 2.0)).astype(np.float32),
    }


def player_features(trk):
    """Per-team proximity to the ball, collective speed and shape."""
    length, width = trk.pitch
    speeds = _central_speed(trk.px, trk.py, trk.fps, cap=MAX_PLAYER_SPEED)
    feats = {}

    distance = np.hypot(trk.px - trk.bx[:, None], trk.py - trk.by[:, None])
    for code, side in ((HOME, "home"), (AWAY, "away")):
        cols = trk.team == code
        px, py, dist = trk.px[:, cols], trk.py[:, cols], distance[:, cols]
        seen = ~np.isnan(px)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feats["d_" + side] = np.nanmin(dist, axis=1).astype(np.float32)
        feats["near_" + side] = np.where(trk.ball_seen, (dist < NEAR_RADIUS).sum(axis=1),
                                         np.nan).astype(np.float32)
        feats["n_" + side] = seen.sum(axis=1).astype(np.float32)
        feats["speed_" + side] = _nanmean(speeds[:, cols], axis=1).astype(np.float32)

        # Shape in the side's own attacking frame: +x is towards the goal it attacks.
        sign = trk.attack_sign if code == HOME else -trk.attack_sign
        forward = sign[:, None] * (px - length / 2.0)
        feats["adv_" + side] = _nanmean(forward, axis=1).astype(np.float32)
        feats["depth_" + side] = _nanstd(px, axis=1).astype(np.float32)
        feats["width_" + side] = _nanstd(py, axis=1).astype(np.float32)
        # Last outfield line: the second-deepest player, so the goalkeeper does not set it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ordered = np.sort(np.where(seen, forward, np.nan), axis=1)
        feats["line_" + side] = ordered[:, 1].astype(np.float32)

    feats["speed_all"] = _nanmean(speeds, axis=1).astype(np.float32)
    feats["speed_all_1s"] = _smooth(feats["speed_all"])
    feats["d_diff"] = feats["d_home"] - feats["d_away"]          # < 0: home closer
    feats["adv_diff"] = feats["adv_home"] - feats["adv_away"]
    feats["width_diff"] = feats["width_home"] - feats["width_away"]
    feats["depth_diff"] = feats["depth_home"] - feats["depth_away"]
    feats["speed_diff"] = feats["speed_home"] - feats["speed_away"]
    return feats


def all_features(trk):
    feats = ball_features(trk)
    feats.update(player_features(trk))
    return feats
