"""Put the broadcast recording on the tactical timeline, by matching where the players are.

Both ATDs are in pitch coordinates, so a broadcast frame and the tactical frame showing the same
instant agree on where every visible player stands. A live TV feed runs in real time -- replays
and close-ups replace live pictures rather than delaying them -- so the mapping is one constant
offset per half, and the search is for that offset:

  coarse  windows of 30 s across the broadcast; for each, the offset (step 1 s) minimising the
          mean distance from every broadcast player to the nearest same-team tactical player
          (a one-sided Chamfer distance), kept only when it beats every far-away offset clearly
  split   the reliable windows fall into two clusters of offsets -- one per half -- split at the
          largest jump between consecutive windows
  fine    per half, the offset within +/-60 frames of the cluster median that minimises the same
          distance over every usable frame of that half, step 1 frame

Nothing in here is a constant read off the video: the offsets are found by the search, and come
out within one frame of the same value in every 2-minute window of each half.

The result is cached in cache/ like the parsed tracking.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from src.tracking import CACHE_DIR, load_metrica
from games import require

# Params
COARSE_WINDOW = 750          # frames, 30 s
COARSE_SAMPLE = 75           # one frame in 75 inside a window
COARSE_STEP = 25             # offset step, 1 s
COARSE_RANGE = (-5000, 45000)
DISTINCT_RATIO = 0.5         # best must be < half of the best offset more than 10 s away
FINE_RADIUS = 60
FINE_SAMPLE = 30
MIN_PLAYERS = 10
CAP = 15.0                   # metres; one unmatched player cannot dominate the distance


def _chamfer(tac, brd, b_frame, t_frames):
    """Mean distance from broadcast players at b_frame to tactical players at each t_frame."""
    total, count = np.zeros(len(t_frames)), 0
    for team in (0, 1):
        bx = brd.px[b_frame, brd.team == team]
        by = brd.py[b_frame, brd.team == team]
        keep = ~np.isnan(bx)
        if not keep.any():
            continue
        tx = tac.px[t_frames][:, tac.team == team]
        ty = tac.py[t_frames][:, tac.team == team]
        d = np.hypot(tx[:, None, :] - bx[keep][None, :, None],
                     ty[:, None, :] - by[keep][None, :, None])
        nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)
        total += np.minimum(nearest, CAP).sum(axis=1)
        count += int(keep.sum())
    return total / max(count, 1)


def _usable(brd, valid):
    return valid & ((~np.isnan(brd.px)).sum(axis=1) >= MIN_PLAYERS)


def _mean_distance(tac, brd, b_frames, offsets):
    scores, ok = np.zeros(len(offsets)), np.ones(len(offsets), dtype=bool)
    for b in b_frames:
        t = b + offsets
        inside = (t >= 0) & (t < tac.n_frames)
        ok &= inside
        s = np.full(len(offsets), CAP)
        s[inside] = _chamfer(tac, brd, b, t[inside])
        scores += s
    scores /= max(len(b_frames), 1)
    scores[~ok] = np.inf
    return scores


def find_offsets(tac, brd, valid):
    """[(first_broadcast_frame, last_broadcast_frame, offset)] -- tactical = broadcast + offset."""
    usable = _usable(brd, valid)
    offsets = np.arange(COARSE_RANGE[0], COARSE_RANGE[1], COARSE_STEP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        windows = []
        for start in range(0, brd.n_frames - COARSE_WINDOW, COARSE_WINDOW):
            frames = [b for b in range(start, start + COARSE_WINDOW, COARSE_SAMPLE) if usable[b]]
            if len(frames) < 5:
                continue
            scores = _mean_distance(tac, brd, frames, offsets)
            best = int(np.argmin(scores))
            far = np.abs(offsets - offsets[best]) > 10 * COARSE_STEP
            if scores[best] < DISTINCT_RATIO * scores[far].min():
                windows.append((start, int(offsets[best])))
        if len(windows) < 4:
            raise RuntimeError("broadcast sync: too few reliable windows ({})".format(len(windows)))

        starts = np.array([w[0] for w in windows])
        coarse = np.array([w[1] for w in windows])
        cut = int(np.argmax(np.abs(np.diff(coarse)))) + 1
        groups = [(0, cut), (cut, len(windows))]
        if np.abs(np.diff(coarse)).max() < 10 * COARSE_STEP:
            groups = [(0, len(windows))]

        out = []
        for index, (a, b) in enumerate(groups):
            centre = int(np.median(coarse[a:b]))
            first = 0 if index == 0 else int(starts[a])
            last = brd.n_frames - 1 if index == len(groups) - 1 else int(starts[b]) - 1
            frames = [f for f in range(first, last + 1, FINE_SAMPLE) if usable[f]]
            fine = np.arange(centre - FINE_RADIUS, centre + FINE_RADIUS + 1)
            scores = _mean_distance(tac, brd, frames, fine)
            out.append((first, last, int(fine[np.argmin(scores)])))
    return out


def broadcast_on_tactical(game_tac=None, game_brd=None):
    """Broadcast signals resampled to tactical frames: dict of (n_tactical,) arrays.

      bc_covered   1 where some broadcast frame maps here
      bc_invalid   1 where that broadcast frame has no pitch homography (replay, close-up, crowd)
      bc_bx, bc_by the broadcast ball, where seen on a valid frame
    """
    tac = game_tac or load_metrica("tactical")
    brd = game_brd or load_metrica("broadcast")
    hom_path = require(["homography"], source="broadcast")[0]
    valid = pd.read_csv(hom_path, usecols=["valid"])["valid"].values.astype(bool)

    stat = os.stat(hom_path)
    cache = os.path.join(CACHE_DIR, "sync_{}_{}.json".format(stat.st_size, int(stat.st_mtime)))
    if os.path.isfile(cache):
        with open(cache) as handle:
            segments = json.load(handle)
    else:
        segments = find_offsets(tac, brd, valid)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache, "w") as handle:
            json.dump(segments, handle)

    n = tac.n_frames
    out = {"bc_covered": np.zeros(n, np.float32), "bc_invalid": np.full(n, np.nan, np.float32),
           "bc_bx": np.full(n, np.nan, np.float32), "bc_by": np.full(n, np.nan, np.float32)}
    for first, last, offset in segments:
        b = np.arange(first, last + 1)
        t = b + offset
        keep = (t >= 0) & (t < n)
        b, t = b[keep], t[keep]
        out["bc_covered"][t] = 1.0
        out["bc_invalid"][t] = (~valid[b]).astype(np.float32)
        ball = valid[b] & brd.ball_seen[b]
        out["bc_bx"][t[ball]] = brd.bx[b[ball]]
        out["bc_by"][t[ball]] = brd.by[b[ball]]
    out["segments"] = segments
    return out
