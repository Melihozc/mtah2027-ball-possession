"""Make clean external tracking look like the automatic ATD it will be applied to.

DFL tracking sees the ball on every frame and every player for the whole match. The Metrica ATD
does neither: the ball is missing in long blackouts (median ~11 s, not flicker), players drop out
and reappear as new tracklets, and a small share of ball positions are false detections far off the
pitch. A model fitted on clean data would learn to lean on a ball that BAR-ATH does not have.

The profile below is measured on the BAR-ATH ATD itself -- run lengths of the ball being seen and
missing, how many players are visible, how often the ball jumps off the pitch. Those are statistics
of the input, computed by code, with no label involved, which is the line RULES.md draws. Position
noise is deliberately not added: measured second differences show the ATD is smoother than DFL.
"""

import numpy as np

from src.tracking import Tracking

# Params
PLAYER_PRESENT_MEAN_S = 35.0   # median tracklet lifetime in the ATD
PLAYER_MISSING_MEAN_S = 2.6    # gives ~93% of players visible, as measured (10.2-10.3 of 11)
OUTLIER_RUN_FRAMES = 10        # false ball detections come in short bursts


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def noise_profile(trk):
    """Run-length samples and rates from an ATD Tracking, over its halves only."""
    seen = trk.ball_seen
    blocks = trk.half_blocks()
    length, width = trk.pitch
    inside = trk.half > 0
    far = seen & inside & ((trk.bx < -3) | (trk.bx > length + 3) |
                           (trk.by < -3) | (trk.by > width + 3))
    return {
        "ball_present_runs": np.concatenate([_runs(seen[a:b]) for _, a, b in blocks]),
        "ball_missing_runs": np.concatenate([_runs(~seen[a:b]) for _, a, b in blocks]),
        "ball_outlier_rate": far.sum() / float((seen & inside).sum()),
    }


def _alternating_mask(n, sample_on, sample_off, rng, p_start_on):
    """Boolean mask of length n built from alternating on/off runs drawn by the two samplers."""
    mask = np.zeros(n, dtype=bool)
    position, on = 0, rng.random() < p_start_on
    # Start part-way into the first run, so the mask is not aligned to the start of the match.
    first = sample_on() if on else sample_off()
    run = int(rng.integers(1, first + 1))
    while position < n:
        if on:
            mask[position:position + run] = True
        position += run
        on = not on
        run = sample_on() if on else sample_off()
    return mask


def degrade(trk, profile, seed):
    """A copy of trk with ATD-like ball blackouts, player dropouts and false ball detections."""
    rng = np.random.default_rng(seed)
    n, n_slots = trk.px.shape
    length, width = trk.pitch
    px, py = trk.px.copy(), trk.py.copy()
    bx, by = trk.bx.copy(), trk.by.copy()

    present, missing = profile["ball_present_runs"], profile["ball_missing_runs"]
    p_seen = present.sum() / float(present.sum() + missing.sum())
    ball_on = _alternating_mask(n, lambda: int(rng.choice(present)),
                                lambda: int(rng.choice(missing)), rng, p_seen)

    # False detections: bursts placed on seen frames, at a random spot well off the pitch.
    n_bursts = int(round(profile["ball_outlier_rate"] * ball_on.sum() / OUTLIER_RUN_FRAMES))
    for start in rng.choice(np.flatnonzero(ball_on), size=n_bursts, replace=False):
        stop = min(n, start + OUTLIER_RUN_FRAMES)
        side = rng.integers(4)
        x = rng.uniform(0, length) if side < 2 else (rng.uniform(-8, -3) if side == 2 else
                                                     rng.uniform(length + 3, length + 8))
        y = rng.uniform(width + 3, width + 30) if side == 0 else (
            rng.uniform(-8, -3) if side == 1 else rng.uniform(0, width))
        bx[start:stop], by[start:stop] = x, y
    bx[~ball_on] = np.nan
    by[~ball_on] = np.nan

    fps = trk.fps
    on_mean, off_mean = PLAYER_PRESENT_MEAN_S * fps, PLAYER_MISSING_MEAN_S * fps
    p_on = on_mean / (on_mean + off_mean)
    for slot in range(n_slots):
        keep = _alternating_mask(n, lambda: int(rng.exponential(on_mean)) + 1,
                                 lambda: int(rng.exponential(off_mean)) + 1, rng, p_on)
        px[~keep, slot] = np.nan
        py[~keep, slot] = np.nan

    return Tracking(px=px, py=py, team=trk.team, bx=bx, by=by, half=trk.half,
                    attack_sign=trk.attack_sign, pitch=trk.pitch, fps=fps,
                    name=trk.name + "_degraded")
