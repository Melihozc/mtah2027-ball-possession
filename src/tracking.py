"""Tracking in one shape, whatever it came from.

Every stage after loading -- features, dead-ball cues, the decoder -- works on a Tracking: player
and ball positions in metres, one row per frame, NaN where nothing was observed. The provided
Metrica ATD and the external DFL matches (IDSSE) both load into it, which is what lets the same
pipeline be tuned on other games and then run unchanged on BAR-ATH.

Conventions
-----------
- x runs along the pitch, 0..length; y across it, 0..width. Metres, not the 0-1 the ATD ships in.
- attack_sign[t] is +1 when the home side attacks towards +x on frame t, -1 when towards 0, and 0
  on frames outside both halves. Anything that needs "forward" multiplies by it.
- half[t] is 1 or 2 inside a half and 0 elsewhere. For BAR-ATH the halves are the graded blocks
  from games.half_video_blocks, so a feature can never be computed across halftime.

The raw ATD parse takes ~20 s, so the arrays are cached under cache/ (gitignored) and keyed on the
source file's size and mtime. Delete the folder to force a re-parse.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

# Params
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
HOME, AWAY = 0, 1


@dataclass
class Tracking:
    px: np.ndarray           # (n, slots) float32, metres along the length
    py: np.ndarray           # (n, slots) float32, metres across
    team: np.ndarray         # (slots,) int8, HOME or AWAY
    bx: np.ndarray           # (n,) float32
    by: np.ndarray           # (n,) float32
    half: np.ndarray         # (n,) int8, 0 outside a half
    attack_sign: np.ndarray  # (n,) int8, +1 home attacks +x, -1 towards 0, 0 outside a half
    pitch: tuple             # (length, width) in metres
    fps: int = 25
    name: str = ""

    @property
    def n_frames(self):
        return len(self.bx)

    @property
    def ball_seen(self):
        return ~np.isnan(self.bx)

    def half_blocks(self):
        """[(half, start, stop)] half-open frame spans, in order."""
        out = []
        for half in (1, 2):
            frames = np.flatnonzero(self.half == half)
            if len(frames):
                out.append((half, int(frames[0]), int(frames[-1]) + 1))
        return out


# -- Metrica FIFA/EPTS ---------------------------------------------------------------------------

def _epts_meta(meta_path):
    """(team per slot, home attacks left-to-right in the first half) from the EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    # First team listed is home -- the FIFA/EPTS convention, and the one starter/baseline.py uses.
    home_id = teams[0].get("id")
    team = np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)

    home_left_to_right = None
    for parameter in teams[0].iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            home_left_to_right = parameter.findtext("Value") == "left_to_right"
    return team, home_left_to_right


def _parse_epts_raw(raw_path, n_slots, n_frames):
    """(players (n, slots*3), ball (n, 3)) as float32, normalised units, NaN where missing.

    One line per frame, "<frame_id>:<x,y,s;x,y,s;...>:<x,y,s>". np.fromstring reads "NaN" natively,
    so each line is two C-level parses rather than a few hundred float() calls.
    """
    players = np.full((n_frames, n_slots * 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            head, body, tail = line.split(":")
            row = int(head) - 1  # frame_id counts from 1, video frames from 0
            players[row] = np.fromstring(body.replace(";", ","), sep=",")
            ball[row] = np.fromstring(tail, sep=",")
    return players, ball


def _cached(path, build):
    """Load arrays from cache/ if the cache matches the source file, else build and store them."""
    stat = os.stat(path)
    key = "{}_{}_{}.npz".format(os.path.basename(path), stat.st_size, int(stat.st_mtime))
    cache_path = os.path.join(CACHE_DIR, key)
    if os.path.isfile(cache_path):
        with np.load(cache_path) as stored:
            return {name: stored[name] for name in stored.files}
    arrays = build()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **arrays)
    return arrays


def load_metrica(source="tactical", game=GAME, data_dir=None):
    """The provided ATD of one recording as a Tracking.

    Tactical frames are video frames, so row i here is row i of a submission. Broadcast is on its
    own timeline: it gets half = 0 and attack_sign = 0 everywhere until it is synchronised.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team, home_left_to_right = _epts_meta(meta_path)
    n_frames = source_frames(source, game)

    arrays = _cached(raw_path, lambda: dict(zip(
        ("players", "ball"), _parse_epts_raw(raw_path, len(team), n_frames))))
    players, ball = arrays["players"], arrays["ball"]

    length, width = GAMES[game]["pitch"]
    half = np.zeros(n_frames, dtype=np.int8)
    attack_sign = np.zeros(n_frames, dtype=np.int8)
    if source == "tactical":
        first_sign = 1 if home_left_to_right else -1
        for number, first, last in half_video_blocks(game):
            half[first:last + 1] = number
            attack_sign[first:last + 1] = first_sign if number == 1 else -first_sign

    return Tracking(
        px=players[:, 0::3] * length, py=players[:, 1::3] * width, team=team,
        bx=ball[:, 0] * length, by=ball[:, 1] * width,
        half=half, attack_sign=attack_sign, pitch=(length, width),
        fps=GAMES[game]["fps"], name="{}_{}".format(game, source))
