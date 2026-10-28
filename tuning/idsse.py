"""IDSSE: seven Bundesliga matches of DFL tracking with per-frame possession, as Trackings.

Source: Bassek, M., Rein, R., Weber, H., Memmert, D. "An integrated dataset of spatiotemporal and
event data in elite soccer." Scientific Data 12, 195 (2025). doi:10.1038/s41597-025-04505-y.
CC BY 4.0. Mirror used: https://huggingface.co/datasets/pysport/idsse-data. Its use for fitting
and tuning was confirmed by the organisers in issue #16 of the challenge repository.

Why this dataset: every ball frame carries BallStatus (1 alive, 0 dead) and BallPossession (1 home,
2 guest) at 25 fps -- the same H/A/D question the challenge asks, answered on other games. That is
what every parameter of the pipeline is chosen on; BAR-ATH never is.

The positions files are ~400 MB of XML each, so they are parsed line by line (one <Frame> per
line) instead of through a DOM, and cached under cache/ like the Metrica ATD.

    python -m tuning.idsse            # parse all seven, print a label summary per match
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.tracking import HOME, AWAY, Tracking, _cached  # noqa: E402

# Params
IDSSE_DIR = os.environ.get("IDSSE_DIR") or os.path.join(
    os.path.dirname(ROOT), "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = ("firstHalf", "secondHalf")
# One empty frame between the halves, labelled X, so nothing temporal can run across halftime.
GAP_FRAMES = 1

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _paths(match):
    def one(prefix):
        found = glob.glob(os.path.join(IDSSE_DIR, "{}_*{}.xml".format(prefix, match)))
        if len(found) != 1:
            raise IOError("IDSSE file {}*{} not found in {}".format(prefix, match, IDSSE_DIR))
        return found[0]
    return one("DFL_02_01_matchinformation"), one("DFL_04_03_positions_raw_observed")


def _match_info(info_path):
    """(home team id, (pitch length, width))."""
    with open(info_path, encoding="utf-8") as handle:
        text = handle.read()
    home = None
    for tag in re.findall(r"<Team [^>]*>", text):
        attrs = dict(_ATTR.findall(tag))
        if attrs.get("Role") == "home":
            home = attrs["TeamId"]
    env = dict(_ATTR.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    return home, (float(env["PitchX"]), float(env["PitchY"]))


def _parse_positions(pos_path, home_id):
    """Raw columns, one array per field, every frame of every FrameSet in file order."""
    sets = []  # (section, kind, person) per FrameSet; kind 0 home, 1 away, 2 ball, -1 skip
    rows = {"set": [], "n": [], "x": [], "y": [], "poss": [], "status": []}
    kind = -1
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if kind < 0:
                    continue
                attrs = dict(_ATTR.findall(line))
                rows["set"].append(len(sets) - 1)
                rows["n"].append(int(attrs["N"]))
                rows["x"].append(float(attrs["X"]))
                rows["y"].append(float(attrs["Y"]))
                rows["poss"].append(int(attrs.get("BallPossession", 0)))
                rows["status"].append(int(attrs.get("BallStatus", -1)))
            elif line.startswith("<FrameSet "):
                attrs = dict(_ATTR.findall(line))
                team = attrs["TeamId"]
                kind = 2 if team == "BALL" else -1 if team == "referee" else (
                    HOME if team == home_id else AWAY)
                sets.append((SECTIONS.index(attrs["GameSection"]), kind, attrs["PersonId"]))
    out = {name: np.array(values) for name, values in rows.items()}
    out["set_section"] = np.array([s[0] for s in sets], dtype=np.int8)
    out["set_kind"] = np.array([s[1] for s in sets], dtype=np.int8)
    persons = sorted({s[2] for s in sets if s[1] in (HOME, AWAY)})
    out["set_slot"] = np.array([persons.index(s[2]) if s[1] in (HOME, AWAY) else -1
                                for s in sets], dtype=np.int16)
    out["slot_team"] = np.array([next(s[1] for s in sets if s[2] == p) for p in persons],
                                dtype=np.int8)
    return out


def load_match(match):
    """(Tracking, labels) for one IDSSE match. labels is an object array of H/A/D/X."""
    info_path, pos_path = _paths(match)
    home_id, pitch = _match_info(info_path)
    raw = _cached(pos_path, lambda: _parse_positions(pos_path, home_id))
    length, width = pitch
    n_slots = len(raw["slot_team"])

    section = raw["set_section"][raw["set"]]
    kind = raw["set_kind"][raw["set"]]
    slot = raw["set_slot"][raw["set"]]

    # Frame index: each half is the ball FrameSet's N range, halves stacked with a gap between.
    starts, offset = {}, 0
    for half in (0, 1):
        ball_n = raw["n"][(kind == 2) & (section == half)]
        starts[half] = (offset, ball_n.min(), ball_n.max())
        offset += int(ball_n.max() - ball_n.min() + 1) + GAP_FRAMES
    n_frames = offset - GAP_FRAMES
    row = np.array([starts[s][0] + n - starts[s][1] for s, n in zip(section, raw["n"])])

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player = kind != 2
    # DFL is centred on the pitch with y up; shift to 0..length, 0..width.
    px[row[player], slot[player]] = raw["x"][player] + length / 2.0
    py[row[player], slot[player]] = width / 2.0 - raw["y"][player]

    ball = kind == 2
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bx[row[ball]] = raw["x"][ball] + length / 2.0
    by[row[ball]] = width / 2.0 - raw["y"][ball]

    labels = np.full(n_frames, "X", dtype=object)
    status, poss = raw["status"][ball], raw["poss"][ball]
    labels[row[ball]] = np.where(status == 0, "D", np.where(poss == 1, "H", np.where(
        poss == 2, "A", "X")))

    half = np.zeros(n_frames, dtype=np.int8)
    attack_sign = np.zeros(n_frames, dtype=np.int8)
    team = raw["slot_team"]
    for index, (first, n_min, n_max) in starts.items():
        last = first + int(n_max - n_min)
        half[first:last + 1] = index + 1
        # Kickoff geometry decides the direction: at the first frame of a half every player is
        # in his own half, so the home side attacks +x exactly when it starts on the low-x side.
        kickoff = px[first:first + 25][:, team == HOME]
        attack_sign[first:last + 1] = 1 if np.nanmean(kickoff) < length / 2.0 else -1

    trk = Tracking(px=px, py=py, team=team, bx=bx, by=by, half=half, attack_sign=attack_sign,
                   pitch=pitch, fps=25, name="IDSSE_" + match)
    return trk, labels


if __name__ == "__main__":
    for match in MATCHES:
        trk, labels = load_match(match)
        graded = labels != "X"
        values, counts = np.unique(labels[graded], return_counts=True)
        share = dict(zip(values, np.round(counts / float(graded.sum()), 3)))
        # Possessions as the challenge counts them: maximal H or A runs, D and X break them.
        breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(labels)]))
        spells = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if labels[a] in ("H", "A")]
        durations = np.array([b - a for a, b in spells]) / 25.0
        per_5min = len(spells) / (graded.sum() / 25.0 / 300.0)
        print("{}  frames {:>6}  slots {:>2}  share {}  possessions {:>4} ({:.1f}/5min)  "
              "mean {:.1f}s  median {:.1f}s  <1s {:.0%}".format(
                  match, len(labels), trk.px.shape[1], share, len(spells), per_5min,
                  durations.mean(), np.median(durations), np.mean(durations < 1.0)))
