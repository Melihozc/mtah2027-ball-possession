"""Regenerate the submission from the data bundle and the committed models. One command:

    python make_submission.py --team <team_name>

reads the tactical ATD, builds the frame table, applies the classifiers and decoder stored in
models/ (fitted and chosen on external matches by tuning/run_external.py), and writes
out/<team_name>.csv. Deterministic on CPU: the same bundle and models give the same file, byte for
byte, and the SHA-256 printed at the end is what to compare.

Frames outside both halves are X: they are never graded, and the tracking there is a camera
pointed at an empty pitch.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from src.decode import decode, to_labels  # noqa: E402
from src.model import PossessionModel, frame_table  # noqa: E402
from src.stack import stack_table  # noqa: E402
from src.tracking import load_metrica  # noqa: E402

# Params
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "out")


def fill_ball_from_broadcast(trk):
    """Where the tactical tracking has no ball, use the synchronised broadcast ball.

    Needs no new parameter: the classifiers simply see the ball on more frames. Only frames where
    the broadcast has a pitch homography count, so replays and close-ups contribute nothing.
    """
    from src.sync_broadcast import broadcast_on_tactical
    bc = broadcast_on_tactical(game_tac=trk)
    fill = ~trk.ball_seen & ~np.isnan(bc["bc_bx"])
    trk.bx[fill] = bc["bc_bx"][fill]
    trk.by[fill] = bc["bc_by"][fill]
    graded = trk.half > 0
    print("broadcast sync {} -> ball seen on {:.1%} of graded frames (+{:.1%})".format(
        bc["segments"], trk.ball_seen[graded].mean(), fill[graded].mean()))
    return trk


def build_labels(use_broadcast=True):
    trk = load_metrica("tactical")
    if use_broadcast:
        trk = fill_ball_from_broadcast(trk)
    table, names = frame_table(trk)
    model = PossessionModel.load(os.path.join(MODELS_DIR, "possession_model.pkl"))
    if names != model.names:
        raise ValueError("feature table does not match the stored model; re-run "
                         "python -m tuning.run_external")
    with open(os.path.join(MODELS_DIR, "decoder.json")) as handle:
        config = json.load(handle)
    params = config["params"]
    p_dead, p_home = model.predict(table)
    if config.get("stacked"):
        stage_two = PossessionModel.load(os.path.join(MODELS_DIR, "stage_two_model.pkl"))
        table2, names2 = stack_table(p_dead, p_home, table, names)
        if names2 != stage_two.names:
            raise ValueError("stage-two table does not match the stored model")
        p_dead, p_home = stage_two.predict(table2)
    return to_labels(decode(p_dead, p_home, trk.half_blocks(), **params))


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="submission", help="names the output file")
    parser.add_argument("--no-broadcast", action="store_true",
                        help="tactical tracking only, no broadcast ball fill")
    args = parser.parse_args()

    labels = build_labels(use_broadcast=not args.no_broadcast)
    n_expected = source_frames("tactical")
    path = os.path.join(OUT_DIR, "{}.csv".format(args.team))
    write(path, labels)

    problem = pm.validate_submission(pm.read_label_csv(path), n_expected)
    if problem:
        raise SystemExit("INVALID: {}".format(problem))
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    runs = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    spells = pm.possession_spells(pm.label_codes(labels), runs)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("wrote {}  rows {}  {}".format(os.path.relpath(path, ROOT), len(labels), counts))
    print("possessions {}  mean {:.1f}s".format(
        len(spells), np.mean([b - a for a, b, _ in spells]) / 25.0))
    print("sha256 {}".format(digest))


if __name__ == "__main__":
    main()
