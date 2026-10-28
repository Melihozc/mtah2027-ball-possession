"""Fit, decode and score on the external matches -- the only place any parameter is chosen.

Leave-one-match-out over the seven IDSSE matches, each degraded to look like the BAR-ATH ATD:

  1. for each match, fit the two classifiers on the other six and predict the held-out one
  2. grid-search the decoder on the seven held-out predictions together, ranked by mean spellF1
     -- the competition metric, from the competition's own evaluation/possession_metrics.py
  3. fit the final classifiers on all seven and store them, with the chosen decoder, in models/

The BAR-ATH ATD is read once, for its noise profile (tuning/degrade.py) -- input statistics, no
label. The public ground truth is never read here.

    python -m tuning.run_external            # ~15 min on a laptop CPU
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from src.decode import decode, to_labels  # noqa: E402
from src.model import GBM_PARAMS, PossessionModel, frame_table  # noqa: E402
from src.stack import stack_table  # noqa: E402
from src.tracking import load_metrica, CACHE_DIR  # noqa: E402
from tuning.degrade import degrade, noise_profile  # noqa: E402
from tuning.fast_metric import spell_f1 as fast_spell_f1  # noqa: E402
from tuning.idsse import MATCHES, load_match  # noqa: E402

# Params
MODELS_DIR = os.path.join(ROOT, "models")
# A first sweep (alpha 0.02-0.2, every cost 2-16) put the optimum on the alpha=0.2 / team_cost=2
# edge and showed stop and restart costs nearly flat, so the grid was widened along those two.
GRID = {
    "alpha": (0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    "team_cost": (0.25, 0.5, 1.0, 2.0, 4.0),
    "stop_cost": (2.0, 8.0, 32.0),
    "restart_cost": (2.0, 8.0, 32.0),
}


def score(labels, pred_labels):
    """(spellF1, macroF1) of predicted labels against H/A/D/X truth, with the official code."""
    gt = pm.label_codes(labels)
    pred = pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell, macro


def baseline_labels(trk, window=50):
    """starter/baseline.py's rule on any Tracking: nearest player to the ball, majority vote."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from baseline import smooth  # noqa: E402
    distance = np.hypot(trk.px - trk.bx[:, None], trk.py - trk.by[:, None])
    filled = np.where(np.isnan(distance), np.inf, distance)
    nearest = trk.team[np.argmin(filled, axis=1)].astype(np.int8)
    codes = np.where(trk.ball_seen & np.isfinite(filled.min(axis=1)), nearest, -1)
    labels = smooth(codes.astype(np.int8), window).astype(object)
    labels[trk.half == 0] = "X"
    return labels


_WORKER = {}


def _init_worker(held_out, gt_codes, blocks):
    _WORKER.update(held_out=held_out, gt_codes=gt_codes, blocks=blocks,
                   runs=[pm.valid_runs(g) for g in gt_codes])


def _grid_score(params):
    """Mean fast spellF1 of one decoder setting over the held-out matches."""
    spells = []
    for (p_dead, p_home), gt, runs, block in zip(_WORKER["held_out"], _WORKER["gt_codes"],
                                                 _WORKER["runs"], _WORKER["blocks"]):
        pred = decode(p_dead, p_home, block, **params)
        spells.append(fast_spell_f1(gt, np.where(pred < 0, pm.X, pred).astype(np.int8), runs))
    return float(np.mean(spells))


def _code_version():
    """Short hash of the code a feature table depends on, so an edit invalidates the cache."""
    digest = hashlib.sha256()
    for path in ("src/features.py", "src/context.py", "tuning/degrade.py", "tuning/idsse.py"):
        with open(os.path.join(ROOT, path), "rb") as handle:
            digest.update(handle.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:10]


def build_tables(profile):
    """Degraded feature tables and labels for every external match, cached."""
    tables, labels, blocks, baselines = [], [], [], []
    version = _code_version()
    for index, match in enumerate(MATCHES):
        cache = os.path.join(CACHE_DIR, "table_{}_degraded_s{}_{}.npz".format(
            match, index, version))
        trk, y = load_match(match)
        noisy = degrade(trk, profile, seed=index)
        if os.path.isfile(cache):
            with np.load(cache, allow_pickle=True) as stored:
                x, names = stored["x"], list(stored["names"])
        else:
            x, names = frame_table(noisy)
            np.savez(cache, x=x, names=np.array(names, dtype=object))
        tables.append(x)
        labels.append(y)
        blocks.append(noisy.half_blocks())
        baselines.append(baseline_labels(noisy))
        print("  {}  {} frames, {} features".format(match, len(y), x.shape[1]), flush=True)
    return tables, labels, blocks, baselines, names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller decoder grid")
    parser.add_argument("--refit", action="store_true", help="ignore cached held-out predictions")
    parser.add_argument("--stack", action="store_true", help="add the second stage (src/stack.py)")
    parser.add_argument("--drop", nargs="+", default=[],
                        help="ablation: drop features whose name starts with any of these; "
                             "scores only, models/ is left untouched")
    args = parser.parse_args()
    started = time.time()

    print("noise profile from the BAR-ATH ATD (input statistics only)")
    profile = noise_profile(load_metrica())
    print("building degraded external tables")
    tables, labels, blocks, baselines, names = build_tables(profile)
    if args.drop:
        keep = [i for i, name in enumerate(names) if not name.startswith(tuple(args.drop))]
        print("ablation: dropping {} of {} features ({})".format(
            len(names) - len(keep), len(names), ", ".join(args.drop)))
        tables = [t[:, keep] for t in tables]
        names = [names[i] for i in keep]

    print("leave-one-match-out fits")
    held_out_path = os.path.join(CACHE_DIR, "held_out_{}_{}.npz".format(
        _code_version(), hashlib.sha256((repr(GBM_PARAMS) + repr(names)).encode()).hexdigest()[:6]))
    if os.path.isfile(held_out_path) and not args.refit:
        with np.load(held_out_path) as stored:
            held_out = [(stored["dead_{}".format(i)], stored["home_{}".format(i)])
                        for i in range(len(MATCHES))]
    else:
        held_out = []
        for index, match in enumerate(MATCHES):
            others = [i for i in range(len(MATCHES)) if i != index]
            model = PossessionModel(names).fit([tables[i] for i in others],
                                               [labels[i] for i in others])
            held_out.append(model.predict(tables[index]))
            print("  {} fitted ({:.0f}s)".format(match, time.time() - started), flush=True)
        np.savez(held_out_path, **{"{}_{}".format(kind, i): pair[k]
                                   for i, pair in enumerate(held_out)
                                   for k, kind in enumerate(("dead", "home"))})

    names2, stage_two_tables = None, None
    if args.stack:
        # Stage two reads stage one's leave-one-match-out predictions. Its own held-out score
        # carries a weak optimism: the stage-one predictions it trains on for match j came from
        # a model that had seen the held-out match's labels. Fine for ranking ideas, which is
        # what it is used for here.
        print("stage two on leave-one-match-out stage-one predictions")
        stacked = [stack_table(p_dead, p_home, x, names)
                   for (p_dead, p_home), x in zip(held_out, tables)]
        names2 = stacked[0][1]
        stage_two_tables = [s for s, _ in stacked]
        held_out = []
        for index, match in enumerate(MATCHES):
            others = [i for i in range(len(MATCHES)) if i != index]
            model2 = PossessionModel(names2).fit([stage_two_tables[i] for i in others],
                                                 [labels[i] for i in others])
            held_out.append(model2.predict(stage_two_tables[index]))
            print("  {} stage two fitted ({:.0f}s)".format(match, time.time() - started),
                  flush=True)

    grid = dict(GRID)
    if args.quick:
        grid = {key: values[::2] for key, values in grid.items()}
    keys = sorted(grid)
    gt_codes = [pm.label_codes(y) for y in labels]
    candidates = [dict(zip(keys, values))
                  for values in itertools.product(*(grid[key] for key in keys))]
    print("decoder grid: {} settings x {} matches".format(len(candidates), len(MATCHES)),
          flush=True)
    # Settings are independent, so they are spread over processes; each worker receives the
    # held-out predictions once, and the result does not depend on how the work was split.
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1),
                             initializer=_init_worker,
                             initargs=(held_out, gt_codes, blocks)) as pool:
        means = list(pool.map(_grid_score, candidates, chunksize=4))
    results = sorted(zip(means, candidates), key=lambda item: -item[0])

    # Re-score the leaders with the official code, and rank them on its numbers.
    official = []
    for _, params in results[:8]:
        per_match = [score(y, to_labels(decode(p_dead, p_home, block, **params)).astype(object))
                     for (p_dead, p_home), y, block in zip(held_out, labels, blocks)]
        official.append((float(np.mean([s for s, _ in per_match])),
                         float(np.mean([m for _, m in per_match])), params, per_match))
    official.sort(key=lambda item: (-item[0], -item[1]))
    results = official

    base = [score(y, b) for y, b in zip(labels, baselines)]
    print("\nbaseline (nearest + smooth50) on degraded IDSSE: spellF1 {:.4f}  macroF1 {:.4f}".format(
        np.mean([s for s, _ in base]), np.mean([m for _, m in base])))
    print("top decoder settings (held-out mean over {} matches):".format(len(MATCHES)))
    for spell, macro, params, _ in results[:8]:
        print("  spellF1 {:.4f}  macroF1 {:.4f}  {}".format(spell, macro, params))
    best_spell, best_macro, best, per_match = results[0]
    for match, (s, m) in zip(MATCHES, per_match):
        print("    {}  spellF1 {:.4f}  macroF1 {:.4f}".format(match, s, m))
    if args.drop:
        print("ablation run: models/ not written ({:.0f}s total)".format(time.time() - started))
        return

    print("\nfinal fit on all {} matches".format(len(MATCHES)))
    final = PossessionModel(names).fit(tables, labels)
    final.save(os.path.join(MODELS_DIR, "possession_model.pkl"))
    stage_two_path = os.path.join(MODELS_DIR, "stage_two_model.pkl")
    if args.stack:
        PossessionModel(names2).fit(stage_two_tables, labels).save(stage_two_path)
    elif os.path.isfile(stage_two_path):
        os.remove(stage_two_path)  # a stale stage two would not match the new stage one
    with open(os.path.join(MODELS_DIR, "decoder.json"), "w") as handle:
        json.dump({"params": best, "stacked": bool(args.stack),
                   "external_spell_f1": best_spell, "external_macro_f1": best_macro,
                   "matches": list(MATCHES)}, handle, indent=2)
    print("wrote models/ ({:.0f}s total)".format(time.time() - started))


if __name__ == "__main__":
    main()
