"""spellF1 exactly as evaluation/possession_metrics.py defines it, fast enough for a grid search.

The official spell_overlaps compares every true possession with every predicted one, which is fine
for scoring a file and slow for scoring thousands of decoder settings: a jittery candidate has
thousands of spells. Spells inside one sequence never overlap each other, so only pairs that
overlap in time can have IoU > 0, and a two-pointer sweep finds all of them in linear time. The
pairs are then sorted and matched exactly as the official code does, so the result is identical
-- test_fast_metric() checks that against the official function.

Used only to rank decoder settings during tuning. Every number reported anywhere comes from the
official module.
"""

import numpy as np

import possession_metrics as pm


def _spells(codes, runs):
    return [(a, b, k) for a, b, k in pm.segments(codes, runs) if k in pm.POSSESSION_LABELS]


def spell_f1(gt_codes, pred_codes, runs, ious=pm.SPELL_IOUS):
    gt, pred = _spells(gt_codes, runs), _spells(pred_codes, runs)
    overlaps = []
    j0 = 0
    for i, (gs, ge, gk) in enumerate(gt):
        while j0 < len(pred) and pred[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j][0] < ge:
            ps, pe, pk = pred[j]
            if pk == gk:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                overlaps.append((inter / float(union), i, j))
            j += 1
    overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
    f1s = []
    for threshold in ious:
        tp, fp, fn = pm.match_spells(overlaps, len(gt), len(pred), threshold)
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def test_fast_metric(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.repeat(rng.integers(0, 3, n // 50), 50)[:n].astype(np.int8)
    gt[:100] = pm.X
    runs = pm.valid_runs(gt)
    for flip in (0.001, 0.01, 0.05):
        pred = gt.copy()
        pred[pred == pm.X] = 0
        noise = rng.random(n) < flip
        pred[noise] = rng.integers(0, 3, noise.sum())
        official = pm.spell_scores(gt, pred, runs)["mean_f1"]
        assert abs(official - spell_f1(gt, pred, runs)) < 1e-12, (flip, official)
    return True
