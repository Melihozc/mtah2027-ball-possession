"""The two per-frame classifiers, and the frame table they read.

  dead model   P(ball dead)            fitted on every graded frame
  team model   P(home has it | alive)  fitted on alive frames only

Both are gradient-boosted trees over the same feature table, because the table is mostly NaN in
exactly the places that matter (no ball for 41% of graded frames) and trees take NaN natively
rather than needing an imputation that would itself be a modelling decision. Fixed seeds and no
early stopping, so a refit on the same data gives the same model, and inference is deterministic
on CPU.

They are fitted on external matches only (tuning/), and the fitted models are stored in models/
so make_submission.py never has to refit.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.context import context_features, matrix
from src.features import all_features

# Params
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
TRAIN_STRIDE = 5   # every 5th frame: neighbours are near-duplicates, and 5x faster to fit


def frame_table(trk):
    """(X, names) for every frame of a Tracking."""
    return matrix(context_features(all_features(trk), trk))


class PossessionModel:
    def __init__(self, names):
        self.names = list(names)
        self.dead = HistGradientBoostingClassifier(**GBM_PARAMS)
        self.team = HistGradientBoostingClassifier(**GBM_PARAMS)

    def fit(self, tables, labels):
        """tables: list of X matrices; labels: matching lists of H/A/D/X arrays."""
        x = np.concatenate([t[::TRAIN_STRIDE] for t in tables])
        y = np.concatenate([l[::TRAIN_STRIDE] for l in labels])
        graded = y != "X"
        self.dead.fit(x[graded], (y[graded] == "D").astype(np.int8))
        alive = np.isin(y, ("H", "A"))
        self.team.fit(x[alive], (y[alive] == "H").astype(np.int8))
        return self

    def predict(self, table):
        """(p_dead, p_home) for every row."""
        return self.dead.predict_proba(table)[:, 1], self.team.predict_proba(table)[:, 1]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=4)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
