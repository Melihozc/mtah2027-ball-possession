# Our solution: how to reproduce

Everything below runs on CPU. Python 3.11, dependencies pinned in `requirements.lock`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.lock
export MTAH_DATA_DIR=/path/to/bundle          # folder holding tactical/ and broadcast/
```

## 1. Regenerate the submission (minutes)

```bash
python make_submission.py --team <team_name>  # -> out/<team_name>.csv, prints its SHA-256
python evaluation/validate_submission.py out/<team_name>.csv
```

Uses only the committed `models/` and the data bundle.

## 2. Rebuild the models (optional, ~10 min)

```bash
# IDSSE (Bassek et al. 2025, CC BY 4.0) from https://huggingface.co/datasets/pysport/idsse-data
# into ../external/idsse, or point IDSSE_DIR at it
python -m tuning.run_external
```

Fits both classifiers and chooses every decoder parameter on seven Bundesliga matches, with
leave-one-match-out and the official `evaluation/possession_metrics.py`. The provided game's ground truth is never
read. The only thing read from BAR-ATH is the tracking noise profile (ball gap lengths), which is
an input statistic, not a label.

## Layout

| path | what |
|---|---|
| `src/tracking.py` | one in-memory format for Metrica ATD and DFL tracking |
| `src/features.py`, `src/context.py` | per-frame features, then windows and gap edges around them |
| `src/model.py` | P(dead) and P(home \| alive), gradient-boosted trees |
| `src/decode.py` | Viterbi over H/A/D with switch costs |
| `src/sync_broadcast.py` | broadcast-to-tactical synchronisation by matching player positions |
| `tuning/` | IDSSE loader, ATD-like degradation, leave-one-match-out tuning |
| `make_submission.py` | data bundle + models -> submission CSV |

Data citation: Bassek, M., Rein, R., Weber, H., Memmert, D. *An integrated dataset of
spatiotemporal and event data in elite soccer.* Scientific Data 12, 195 (2025).
doi:10.1038/s41597-025-04505-y.
