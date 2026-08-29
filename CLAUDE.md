# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university group project (RMIT COSC2753 Assignment 2, due 12 Sep 2026): one visual
understanding problem over 60×80 fashion product photos, with four label views
(`articleType`, `season`, `gender`, `usage`) plus a top-K visual search engine.
It is a research/report codebase, not an application — the deliverable is a report
backed by `outputs/results.csv`, a filled `styles_prediction.csv`, and a demo app.

If need project context, task background refer to COSC2753_2026B_Assignment 2.pdf.

`PLAN.md` (gitignored, present locally) holds the goals, eight falsifiable hypotheses
and the phase-by-phase build order. `README.md` is the team-facing onboarding doc.
Read both before making design decisions — much of what looks like a missing feature
is a deliberate, documented choice.

## Commands

```bash
pip install -r requirements.txt          # Python 3.11, versions pinned deliberately

python -m src.data                       # smoke test: prints row counts + per-target missing labels
python -m tasks.task3_gender_usage.train_gender
python -m tasks.task3_gender_usage.train_usage
```

Always run scripts as modules from the repo root (`python -m tasks.x.y`), never
`python tasks/x/y.py` — the latter fails with `No module named 'src'`.

There is no test suite, linter, or CI. Verification is running a task script and
checking `outputs/results.csv`.

First run of anything that loads images decodes the whole dataset into
`cache/images_train.npy` (~550 MB, a few minutes); every later run is fast.
Delete `cache/` to force a rebuild.

## Setup requirement

The dataset is gitignored and must be extracted to
`A2_FashionDataset/FashionDataset/{train,test}/` (see README). Without it every
script fails at import of the CSV.

## Architecture

Three shared modules under `src/`, one file per task under `tasks/`. The split matters:

| `src/` — shared, **frozen after Week 1** | `tasks/<task>/train*.py` — per-member |
|---|---|
| `config.py` — paths, `SEED=42`, `IMG_SHAPE=(80,60,3)`, `STRATIFY_ON`, `VAL_SIZE` | class merging / relabelling |
| `data.py` — load, clean, cache, normalise, the one train/val split | class weighting, augmentation, tuning |
| `models.py` — `build_cnn()`, `default_callbacks()` | anything you might want to undo and compare against |
| `evaluate.py` — metrics, `outputs/results.csv`, confusion plots | |

Rule: **if it fixes the data it goes in `src/`; if it's an experiment it goes in the
task file.** Editing `src/data.py`, `src/models.py` or `src/evaluate.py` now
invalidates every member's already-recorded numbers — treat changes there as
requiring an explicit decision, not a routine refactor. Merging classes inside the
shared loader in particular would destroy the merged-vs-unmerged comparison that the
report depends on.

### Data flow

`load_metadata()` → drops `Unnamed:*` junk columns, inner-joins the CSV against
images actually on disk, and attaches a frozen `_split` column (stratified on
`articleType`; the 7 single-example classes are routed to train).
`get_split(target)` filters only rows missing *that* target's label, fits and
**saves** a `LabelEncoder` to `models/`, then returns
`(X_train, X_val, y_train, y_val, le)`. Every target shares the same row assignment
— that is what makes results comparable across tasks.

Entry points: `get_split(target)` for classification, `get_images_only()` for
Task 4 retrieval, `load_test_images()` for submission (returns images in
`styles_prediction.csv` row order).

### Conventions that are load-bearing

- **Macro-F1 is the headline metric, never accuracy.** Always predicting "Casual"
  scores 0.769 accuracy on `usage`. `evaluate_model()` prints both plus the
  majority baseline and warns when accuracy far exceeds macro-F1.
- `evaluate_model(y_true, y_pred, target, model_name, notes)` appends to
  `outputs/results.csv`, replacing any prior row with the same `(target, model)` —
  so re-running a script is idempotent. `model_name` is what identifies the row in
  the report; keep it short and specific (`cnn_baseline`, `cnn_weighted`, `svm_hog`).
- A merged-class experiment is recorded under a **synthetic target name**
  (`gender_3class`, `usage_5class`) with its own local `LabelEncoder`, so folded and
  directly-trained variants sit side by side in the same results table. See
  [train_gender.py](tasks/task3_gender_usage/train_gender.py) for the pattern.
- Normalisation stats are computed on **train only** and passed back so val/test
  reuse them; they are saved to `models/channel_stats.joblib`. Prediction must load
  the saved encoder and stats, never refit — refitting reorders classes and yields a
  plausible-looking, completely wrong submission.
- All models are built through `build_cnn()` with arguments, not by copying and
  editing the function, so score differences attribute to the task rather than the
  architecture. `default_callbacks(target)` fixes the stopping rule and writes
  `models/cnn_<target>.keras`.

### Gotchas already discovered (documented in the code — don't "fix" them)

- `normalise()` accumulates in float64 on purpose: float32 saturates over ~24M
  values per channel and silently returns identical wrong means.
- `ModelCheckpoint` receives `str(path)`, not a `Path` — Keras 2.15 calls
  `.endswith()` on it.
- `protobuf<5.0.0` is pinned because TF 2.15 breaks on protobuf ≥5.
- ~0.9% of images are grayscale (PIL mode `L`) and a handful are not 60×80; both are
  handled during cache build.

`cache/`, `models/`, `outputs/`, the dataset and `PLAN.md` are all gitignored, so
generated artefacts never enter git. `asm(ignore).ipynb` is scratch — ignore it.
