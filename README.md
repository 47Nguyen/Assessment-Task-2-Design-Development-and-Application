# COSC2753 Assignment 2 — Fashion Intelligence System

Machine learning system that predicts item type, season, gender and usage from a
fashion product image, plus a visual search engine for finding similar items.

## Setup

**1. Install dependencies** (Python 3.11)

```bash
pip install -r requirements.txt
```

**2. Add the dataset**

Download `A2_Fashion.zip` from Canvas and extract it into the repo root so the
layout is:

```
A2_FashionDataset/FashionDataset/
├── train/
│   ├── styles_train.csv
│   └── images_train/        ~38.6k .jpg
└── test/
    ├── styles_prediction.csv
    └── images_test/         5,829 .jpg
```

The dataset is gitignored — it is never committed.

**3. Check it works**

```bash
python -m src.data
```

You should see:

```
metadata: 38617 rows in CSV -> 38612 usable (5 dropped: no matching image)
          30890 train / 7722 val
```

## Starting your task

Each person has a folder under `tasks/` with a README explaining their task,
the data, the steps, and the traps. **Read your README first.**

| Folder | Task | Owner |
|---|---|---|
| [tasks/task1_article_type/](tasks/task1_article_type/) | Item type — 124 classes | M1 + M2 |
| [tasks/task2_season/](tasks/task2_season/) | Season — 4 classes | M3 |
| [tasks/task3_gender_usage/](tasks/task3_gender_usage/) | Gender & occasion | M3 |
| [tasks/task4_visual_search/](tasks/task4_visual_search/) | Visual search | M4 |

Copy the template into your folder and start there:

```bash
cp tasks/_template.py tasks/task2_season/train.py
```

Change the `target_value` line, then run it **from the project root** using `-m` and
dots instead of slashes:

```bash
python -m tasks.task2_season.train        # correct
python tasks/task2_season/train.py        # fails: "No module named 'src'"
```

The `-m` form tells Python to treat the project folder as the starting point,
which is what lets `from src...` find the shared code. It means nobody has to
edit a file path — the same command works on everyone's machine.

## Loading data

Everyone loads data the same way. **Do not write your own train/val split** — if
two people split differently, our results can't be compared in the report.

```python
from src.data import get_split, get_images_only

# Classification tasks — same train/val rows for every target
X_train, X_val, y_train, y_val, le = get_split("articleType")
X_train, X_val, y_train, y_val, le = get_split("season")
X_train, X_val, y_train, y_val, le = get_split("gender")
X_train, X_val, y_train, y_val, le = get_split("usage")

# Task 4 — all images, no labels needed
X, meta = get_images_only()
```

The first call decodes every image into `cache/images_train.npy` (~550 MB, a few
minutes). Every call after that is fast. The cache is gitignored, so each person
builds it once on their own machine.

`get_split` returns images already cleaned and normalised, and saves the label
encoder to `models/`.

## Building and evaluating a model

```python
from src.models import build_cnn, default_callbacks
from src.evaluate import evaluate_model, plot_confusion, per_class_report

model = build_cnn(n_classes=len(le.classes_))
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=30, batch_size=128,
          callbacks=default_callbacks("season"))

y_pred = model.predict(X_val).argmax(axis=1)
evaluate_model(y_val, y_pred, "season", "cnn_baseline", notes="lr 1e-3")
```

`evaluate_model` saves a row to `outputs/results.csv`. That file becomes the
comparison table in the report, so it builds itself as everyone works.

**Macro-F1 is our main metric, not accuracy.** Always predicting "Casual" scores
76.8% accuracy on `usage` while learning nothing.

## What goes where

| In `src/` (shared) | In your `train.py` |
|---|---|
| Fixing the data — junk columns, missing images, grayscale files, odd sizes, normalisation, the split | Anything you might want to undo and compare against |
| The CNN everyone shares | Merging rare classes |
| The metrics everyone reports | Class weighting, augmentation, tuning |

Rule of thumb: **if it fixes the data it goes in `src/`; if it's an experiment it
goes in your file.** Merging classes inside the shared loader would make the
merged-vs-unmerged comparison impossible, and that comparison is part of the
report.

> **`src/data.py`, `src/models.py` and `src/evaluate.py` are frozen at the end of
> Week 1.** Changing them later invalidates everyone's numbers.

## Data quality notes

Measured from the actual files. These justify the cleaning steps and belong in
the report's preprocessing section.

| Finding | Detail | Handled by |
|---|---|---|
| Junk columns | `Unnamed: 10`, `Unnamed: 11` from stray commas | dropped on load |
| ID/file mismatch | 5 CSV rows have no image; 1 image has no CSV row | inner join |
| Grayscale images | 343 train (0.9%), 88 test (1.5%) in PIL mode `L` | converted to RGB |
| Irregular sizes | 17 train, 6 test images are not 60×80 (e.g. 53×80, 60×60) | resized |
| Missing labels | `season` 20, `usage` 72; `articleType` and `gender` 0 | filtered per target |
| Single-example classes | 7 `articleType` classes have 1 image — can't be split | routed to train |
| Class coverage | 110/124 `articleType` and 7/8 `usage` classes appear in val | unavoidable; report it |

## Baselines to beat

| Target | Classes | Majority baseline |
|---|---:|---:|
| `articleType` | 124 | 0.176 |
| `season` | 4 | 0.496 |
| `gender` | 5 | 0.542 |
| `usage` | 8 | 0.769 |

## Repo layout

```
src/
  config.py     paths, seed, constants
  data.py       loading, cleaning, train/val split   <- shared, frozen after Week 1
  models.py     build_cnn()                          <- shared, frozen after Week 1
  evaluate.py   metrics                              <- shared, frozen after Week 1
tasks/
  _template.py           copy this to start
  task1_article_type/    README + your train.py
  task2_season/
  task3_gender_usage/
  task4_visual_search/
cache/          decoded images (gitignored)
models/         trained models + label encoders (gitignored)
outputs/        results.csv, figures
```

## Team

| Member | Task | Also owns |
|---|---|---|
| M1 | Task 1 — architecture & tuning | `src/data.py` |
| M2 | Task 1 — class imbalance & error analysis | `src/models.py`, `src/evaluate.py` |
| M3 | Tasks 2 & 3 — season, gender, usage | the demo app, later |
| M4 | Task 4 — visual search | |
