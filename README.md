# COSC2753 Assignment 2 — Fashion Intelligence System

Machine learning system that predicts item type, season, gender and usage from a
fashion product image, plus a visual search engine for finding similar items.

## 1. Python version

Python 3.11

## 2. Install requirements

```bash
pip install -r requirements.txt
```

## 3. Add the dataset

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

## 4. Run each task

Run these from the project root.

**Task 1 — Item type**

```bash
python -m tasks.task1_article_type.train
```

**Task 2 — Season**

```bash
python -m tasks.task2_season.train --tune
python -m tasks.task2_season.predict
```

**Task 3 — Gender & usage**

```bash
python -m tasks.task3_gender_usage.train
python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --export
```

**Task 4 — Visual search**

```bash
python tasks/task4_visual_search/task-4.py
```

Tasks 1–3 all write into the same shared submission file,
`outputs/COSC253_A2_SG_G9.csv`, each filling in only its own column
(`articleType`, `season`, or `gender`/`usage`). Run them in any order — each
run updates the file in place without touching the other tasks' columns.
Task 4's output is separate: `outputs/task_4/task4_topk_predictions_tuned.csv`.

Each task's folder under `tasks/` has its own README with more detail if you
get stuck.
