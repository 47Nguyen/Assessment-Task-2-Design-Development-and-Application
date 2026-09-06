# Task 3 — Gender & Usage Classification (`gender`, `usage`)

**Report section 3.3**

> Predict two catalogue labels from one fashion image: who the item is for
> (`gender`) and when it is suitable (`usage`).

Task 3 trains two separate Multi-Layer Perceptron (MLP) classifiers. Keeping the
targets separate avoids creating many rare `gender × usage` combinations and
allows each target to keep its own class vocabulary.

## Start here

Open a PowerShell terminal at the repository root and activate Python 3.11:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The three Task 3 scripts are already prepared. Do **not** copy `_template.py`
over them. Always use `python -m` from the repository root.

| File | Purpose |
|---|---|
| `data.py` | Audit data, standardise images and create the split |
| `train.py` | Train, compare, select and evaluate MLP candidates |
| `predict.py` | Reload final models, predict one image or export a CSV |

Required dataset layout:

```text
A2_FashionDataset/FashionDataset/
├── train/
│   ├── styles_train.csv
│   └── images_train/<id>.jpg
└── test/
    ├── styles_prediction.csv
    └── images_test/<id>.jpg
```

## What the data looks like

After matching metadata to readable image files, Task 3 has **38,612 usable
training images**.

### `gender`

| Class | Images | Share |
|---|---:|---:|
| Men | 20,913 | 54.2% |
| Women | 14,160 | 36.7% |
| Unisex | 2,080 | 5.4% |
| Boys | 814 | 2.1% |
| Girls | 645 | 1.7% |

- 5 classes and no missing labels
- Majority baseline: approximately **54.0% holdout accuracy**

### `usage`

| Class | Images | Share |
|---|---:|---:|
| Casual | 29,636 | 76.8% |
| Sports | 3,940 | 10.2% |
| Ethnic | 2,570 | 6.7% |
| Formal | 2,300 | 6.0% |
| Smart Casual | 55 | 0.1% |
| Travel | 25 | 0.1% |
| Party | 13 | <0.1% |
| Home | 1 | <0.1% |

- 8 classes and 72 rows with a missing usage label
- Majority baseline: approximately **76.9% holdout accuracy**
- `Home` has no validation or holdout example, so both complete-vocabulary and
  supported-class macro-F1 are recorded

## The thing to understand before you start

Accuracy alone is misleading for `usage`: a model that always predicts
`Casual` already reaches about 77% accuracy. The main selection metric is
therefore **validation macro-F1**, which gives each class equal importance.

The code found **636 exact-duplicate groups**. Every duplicate group stays in a
single partition to prevent the same decoded image appearing in both training
and evaluation. The resulting shared split is:

| Split | Images | Purpose |
|---|---:|---|
| Train | 27,029 | Learn model parameters |
| Validation | 5,777 | Compare candidates and select the winner |
| Holdout | 5,806 | Evaluate the frozen winner once |

Task 3 does not use Task 2's split, so their raw scores are not a controlled
same-split model comparison.

## Steps

### 1. Audit the data (optional)

```powershell
python -m tasks.task3_gender_usage.data --output-dir outputs/task3/data_check_3
```

This creates audit reports and `split_manifest.csv`; it does not train a model.
The folder must be new. If it already exists, choose another name.

### 2. Check the complete training pipeline quickly

```powershell
python -m tasks.task3_gender_usage.train --smoke --run-dir outputs/task3/smoke_check_3
```

Smoke mode uses a small subset and two epochs. It verifies execution only; do
not use smoke-test scores in the report. The initial audit still scans and
hashes every train/test image to detect duplicate leakage, so there may be a
short period with no new terminal output before the two-epoch training begins.

### 3. Train a new full run only when needed

```powershell
python -m tasks.task3_gender_usage.train --run-dir outputs/task3/final_run_2
```

Each run directory must be new because the program intentionally refuses to
overwrite models or evidence. A verified full run already exists at
`outputs/task3/final_run`, so retraining is not required just to predict.

For each target, training compares:

1. Majority-class baseline.
2. `mlp_default`: one 256-unit sigmoid hidden layer.
3. `mlp_regularized`: 256/128 ReLU layers with dropout.
4. `mlp_weighted`: the regularized network with softened class weights.

The highest validation macro-F1 wins. Ties prefer fewer parameters and then a
stable model name. Holdout scores are never used for model selection.

### 4. Predict one image with the verified models

```powershell
python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --image A2_FashionDataset/FashionDataset/test/images_test/52003.jpg
```

The command prints one `gender` and one `usage` label plus softmax scores.
Softmax scores are not calibrated guarantees of correctness.

### 5. Export predictions for all 5,829 test images

```powershell
python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --output outputs/task3/final_run/styles_prediction_task3_check_2.csv
```

To preserve predictions already filled by teammates, pass their official-layout
CSV as a template and write to a different file:

```powershell
python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --template outputs/team_predictions_before_task3.csv --output outputs/team_predictions_with_task3.csv
```

`predict.py` changes only `gender` and `usage`. It verifies official column and
ID order and refuses to overwrite the source or an existing destination.

## Verified results

| Target | Selected model | Holdout accuracy | Holdout macro-F1 |
|---|---|---:|---:|
| Gender | `mlp_default` | 0.8250 | 0.6186 |
| Usage (official 8 classes) | `mlp_default` | 0.8293 | 0.3141 |

The optional five-class usage experiment merges the four extremely small
labels into `Other`. Its retrained model reaches holdout macro-F1 **0.5165**,
compared with **0.5026** when the official eight-class probabilities are folded.
This is analysis evidence only; submission predictions still use all 8 labels.

## Main outputs

```text
outputs/task3/final_run/
├── models/
│   ├── gender_final.keras
│   ├── gender_final.json
│   ├── usage_final.keras
│   └── usage_final.json
├── audit.json
├── class_coverage.csv
├── results_task3.csv
├── selected_models.json
├── split_manifest.csv
├── gender_final_holdout_report.csv
├── gender_final_holdout_confusion.png
├── usage_final_holdout_report.csv
├── usage_final_holdout_confusion.png
└── styles_prediction_task3.csv
```

Keep each `.keras` model with its matching `.json`; the JSON stores the image
size and exact class order required by `predict.py`.

## Watch out for

- Do not type backslashes before underscores. Use `task3_gender_usage`, not
  `task3\_gender\_usage`.
- Do not run `python tasks/task3_gender_usage/train.py`; use the `python -m`
  command from the repository root.
- Do not run `python -m venv .venv` while `.venv` is active. If packages say
  `Requirement already satisfied`, the environment is already installed.
- `FileExistsError` for a run or CSV means the safety check worked. Choose a
  new output name instead of deleting a verified result.
- Do not report accuracy alone. Include macro-F1, per-class results, confusion
  matrices and the class imbalance limitation.
- `outputs/` is ignored by Git. Include required models/results separately in
  the final submission ZIP; do not assume `git push` uploads them.

## Checklist

- [x] Data quality audit and duplicate-aware split saved
- [x] Majority baselines recorded for both targets
- [x] Three MLP candidates compared for both targets
- [x] Validation macro-F1 used for selection
- [x] Accuracy, macro-F1 and per-class reports saved
- [x] Confusion matrices and error examples saved
- [x] Five-class usage improvement investigated
- [x] Final models reloaded and tested on RGB, grayscale and RGBA images
- [x] Test CSV exported with 5,829 IDs in official order
- [ ] Team reviews the local changes before commit
- [ ] Task 1 and Task 2 predictions are merged into the final team CSV
- [ ] Code, final models, evidence and README are included in the submission ZIP

No notebook is required for this script-based workflow. `task3.ipynb` may be
kept as optional demonstration material, but `data.py`, `train.py`,
`predict.py`, the final models, evidence and this README are the core files.
