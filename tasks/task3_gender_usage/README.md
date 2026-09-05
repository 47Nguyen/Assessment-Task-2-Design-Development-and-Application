# Task 3: Fashion Gender and Usage (MLP)

Task 3 predicts **both** catalogue `gender` (5 classes) and `usage` (8 classes)
from a product image. Two independently trained MLPs keep these targets separate.
These labels describe product marketing, not a person's gender identity.

## Start in VSCode

1. Open the repository folder and select its Python 3.11 `.venv` interpreter.
2. Install the existing root requirements if needed: `python -m pip install -r requirements.txt`.
3. Open `tasks/task3_gender_usage/task3.ipynb`, select the same kernel and Run All.
4. The notebook uses `MODE = "train"` by default. Use `"review"` with an existing
   `RUN_DIR` to inspect a completed run without training again, or `"smoke"` for a small check.

Dataset layout from the repository root:

```text
A2_FashionDataset/FashionDataset/
  train/styles_train.csv
  train/images_train/<id>.jpg
  test/styles_prediction.csv
  test/images_test/<id>.jpg
```

No GPU, downloads or pretrained model weights are required. CPU time depends on
the machine. The full experiment trains six candidate MLPs plus one optional
merged-label experiment, up to 20 epochs each with early stopping. Raw images
are held as compact uint8 arrays; the model normalizes pixels internally.

## Command-Line Alternatives

Run from the repository root:

```powershell
python -m tasks.task3_gender_usage.train --smoke
python -m tasks.task3_gender_usage.train --run-dir outputs/task3/my_full_run
python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/my_full_run/models --output outputs/task3/my_full_run/styles_prediction_task3.csv
```

A run never overwrites another run. Choose a new run name when retraining.
`--smoke` uses a small stratified subset and two epochs; its scores are **not report results**.
The notebook's upload widget demonstrates both predictions using the same
`Task3Predictor` API as the command-line script. Scores are not calibrated confidence.

## What The Experiment Does

- Audits all train/test images: missing files, unreadable images, grayscale,
  unusual dimensions, missing labels, exact decoded-RGB duplicates and conflicting labels.
- Converts to RGB and resizes to 32 high x 24 wide, preserving the usual 4:3 aspect ratio.
- Uses one deterministic, approximately 70/15/15 train/validation/holdout split
  for both targets. Exact duplicate groups stay together. Joint gender/usage
  strata with fewer than three groups, and target-conflicting groups, stay in train.
  Reports the resulting coverage; `Home` cannot have a meaningful holdout score.
- Compares a majority predictor, a sigmoid MLP, a deeper ReLU/dropout MLP and
  the same deeper MLP with capped square-root inverse-frequency training weights.
- Selects epochs and the final MLP using **validation** macro-F1 only; evaluates
  only the selected original-label MLP on holdout. Holdout never drives tuning.
- Reports accuracy, fixed-vocabulary macro-F1, supported-class macro-F1, weighted F1,
  per-class support, confusion matrices, errors, training curves and batch latency.
- Tests merging rare usage labels separately: compare retrained five-class
  predictions with the eight-class model's probabilities folded to the **same**
  five labels, on the **same** images. Final submissions retain all eight labels.
- Saves models and class mappings, checks save/reload equivalence, and creates
  a new CSV preserving official ID order and all non-Task-3 columns.

The sigmoid-to-deeper comparison changes several architectural settings together;
it is an approach comparison, not proof of which single setting caused a difference.
The weighted-versus-unweighted deeper models isolate class weighting.

## Files And Outputs

`data.py` handles auditing, splitting and image preprocessing. `train.py` handles
experiments and evaluation. `predict.py` exposes reloadable inference and notebook upload.
The notebook presents the analysis and calls those same functions, without duplicated training code.

Each run is isolated under `outputs/task3/<run>/` (already gitignored):

| Output | Purpose |
| --- | --- |
| `audit.json`, `audit_images_*.csv` | Actual dataset checks |
| `split_manifest.csv`, `class_coverage.csv` | Reproduce the split and inspect rare labels |
| `results_task3.csv`, `selected_models.json` | Comparisons and evidence-based selection |
| `*_report.csv`, `*_confusion.png`, `*_errors.png`, `*_learning.png` | Report evidence |
| `models/gender_final.keras`, `models/usage_final.keras` | Reloadable selected models |
| `models/*_final.json`, `config.json`, `provenance.json` | Classes, preprocessing and reproducibility |
| `styles_prediction_task3.csv` | Task 3 contribution, **not** a complete team submission |

Model files are gitignored, but must be included in the final assignment ZIP.
Do not upload data, models, predictions or code until the owner reviews them.

## Integration And Limitations

The root README currently references a deleted `src` package. This Task 3 pipeline
does not restore it or change Tasks 1/2/4. Its split is explicit and saved; it is
**not** the old shared split. Teammates must align row IDs before claiming controlled
comparisons across tasks. Using the same seed alone does not produce the same split.

Flattening preserves pixel positions; it does not erase image information. MLPs
can learn spatial patterns, but lack CNN locality/weight-sharing and are sensitive
to alignment. Low image resolution and marketing labels limit generalization.
Exact hashing does not catch near duplicates. Tiny validation classes have noisy
scores; absent classes are explicitly counted as zero in fixed-vocabulary macro-F1,
not claimed to have been independently tested.

An untouched holdout is an **internal** generalization check. It is not external
real-world validation. The notebook discusses related published work, but it uses
different datasets/labels and cannot justify a direct numerical benchmark claim.
Team members still need independently labelled external product images for a
credible deployment claim; do not guess labels from appearance or call the blind
assignment test set a labelled evaluation set.

## Sources

- [TensorFlow EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
- [scikit-learn F1 definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Liu et al., DeepFashion (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html)
- [Zakizadeh et al., Improving the Annotation of DeepFashion Images (2018)](https://arxiv.org/abs/1807.11674)

Review and understand the code and verify the course's AI-use/disclosure rules
before submitting. This Task 3 implementation does not complete the team's
five-page report, Tasks 1/2/4, or a combined application automatically.
