# Task 1 — Item Type Classification (`articleType`)

**Owners: M1 + M2** · Report section 3.1

> Predict what type of fashion item is in the image — T-shirt, Shirt, Jeans,
> Watches, Handbags, Casual Shoes, and 118 others.

This is the hardest task, which is why two people are on it.

## Start here

```bash
cp ../_template.py train.py
```

Change `target_value = "articleType"` and run it:

```bash
python -m tasks.task1_article_type.train
```

## What the data looks like

| | |
|---|---|
| Classes | **124** |
| Train / val images | 30,890 / 7,722 |
| Biggest class | Tshirts, 17.6% |
| Baseline to beat | **0.176** accuracy |
| Classes with under 100 images | **79** |
| Classes with only 1 image | 7 |

**Important:** if you got every class with 100+ images perfectly right and
scored zero on all the rest, macro-F1 would still only be **0.368**. That's the
ceiling built into the data.

> **A macro-F1 around 0.4 is a good result here, not a bug.**
> Don't spend a week hunting for a problem that's actually the class imbalance.

## Steps

**1. Baseline.** What score do you get always predicting "Tshirts"? Every model
has to beat this.

**2. Simple models first.** Before the CNN, try classical ML so you have
something to compare against:
- Features: colour histograms, HOG (`skimage.feature.hog`)
- Models: Logistic Regression, SVM, Random Forest, k-NN

**3. CNN.** Use `build_cnn()` from `src/models.py`.

**4. Tune it.** Change one thing at a time, in this order:

| Order | What | Try |
|---|---|---|
| 1 | `learning_rate` | 1e-2, 1e-3, 1e-4 |
| 2 | `filters` | (32,64), (32,64,128), (64,128,256) |
| 3 | `dropout` | 0.2, 0.3, 0.5 |

Log every run with `evaluate_model()` under a different `model_name`. Those rows
become the report's tuning table.

**5. Fix the imbalance (M2's part).** This is the big one for marks:

| Try this | Name it | Question it answers |
|---|---|---|
| Class weights | `cnn_weighted` | Does the easy fix help? |
| Oversample rare classes | `cnn_oversampled` | Is repeating better than weighting? |
| Merge rare classes into `subCategory` | `cnn_merged` | Is a clean 41-class model better than a broken 124-class one? |

`class_weights(y_train)` from `src/evaluate.py` gives you the weights.

**6. Look at the mistakes.** Actually display misclassified images. You'll
quickly see that "Casual Shoes" and "Sports Shoes" look nearly identical at
60×80 — that's an argument the confusion matrix alone can't make.

## Splitting the work

- **M1** — baselines, CNN, tuning. Have a working model by **end of Week 2**.
- **M2** — imbalance experiments and error analysis, built on M1's model.

## Watch out for

- Don't drop rare classes just to make the score look better — that changes the
  question. Report the real number and explain why the ceiling exists.
- Don't merge classes inside `src/data.py`. Merging is an experiment, so it has
  to be reversible for comparison.
- 14 classes have no validation images at all. Mention it as a limitation.

## Checklist

- [ ] Baseline logged
- [ ] At least 2 classical models tried
- [ ] CNN beats them
- [ ] Tuning runs logged in `outputs/results.csv`
- [ ] At least 3 imbalance experiments compared
- [ ] Confusion matrix + misclassified image examples
- [ ] Model saved to `models/cnn_articleType.keras`
- [ ] Notes written for the report
