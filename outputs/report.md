# Fashion Intelligence System — Report

COSC2753 Assignment 2

Four models over a catalogue of fashion product images: article type, season,
gender and occasion of use, plus a visual search engine that retrieves similar
items without using labels at query time.

Every number in this report is read from a saved artefact in `outputs/` or
`models/`. Source files are cited per section.

---

## 1. Evaluation contract

All four tasks share one train/validation split, frozen after Week 1, so results
are directly comparable: **30,890 train / 7,722 validation** rows from 38,612
usable images.

**Macro-F1 is the headline metric, not accuracy.** Every target here is
imbalanced, and accuracy rewards a model for ignoring the tail. On `usage`, a
model that always predicts "Casual" scores 0.769 accuracy and learns nothing.
Macro-F1 weights all classes equally, so that model scores 0.109. Balanced
accuracy is reported alongside as a second imbalance-aware view.

Majority-class baselines, which every model must beat to be worth anything:

| Target | Classes | Baseline accuracy |
|---|---:|---:|
| `articleType` | 124 | 0.178 |
| `season` | 4 | 0.497 |
| `gender` | 5 | 0.542 |
| `usage` | 8 | 0.769 |

---

## 2. Data and preprocessing

The dataset needed real repair before any model could be trained. Each finding
below was measured from the files, not assumed.

| Finding | Detail | Handling |
|---|---|---|
| Junk columns | `Unnamed: 10`, `Unnamed: 11`, created by stray commas in the CSV | Dropped on load |
| ID/file mismatch | 5 CSV rows have no image; 1 image has no CSV row | Inner join on ID |
| Grayscale images | 343 train (0.9%), 88 test (1.5%) stored in PIL mode `L` | Converted to RGB |
| Irregular sizes | 17 train, 6 test images not 60×80 (e.g. 53×80, 60×60) | Resized to 60×80 |
| Missing labels | `season` 20, `usage` 72; `articleType` and `gender` complete | Filtered per target |
| Single-example classes | 7 `articleType` classes have exactly 1 image | Routed to train — cannot be split |

Images are decoded once to a cached array, converted to float32 and normalised.
Cleaning lives in the shared loader; anything a task might want to undo and
compare against — class merging, weighting, augmentation — lives in that task's
own file, so every such comparison remains possible.

**Coverage caveat.** Only 110 of 124 `articleType` classes and 7 of 8 `usage`
classes appear in the validation set at all. Macro-F1 is therefore computed over
the classes present, and the 14 absent article types are invisible to
validation. This is unavoidable given classes with one or two examples, but it
means Task 1's macro-F1 is measured on an easier problem than the full label
space.

---

## 3. Task 1 — Article type (124 classes)

Source: `outputs/task_1/results.csv`

The hardest target in the assignment: 124 classes with a severe long tail, seven
of which have a single example.

| Model | Macro-F1 | Balanced acc | Accuracy |
|---|---:|---:|---:|
| Majority baseline | 0.003 | 0.009 | 0.178 |
| **Custom CNN** | **0.419** | **0.464** | 0.718 |
| CNN + class weighting | 0.055 | 0.088 | 0.135 |
| CNN + oversampling | 0.091 | 0.125 | 0.335 |
| CNN, merged `subCategory` | 0.180 | 0.177 | 0.685 |

The plain custom CNN reaches macro-F1 0.419 against a baseline of 0.003 — a
150× improvement, and the clearest evidence in the report that the model learned
genuine visual structure rather than the class prior. Accuracy 0.718 against
0.178 confirms it.

### Imbalance handling — a negative result

Both standard remedies for class imbalance made the model substantially **worse**:
class weighting fell to macro-F1 0.055 and oversampling to 0.091, against 0.419
for no treatment at all. Accuracy collapsed in step (0.135 and 0.335 against
0.718), so this is not the usual macro-F1-up/accuracy-down trade — the models
were simply worse.

The likely mechanism is the shape of the tail. With 124 classes and singleton
members, inverse-frequency weighting hands enormous gradient weight to classes
with one or two examples, and the model chases that noise instead of the
learnable structure in the head of the distribution. Oversampling does the same
thing by duplication. On a 4-class problem these techniques work (see Task 2,
where balanced weights help); on a 124-class long tail with singletons they
destabilise training.

### Hyperparameter sweep

| Variant | Macro-F1 |
|---|---:|
| lr 1e-3, dropout 0.4 | 0.135 |
| lr 5e-4, dropout 0.4 | 0.069 |
| lr 1e-4, dropout 0.4 | 0.137 |
| lr 1e-3, dropout 0.3 | 0.147 |
| lr 1e-3, dropout 0.5 | 0.194 |

Within the sweep, dropout 0.5 at lr 1e-3 is best (0.194), and dropout is a
stronger lever than learning rate. But every variant scores far below the
original `cnn_custom` run at 0.419. That gap is too large to be explained by
hyperparameters alone and points to the sweep runs being under-trained relative
to the original — fewer epochs, or stopping earlier. **The sweep is therefore
valid for ranking variants against each other, but not for concluding that the
original configuration was beaten or that these hyperparameters are poor.** It is
reported as measured rather than quietly dropped.

### Class merging

Collapsing 124 article types into the coarser `subCategory` target gives
macro-F1 0.180 at accuracy 0.685. The accuracy is comparable to the fine-grained
model while macro-F1 is lower, indicating the merged classes are themselves
imbalanced and the model leans on the larger ones. Merging did not rescue the
tail; it relabelled it.

---

## 4. Task 2 — Season (4 classes)

Source: `outputs/task_2/results.csv`

The cleanest experimental design in the project: a feature ablation establishing
what signal exists, then a model comparison, then a hyperparameter study with
one variable moved at a time.

### Feature ablation

| Model | Features | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Majority baseline | — | 0.166 | 0.497 |
| Logistic regression | mean R, G, B only (3 features) | 0.248 | 0.281 |
| Logistic regression | HOG + 16-bin RGB histogram | 0.578 | 0.614 |
| **Random forest** | HOG + colour histogram | **0.734** | **0.732** |

The mean-RGB model is the informative failure. With only three features it
reaches macro-F1 0.248 — above the baseline's 0.166, so average colour does
carry *some* seasonal signal — but its accuracy of 0.281 is far *below* the
0.497 baseline. Average colour alone is not enough to predict season, which
justifies the full HOG-plus-histogram representation: shape and colour
*distribution* matter, not average hue. Adding them lifts macro-F1 from 0.248 to
0.578, and switching to a random forest lifts it again to 0.734.

### Hyperparameter study

| Variant | Macro-F1 | Balanced acc | Accuracy |
|---|---:|---:|---:|
| n=100, depth None, balanced | 0.728 | 0.703 | 0.723 |
| n=300, depth None, balanced | 0.731 | 0.706 | 0.728 |
| **n=800, depth None, balanced** | **0.734** | **0.708** | 0.732 |
| n=800, depth 20, balanced | 0.728 | 0.708 | 0.722 |
| n=800, depth 10, balanced | 0.653 | 0.683 | 0.649 |
| n=800, depth None, no weights | 0.699 | 0.640 | 0.730 |

Three findings, each from moving one variable:

**Tree count saturates.** 100 → 800 trees gains 0.006 macro-F1. The forest is
already converged at 100; the extra 700 trees buy nothing but compute.

**Depth restriction hurts.** Capping depth at 10 costs 0.081 macro-F1, at 20 it
costs 0.006. The trees need depth to separate these classes, and there is no
overfitting penalty for letting them grow — unrestricted is both simplest and
best.

**Class weighting is the real lever, and it behaves exactly as theory predicts.**
Removing balanced weights moves accuracy *up* slightly (0.730 vs 0.732,
effectively unchanged) while macro-F1 drops 0.035 and balanced accuracy drops
0.068. The unweighted forest is quietly trading minority-class recall for
majority-class hits — invisible in accuracy, plain in the imbalance-aware
metrics. This is the clearest demonstration in the report of why macro-F1 is the
selection metric.

Note the contrast with Task 1: balanced weighting helps on 4 classes and is
destructive on 124 with singletons. The technique is not universally good or bad;
it depends on the shape of the tail.

---

## 5. Task 3 — Gender and usage

Source: `models/task_3/gender_final.json`, `models/task_3/usage_final.json`

Both targets use the same MLP (~591k parameters, 32×24 input, batch 128, Adam,
early stopping on validation macro-F1). Training is fast — 14.4 s and 8.6 s
respectively. Neither run used class weights (`class_weights: null`).

| Target | Classes | Best epoch | Val macro-F1 | Val accuracy |
|---|---:|---:|---:|---:|
| `gender` | 5 | 18 | **0.623** | 0.818 |
| `usage` | 8 | 7 | **0.325** | 0.835 |

**Gender** works. Macro-F1 climbs from 0.283 at epoch 1 to 0.623 at epoch 18,
with validation accuracy 0.818 against a 0.542 baseline. The macro-F1 curve is
noisy in the last third (0.623, 0.588, 0.580 over the final three epochs), so the
selected epoch sits on a fluctuation and the true performance is likely nearer
0.59–0.62. Selection on best validation macro-F1 without a held-out test set
mildly overstates the result.

**Usage does not work, and the history shows exactly why.** For the first three
epochs validation macro-F1 sat at precisely 0.1086 while accuracy sat at 0.7685
— the model had learned to predict "Casual" and nothing else, which is the
majority baseline reproduced exactly. It then improved to 0.325 by epoch 7 and
plateaued. Final accuracy of 0.835 looks respectable only because the majority
class is 77% of the data; macro-F1 of 0.325 across 8 classes is the honest
number, and it means most occasion categories are being missed.

This is the one result in the project that is not deployable. The obvious untried
fix is class weighting, which is disabled in both runs and which Task 2 shows to
be effective on a small, imbalanced label set — a much closer analogue to
`usage` (8 classes) than Task 1's 124-class tail. It was not run, and the report
does not claim a result it did not measure.

---

## 6. Task 4 — Visual search

Sources: `outputs/task_4/results_task4.csv`, `pk_curve.csv`, `cluster_purity.csv`

A retrieval problem, not a classification one. Instead of assigning a fixed
label, the model learns an embedding space where visually similar garments are
close together, so a query image can be answered with its nearest neighbours.
This answers "show me items like this one" — a question no fixed label set can.

**Architecture.** Three Conv64 + pooling blocks into a 64-dimensional dense
layer, followed by L2 normalisation. The L2 step matters: without it the network
can minimise triplet loss by inflating embedding magnitudes rather than learning
directions, a shortcut that improves the loss without improving retrieval.
Constraining all embeddings to the unit sphere removes it. Triplet loss, margin
0.5, Adam 1e-4.

**Evaluation.** `articleType` serves as the relevance proxy — a retrieval counts
as correct if it shares the query's article type. Precision@K and mAP@10 are
reported; mAP is the selection metric because it rewards placing correct items
higher, which is what a user of a search interface actually experiences. K=5 was
chosen from the precision curve (`pk_curve.csv`): precision falls smoothly from
0.668 at K=1 to 0.579 at K=20, retaining 0.620 at K=5 — enough results to be
useful, few enough to display.

### The experiment: random vs semi-hard negative mining

Random negative sampling degrades as training proceeds: once the easy negatives
are separated, most sampled triplets already satisfy the margin and contribute
no gradient. By epoch 4 only 18% of the baseline's triplets were still active.
Semi-hard mining re-selects negatives each epoch from a 50-candidate pool, after
a one-epoch random warm-up, choosing negatives that are harder than the positive
but still inside the margin.

Tuning ran on the 30 largest article types so both samplers faced matched
conditions; the selected model was then re-indexed over the full 118-type
catalogue.

| Metric (30-type subsample, 500 queries) | Random | Semi-hard | Change |
|---|---:|---:|---:|
| P@1 | 0.718 | 0.826 | +10.8 pp |
| P@5 | 0.702 | 0.807 | +10.5 pp |
| P@10 | 0.693 | 0.806 | +11.3 pp |
| **mAP@10** | **0.611** | **0.762** | **+24.6% rel** |
| Validation loss | 0.089 | 0.056 | −37% |
| Active fraction | 0.181 | 0.115 | −6.6 pp |
| Mean positive distance | 0.470 | 0.552 | — |
| Mean negative distance | 1.817 | 1.991 | +0.174 |
| Training time (s) | 2,892 | 6,274 | 2.2× |

Precision stays near-flat from K=1 to K=10 for the semi-hard model (0.826 →
0.806) while the baseline decays (0.718 → 0.693), so the gain is in the whole
ranking rather than the top hit alone.

Within the semi-hard run, training loss rose from 0.114 in the warm-up epoch to
0.207 once mining began — direct confirmation that the mined triplets were
harder — while validation loss still finished lower (0.056 vs 0.089).

**An honest caveat.** The semi-hard model's active fraction ends *lower* than the
baseline's (0.115 vs 0.181), not higher. Mining did not keep more triplets
active; it made the active ones more informative, and the model converged to a
better-separated space. The negative distance confirms this: 1.991 against 1.817.
The tidier story — "mining keeps more triplets alive" — is not what the
diagnostics say.

The 2.2× training cost is from epoch *count* (10 vs 4), not mining overhead:
per-epoch cost was actually lower for semi-hard (627 s vs 723 s).

### Final evaluation — full 118-type catalogue

Same 500 validation queries, full catalogue:

| Model | Precision@5 |
|---|---:|
| Random negatives | 0.602 |
| **Semi-hard negatives** | **0.696** |

Absolute gain +9.4 pp, relative +15.5%. Paired t-test **t = 6.70, p = 5.75 ×
10⁻¹¹** over n = 500; Cohen's d = 0.30. Per-query, 196 improved, 88 worsened,
216 unchanged — the gain is broad, not driven by a handful of queries.

### Independent structural evidence

k-means (k=30) over the learned embeddings, with `articleType` never shown to the
clustering, gives mean cluster purity **0.490** (range 0.163–0.944) against
1/118 = 0.008 for random assignment — roughly 60× chance. The tightest clusters
are visually distinctive categories (Watches, 0.944); the loosest are broad ones
(Tops, 0.246), which is the expected pattern. Because the labels play no part in
forming the clusters, this is evidence that the embedding space encodes article
type structure on its own, independent of the retrieval metrics.

---

## 7. Ultimate judgement

| Task | Selected model | Evidence |
|---|---|---|
| **1 — Article type** | Custom CNN, no imbalance treatment | Macro-F1 0.419 vs 0.003 baseline; both weighting (0.055) and oversampling (0.091) were measured and rejected |
| **2 — Season** | Random forest, n=800, unrestricted depth, balanced weights | Macro-F1 0.734 vs 0.166 baseline; balanced weights add 0.035 macro-F1 at no accuracy cost |
| **3 — Gender** | MLP, epoch 18 | Macro-F1 0.623, accuracy 0.818 vs 0.542 baseline |
| **3 — Usage** | *None deployable* | Macro-F1 0.325 across 8 classes; model reproduces the majority baseline for its first 3 epochs and never fully escapes it |
| **4 — Visual search** | Semi-hard triplet model | Full-catalogue P@5 0.696 vs 0.602, p < 10⁻¹⁰; mAP@10 +24.6% on matched subsample; inference cost unchanged |

Task 4's selection carries no deployment penalty: retrieval is one forward pass
plus a scan of a 7.9 MB index, identical for both models. The entire 2.2× cost
was paid once, at training time.

---

## 8. Limitations and future work

**Validation coverage.** 14 of 124 article types and 1 of 8 usage classes never
appear in validation, so their performance is unmeasured. Reported macro-F1 for
Task 1 is computed on an easier label space than the full problem.

**No held-out test set.** Models were selected on validation macro-F1 and are
reported on the same split. Task 3's gender result in particular sits on a noisy
epoch-to-epoch fluctuation, so the selected 0.623 likely overstates true
performance by a few points. A three-way split would fix this.

**Task 1's imbalance problem is unsolved.** Both standard remedies failed. The
untried approaches are a hierarchical classifier (predict `subCategory` first,
then article type within it) and focal loss, which down-weights easy examples
without handing singleton classes disproportionate gradient.

**Task 1's sweep is inconclusive.** The variants are not comparable to the
original run. Re-running them under matched epoch budgets is cheap and would
make the hyperparameter conclusion defensible.

**Task 3's usage model needs class weights.** Task 2 shows balanced weighting
recovering minority-class performance on a small imbalanced label set at
negligible accuracy cost, and `usage` is the same shape of problem. Training
takes 9 seconds.

**Task 4's relevance proxy.** Retrieval quality is judged by whether results
share the query's `articleType`. A returned item that is genuinely similar but
differently typed — a Kurta retrieved for a Tunic query — is scored as an error.
Reported precision is therefore a lower bound on perceived quality; human
relevance judgements would measure it properly.

---

## Appendix — Figures

**Task 1** (`outputs/task_1/`)
- `confusion_cnn_merged.png` — subCategory confusion matrix
- `confusion_cnn_weighted.png`, `confusion_cnn_oversampled.png` — the failed
  imbalance treatments; the weighted matrix shows the collapse directly

**Task 2** (`outputs/task_2/`)
- `confusion_season_base.png`, `confusion_season_tune.png` — before/after tuning
- `confusion_season_rf_tune_{n100,n300,n800,depth10,depth20,noweights}.png` —
  the one-variable-at-a-time study
- `season_feature_importance_base.png` — which HOG/colour features carry signal
- `season_near_duplicates_base.png` — near-duplicate check

**Task 4** (`outputs/task_4/`)
- `task4_query_grid.png`, `task4_query_grid_tuned.png` — Top-5 retrievals for the
  same unseen query under both models
- `figures/fig2_training_curves.png` — per-epoch validation loss and active
  fraction, both runs; supports the mining caveat in §6
- `figures/fig3_pk_curve.png` — Precision@K, K=1..20, K=5 marked; justifies K=5
- `figures/fig4_cluster_purity.png` — purity per cluster against chance
- `figures/fig5_precision_comparison.png` — subsample P@K and full-catalogue P@5
  with standard errors
- `task4_training_diagnostics.png` — training diagnostics
