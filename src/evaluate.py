"""Shared evaluation - one set of metrics for the whole group.

Everyone calls evaluate_model(), which saves a row to outputs/results.csv.
That file becomes the comparison table in the report, and it builds itself as
people work instead of being pieced together at the end.

Why macro-F1 and not accuracy:
    Always predicting "Casual" gets 0.769 accuracy on `usage` while learning
    nothing at all. Macro-F1 treats every class equally, so it catches that.
    Every score here is shown next to the majority-class baseline, which is the
    floor a model has to beat to have done anything.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from src.config import OUTPUT_DIR

RESULTS_CSV = OUTPUT_DIR / "results.csv"


def majority_baseline(y):
    """Score you'd get by always predicting the most common class."""
    return float(pd.Series(y).value_counts(normalize=True).iloc[0])


def evaluate_model(y_true, y_pred, target, model_name, notes=""):
    """Score a model and add it to outputs/results.csv.

    Args:
        target:     "articleType" | "season" | "gender" | "usage"
        model_name: short and specific - "cnn_baseline", "cnn_weighted",
                    "svm_hog". This is what identifies the row in the report.
        notes:      settings you'd need to reproduce it
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    baseline = majority_baseline(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    row = {
        "target": target,
        "model": model_name,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy,
        "majority_baseline": baseline,
        "beats_baseline": accuracy > baseline,
        "notes": notes,
    }

    print(f"\n{model_name}  ({target})")
    print(f"  macro-F1      {macro_f1:.4f}   <- main metric")
    print(f"  balanced acc  {row['balanced_acc']:.4f}")
    print(f"  accuracy      {accuracy:.4f}  (baseline {baseline:.4f})")

    if not row["beats_baseline"]:
        print("  WARNING: does not beat the majority-class baseline")
    if accuracy - macro_f1 > 0.25:
        print(f"  NOTE: accuracy is {accuracy - macro_f1:.2f} higher than macro-F1 - "
              f"the model is probably just predicting the common classes")

    _save_result(row)
    return row


def _save_result(row):
    """Add a row to outputs/results.csv, replacing any earlier run with the
    same target and model name so re-running doesn't create duplicates."""
    df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        old = old[~((old["target"] == row["target"]) & (old["model"] == row["model"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)


def results_table(target=None):
    """Show all results so far, best macro-F1 first."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV)
    if target:
        df = df[df["target"] == target]
    return df.sort_values(["target", "macro_f1"], ascending=[True, False])


def per_class_report(y_true, y_pred, label_encoder, top_n=None):
    """Precision / recall / F1 for each class, worst recall first.

    This is where the long tail shows up: rare classes sit at zero recall no
    matter how good the overall score looks.
    """
    report = classification_report(
        y_true, y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    df = df.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg")
                        if i in df.index])
    df = df.sort_values("recall")
    return df.head(top_n) if top_n else df


def plot_confusion(y_true, y_pred, label_encoder, target,
                   max_classes=25, save=True):
    """Confusion matrix, normalised by row.

    For targets with many classes it shows only the most common ones - a
    124x124 grid is unreadable in a report.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.arange(len(label_encoder.classes_))
    title = target

    if len(classes) > max_classes:
        keep = pd.Series(y_true).value_counts().head(max_classes).index.values
        mask = np.isin(y_true, keep) & np.isin(y_pred, keep)
        y_true, y_pred = y_true[mask], y_pred[mask]
        classes = np.sort(keep)
        title = f"{target} - top {max_classes} classes"

    names = [label_encoder.classes_[i] for i in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    size = max(6, len(classes) * 0.42)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, xticklabels=names, yticklabels=names, cmap="Blues",
                vmin=0, vmax=1, annot=len(classes) <= 10, fmt=".2f",
                square=True, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / f"confusion_{target}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
    return fig


def class_weights(y):
    """Weights that make rare classes count more during training.

    Not applied by default - whether it helps is an experiment you run and
    compare against a run without it.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y))
    return dict(zip(classes.tolist(), weights.tolist()))
