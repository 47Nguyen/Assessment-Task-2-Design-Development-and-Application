"""Task 3b - Usage / occasion classification.
Casual is 77% of the data, so always guessing Casual gives 0.769 accuracy while
the macro-F1 ceiling is only about 0.50. Always report both.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import OUTPUT_DIR
from src.data import get_split
from src.models import build_cnn, default_callbacks
from src.evaluate import (
    evaluate_model, per_class_report, plot_confusion, class_weights,
)

target = "usage"
epochs = 30
batch_size = 128


# load
# get_split() drops the 72 rows with no usage label.
X_train, X_val, y_train, y_val, le = get_split(target)

print(f"\nTarget: {target}")
print(f"Train images: {X_train.shape}")
print(f"Val images:   {X_val.shape}")
print(f"Classes:      {list(le.classes_)}")

# Home has 1 image in total, so it can't be scored - worth showing.
print("\nHow many of each class are in each split:")
for class_id, class_name in enumerate(le.classes_):
    n_train = int((y_train == class_id).sum())
    n_val = int((y_val == class_id).sum())
    flag = "   <- not in val, cannot be scored" if n_val == 0 else ""
    print(f"  {class_name:<14} train {n_train:>6}   val {n_val:>5}{flag}")


# 1. baseline: always predict the biggest class
most_common = np.bincount(y_train).argmax()
print(f"\nMost common class in train: {le.classes_[most_common]}")

y_pred_baseline = np.full(len(y_val), most_common)

evaluate_model(y_val, y_pred_baseline, target, "baseline_majority",
               notes="always predicts Casual - high accuracy, useless model")


# 2. plain CNN
print("\n" + "=" * 70)
print("Training the plain CNN")
print("=" * 70)

model = build_cnn(n_classes=len(le.classes_))
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=default_callbacks(target),   # saves models/cnn_usage.keras
)

y_pred_cnn = model.predict(X_val).argmax(axis=1)

evaluate_model(y_val, y_pred_cnn, target, "cnn_baseline",
               notes=f"build_cnn defaults, lr 1e-3, batch {batch_size}")


# 3. class-weighted CNN
print("\n" + "=" * 70)
print("Training the class-weighted CNN")
print("=" * 70)

weights = class_weights(y_train)
print("\nClass weights:")
for class_id, weight in weights.items():
    print(f"  {le.classes_[class_id]:<14} {weight:.2f}")

model_weighted = build_cnn(n_classes=len(le.classes_))

model_weighted.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=weights,
    callbacks=default_callbacks(f"{target}_weighted"),
)

y_pred_weighted = model_weighted.predict(X_val).argmax(axis=1)

evaluate_model(y_val, y_pred_weighted, target, "cnn_weighted",
               notes="same CNN, balanced class weights")


# 4. simplified 5-class
# Smart Casual, Travel, Party and Home are under 0.3% of the data between them.
print("\n" + "=" * 70)
print("Training the simplified 5-class model")
print("=" * 70)

mapping = {
    "Casual": "Casual",
    "Sports": "Sports",
    "Ethnic": "Ethnic",
    "Formal": "Formal",
    "Smart Casual": "Other",
    "Travel": "Other",
    "Party": "Other",
    "Home": "Other",
}


def to_simple(y_numbers):
    """8-class numbers -> 5-class words."""
    words = le.inverse_transform(y_numbers)
    return np.array([mapping[w] for w in words])


le_simple = LabelEncoder().fit(["Casual", "Ethnic", "Formal", "Other", "Sports"])

y_train_simple = le_simple.transform(to_simple(y_train))
y_val_simple = le_simple.transform(to_simple(y_val))

print(f"Simplified classes: {list(le_simple.classes_)}")

model_simple = build_cnn(n_classes=len(le_simple.classes_))

model_simple.fit(
    X_train, y_train_simple,
    validation_data=(X_val, y_val_simple),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=default_callbacks(f"{target}_simple"),
)

y_pred_simple = model_simple.predict(X_val).argmax(axis=1)


# 5. fair comparison
# 8-class and 5-class macro-F1 aren't comparable, so fold the 8-class model's
# answers down to 5 and score both in the same space.
print("\n" + "=" * 70)
print("Fair comparison, both scored as 5 classes")
print("=" * 70)

y_pred_cnn_folded = le_simple.transform(to_simple(y_pred_cnn))

evaluate_model(y_val_simple, y_pred_cnn_folded, "usage_5class", "cnn_8class_folded",
               notes="trained on 8 classes, answers merged down to 5 afterwards")

evaluate_model(y_val_simple, y_pred_simple, "usage_5class", "cnn_trained_as_5",
               notes="trained directly on the 5 merged classes")


# 6. per-class tables
# The tiny classes show up here as a row of zeros - that's the case for merging.
print("\n" + "=" * 70)
print("Per-class results (worst recall first)")
print("=" * 70)


def real_classes_only(table):
    # drop summary rows so they don't look like real classes
    return table.drop(index=[i for i in ("micro avg", "samples avg")
                             if i in table.index])


for name, preds in [("cnn_baseline", y_pred_cnn),
                    ("cnn_weighted", y_pred_weighted)]:
    table = real_classes_only(per_class_report(y_val, preds, le))
    print(f"\n--- {name} ---")
    print(table.round(3))
    table.to_csv(OUTPUT_DIR / f"per_class_{target}_{name}.csv")

table_simple = real_classes_only(
    per_class_report(y_val_simple, y_pred_simple, le_simple))
print("\n--- cnn_trained_as_5 ---")
print(table_simple.round(3))
table_simple.to_csv(OUTPUT_DIR / f"per_class_{target}_simple.csv")


# 7. confusion matrices
# Expect a strong vertical stripe on Casual swallowing the other classes.
plot_confusion(y_val, y_pred_cnn, le, target)
plot_confusion(y_val, y_pred_weighted, le, f"{target}_weighted")
plot_confusion(y_val_simple, y_pred_simple, le_simple, f"{target}_5class")

print("\nDone. Results are in outputs/results.csv, figures in outputs/.")
