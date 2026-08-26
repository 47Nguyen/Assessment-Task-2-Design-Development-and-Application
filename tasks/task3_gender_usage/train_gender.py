"""Task 3a - Gender classification.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import OUTPUT_DIR
from src.data import get_split
from src.models import build_cnn, default_callbacks
from src.evaluate import (
    evaluate_model, per_class_report, plot_confusion, class_weights,
)

target = "gender"
epochs = 30
batch_size = 128


# load
X_train, X_val, y_train, y_val, le = get_split(target)

print(f"\nTarget: {target}")
print(f"Train images: {X_train.shape}")
print(f"Val images:   {X_val.shape}")
print(f"Classes:      {list(le.classes_)}")


# 1. baseline: always predict the biggest class
# Taken from train, not val - at prediction time we can't see the val labels.
most_common = np.bincount(y_train).argmax()
print(f"\nMost common class in train: {le.classes_[most_common]}")

y_pred_baseline = np.full(len(y_val), most_common)

evaluate_model(y_val, y_pred_baseline, target, "baseline_majority",
               notes="always predicts the biggest class")


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
    callbacks=default_callbacks(target),   # saves models/cnn_gender.keras
)

y_pred_cnn = model.predict(X_val).argmax(axis=1)   # probabilities -> class id

evaluate_model(y_val, y_pred_cnn, target, "cnn_baseline",
               notes=f"build_cnn defaults, lr 1e-3, batch {batch_size}")


# 3. class-weighted CNN
# Makes rare classes cost more. Accuracy should drop, macro-F1 should rise.
print("\n" + "=" * 70)
print("Training the class-weighted CNN")
print("=" * 70)

weights = class_weights(y_train)
print("\nClass weights:")
for class_id, weight in weights.items():
    print(f"  {le.classes_[class_id]:<8} {weight:.2f}")

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


# 4. simplified 3-class: Men / Women / Other
# Unisex is intent, and Boys/Girls differ by size, which a 60x80 crop can't show.
print("\n" + "=" * 70)
print("Training the simplified 3-class model (Men / Women / Other)")
print("=" * 70)

simple_map = {
    "Men": "Men",
    "Women": "Women",
    "Unisex": "Other",
    "Boys": "Other",
    "Girls": "Other",
}


def to_simple(y_numbers):
    """5-class numbers -> 3-class words."""
    words = le.inverse_transform(y_numbers)
    return np.array([simple_map[w] for w in words])


le_simple = LabelEncoder().fit(["Men", "Other", "Women"])

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
# 3-class and 5-class macro-F1 aren't comparable (different denominators), so
# fold the 5-class model's answers down to 3 and score both in the same space.
print("\n" + "=" * 70)
print("Fair comparison, both scored as 3 classes")
print("=" * 70)

y_pred_cnn_folded = le_simple.transform(to_simple(y_pred_cnn))

evaluate_model(y_val_simple, y_pred_cnn_folded, "gender_3class", "cnn_5class_folded",
               notes="trained on 5 classes, answers merged down to 3 afterwards")

evaluate_model(y_val_simple, y_pred_simple, "gender_3class", "cnn_trained_as_3",
               notes="trained directly on Men/Women/Other")


# 6. per-class tables
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
print("\n--- cnn_trained_as_3 ---")
print(table_simple.round(3))
table_simple.to_csv(OUTPUT_DIR / f"per_class_{target}_simple.csv")


# 7. confusion matrices
# Watch whether Boys falls into Men and Girls into Women.
plot_confusion(y_val, y_pred_cnn, le, target)
plot_confusion(y_val, y_pred_weighted, le, f"{target}_weighted")
plot_confusion(y_val_simple, y_pred_simple, le_simple, f"{target}_3class")

print("\nDone. Results are in outputs/results.csv, figures in outputs/.")
