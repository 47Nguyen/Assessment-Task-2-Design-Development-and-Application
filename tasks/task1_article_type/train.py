"""
Task 1: Fashion Item Type Classification (articleType)
======================================================
Owner: Member 1
Target: "articleType" (124 classes)

HOW TO RUN:
    From the project root folder:
    python -m tasks.task1_article_type.train

OPTIONS:
    python -m tasks.task1_article_type.train --help
    python -m tasks.task1_article_type.train --epochs 30 --batch-size 128
    python -m tasks.task1_article_type.train --tune               # Run hyperparameter grid
    python -m tasks.task1_article_type.train --skip-baselines     # Fast run without classical ML
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. Setup & Shared Imports (Frozen APIs)
# ---------------------------------------------------------------------------
from src.config import CACHE_DIR, MODEL_DIR, OUTPUT_DIR, SEED
from src.data import get_split
from src.evaluate import (
    class_weights,
    evaluate_model,
    per_class_report,
    plot_confusion,
)
from src.models import build_cnn

# Set deterministic random seeds
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

TARGET_VALUE = "articleType"


# ---------------------------------------------------------------------------
# 2. Classical ML Baselines (HOG Feature Extraction + Linear Classifiers)
# ---------------------------------------------------------------------------
def extract_hog_features(images: np.ndarray, split_name: str = "train") -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) features from image array.

    Uses caching to avoid re-extracting features on repeated runs.
    Image size: 80x60 -> Grayscale -> HOG feature vector.
    """
    cache_path = CACHE_DIR / f"hog_features_{split_name}.npy"
    if cache_path.exists():
        print(f"Loading cached HOG features from {cache_path.name}...")
        return np.load(cache_path)

    print(f"Extracting HOG features for {split_name} set ({len(images)} images)...")
    hog_list = []
    for i, img in enumerate(images):
        # Convert normalized float RGB (or uint8) to grayscale float in [0, 1]
        # Images loaded from get_split are normalized float32
        gray_img = (
            rgb2gray(img)
            if img.ndim == 3 and img.shape[2] == 3
            else img.squeeze()
        )
        feat = hog(
            gray_img,
            orientations=8,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
        )
        hog_list.append(feat)
        if (i + 1) % 10000 == 0 or (i + 1) == len(images):
            print(f"  Processed {i + 1}/{len(images)} images")

    hog_features = np.asarray(hog_list, dtype=np.float32)
    np.save(cache_path, hog_features)
    print(f"Saved HOG features to {cache_path.name} (shape: {hog_features.shape})")
    return hog_features


def run_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder,
):
    """Establish non-deep learning baseline benchmarks.

    1. Majority Class Baseline (predict most frequent class: Tshirts)
    2. Logistic Regression on HOG features
    3. Linear SVM on HOG features
    """
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING CLASSICAL ML BASELINES")
    print("=" * 60)

    # 1. Majority class baseline
    majority_class = int(np.bincount(y_train).argmax())
    majority_name = label_encoder.classes_[majority_class]
    y_pred_maj = np.full_like(y_val, fill_value=majority_class)
    print(f"\n[Baseline 1] Majority Class: '{majority_name}' (ID: {majority_class})")
    evaluate_model(
        y_val,
        y_pred_maj,
        TARGET_VALUE,
        "majority_baseline",
        notes=f"Always predicts most frequent class: '{majority_name}'",
    )

    # Extract HOG features for classical models
    X_train_hog = extract_hog_features(X_train, split_name="train")
    X_val_hog = extract_hog_features(X_val, split_name="val")

    # 2. Logistic Regression (via SGDClassifier for fast & scalable multi-class training)
    print("\n[Baseline 2] Training Logistic Regression (HOG + SGD log_loss)...")
    logreg_pipe = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=SEED,
            tol=1e-3,
            n_jobs=-1,
        ),
    )
    logreg_pipe.fit(X_train_hog, y_train)
    y_pred_lr = logreg_pipe.predict(X_val_hog)
    evaluate_model(
        y_val,
        y_pred_lr,
        TARGET_VALUE,
        "logreg_hog",
        notes="HOG features (8 orientations, 10x10 cell) + Logistic Regression (SGD)",
    )

    # 3. Linear SVM (via SGDClassifier hinge loss)
    print("\n[Baseline 3] Training Linear SVM (HOG + SGD hinge)...")
    svm_pipe = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=SEED,
            tol=1e-3,
            n_jobs=-1,
        ),
    )
    svm_pipe.fit(X_train_hog, y_train)
    y_pred_svm = svm_pipe.predict(X_val_hog)
    evaluate_model(
        y_val,
        y_pred_svm,
        TARGET_VALUE,
        "svm_hog",
        notes="HOG features (8 orientations, 10x10 cell) + Linear SVM (SGD)",
    )


# ---------------------------------------------------------------------------
# 3. Data Augmentation & tf.data Pipeline (Crucial for 60x80 images)
# ---------------------------------------------------------------------------
# Mild data augmentation specifically chosen for low-resolution 60x80 fashion images.
# Strong zoom/rotation would destroy essential garment silhouettes and collar details.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ],
    name="data_augmentation",
)


def create_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 128,
    augment: bool = True,
):
    """Build optimized tf.data.Dataset pipelines for training and validation.

    Augmentation is applied WITH training=True strictly during training.
    Validation data remains untouched for fair and consistent evaluation.
    """
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=min(len(X_train), 10000), seed=SEED)
    train_ds = train_ds.batch(batch_size)

    if augment:
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# 4. Dynamic Class Weights (Anti-Bias Capping Fix)
# ---------------------------------------------------------------------------
def compute_capped_class_weights(
    y_train: np.ndarray,
    max_weight_cap: float = 10.0,
    min_weight: float = 1.0,
) -> tuple[dict, dict]:
    """Compute balanced class weights and apply upper bound clipping ("hãm phanh").

    Raw inverse-frequency weights for rare classes can reach 249.11, causing
    unstable gradients and pushing the network to falsely predict rare items
    (e.g., 'Jewellery Set') on everyday clothing. Capping at 10.0 prevents this.
    """
    raw_weights = class_weights(y_train)
    raw_values = np.array(list(raw_weights.values()))

    # Apply clipping/smoothing
    capped_weights = {
        cls: float(np.clip(w, min_weight, max_weight_cap))
        for cls, w in raw_weights.items()
    }
    capped_values = np.array(list(capped_weights.values()))

    print("\n" + "=" * 60)
    print("DYNAMIC CLASS WEIGHTS SUMMARY (Anti-Bias 'Hãm Phanh' Fix)")
    print("=" * 60)
    print(
        f"Raw weights:    min={raw_values.min():.2f}, "
        f"median={np.median(raw_values):.2f}, max={raw_values.max():.2f}"
    )
    print(
        f"Capped weights: min={capped_values.min():.2f}, "
        f"median={np.median(capped_values):.2f}, max={capped_values.max():.2f} (cap: {max_weight_cap})"
    )

    return capped_weights, raw_weights


# ---------------------------------------------------------------------------
# 5. CNN Model Training & Evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder,
    epochs: int = 30,
    batch_size: int = 128,
    patience: int = 5,
    learning_rate: float = 1e-3,
    filters: tuple = (32, 64, 128),
    dropout: float = 0.3,
    use_class_weights: bool = True,
    max_weight_cap: float = 10.0,
    model_name: str = "cnn_baseline",
    save_best_model: bool = True,
):
    """Build, train, and evaluate the CNN model."""
    n_classes = len(label_encoder.classes_)
    print("\n" + "=" * 60)
    print(f"STEP 5: TRAINING CNN ({model_name})")
    print(
        f"Config: lr={learning_rate}, filters={filters}, dropout={dropout}, "
        f"epochs={epochs}, batch_size={batch_size}, weighted={use_class_weights}"
    )
    print("=" * 60)

    train_ds, val_ds = create_tf_datasets(
        X_train, y_train, X_val, y_val, batch_size=batch_size, augment=True
    )

    model = build_cnn(
        n_classes=n_classes,
        filters=filters,
        dropout=dropout,
        learning_rate=learning_rate,
        name=model_name,
    )
    model.summary()

    checkpoint_path = str(MODEL_DIR / f"cnn_{TARGET_VALUE}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if save_best_model:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

    weights_dict = None
    if use_class_weights:
        weights_dict, _ = compute_capped_class_weights(
            y_train, max_weight_cap=max_weight_cap
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=weights_dict,
        verbose=1,
    )

    # Validation Evaluation
    print("\nRunning inference on validation set...")
    val_probs = model.predict(val_ds)
    y_pred = np.argmax(val_probs, axis=1)

    notes = (
        f"CNN (lr={learning_rate}, filters={filters}, dropout={dropout}, "
        f"aug=mild, weights={'capped' if use_class_weights else 'none'})"
    )
    eval_row = evaluate_model(
        y_val, y_pred, TARGET_VALUE, model_name, notes=notes
    )

    # Plot and save confusion matrix
    plot_confusion(y_val, y_pred, label_encoder, TARGET_VALUE, max_classes=25)

    # Top 10 hardest classes (lowest recall)
    hardest_classes = per_class_report(
        y_val, y_pred, label_encoder, top_n=10
    )
    print("\n--- Top 10 Hardest Classes (Lowest Recall) ---")
    print(hardest_classes[["precision", "recall", "f1-score", "support"]])

    # Save fitted label encoder with joblib
    le_path = MODEL_DIR / f"label_encoder_{TARGET_VALUE}.joblib"
    joblib.dump(label_encoder, le_path)
    print(f"\nSaved LabelEncoder to {le_path.name}")

    return model, history, eval_row


# ---------------------------------------------------------------------------
# 6. Hyperparameter Tuning Grid (--tune)
# ---------------------------------------------------------------------------
def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder,
    epochs: int = 15,
    batch_size: int = 128,
    patience: int = 4,
):
    """Systematically tune learning rate, filter blocks, and dropout rates.

    All runs are logged to outputs/results.csv for direct report inclusion.
    """
    print("\n" + "=" * 60)
    print("STEP 6: HYPERPARAMETER TUNING GRID")
    print("=" * 60)

    learning_rates = [1e-2, 1e-3, 1e-4]
    filter_configs = [
        (32, 64),
        (32, 64, 128),
        (64, 128, 256),
    ]
    dropout_rates = [0.2, 0.3, 0.5]

    total_runs = (
        len(learning_rates) + len(filter_configs) - 1 + len(dropout_rates) - 1
    )
    print(
        f"Starting sequential tuning across LR, Filters, and Dropout ({total_runs} runs)..."
    )

    # Phase 1: Tune Learning Rate (baseline: filters=(32,64,128), dropout=0.3)
    best_lr = 1e-3
    best_f1 = -1.0
    for lr in learning_rates:
        model_name = f"cnn_tune_lr_{lr:.0e}"
        _, _, row = train_and_evaluate_cnn(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=lr,
            filters=(32, 64, 128),
            dropout=0.3,
            use_class_weights=True,
            model_name=model_name,
            save_best_model=False,
        )
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_lr = lr

    print(f"\n>>> Best Learning Rate selected: {best_lr} (macro-F1: {best_f1:.4f})")

    # Phase 2: Tune Filter Depth with best_lr
    best_filters = (32, 64, 128)
    for f in filter_configs:
        if f == (32, 64, 128):
            continue  # Already tested with best_lr
        filter_str = "_".join(map(str, f))
        model_name = f"cnn_tune_filters_{filter_str}"
        _, _, row = train_and_evaluate_cnn(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=best_lr,
            filters=f,
            dropout=0.3,
            use_class_weights=True,
            model_name=model_name,
            save_best_model=False,
        )
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_filters = f

    print(f"\n>>> Best Filter Config selected: {best_filters} (macro-F1: {best_f1:.4f})")

    # Phase 3: Tune Dropout Rate with best_lr and best_filters
    best_dropout = 0.3
    for d in dropout_rates:
        if d == 0.3:
            continue  # Already tested
        model_name = f"cnn_tune_dropout_{d}"
        _, _, row = train_and_evaluate_cnn(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=best_lr,
            filters=best_filters,
            dropout=d,
            use_class_weights=True,
            model_name=model_name,
            save_best_model=False,
        )
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_dropout = d

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETED")
    print(
        f"Optimal configuration found: lr={best_lr}, filters={best_filters}, "
        f"dropout={best_dropout} (Best macro-F1: {best_f1:.4f})"
    )
    print("=" * 60)

    # Train and save final optimal model
    print("\nTraining final model with optimal hyperparameters...")
    train_and_evaluate_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        label_encoder,
        epochs=epochs + 10,
        batch_size=batch_size,
        patience=patience + 2,
        learning_rate=best_lr,
        filters=best_filters,
        dropout=best_dropout,
        use_class_weights=True,
        model_name="cnn_articleType_best",
        save_best_model=True,
    )


# ---------------------------------------------------------------------------
# 7. Main Execution Flow
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Item Type (articleType) Classification Models for Task 1 (Member 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate before dense head (default: 0.3)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip classical ML baselines (majority, LogReg, SVM)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Train CNN without class weighting",
    )
    parser.add_argument(
        "--max-weight-cap",
        type=float,
        default=10.0,
        help="Upper bound cap for class weights to avoid gradient spikes (default: 10.0)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run full hyperparameter tuning grid across LR, filters, and dropout",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TASK 1: ARTICLE TYPE CLASSIFICATION — TRAINING PIPELINE")
    print(f"Target: {TARGET_VALUE}")
    print("=" * 60)

    # 1. Load Data using shared get_split
    X_train, X_val, y_train, y_val, label_encoder = get_split(TARGET_VALUE)

    print(f"\n[Data Summary]")
    print(f"  Train images: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  Val images:   {X_val.shape} (dtype: {X_val.dtype})")
    print(f"  Classes:      {len(label_encoder.classes_)}")
    print(f"  Sample classes: {list(label_encoder.classes_)[:8]}")

    # 2. Classical Baselines
    if not args.skip_baselines and not args.tune:
        run_classical_baselines(X_train, y_train, X_val, y_val, label_encoder)

    # 3. Hyperparameter Tuning Mode
    if args.tune:
        run_hyperparameter_tuning(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder,
            epochs=min(args.epochs, 20),
            batch_size=args.batch_size,
            patience=args.patience,
        )
    else:
        # 4. Standard CNN Training
        train_and_evaluate_cnn(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.lr,
            dropout=args.dropout,
            use_class_weights=not args.no_class_weights,
            max_weight_cap=args.max_weight_cap,
            model_name="cnn_articleType",
            save_best_model=True,
        )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY! ")
    print(f"Results logged to: {OUTPUT_DIR / 'results.csv'}")
    print(f"Model saved to:    {MODEL_DIR / f'cnn_{TARGET_VALUE}.keras'}")
    print(f"Encoder saved to:  {MODEL_DIR / f'label_encoder_{TARGET_VALUE}.joblib'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
