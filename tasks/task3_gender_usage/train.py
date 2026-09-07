"""
Task 3: Train image-only MLP models for gender and usage.

HOW TO RUN (from the repository root):
    python -m tasks.task3_gender_usage.train --run-dir outputs/task3/final_run
    python -m tasks.task3_gender_usage.train --smoke

FLOW: audit data -> split -> train candidates -> select on validation ->
      evaluate the selected model on holdout -> save models and evidence.

No pretrained weights, notebook, or shared src/ module is needed.
"""

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Set these before importing TensorFlow. Limit routine informational messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")  # Save figures without opening windows during training.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import (
    TARGETS,
    USAGE_MERGE,
    audit_data,
    find_root,
    load_pixels,
    make_split,
    prepare_image,
    target_arrays,
)


# ---------------------------------------------------------------------------
# 2. Choose the experiment settings
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Settings shared by every candidate, so the comparison is consistent."""

    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    height: int = 32
    width: int = 24
    patience: int = 5
    threads: int = 4
    smoke: bool = False
    merged_experiment: bool = True


# Both targets try these same three candidates, each trained from scratch.
# Regularization and weighting are experiments, not guaranteed improvements.
VARIANTS = {
    "mlp_default": {
        "hidden": (256,),
        "activation": "sigmoid",
        "dropout": 0.0,
        "weighted": False,
    },
    "mlp_regularized": {
        "hidden": (256, 128),
        "activation": "relu",
        "dropout": 0.3,
        "weighted": False,
    },
    "mlp_weighted": {
        "hidden": (256, 128),
        "activation": "relu",
        "dropout": 0.3,
        "weighted": True,
    },
}


def configure(config):
    """Check settings and request repeatable runs on the same environment."""
    positive_settings = (
        config.epochs, config.batch_size, config.height,
        config.width, config.patience, config.threads,
    )
    if min(positive_settings) < 1:
        raise ValueError("Epochs, batch size, dimensions, patience and threads must be positive.")

    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # A notebook may already have started TensorFlow: thread counts cannot
        # then be changed. This does not disable the seed/determinism settings.
        pass
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# 3. Build the MLP and calculate training-only class weights
# ---------------------------------------------------------------------------
def build_mlp(classes, variant, config):
    """Convert RGB pixels into one softmax score per catalogue class."""
    specification = VARIANTS[variant]
    network = [
        tf.keras.layers.Input(shape=(config.height, config.width, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),  # uint8 [0,255] -> float [0,1]
        tf.keras.layers.Flatten(),             # 32 x 24 x 3 -> 2,304 inputs
    ]
    for units in specification["hidden"]:
        network.append(
            tf.keras.layers.Dense(units, activation=specification["activation"])
        )
        if specification["dropout"]:
            network.append(tf.keras.layers.Dropout(specification["dropout"]))
    network.append(tf.keras.layers.Dense(len(classes), activation="softmax"))

    model = tf.keras.Sequential(network)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # Labels are integer class IDs.
        metrics=["accuracy"],
    )
    return model


def training_class_weights(labels, n_classes):
    """Give rare TRAINING classes more influence without full inverse weighting."""
    counts = np.bincount(labels, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("A training class is absent.")

    # Square root softens the inverse-frequency weighting. Cap BEFORE making
    # the sample-weighted mean equal to 1; final weights can therefore exceed 5.
    weights = np.minimum(np.sqrt(len(labels) / (n_classes * counts)), 5.0)
    weights /= np.average(weights, weights=counts)
    return {index: float(weight) for index, weight in enumerate(weights)}


# ---------------------------------------------------------------------------
# 4. Evaluate predictions and save figures for the report
# ---------------------------------------------------------------------------
def score_predictions(truth, predicted, classes):
    """Use the same class vocabulary for all candidates, including rare classes."""
    vocabulary = np.arange(len(classes))
    supported = np.unique(truth)
    report = classification_report(
        truth, predicted, labels=vocabulary, target_names=classes,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(
            truth, predicted, labels=vocabulary, average="macro", zero_division=0,
        )),
        "macro_f1_supported": float(f1_score(
            truth, predicted, labels=supported, average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            truth, predicted, labels=vocabulary, average="weighted", zero_division=0,
        )),
        "n_evaluated": len(truth),
        "n_classes": len(classes),
        "n_supported_classes": len(supported),
    }
    return metrics, report


def draw_history(history, path):
    """Show training/validation loss, accuracy and validation macro-F1."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Cross-entropy")
    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(history["val_macro_f1"], label="Validation")
    axes[2].set_title("Fixed-vocabulary macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch (zero-based)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_error_examples(truth, predicted, classes, metadata, path):
    """Display the first 12 mistakes, not a hand-picked set of successes."""
    errors = np.flatnonzero(truth != predicted)[:12]
    if len(errors) == 0:
        return

    figure, axes = plt.subplots(3, 4, figsize=(10, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(errors):
            position = errors[index]
            row = metadata.iloc[position]
            # This larger image is for display only, not the model input.
            axis.imshow(prepare_image(row.image_path, (80, 60)))
            axis.set_title(
                f"{row.id}\n{classes[truth[position]]} -> {classes[predicted[position]]}",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_evaluation(truth, predicted, classes, metadata, output_dir, prefix):
    """Save metrics, per-class report, confusion matrix and individual errors."""
    metrics, report = score_predictions(truth, predicted, classes)
    pd.DataFrame(report).T.to_csv(output_dir / f"{prefix}_report.csv")

    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        output_dir / f"{prefix}_confusion.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted", ylabel="Actual", title=prefix,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(classes)):
        for column in range(len(classes)):
            text_color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center", color=text_color, fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_confusion.png", dpi=140)
    plt.close(figure)

    prediction_rows = metadata[["id", "duplicate_group"]].copy()
    prediction_rows["actual"] = np.asarray(classes)[truth]
    prediction_rows["predicted"] = np.asarray(classes)[predicted]
    prediction_rows.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    save_error_examples(
        truth, predicted, classes, metadata, output_dir / f"{prefix}_errors.png"
    )
    return metrics


# ---------------------------------------------------------------------------
# 5. Train one candidate and keep its best validation checkpoint
# ---------------------------------------------------------------------------
class ValidationMacroF1(tf.keras.callbacks.Callback):
    """Calculate macro-F1 at each epoch BEFORE checkpoint/early-stop callbacks."""

    def __init__(self, pixels, labels, classes, batch_size):
        super().__init__()
        self.pixels = pixels
        self.labels = labels
        self.classes = classes
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            self.pixels, batch_size=self.batch_size, verbose=0
        )
        predicted = probabilities.argmax(axis=1)
        value = f1_score(
            self.labels, predicted, labels=np.arange(len(self.classes)),
            average="macro", zero_division=0,
        )
        if logs is not None:
            logs["val_macro_f1"] = float(value)


def fit_variant(pixels, partitions, classes, variant, config, directory, target):
    """Train from scratch; return the saved best model, val scores and settings."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    model = build_mlp(classes, variant, config)
    train_indices, train_labels = partitions["train"]
    val_indices, val_labels = partitions["validation"]

    weights = None
    if VARIANTS[variant]["weighted"]:
        weights = training_class_weights(train_labels, len(classes))

    model_path = directory / f"{target}_{variant}.keras"
    macro_f1 = ValidationMacroF1(
        pixels[val_indices], val_labels, classes, config.batch_size
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_macro_f1", mode="max", save_best_only=True
    )
    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=config.patience,
        restore_best_weights=True, min_delta=1e-4,
    )

    started = time.perf_counter()
    print(f"Training {target}: {variant}", flush=True)
    history = model.fit(
        pixels[train_indices], train_labels,
        validation_data=(pixels[val_indices], val_labels),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=[macro_f1, checkpoint, stopping],
        class_weight=weights,
        shuffle=True,
        verbose=2,
    ).history
    history = {
        key: [float(value) for value in values] for key, values in history.items()
    }

    # Use the checkpoint, not the last epoch. EarlyStopping uses min_delta,
    # whereas ModelCheckpoint retains any strictly higher validation macro-F1.
    model = tf.keras.models.load_model(str(model_path), compile=False)
    probabilities = model.predict(
        pixels[val_indices], batch_size=config.batch_size, verbose=0
    )

    # A load-consistency check: two loads of this checkpoint must agree.
    # This is NOT a before-save vs after-save accuracy evaluation.
    restored = tf.keras.models.load_model(str(model_path), compile=False)
    sample = pixels[val_indices[:4]]
    np.testing.assert_allclose(
        model(sample, training=False).numpy(),
        restored(sample, training=False).numpy(),
        rtol=1e-5, atol=1e-6,
    )

    details = {
        "target": target,
        "variant": variant,
        "classes": classes,
        "config": asdict(config),
        "class_weights": weights,
        "history": history,
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmax(history["val_macro_f1"]) + 1),
        "parameters": model.count_params(),
        # Includes fit plus checkpoint loading and validation checks.
        "training_seconds": time.perf_counter() - started,
    }
    (directory / f"{target}_{variant}.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    return model, probabilities, details


# ---------------------------------------------------------------------------
# 6. Run the optional five-class usage experiment (not submission labels)
# ---------------------------------------------------------------------------
def fold_usage(probabilities, classes, merged_classes):
    """Sum original scores into merged classes BEFORE choosing the largest."""
    folded = np.zeros((len(probabilities), len(merged_classes)), dtype=np.float32)
    for index, name in enumerate(classes):
        merged_name = USAGE_MERGE.get(name, "Other")
        merged_index = merged_classes.index(merged_name)
        folded[:, merged_index] += probabilities[:, index]
    return folded


def run_usage_experiment(model, pixels, metadata, classes, winner, config, run_dir):
    """Compare folded 8-class scores vs a retrained 5-class model fairly."""
    merged_classes = sorted({USAGE_MERGE.get(name, "Other") for name in classes})
    partitions = target_arrays(metadata, "usage_5class", merged_classes)
    merged_model, _, details = fit_variant(
        pixels, partitions, merged_classes, winner,
        config, run_dir / "models", "usage_5class",
    )
    draw_history(details["history"], run_dir / "usage_5class_learning.png")

    results = []
    for split in ("validation", "holdout"):
        positions, labels = partitions[split]
        original_scores = model.predict(pixels[positions], verbose=0)
        folded_scores = fold_usage(original_scores, classes, merged_classes)
        retrained_scores = merged_model.predict(pixels[positions], verbose=0)
        for name, scores in (
            ("folded_8class", folded_scores),
            ("retrained_5class", retrained_scores),
        ):
            metrics = save_evaluation(
                labels, scores.argmax(axis=1), merged_classes,
                metadata.iloc[positions], run_dir, f"usage_5class_{name}_{split}",
            )
            results.append({
                "target": "usage_5class", "model": name, "split": split, **metrics
            })
    # The official usage_final model is unchanged and still has 8 classes.
    return results


# ---------------------------------------------------------------------------
# 7. Compare candidates and evaluate the selected model for one target
# ---------------------------------------------------------------------------
def train_target(target, pixels, metadata, config, run_dir, results):
    """Select on validation only; use holdout to assess the frozen choice."""
    models_dir = run_dir / "models"
    training_rows = metadata["split"].eq("train")
    classes = sorted(metadata.loc[training_rows, target].dropna().unique().tolist())
    partitions = target_arrays(metadata, target, classes)
    val_indices, val_labels = partitions["validation"]
    holdout_indices, holdout_labels = partitions["holdout"]

    coverage = []
    for split, (_, labels) in partitions.items():
        for index, name in enumerate(classes):
            coverage.append({
                "target": target, "split": split, "class": name,
                "count": int((labels == index).sum()),
            })
    print(f"\nTarget: {target} | Classes: {classes}", flush=True)
    print("Partition sizes:", {name: len(value[0]) for name, value in partitions.items()})

    # Baseline: always predict the most frequent TRAINING label.
    train_labels = partitions["train"][1]
    majority = int(np.bincount(train_labels, minlength=len(classes)).argmax())
    for split in ("validation", "holdout"):
        positions, labels = partitions[split]
        metrics = save_evaluation(
            labels, np.full(len(labels), majority), classes,
            metadata.iloc[positions], run_dir, f"{target}_majority_{split}",
        )
        results.append({"target": target, "model": "majority", "split": split, **metrics})

    candidates = []
    for variant in VARIANTS:
        _, probabilities, details = fit_variant(
            pixels, partitions, classes, variant, config, models_dir, target
        )
        metrics = save_evaluation(
            val_labels, probabilities.argmax(axis=1), classes,
            metadata.iloc[val_indices], run_dir, f"{target}_{variant}_validation",
        )
        draw_history(details["history"], run_dir / f"{target}_{variant}_learning.png")
        row = {
            "target": target, "model": variant, "split": "validation", **metrics,
            "parameters": details["parameters"],
            "epochs": details["epochs_run"],
            "training_seconds": details["training_seconds"],
        }
        results.append(row)
        candidates.append(row)
        # Save partial progress too, in case a later training step fails.
        pd.DataFrame(results).to_csv(run_dir / "results_task3.csv", index=False)

    # Deterministic tie-break: higher F1, then fewer parameters, then name.
    winner = sorted(
        candidates,
        key=lambda item: (-item["macro_f1"], item["parameters"], item["model"]),
    )[0]
    winner_name = winner["model"]
    shutil.copy2(
        models_dir / f"{target}_{winner_name}.keras",
        models_dir / f"{target}_final.keras",
    )
    details = json.loads(
        (models_dir / f"{target}_{winner_name}.json").read_text(encoding="utf-8")
    )
    details["selection_rule"] = (
        "Highest validation macro-F1; ties: fewer parameters, then stable name. Holdout not used."
    )
    (models_dir / f"{target}_final.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    # Do not retrain on holdout or choose another model after seeing this score.
    model = tf.keras.models.load_model(str(models_dir / f"{target}_final.keras"), compile=False)
    started = time.perf_counter()
    probabilities = model.predict(pixels[holdout_indices], batch_size=config.batch_size, verbose=0)
    inference_seconds = time.perf_counter() - started
    metrics = save_evaluation(
        holdout_labels, probabilities.argmax(axis=1), classes,
        metadata.iloc[holdout_indices], run_dir, f"{target}_final_holdout",
    )
    results.append({
        "target": target, "model": winner_name, "split": "holdout", **metrics,
        "batch_inference_ms_per_image": inference_seconds * 1000 / len(holdout_indices),
    })
    selected = {
        "variant": winner_name,
        "validation_macro_f1": winner["macro_f1"],
        "classes": classes,
        "holdout": metrics,
    }

    if target == "usage" and config.merged_experiment:
        results.extend(run_usage_experiment(
            model, pixels, metadata, classes, winner_name, config, run_dir
        ))
    return selected, coverage


# ---------------------------------------------------------------------------
# 8. Run the complete training pipeline
# ---------------------------------------------------------------------------
def run_experiment(repo_root=None, run_dir=None, config=None):
    """Run both targets in a NEW folder; never overwrite an existing run."""
    config = config or Config()
    configure(config)
    root = find_root(repo_root)
    if run_dir is None:
        run_dir = root / "outputs/task3" / time.strftime("run_%Y%m%d_%H%M%S")
    else:
        run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run already exists; choose a new directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print("TASK 3: GENDER AND USAGE CLASSIFICATION - TRAINING PIPELINE", flush=True)
    print("\nSTEP 1: AUDIT DATA AND CREATE THE SHARED TASK 3 SPLIT", flush=True)
    metadata, _ = audit_data(root, run_dir)
    metadata = make_split(metadata, config.seed)
    metadata.to_csv(run_dir / "split_manifest.csv", index=False)

    if config.smoke:
        print("SMOKE TEST ONLY: small subset; do not report these scores as final results.")
        subsets = []
        groups = metadata.groupby(["split", "gender", "usage"], dropna=False, sort=True)
        for _, group in groups:
            subsets.append(group.sample(n=min(50, len(group)), random_state=config.seed))
        metadata = pd.concat(subsets).sort_values("id").reset_index(drop=True)
        metadata.to_csv(run_dir / "smoke_manifest.csv", index=False)

    print("\nSTEP 2: LOAD RGB PIXELS (NORMALIZATION IS INSIDE THE MODEL)", flush=True)
    pixels = load_pixels(metadata, (config.height, config.width))
    print(f"Images: {pixels.shape} | dtype: {pixels.dtype}", flush=True)

    print("\nSTEP 3: TRAIN, SELECT AND EVALUATE EACH TARGET", flush=True)
    results = []
    selected = {}
    coverage = []
    for target in TARGETS:
        target_choice, target_coverage = train_target(
            target, pixels, metadata, config, run_dir, results
        )
        selected[target] = target_choice
        coverage.extend(target_coverage)

    print("\nSTEP 4: SAVE RESULTS AND REPRODUCIBILITY INFORMATION", flush=True)
    pd.DataFrame(coverage).to_csv(run_dir / "class_coverage.csv", index=False)
    pd.DataFrame(results).to_csv(run_dir / "results_task3.csv", index=False)
    (run_dir / "selected_models.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    source_hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    train_csv = root / "A2_FashionDataset/FashionDataset/train/styles_train.csv"
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "metadata_sha256": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "smoke_only": config.smoke,
        "external_evaluation_performed": False,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(pd.DataFrame(results)[["target", "model", "split", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"\nSaved Task 3 run to {run_dir}", flush=True)
    return run_dir


# ---------------------------------------------------------------------------
# 9. Read terminal options and start the pipeline
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Task 3 gender and usage MLP models.")
    parser.add_argument("--repo-root", type=Path, help="Repository containing A2_FashionDataset.")
    parser.add_argument("--run-dir", type=Path, help="New output directory; default: timestamped folder.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per candidate (default: 20).")
    parser.add_argument("--batch-size", type=int, default=128, help="Images per batch (default: 128).")
    parser.add_argument("--seed", type=int, default=42, help="Split and training seed (default: 42).")
    parser.add_argument("--smoke", action="store_true", help="Check the pipeline with a subset and 2 epochs.")
    parser.add_argument("--skip-merged", action="store_true", help="Skip the extra 5-class usage experiment.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        smoke=args.smoke,
        merged_experiment=not args.skip_merged,
    )
    run_experiment(args.repo_root, args.run_dir, config)


if __name__ == "__main__":
    main()
