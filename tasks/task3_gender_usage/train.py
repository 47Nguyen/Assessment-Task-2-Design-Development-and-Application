"""Train and evaluate Task 3 MLPs without shared src or external weights."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import TARGETS, USAGE_MERGE, audit_data, find_root, load_pixels, make_split, target_arrays


@dataclass
class Config:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    height: int = 32
    width: int = 24
    patience: int = 5
    threads: int = 4
    smoke: bool = False
    merged_experiment: bool = True


VARIANTS = {
    "mlp_default": {"hidden": (256,), "activation": "sigmoid", "dropout": 0.0, "weighted": False},
    "mlp_regularized": {"hidden": (256, 128), "activation": "relu", "dropout": 0.3, "weighted": False},
    "mlp_weighted": {"hidden": (256, 128), "activation": "relu", "dropout": 0.3, "weighted": True},
}


def configure(config):
    if min(config.epochs, config.batch_size, config.height, config.width, config.patience, config.threads) < 1:
        raise ValueError("Epochs, batch size, dimensions, patience and threads must be positive.")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()


def build_mlp(classes, variant, config):
    specification = VARIANTS[variant]
    network = [tf.keras.layers.Input(shape=(config.height, config.width, 3)),
               tf.keras.layers.Rescaling(1.0 / 255), tf.keras.layers.Flatten()]
    for units in specification["hidden"]:
        network.append(tf.keras.layers.Dense(units, activation=specification["activation"]))
        if specification["dropout"]:
            network.append(tf.keras.layers.Dropout(specification["dropout"]))
    network.append(tf.keras.layers.Dense(len(classes), activation="softmax"))
    model = tf.keras.Sequential(network)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def score_predictions(truth, predicted, classes):
    vocabulary = np.arange(len(classes))
    supported = np.unique(truth)
    report = classification_report(truth, predicted, labels=vocabulary, target_names=classes,
                                   output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(truth, predicted)),
               "macro_f1": float(f1_score(truth, predicted, labels=vocabulary, average="macro", zero_division=0)),
               "macro_f1_supported": float(f1_score(truth, predicted, labels=supported, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(truth, predicted, labels=vocabulary, average="weighted", zero_division=0)),
               "n_evaluated": len(truth), "n_classes": len(classes), "n_supported_classes": len(supported)}
    return metrics, report


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, pixels, labels, classes, batch_size):
        super().__init__()
        self.pixels, self.labels, self.classes, self.batch_size = pixels, labels, classes, batch_size

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.pixels, batch_size=self.batch_size, verbose=0).argmax(axis=1)
        value = f1_score(self.labels, predicted, labels=np.arange(len(self.classes)), average="macro", zero_division=0)
        logs["val_macro_f1"] = float(value)


def draw_history(history, path):
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


def save_evaluation(truth, predicted, classes, metadata, output_dir, prefix):
    metrics, report = score_predictions(truth, predicted, classes)
    pd.DataFrame(report).T.to_csv(output_dir / f"{prefix}_report.csv")
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / f"{prefix}_confusion.csv")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
             xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual", title=prefix)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_confusion.png", dpi=140)
    plt.close(figure)
    prediction_rows = metadata[["id", "duplicate_group"]].copy()
    prediction_rows["actual"] = np.asarray(classes)[truth]
    prediction_rows["predicted"] = np.asarray(classes)[predicted]
    prediction_rows.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    errors = np.flatnonzero(truth != predicted)[:12]
    if len(errors):
        figure, axes = plt.subplots(3, 4, figsize=(10, 8))
        from .data import prepare_image
        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index < len(errors):
                position = errors[index]
                axis.imshow(prepare_image(metadata.iloc[position].image_path, (80, 60)))
                axis.set_title(f"{metadata.iloc[position].id}\n{classes[truth[position]]} -> {classes[predicted[position]]}", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{prefix}_errors.png", dpi=120)
        plt.close(figure)
    return metrics


def fit_variant(pixels, partitions, classes, variant, config, directory, target):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    model = build_mlp(classes, variant, config)
    train_indices, train_labels = partitions["train"]
    val_indices, val_labels = partitions["validation"]
    weights = None
    if VARIANTS[variant]["weighted"]:
        counts = np.bincount(train_labels, minlength=len(classes))
        if (counts == 0).any():
            raise ValueError("A training class is absent.")
        balanced = np.minimum(np.sqrt(len(train_labels) / (len(classes) * counts)), 5.0)
        balanced /= np.average(balanced, weights=counts)
        weights = {index: float(value) for index, value in enumerate(balanced)}
    callback = ValidationMacroF1(pixels[val_indices], val_labels, classes, config.batch_size)
    model_path = directory / f"{target}_{variant}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_macro_f1", mode="max", save_best_only=True)
    stopping = tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=config.patience,
                                                restore_best_weights=True, min_delta=1e-4)
    started = time.perf_counter()
    print(f"Training {target}: {variant}", flush=True)
    history = model.fit(pixels[train_indices], train_labels, validation_data=(pixels[val_indices], val_labels),
                        batch_size=config.batch_size, epochs=config.epochs, callbacks=[callback, checkpoint, stopping],
                        class_weight=weights, shuffle=True, verbose=2).history
    history = {key: [float(value) for value in values] for key, values in history.items()}
    model = tf.keras.models.load_model(str(model_path), compile=False)
    probabilities = model.predict(pixels[val_indices], batch_size=config.batch_size, verbose=0)
    restored = tf.keras.models.load_model(str(model_path), compile=False)
    np.testing.assert_allclose(model(pixels[val_indices[:4]], training=False).numpy(),
                               restored(pixels[val_indices[:4]], training=False).numpy(), rtol=1e-5, atol=1e-6)
    details = {"target": target, "variant": variant, "classes": classes, "config": asdict(config),
               "class_weights": weights, "history": history, "epochs_run": len(history["loss"]),
               "best_epoch": int(np.argmax(history["val_macro_f1"]) + 1),
               "parameters": model.count_params(), "training_seconds": time.perf_counter() - started}
    (directory / f"{target}_{variant}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return model, probabilities, details


def fold_usage(probabilities, classes, merged_classes):
    folded = np.zeros((len(probabilities), len(merged_classes)), dtype=np.float32)
    for index, name in enumerate(classes):
        folded[:, merged_classes.index(USAGE_MERGE.get(name, "Other"))] += probabilities[:, index]
    return folded


def run_experiment(repo_root=None, run_dir=None, config=None):
    config = config or Config()
    configure(config)
    root = find_root(repo_root)
    run_dir = Path(run_dir).resolve() if run_dir else root / "outputs/task3" / time.strftime("run_%Y%m%d_%H%M%S")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run already exists; choose a new directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    metadata, audit = audit_data(root, run_dir)
    metadata = make_split(metadata, config.seed)
    metadata.to_csv(run_dir / "split_manifest.csv", index=False)
    if config.smoke:
        subsets = []
        for _, section in metadata.groupby(["split", "gender", "usage"], dropna=False, sort=True):
            subsets.append(section.sample(n=min(50, len(section)), random_state=config.seed))
        metadata = pd.concat(subsets).sort_values("id").reset_index(drop=True)
        metadata.to_csv(run_dir / "smoke_manifest.csv", index=False)
    pixels = load_pixels(metadata, (config.height, config.width))
    results, selected, coverage = [], {}, []
    for target in TARGETS:
        classes = sorted(metadata.loc[metadata.split.eq("train"), target].dropna().unique().tolist())
        partitions = target_arrays(metadata, target, classes)
        for split, (_, labels) in partitions.items():
            for index, name in enumerate(classes):
                coverage.append({"target": target, "split": split, "class": name, "count": int((labels == index).sum())})
        val_indices, val_labels = partitions["validation"]
        holdout_indices, holdout_labels = partitions["holdout"]
        majority = int(np.bincount(partitions["train"][1], minlength=len(classes)).argmax())
        for split in ("validation", "holdout"):
            positions, labels = partitions[split]
            metrics = save_evaluation(labels, np.full(len(labels), majority), classes, metadata.iloc[positions],
                                      run_dir, f"{target}_majority_{split}")
            results.append({"target": target, "model": "majority", "split": split, **metrics})
        candidates = []
        for variant in VARIANTS:
            model, probabilities, details = fit_variant(pixels, partitions, classes, variant, config, models_dir, target)
            metrics = save_evaluation(val_labels, probabilities.argmax(axis=1), classes, metadata.iloc[val_indices],
                                      run_dir, f"{target}_{variant}_validation")
            draw_history(details["history"], run_dir / f"{target}_{variant}_learning.png")
            row = {"target": target, "model": variant, "split": "validation", **metrics,
                   "parameters": details["parameters"], "epochs": details["epochs_run"],
                   "training_seconds": details["training_seconds"]}
            results.append(row)
            candidates.append(row)
            pd.DataFrame(results).to_csv(run_dir / "results_task3.csv", index=False)
        winner = sorted(candidates, key=lambda item: (-item["macro_f1"], item["parameters"], item["model"]))[0]
        winner_name = winner["model"]
        shutil.copy2(models_dir / f"{target}_{winner_name}.keras", models_dir / f"{target}_final.keras")
        details = json.loads((models_dir / f"{target}_{winner_name}.json").read_text(encoding="utf-8"))
        details["selection_rule"] = "Highest validation macro-F1; ties: fewer parameters, then stable name. Holdout not used."
        (models_dir / f"{target}_final.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        selected[target] = {"variant": winner_name, "validation_macro_f1": winner["macro_f1"], "classes": classes}
        model = tf.keras.models.load_model(str(models_dir / f"{target}_final.keras"), compile=False)
        started = time.perf_counter()
        probabilities = model.predict(pixels[holdout_indices], batch_size=config.batch_size, verbose=0)
        inference_seconds = time.perf_counter() - started
        metrics = save_evaluation(holdout_labels, probabilities.argmax(axis=1), classes, metadata.iloc[holdout_indices],
                                  run_dir, f"{target}_final_holdout")
        results.append({"target": target, "model": winner_name, "split": "holdout", **metrics,
                        "batch_inference_ms_per_image": inference_seconds * 1000 / len(holdout_indices)})
        selected[target]["holdout"] = metrics
        if target == "usage" and config.merged_experiment:
            merged_classes = sorted({USAGE_MERGE.get(name, "Other") for name in classes})
            merged_partitions = target_arrays(metadata, "usage_5class", merged_classes)
            merged_model, _, merged_details = fit_variant(pixels, merged_partitions, merged_classes, winner_name,
                                                         config, models_dir, "usage_5class")
            draw_history(merged_details["history"], run_dir / "usage_5class_learning.png")
            for split in ("validation", "holdout"):
                positions, labels = merged_partitions[split]
                for name, scores in (("folded_8class", fold_usage(model.predict(pixels[positions], verbose=0), classes, merged_classes)),
                                     ("retrained_5class", merged_model.predict(pixels[positions], verbose=0))):
                    metrics = save_evaluation(labels, scores.argmax(axis=1), merged_classes, metadata.iloc[positions],
                                              run_dir, f"usage_5class_{name}_{split}")
                    results.append({"target": "usage_5class", "model": name, "split": split, **metrics})
    pd.DataFrame(coverage).to_csv(run_dir / "class_coverage.csv", index=False)
    pd.DataFrame(results).to_csv(run_dir / "results_task3.csv", index=False)
    (run_dir / "selected_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    provenance = {"python": platform.python_version(), "platform": platform.platform(), "tensorflow": tf.__version__,
                  "numpy": np.__version__, "pandas": pd.__version__,
                  "metadata_sha256": hashlib.sha256((root / "A2_FashionDataset/FashionDataset/train/styles_train.csv").read_bytes()).hexdigest(),
                  "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
                  "smoke_only": config.smoke, "external_evaluation_performed": False}
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(pd.DataFrame(results)[["target", "model", "split", "accuracy", "macro_f1"]].to_string(index=False), flush=True)
    print(f"Saved Task 3 run to {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-merged", action="store_true")
    args = parser.parse_args()
    config = Config(epochs=2 if args.smoke else args.epochs, batch_size=args.batch_size,
                    seed=args.seed, smoke=args.smoke, merged_experiment=not args.skip_merged)
    run_experiment(args.repo_root, args.run_dir, config)


if __name__ == "__main__":
    main()
