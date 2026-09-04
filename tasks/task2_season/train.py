"""
Task 2: Fashion Season Classification (season)
================================================
Owner: Member 3
Target: "season" (4 classes: Summer, Fall, Winter, Spring)

MODEL: Random Forest on HOG + colour-histogram features, not the shared CNN.
Task 1 uses build_cnn() and Task 3 uses an MLP, so per-task model choice is
already how this team works. Random Forest fits this task specifically because
season describes which catalogue an item shipped in, not what it looks like
(see tasks/task2_season/README.md) - the goal is to show the ceiling is in the
label, not in model capacity, and a cheap model reaching the same ceiling as an
expensive one is itself evidence for that. Random Forest also hands us feature
importances for free, which is exactly the "does colour explain most of it?"
evidence the README asks for.

Random Forest has no learning rate or filters, so the tuning experiments below
use its own equivalents:
    n_estimators  <- capacity, plays the role "filters" plays for a CNN
    max_depth     <- model complexity, plays the role "dropout" plays for a CNN
    class_weight  <- same idea as CNN class weighting, Spring is only 4% of the data

HOW TO RUN:
    From the project root folder:
    python -m tasks.task2_season.train

OPTIONS:
    python -m tasks.task2_season.train --help
    python -m tasks.task2_season.train --n-estimators 300 --max-depth 20
    python -m tasks.task2_season.train --tune               # Run hyperparameter grid
    python -m tasks.task2_season.train --skip-baselines     # Fast run without classical ML
"""

import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# these four imports are the "frozen" shared code everyone in the team uses -
# they load the data, split it the same way for every task, and score models
# the same way, so results across tasks/people are actually comparable
from src.config import CACHE_DIR, MODEL_DIR, OUTPUT_DIR, SEED
from src.data import get_split, load_metadata
from src.evaluate import (
    class_weights,
    evaluate_model,
    per_class_report,
    plot_confusion,
)

np.random.seed(SEED)

TARGET_VALUE = "season"
N_COLOR_FEATURES = 16 * 3   # 16-bin histogram x 3 RGB channels, used later to
                            # split the feature vector back into "HOG part" and
                            # "colour part" for the feature-importance chart


def extract_features(images, split_name="train"):
    # turns a stack of raw images into a table of numbers a classic ML model
    # (not a CNN) can actually use: HOG describes shape/edges, the colour
    # histogram describes what colours are present and how much of each.
    # this is slow (a loop over every image) so the result is cached to disk -
    # the first run takes a while, every run after that just loads the file.
    cache_path = CACHE_DIR / f"features_{TARGET_VALUE}_{split_name}.npy"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path.name}...")
        return np.load(cache_path)

    print(f"Extracting features for {split_name} set ({len(images)} images)...")
    feats = []
    for i, img in enumerate(images):
        img = np.asarray(img, dtype=np.float32)

        # HOG needs a single-channel (grayscale) image, not RGB
        gray_img = rgb2gray(img / 255.0)
        hog_feat = hog(
            gray_img,
            orientations=8,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
        )

        # one 16-bucket histogram per colour channel (R, then G, then B),
        # normalised so it doesn't matter how big the image is
        hist_feats = []
        for c in range(3):
            h, _ = np.histogram(img[:, :, c], bins=16, range=(0, 255))
            hist_feats.append(h / (h.sum() + 1e-8))

        feats.append(np.concatenate([hog_feat, *hist_feats]))

        if (i + 1) % 10000 == 0 or (i + 1) == len(images):
            print(f"  Processed {i + 1}/{len(images)} images")

    features = np.asarray(feats, dtype=np.float32)
    np.save(cache_path, features)
    print(f"Saved features to {cache_path.name} (shape: {features.shape})")
    return features


def run_baselines(X_train_feat, y_train, X_val_feat, y_val, X_train_raw, X_val_raw,
                   label_encoder):
    # three numbers to beat before the Random Forest counts as having learned
    # anything real:
    #   1. majority baseline - the score from just always guessing "Summer"
    #   2. logistic regression on the full HOG + colour features - a simple
    #      linear model, so if it already does fine, we don't need Random
    #      Forest's extra complexity to explain the result
    #   3. logistic regression on ONLY the average colour of the image (3
    #      numbers) - this one is used later as evidence: if this crude model
    #      is nearly as good as the real one, colour is doing most of the work

    print("\nSTEP 2: RUNNING BASELINES")

    majority_class = int(np.bincount(y_train).argmax())
    majority_name = label_encoder.classes_[majority_class]
    y_pred_maj = np.full_like(y_val, fill_value=majority_class)
    print(f"\n[Baseline 1] Majority Class: '{majority_name}' (ID: {majority_class})")
    evaluate_model(
        y_val, y_pred_maj, TARGET_VALUE, "baseline_majority",
        notes=f"Always predicts most frequent class: '{majority_name}'",
    )

    print("\n[Baseline 2] Logistic Regression (HOG + colour histogram)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)
    evaluate_model(
        y_val, y_pred_lr, TARGET_VALUE, "logreg_hog_colour",
        notes="HOG (8 orient, 10x10 cell) + 16-bin RGB histogram, StandardScaler",
    )

    print("\n[Baseline 3] Logistic Regression on mean RGB only (colour evidence)...")
    # collapse each image down to just its average red, green, blue value -
    # the crudest possible colour-only feature, no shape information at all
    mean_rgb_train = X_train_raw.reshape(len(X_train_raw), -1, 3).mean(axis=1)
    mean_rgb_val = X_val_raw.reshape(len(X_val_raw), -1, 3).mean(axis=1)

    color_scaler = StandardScaler()
    mean_rgb_train_scaled = color_scaler.fit_transform(mean_rgb_train)
    mean_rgb_val_scaled = color_scaler.transform(mean_rgb_val)

    color_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    color_lr.fit(mean_rgb_train_scaled, y_train)
    y_pred_color = color_lr.predict(mean_rgb_val_scaled)
    evaluate_model(
        y_val, y_pred_color, TARGET_VALUE, "logreg_mean_rgb",
        notes="3 features only: mean R,G,B per image - crudest possible colour signal",
    )
    print("\nIf logreg_mean_rgb lands close to the Random Forest below, that is "
          "strong evidence colour alone explains most of what any model here "
          "can learn (see outputs/results.csv for the comparison).")


def train_and_evaluate_rf(
    X_train_feat, y_train, X_val_feat, y_val, label_encoder,
    n_estimators=300, max_depth=None, use_class_weights=True,
    model_name="rf_baseline", save_best_model=True,
):
    # this is the one function that actually builds a Random Forest, trains
    # it, scores it on the validation set, and (optionally) saves it - every
    # other function in this file just calls this one with different settings
    print(f"\nSTEP 4: TRAINING RANDOM FOREST ({model_name})")
    print(f"Config: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"weighted={use_class_weights}")

    # class_weights() makes rare classes (Spring is only ~4% of the data)
    # count for more during training, so the forest doesn't just ignore them
    weights_dict = class_weights(y_train) if use_class_weights else None
    if weights_dict:
        print("\nClass weights:")
        for class_id, weight in weights_dict.items():
            print(" ", label_encoder.classes_[class_id], round(weight, 2))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=weights_dict,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_feat, y_train)
    y_pred = rf.predict(X_val_feat)

    notes = (f"n_estimators={n_estimators}, max_depth={max_depth}, "
             f"weights={'balanced' if use_class_weights else 'none'}")
    eval_row = evaluate_model(y_val, y_pred, TARGET_VALUE, model_name, notes=notes)

    plot_confusion(y_val, y_pred, label_encoder, TARGET_VALUE)

    print("\n--- Per-class results ---")
    print(per_class_report(y_val, y_pred, label_encoder).round(3))

    if save_best_model:
        # save the model together with the label encoder, so a later script
        # can load both and know which number means which season
        model_path = MODEL_DIR / f"rf_{TARGET_VALUE}.joblib"
        joblib.dump({"model": rf, "label_encoder": label_encoder}, model_path)
        print(f"\nSaved model to {model_path.name}")

    return rf, eval_row


def run_hyperparameter_tuning(X_train_feat, y_train, X_val_feat, y_val, label_encoder):
    # tries a handful of settings one group at a time instead of every possible
    # combination at once (that would be way too many runs) - find the best
    # n_estimators first, then keep it fixed while trying different max_depth,
    # then keep both fixed while checking if class weighting actually helps.
    # every attempt gets logged to outputs/results.csv either way.
    print("\nSTEP 5: HYPERPARAMETER TUNING GRID")

    n_estimators_grid = [100, 300, 800]
    max_depth_grid = [None, 20, 10]

    # round 1: how many trees? (max_depth=None, class weights on for all of these)
    best_n_estimators = 300
    best_f1 = -1.0
    for n in n_estimators_grid:
        _, row = train_and_evaluate_rf(
            X_train_feat, y_train, X_val_feat, y_val, label_encoder,
            n_estimators=n, max_depth=None, use_class_weights=True,
            model_name=f"rf_tune_n{n}", save_best_model=False,
        )
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_n_estimators = n

    print(f"\n>>> Best n_estimators: {best_n_estimators} (macro-F1: {best_f1:.4f})")

    # round 2: how deep should each tree be allowed to grow? uses whatever
    # n_estimators won round 1. skip None since that was already tried above.
    best_max_depth = None
    for d in max_depth_grid:
        if d is None:
            continue
        _, row = train_and_evaluate_rf(
            X_train_feat, y_train, X_val_feat, y_val, label_encoder,
            n_estimators=best_n_estimators, max_depth=d, use_class_weights=True,
            model_name=f"rf_tune_depth{d}", save_best_model=False,
        )
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_max_depth = d

    print(f"\n>>> Best max_depth: {best_max_depth} (macro-F1: {best_f1:.4f})")

    # round 3: does turning class weighting off actually do better? uses
    # whatever n_estimators/max_depth won rounds 1 and 2.
    _, row_no_weights = train_and_evaluate_rf(
        X_train_feat, y_train, X_val_feat, y_val, label_encoder,
        n_estimators=best_n_estimators, max_depth=best_max_depth,
        use_class_weights=False, model_name="rf_tune_noweights",
        save_best_model=False,
    )
    best_use_weights = True
    if row_no_weights["macro_f1"] > best_f1:
        best_f1 = row_no_weights["macro_f1"]
        best_use_weights = False

    print("\nHYPERPARAMETER TUNING COMPLETED")
    print(f"Optimal configuration: n_estimators={best_n_estimators}, "
          f"max_depth={best_max_depth}, class_weights={best_use_weights} "
          f"(Best macro-F1: {best_f1:.4f})")

    # train one final model with whatever combination won, and this time
    # actually save it to disk
    print("\nTraining final model with optimal hyperparameters...")
    train_and_evaluate_rf(
        X_train_feat, y_train, X_val_feat, y_val, label_encoder,
        n_estimators=best_n_estimators, max_depth=best_max_depth,
        use_class_weights=best_use_weights, model_name="rf_best",
        save_best_model=True,
    )


def run_evidence(rf, X_val_raw, label_encoder, n_hog_features):
    # this task isn't really about getting a high score - the README explains
    # that "season" is more of a catalogue label than something visible in the
    # photo. so instead of chasing accuracy, this function builds two pieces
    # of evidence for the report: (1) is the model mostly using colour rather
    # than shape, and (2) can we find near-identical items that got different
    # season labels, which would prove the label isn't really in the image.
    print("\nSTEP 6: EVIDENCE - feature importance & near-duplicate images")

    # rf.feature_importances_ has one number per feature, in the same order
    # the features were built in extract_features() - HOG numbers first, then
    # the colour histogram numbers - so slicing it at n_hog_features separates
    # "how much the model relied on shape" from "how much it relied on colour"
    importances = rf.feature_importances_
    hog_importance = importances[:n_hog_features].sum()
    color_importance = importances[n_hog_features:].sum()
    print(f"\nTotal importance from HOG (shape) features:     {hog_importance:.3f}")
    print(f"Total importance from colour histogram features: {color_importance:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["HOG (shape)", "Colour histogram"], [hog_importance, color_importance],
           color=["steelblue", "salmon"])
    ax.set_ylabel("summed feature importance")
    ax.set_title("Random Forest: shape vs colour importance")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "season_feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_DIR / 'season_feature_importance.png'}")

    # now look for pairs of validation images that are the same articleType
    # (so they're already similar kinds of item) but were given different
    # season labels, and pick the pair whose pixels are closest together -
    # if two nearly-identical photos have different labels, that's direct
    # proof the season label isn't something the image can tell you
    df = load_metadata(verbose=False)
    df = df[df[TARGET_VALUE].notna()].reset_index(drop=True)
    va_ids = df[df["_split"] == "val"]["id"]
    df_va = df[df["_split"] == "val"].reset_index(drop=True)
    va_id_to_idx = {str(img_id): i for i, img_id in enumerate(va_ids)}

    best_pairs = []
    for article, group in df_va.groupby("articleType"):
        if len(group) < 2 or group[TARGET_VALUE].nunique() < 2:
            continue
        idx = group.index.to_numpy()
        imgs = np.stack([X_val_raw[va_id_to_idx[str(i)]] for i in df_va.loc[idx, "id"]])
        imgs_flat = imgs.reshape(len(imgs), -1).astype(np.float32)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if group.loc[idx[a], TARGET_VALUE] == group.loc[idx[b], TARGET_VALUE]:
                    continue
                # mean absolute pixel difference - smaller means more similar
                dist = np.abs(imgs_flat[a] - imgs_flat[b]).mean()
                best_pairs.append((dist, idx[a], idx[b], article))

    best_pairs.sort(key=lambda t: t[0])
    n_show = min(4, len(best_pairs))
    if n_show == 0:
        print("No cross-season near-duplicate pairs found.")
        return

    fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    for row_i, (dist, a, b, article) in enumerate(best_pairs[:n_show]):
        for col, i in enumerate((a, b)):
            img_id = df_va.loc[i, "id"]
            season_label = df_va.loc[i, TARGET_VALUE]
            axes[row_i, col].imshow(X_val_raw[va_id_to_idx[str(img_id)]])
            axes[row_i, col].set_title(f"{article}\nseason: {season_label}", fontsize=9)
            axes[row_i, col].axis("off")
    plt.suptitle("Near-identical items labelled with different seasons")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "season_near_duplicates.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_DIR / 'season_near_duplicates.png'} "
          f"({n_show} pairs, closest pixel-distance {best_pairs[0][0]:.1f})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Season Classification Models for Task 2 (Member 3)"
    )
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees (default: 300)")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max tree depth, unset = no limit (default: None)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip majority/logreg baselines")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Train RF without class weighting")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning grid across n_estimators, max_depth, class weights")
    return parser.parse_args()


def main():
    args = parse_args()

    print("TASK 2: SEASON CLASSIFICATION — TRAINING PIPELINE")
    print(f"Target: {TARGET_VALUE}")

    # normalised=False because the Random Forest trains on HOG/colour features
    # built from the original 0-255 pixel values, not the CNN's standardised
    # ones - HOG and colour histograms expect the raw image
    X_train_raw, X_val_raw, y_train, y_val, label_encoder = get_split(
        TARGET_VALUE, normalised=False)

    print("\n[Data Summary]")
    print(f"  Train images: {X_train_raw.shape} (dtype: {X_train_raw.dtype})")
    print(f"  Val images:   {X_val_raw.shape} (dtype: {X_val_raw.dtype})")
    print(f"  Classes:      {list(label_encoder.classes_)}")

    X_train_feat = extract_features(X_train_raw, split_name="train")
    X_val_feat = extract_features(X_val_raw, split_name="val")
    n_hog_features = X_train_feat.shape[1] - N_COLOR_FEATURES
    print(f"  Feature vector length: {X_train_feat.shape[1]} "
          f"({n_hog_features} HOG + {N_COLOR_FEATURES} colour)")

    if not args.skip_baselines and not args.tune:
        run_baselines(X_train_feat, y_train, X_val_feat, y_val,
                      X_train_raw, X_val_raw, label_encoder)

    if args.tune:
        run_hyperparameter_tuning(X_train_feat, y_train, X_val_feat, y_val, label_encoder)
    else:
        rf, _ = train_and_evaluate_rf(
            X_train_feat, y_train, X_val_feat, y_val, label_encoder,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            use_class_weights=not args.no_class_weights,
            model_name="rf_baseline",
            save_best_model=True,
        )
        # only build the evidence for a single, real run - not useful to
        # repeat this after every combination tried during --tune
        run_evidence(rf, X_val_raw, label_encoder, n_hog_features)

    print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results logged to: {OUTPUT_DIR / 'results.csv'}")
    print(f"Model saved to:    {MODEL_DIR / f'rf_{TARGET_VALUE}.joblib'}")


if __name__ == "__main__":
    main()