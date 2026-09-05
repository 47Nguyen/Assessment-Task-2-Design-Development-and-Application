"""
Task 2: Fashion Season Classification (season)
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42

# ROOT is the project root: this file sits at ROOT/tasks/task2_season/train.py
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "A2_FashionDataset" / "FashionDataset"
TRAIN_CSV = DATA_ROOT / "train" / "styles_train.csv"
TRAIN_IMAGES = DATA_ROOT / "train" / "images_train"

CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
for _d in (CACHE_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)

IMG_WIDTH = 60
IMG_HEIGHT = 80
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

STRATIFY_ON = "articleType"   # finest-grained label, balancing it roughly
                              # balances the others through correlation
VAL_SIZE = 0.2

RESULTS_CSV = OUTPUT_DIR / "results.csv"

TARGET_VALUE = "season"
N_COLOR_FEATURES = 16 * 3   # 16-bin histogram x 3 RGB channels


def load_metadata(verbose=True):
    # reads styles_train.csv, drops the junk trailing columns caused by stray
    # commas in the file, keeps only rows that actually have a matching image
    # on disk, and tags every row "train" or "val" so the split is fixed once
    # and reused everywhere below
    df = pd.read_csv(TRAIN_CSV)
    n_raw = len(df)

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    on_disk = {p.stem for p in TRAIN_IMAGES.glob("*.jpg")}
    df = df[df["id"].astype(str).isin(on_disk)].reset_index(drop=True)

    df = _attach_split(df)

    if verbose:
        print(f"metadata: {n_raw} rows in CSV -> {len(df)} usable "
              f"({n_raw - len(df)} dropped: no matching image)")
        print(f"          {(df['_split'] == 'train').sum()} train / "
              f"{(df['_split'] == 'val').sum()} val")
    return df


def _attach_split(df):
    # stratified 80/20 split on articleType, same seed every run. a handful
    # of articleType classes have only 1 example and can't be stratified -
    # those rows go to train by default, which means they can never show up
    # in validation (worth mentioning in the report, not a bug)
    counts = df[STRATIFY_ON].value_counts()
    too_rare = counts[counts < 2].index
    rare_mask = df[STRATIFY_ON].isin(too_rare)

    splittable = df[~rare_mask]
    train_idx, val_idx = train_test_split(
        splittable.index,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=splittable[STRATIFY_ON],
    )

    df = df.copy()
    df["_split"] = "train"
    df.loc[val_idx, "_split"] = "val"
    return df


def _build_image_cache(ids, image_dir, cache_path):
    # decodes every jpg in image_dir into one big uint8 array and saves it to
    # disk - slow the first time (~1-2 minutes for the full dataset), instant
    # every run after that. ~38.6k x 80 x 60 x 3 uint8 is about 550 MB.
    print(f"building image cache -> {cache_path.name} ({len(ids)} images, one-off)")
    arr = np.empty((len(ids), *IMG_SHAPE), dtype=np.uint8)
    n_grey, n_resized = 0, 0
    for i, img_id in enumerate(ids):
        with Image.open(image_dir / f"{img_id}.jpg") as im:
            # ~0.9% of files are grayscale - without converting, they come
            # back 2-D and break when stacked with the RGB ones
            if im.mode != "RGB":
                im = im.convert("RGB")
                n_grey += 1
            # most images are a uniform 60x80, a handful are not
            if im.size != (IMG_WIDTH, IMG_HEIGHT):
                im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
                n_resized += 1
            arr[i] = np.asarray(im)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(ids)}")
    print(f"  done: {n_grey} grayscale converted, {n_resized} resized to "
          f"{IMG_WIDTH}x{IMG_HEIGHT}")
    np.save(cache_path, arr)
    return arr


def load_images(ids):
    # returns decoded uint8 images for the given ids, building/reusing the
    # on-disk cache so repeated runs don't re-read every jpg from scratch
    ids = [str(i) for i in ids]
    cache_path = CACHE_DIR / "images_train.npy"
    index_path = CACHE_DIR / "index_train.npy"

    if cache_path.exists() and index_path.exists():
        cached_ids = np.load(index_path, allow_pickle=True)
        arr = np.load(cache_path, mmap_mode="r")
        lookup = {img_id: i for i, img_id in enumerate(cached_ids)}
        return np.stack([arr[lookup[i]] for i in ids])

    all_ids = sorted(p.stem for p in TRAIN_IMAGES.glob("*.jpg"))
    arr = _build_image_cache(all_ids, TRAIN_IMAGES, cache_path)
    np.save(index_path, np.array(all_ids, dtype=object))
    lookup = {img_id: i for i, img_id in enumerate(all_ids)}
    return np.stack([arr[lookup[i]] for i in ids])


def get_split(target, verbose=True):
    # the one function every task calls - loads the raw images and labels for
    # `target`, using the same train/val row assignment as every other task.
    # returns (X_train, X_val, y_train, y_val, label_encoder). Random Forest
    # works on raw uint8 images here because extract_features() below turns
    # them into HOG/colour features itself - there's no CNN-style pixel
    # normalisation step needed.
    df = load_metadata(verbose=False)

    n_before = len(df)
    df = df[df[target].notna()].reset_index(drop=True)
    if verbose and n_before != len(df):
        print(f"{target}: dropped {n_before - len(df)} rows with a missing label")

    le = LabelEncoder().fit(df[target])
    joblib.dump(le, MODEL_DIR / f"label_encoder_{target}.joblib")

    tr = df[df["_split"] == "train"]
    va = df[df["_split"] == "val"]

    X_train = load_images(tr["id"])
    X_val = load_images(va["id"])
    y_train = le.transform(tr[target])
    y_val = le.transform(va[target])

    if verbose:
        print(f"{target}: {len(le.classes_)} classes | "
              f"train {X_train.shape[0]} / val {X_val.shape[0]} | "
              f"majority baseline {pd.Series(y_train).value_counts(normalize=True).iloc[0]:.3f}")
    return X_train, X_val, y_train, y_val, le


def majority_baseline(y):
    return float(pd.Series(y).value_counts(normalize=True).iloc[0])


def evaluate_model(y_true, y_pred, target, model_name, notes=""):
    # scores a model and appends/updates one row in outputs/results.csv -
    # macro-F1 is the headline number because accuracy alone can look good
    # while a model has just learned to always guess the biggest class
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
    # keeps outputs/results.csv as one row per (target, model) combo - running
    # the same model_name twice replaces the old row instead of duplicating it
    df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        old = old[~((old["target"] == row["target"]) & (old["model"] == row["model"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)


def per_class_report(y_true, y_pred, label_encoder, top_n=None):
    # precision/recall/F1 broken down per class, worst recall first - this is
    # where a class that the model basically ignores becomes visible
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


def plot_confusion(y_true, y_pred, label_encoder, target, max_classes=25):
    # confusion matrix normalised by row (so each row sums to 1 = "of the
    # items that were truly class X, what fraction got predicted as what")
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

    path = OUTPUT_DIR / f"confusion_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return fig


def class_weights(y):
    # weights that make rare classes (Spring is only ~4% of the data) count
    # for more during training, so the model doesn't just ignore them
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y))
    return dict(zip(classes.tolist(), weights.tolist()))


# ===========================================================================
# Task 2 specific code starts here
# ===========================================================================

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

    X_train_raw, X_val_raw, y_train, y_val, label_encoder = get_split(TARGET_VALUE)

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