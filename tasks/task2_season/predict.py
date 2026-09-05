"""
Task 2: Predict season on the test set

HOW TO RUN:
    From the project root folder (same place train.py runs from):
    python -m tasks.task2_season.predict

Requires models/rf_season.joblib to already exist - run train.py first.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

# same path setup as train.py, so this script finds the same dataset/model
# without needing src/
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "A2_FashionDataset" / "FashionDataset"
TEST_CSV = DATA_ROOT / "test" / "styles_prediction.csv"
TEST_IMAGES = DATA_ROOT / "test" / "images_test"

CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
for _d in (CACHE_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)

IMG_WIDTH = 60
IMG_HEIGHT = 80
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

TARGET_VALUE = "season"
OUTPUT_CSV = OUTPUT_DIR / "task2_season_predictions.csv"


def load_test_images():
    # same idea as train.py's image cache - decode every test jpg once and
    # save it, so re-running this script doesn't re-read 5,829 files from disk
    cache_path = CACHE_DIR / "images_test.npy"
    index_path = CACHE_DIR / "index_test.npy"

    ids = [str(i) for i in pd.read_csv(TEST_CSV)["id"]]

    if cache_path.exists() and index_path.exists():
        cached_ids = np.load(index_path, allow_pickle=True)
        arr = np.load(cache_path, mmap_mode="r")
        lookup = {img_id: i for i, img_id in enumerate(cached_ids)}
        return np.stack([arr[lookup[i]] for i in ids]), ids

    print(f"building test image cache -> {cache_path.name} ({len(ids)} images, one-off)")
    arr = np.empty((len(ids), *IMG_SHAPE), dtype=np.uint8)
    n_grey, n_resized = 0, 0
    for i, img_id in enumerate(ids):
        with Image.open(TEST_IMAGES / f"{img_id}.jpg") as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
                n_grey += 1
            if im.size != (IMG_WIDTH, IMG_HEIGHT):
                im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
                n_resized += 1
            arr[i] = np.asarray(im)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(ids)}")
    print(f"  done: {n_grey} grayscale converted, {n_resized} resized to {IMG_WIDTH}x{IMG_HEIGHT}")

    np.save(cache_path, arr)
    np.save(index_path, np.array(ids, dtype=object))
    return arr, ids


def extract_features(images):
    # MUST match train.py's extract_features exactly (same HOG params, same
    # histogram bins) - the model was fit on features built this way, so
    # predicting with differently-shaped features would silently produce
    # garbage rather than an error
    cache_path = CACHE_DIR / f"features_{TARGET_VALUE}_test.npy"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path.name}...")
        return np.load(cache_path)

    print(f"Extracting features for test set ({len(images)} images)...")
    feats = []
    for i, img in enumerate(images):
        img = np.asarray(img, dtype=np.float32)
        gray_img = rgb2gray(img / 255.0)
        hog_feat = hog(
            gray_img,
            orientations=8,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
        )
        hist_feats = []
        for c in range(3):
            h, _ = np.histogram(img[:, :, c], bins=16, range=(0, 255))
            hist_feats.append(h / (h.sum() + 1e-8))
        feats.append(np.concatenate([hog_feat, *hist_feats]))
        if (i + 1) % 2000 == 0 or (i + 1) == len(images):
            print(f"  Processed {i + 1}/{len(images)} images")

    features = np.asarray(feats, dtype=np.float32)
    np.save(cache_path, features)
    print(f"Saved features to {cache_path.name} (shape: {features.shape})")
    return features


def main():
    print("TASK 2: SEASON PREDICTION ON TEST SET")

    model_path = MODEL_DIR / f"rf_{TARGET_VALUE}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run 'python -m tasks.task2_season.train' first"
        )
    saved = joblib.load(model_path)
    rf = saved["model"]
    label_encoder = saved["label_encoder"]
    print(f"Loaded model from {model_path}")

    X_test_raw, ids = load_test_images()
    print(f"Test images: {X_test_raw.shape}")

    X_test_feat = extract_features(X_test_raw)
    print(f"Feature vector length: {X_test_feat.shape[1]}")

    y_pred = rf.predict(X_test_feat)
    season_labels = label_encoder.inverse_transform(y_pred)

    print("\nPredicted season distribution:")
    print(pd.Series(season_labels).value_counts())

    result = pd.DataFrame({"id": ids, "season": season_labels})
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to {OUTPUT_CSV}")
    print("(id, season only - merge this into the shared submission file separately)")


if __name__ == "__main__":
    main()