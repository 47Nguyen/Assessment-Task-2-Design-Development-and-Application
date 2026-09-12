"""
Task 1: Predict articleType on the test set

HOW TO RUN:
    From the project root folder (same place train.py runs from):
    python -m tasks.task1_article_type.predict

Requires models/task_1/cnn_articleType.keras and
models/task_1/label_encoder_articleType.joblib to already exist - run
train.py first.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "A2_FashionDataset" / "FashionDataset"
TEST_CSV = DATA_ROOT / "test" / "styles_prediction.csv"
TEST_IMAGES = DATA_ROOT / "test" / "images_test"

# Tasks 1-3 all fill in different columns of this one file
SUBMISSION_PATH = ROOT / "outputs" / "COSC2753_A2_SG_G9_Task1-3.csv"

MODEL_DIR = ROOT / "models" / "task_1"

IMG_WIDTH = 60
IMG_HEIGHT = 80

TARGET_VALUE = "articleType"
BATCH_SIZE = 256

def load_test_images(ids):

    print(f"Loading {len(ids)} test images...")
    arr = np.empty((len(ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    for i, img_id in enumerate(ids):
        img = tf.keras.preprocessing.image.load_img(
            TEST_IMAGES / f"{img_id}.jpg", target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        arr[i] = tf.keras.preprocessing.image.img_to_array(img)
        if (i + 1) % 2000 == 0 or (i + 1) == len(ids):
            print(f"  {i + 1}/{len(ids)}")
    return arr / 255.0

def update_submission(ids, columns):
    source = SUBMISSION_PATH if SUBMISSION_PATH.exists() else TEST_CSV
    table = pd.read_csv(source, dtype=str, keep_default_na=False)

    ids = [str(i) for i in ids]
    lookup = {column: dict(zip(ids, values)) for column, values in columns.items()}
    for column, values_by_id in lookup.items():
        table[column] = [
            values_by_id.get(row_id, existing)
            for row_id, existing in zip(table["id"], table[column])
        ]

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUBMISSION_PATH, index=False)
    print(f"Updated {list(columns)} for {len(ids)} rows in {SUBMISSION_PATH}")
    return SUBMISSION_PATH

def main():
    print("TASK 1: ARTICLE TYPE PREDICTION ON TEST SET")

    model_path = MODEL_DIR / f"cnn_{TARGET_VALUE}.keras"
    encoder_path = MODEL_DIR / f"label_encoder_{TARGET_VALUE}.joblib"
    for path in (model_path, encoder_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run 'python -m tasks.task1_article_type.train' first"
            )

    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    print(f"Loaded model from {model_path}")

    ids = [str(i) for i in pd.read_csv(TEST_CSV)["id"]]
    X_test = load_test_images(ids)
    print(f"Test images: {X_test.shape}")

    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    article_labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))

    print("\nPredicted articleType distribution:")
    print(pd.Series(article_labels).value_counts())

    update_submission(ids, {TARGET_VALUE: article_labels})

if __name__ == "__main__":
    main()
