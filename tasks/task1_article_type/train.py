# ===========================================================================
# TASK 1 (ARTICLE TYPE CLASSIFICATION) - 100% SELF-CONTAINED TRAINING SCRIPT
# This script has been adjusted to run without the 'src' folder (frozen APIs).
# All data loading, cleaning, CNN building, and evaluation logic are now local.
# ===========================================================================

import argparse
import inspect
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. Setup & Configuration
# ---------------------------------------------------------------------------
SEED = 42
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("cache")
ROOT = Path(".")

# Ensure folders exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set deterministic random seeds
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# import data 
folder_path = ""

# ---------------------------------------------------------------------------
# 2. Choose your target
# ---------------------------------------------------------------------------
target_value = "articleType"     # <-- CHANGE THIS

# ---------------------------------------------------------------------------
# 2. Self-contained replacement for src.data.get_split
# ---------------------------------------------------------------------------
def get_split_self_contained(target_value="articleType", normalised=True, verbose=True):
    """
    100% self-contained data loader and split generator.
    Replaces get_split from src.data when the src folder is deleted.
    """
    cache_prefix = CACHE_DIR / f"task1_{target_value}"
    X_train_path = f"{cache_prefix}_X_train.npy"
    X_val_path = f"{cache_prefix}_X_val.npy"
    y_train_path = f"{cache_prefix}_y_train.npy"
    y_val_path = f"{cache_prefix}_y_val.npy"
    encoder_path = MODEL_DIR / f"label_encoder_{target_value}.joblib"
    
    if os.path.exists(X_train_path) and os.path.exists(X_val_path) and os.path.exists(y_train_path) and os.path.exists(y_val_path) and os.path.exists(encoder_path):
        if verbose:
            print("Loading cached dataset...")
        X_train = np.load(X_train_path)
        X_val = np.load(X_val_path)
        y_train = np.load(y_train_path)
        y_val = np.load(y_val_path)
        le = joblib.load(encoder_path)
    else:
        if verbose:
            print("Loading raw styles_train.csv and decoding images...")
        
        base_dir = folder_path if folder_path != "" else "A2_FashionDataset/FashionDataset"
        csv_path = os.path.join(base_dir, "train", "styles_train.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"styles_train.csv not found at: {csv_path}. Please check the path or set folder_path at the top of the file!")
            
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df = df.iloc[:, :10]  # Drop junk columns (Unnamed: 10, Unnamed: 11) from stray commas
        
        # Filter out missing/invalid images
        img_dir = os.path.join(base_dir, "train", "images_train")
        df['img_path'] = df['id'].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
        valid_mask = df['img_path'].apply(os.path.exists)
        df = df[valid_mask].reset_index(drop=True)
        
        # Filter out missing labels if any
        df = df.dropna(subset=[target_value]).reset_index(drop=True)
        
        # Fit LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df[target_value])
        joblib.dump(le, encoder_path)
        
        # Split train/val 80/20
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
        
        # Load images
        def load_images_from_df(dataframe):
            images = []
            from tqdm import tqdm
            for path in tqdm(dataframe['img_path'], desc="Reading & resizing images to 60x80"):
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                        img = img.resize((60, 80))  # width=60, height=80 -> shape (80, 60, 3)
                        images.append(np.array(img, dtype=np.uint8))
                except Exception as e:
                    # Fallback to black image if load fails
                    images.append(np.zeros((80, 60, 3), dtype=np.uint8))
            return np.array(images, dtype=np.uint8)
            
        X_train_raw = load_images_from_df(train_df)
        X_val_raw = load_images_from_df(val_df)
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        
        # Cache raw arrays
        np.save(X_train_path, X_train_raw)
        np.save(X_val_path, X_val_raw)
        np.save(y_train_path, y_train)
        np.save(y_val_path, y_val)
        
        X_train = X_train_raw
        X_val = X_val_raw
        
    if normalised:
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        
    return X_train, X_val, y_train, y_val, le

# ---------------------------------------------------------------------------
# 3. Self-contained replacements for models & callbacks
# ---------------------------------------------------------------------------
def build_cnn(n_classes, filters=(32, 64, 128), dropout=0.3):
    """
    Self-contained CNN Builder.
    Replaces build_cnn() from src.models.
    """
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=(80, 60, 3), name="image")
    x = inputs
    
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, (3, 3), padding='same', name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation('relu', name=f"relu{i}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i}")(x)
        
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="embedding")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation='softmax', name="prediction")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="cnn_articleType")

def get_default_callbacks(target_value, patience=5):
    """
    Self-contained Callbacks Generator.
    Replaces default_callbacks() from src.models.
    """
    checkpoint_path = f"models/cnn_{target_value}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

# ---------------------------------------------------------------------------
# 4. Self-contained replacement for src.evaluate.evaluate_model & confusion
# ---------------------------------------------------------------------------
def evaluate_model(y_true, y_pred, task_name, model_name, notes=""):
    """
    Self-contained evaluation and metrics logger.
    Replaces evaluate_model() from src.evaluate.
    """
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\nMODEL EVALUATION RESULTS: {model_name} ({task_name})")
    print(f"  macro-F1:      {macro_f1:.4f}   <- Primary Metric")
    print(f"  balanced acc:  {balanced_acc:.4f}")
    print(f"  accuracy:      {acc:.4f}")
    
    # Save a row to outputs/results.csv
    res_path = OUTPUT_DIR / "results.csv"
    row = {
        "task": task_name,
        "model_name": model_name,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "accuracy": acc,
        "notes": notes
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(res_path):
        try:
            df_all = pd.read_csv(res_path)
            df_all = pd.concat([df_all, df_row], ignore_index=True)
        except Exception:
            df_all = df_row
    else:
        df_all = df_row
    df_all.to_csv(res_path, index=False)
    print(f"Results saved to: {res_path}")
    return row

def compute_capped_class_weights(y_train, max_weight_cap=10.0, min_weight=1.0):
    """
    Balanced class weighting with clipping.
    """
    counts = np.bincount(y_train)
    total = len(y_train)
    n_classes = len(counts)
    
    raw_weights = total / (n_classes * np.maximum(counts, 1))
    capped_weights = np.clip(raw_weights, min_weight, max_weight_cap)
    
    weights_dict = {i: float(w) for i, w in enumerate(capped_weights)}
    return weights_dict

# ---------------------------------------------------------------------------
# 5. Core Feature Extraction (Classical ML)
# ---------------------------------------------------------------------------
def extract_features(images):
    """
    HOG + Color Histogram feature extractor for baselines.
    """
    features = []
    print(f"Extracting HOG + Color Hist features for {len(images)} images...")
    for idx, img in enumerate(images):
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(images)} images...")
        
        # Grayscale for HOG
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        # HOG Feature
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(15, 20),
                       cells_per_block=(1, 1), visualize=False)
        
        # Color Histograms (8 bins per RGB channel)
        hist_r, _ = np.histogram(img[:,:,0], bins=8, range=(0, 1))
        hist_g, _ = np.histogram(img[:,:,1], bins=8, range=(0, 1))
        hist_b, _ = np.histogram(img[:,:,2], bins=8, range=(0, 1))
        color_hist = np.concatenate([hist_r, hist_g, hist_b])
        
        combined = np.concatenate([hog_feat, color_hist])
        features.append(combined)
        
    return np.array(features)

# ---------------------------------------------------------------------------
# 6. Main Execution Flow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 1 train script - Adjusted (Self-Contained)")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning grid')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip classical ML baselines')
    args = parser.parse_args()

    # Load data locally
    print("Loading data split...")
    X_train, X_val, y_train, y_val, label_encoder = get_split_self_contained(target_value)
    n_classes = len(label_encoder.classes_)
    
    print(f"\nTarget: {target_value}")
    print(f"Train images: {X_train.shape}")
    print(f"Val images:   {X_val.shape}")
    print(f"Classes:      {n_classes}")

    # Step 1: Baseline
    print("\n--- Step 1: Training Baseline ---")
    majority_class_idx = np.bincount(y_train.astype(int)).argmax()
    majority_class_name = label_encoder.classes_[majority_class_idx]
    print(f"Majority class: {majority_class_name} (index {majority_class_idx})")
    
    y_pred_baseline = np.full_like(y_val, majority_class_idx)
    evaluate_model(y_val, y_pred_baseline, target_value, "majority_baseline", 
                   notes=f"Always predicts: {majority_class_name}")

    # Step 2: Classical ML
    if not args.skip_baselines:
        print("\n--- Step 2: Training Classical ML Models ---")
        X_train_feats = extract_features(X_train)
        X_val_feats = extract_features(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feats)
        X_val_scaled = scaler.transform(X_val_feats)

        # Logistic Regression
        print("Training Logistic Regression (SGD)...")
        clf_lr = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
        clf_lr.fit(X_train_scaled, y_train)
        y_pred_lr = clf_lr.predict(X_val_scaled)
        evaluate_model(y_val, y_pred_lr, target_value, "logistic_regression", 
                       notes="SGDClassifier on HOG+ColorHist")

        # Linear SVM
        print("Training Linear SVM (SGD)...")
        clf_svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
        clf_svm.fit(X_train_scaled, y_train)
        y_pred_svm = clf_svm.predict(X_val_scaled)
        evaluate_model(y_val, y_pred_svm, target_value, "linear_svm", 
                       notes="SGDClassifier with hinge loss on HOG+ColorHist")

    # Step 3: CNN Baseline
    print("\n--- Step 3: Training Baseline CNN Model ---")
    from tensorflow.keras.optimizers import Adam
    
    model = build_cnn(n_classes=n_classes)
    
    # Compute capped weights to avoid gradient explosion
    weights_dict = compute_capped_class_weights(y_train, max_weight_cap=10.0)
    
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    print("Starting Reference CNN Training...")
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=args.epochs, 
              batch_size=args.batch_size, 
              class_weight=weights_dict,
              callbacks=get_default_callbacks(target_value))
    
    y_pred_cnn = model.predict(X_val).argmax(axis=1)
    evaluate_model(y_val, y_pred_cnn, target_value, "cnn_baseline", 
                   notes="lr 1e-3, local architecture")
    
    model.save("models/cnn_articleType.keras")
    print("CNN model successfully saved to: models/cnn_articleType.keras")

    # Step 4: Hyperparameter Tuning Grid
    if args.tune:
        print("\n--- Step 4: Running Hyperparameter Tuning (OFAAT) ---")
        
        lr_candidates = [1e-2, 1e-3, 1e-4]
        filter_candidates = [(32, 64), (32, 64, 128), (64, 128, 256)]
        dropout_candidates = [0.2, 0.3, 0.5]
        
        def run_tuning_experiment(lr=1e-3, filters=(32, 64, 128), dropout=0.3, run_name="cnn_tuned"):
            print(f"\nTuning: {run_name} (lr={lr}, filters={filters}, dropout={dropout})")
            m = build_cnn(n_classes=n_classes, filters=filters, dropout=dropout)
            m.compile(optimizer=Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
            
            # Fast train 10 epochs for comparison
            m.fit(X_train, y_train, 
                  validation_data=(X_val, y_val), 
                  epochs=10, 
                  batch_size=args.batch_size, 
                  class_weight=weights_dict,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
            
            y_pred_exp = m.predict(X_val).argmax(axis=1)
            evaluate_model(y_val, y_pred_exp, target_value, run_name, 
                           notes=f"lr={lr}, filters={filters}, dropout={dropout}")
            return m

        # 1. Tuning Learning Rate
        for lr in lr_candidates:
            run_tuning_experiment(lr=lr, run_name=f"cnn_lr_{str(lr).replace('.', '_')}")
            
        # 2. Tuning Filters
        for filters in filter_candidates:
            run_tuning_experiment(filters=filters, run_name=f"cnn_filters_{'_'.join(map(str, filters))}")
            
        # 3. Tuning Dropout
        for d in dropout_candidates:
            run_tuning_experiment(dropout=d, run_name=f"cnn_dropout_{str(d).replace('.', '_')}")
            
        print("\nTuning Grid completed!")

if __name__ == "__main__":
    main()
