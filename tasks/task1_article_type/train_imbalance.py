import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Architecture Decisions
# - Use local directories instead of importing from src/ to keep script self-contained.
# - Cache data as .npy arrays to speed up IO reading operations.
# ---------------------------------------------------------------------------
SEED = 42
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("cache")

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = "A2_FashionDataset/FashionDataset/train/styles_train.csv"
IMG_DIR = "A2_FashionDataset/FashionDataset/train/images_train"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log_result(model_name, target, macro_f1, balanced_acc, accuracy):
    """Log results to CSV file. Overwrite if model_name and target already exist."""
    csv_file = OUTPUT_DIR / "results.csv"
    new_row = pd.DataFrame([{
        "model": model_name,
        "target": target,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "accuracy": accuracy
    }])
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        # Remove old record for this model
        df = df[~((df["target"] == target) & (df["model"] == model_name))]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
        
    df.to_csv(csv_file, index=False)
    print(f"[*] Updated results for {model_name} in {csv_file}")

def plot_and_save_confusion_matrix(y_true, y_pred, classes, model_name):
    """Plot and save Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 24) if len(classes) > 50 else (12, 12))
    
    # Hide labels if there are too many to prevent clutter
    if len(classes) > 50:
        sns.heatmap(cm, cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(f"Confusion Matrix: {model_name}\n(Labels hidden due to high cardinality)", fontsize=16)
    else:
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"confusion_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[*] Saved confusion matrix to {out_path}")

def build_reference_cnn(n_classes):
    """
    Standard CNN architecture required for Imbalance experiments.
    Uses 3 basic Conv2D blocks + GlobalAveragePooling (prevents overfitting better than Flatten).
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=(80, 60, 3)),
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data(target_column="articleType"):
    """
    Load data, filter invalid images, and cache npy arrays.
    Performs exact 80/20 split with SEED=42.
    """
    cache_prefix = CACHE_DIR / f"task1_{target_column}"
    paths = {
        "X_train": f"{cache_prefix}_X_train.npy",
        "X_val": f"{cache_prefix}_X_val.npy",
        "y_train": f"{cache_prefix}_y_train.npy",
        "y_val": f"{cache_prefix}_y_val.npy",
        "encoder": MODEL_DIR / f"label_encoder_{target_column}.joblib"
    }
    
    all_cached = all(os.path.exists(p) for p in paths.values())
    if all_cached:
        print(f"[*] Loading data {target_column} from Cache...")
        X_train = np.load(paths["X_train"])
        X_val = np.load(paths["X_val"])
        y_train = np.load(paths["y_train"])
        y_val = np.load(paths["y_val"])
        le = joblib.load(paths["encoder"])
    else:
        print(f"[*] Processing data from scratch for {target_column} (will cache afterwards)...")
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Cannot find {CSV_PATH}")
            
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
        df = df.iloc[:, :10] # Keep only first 10 columns like original code
        
        # Create image paths
        df['img_path'] = df['id'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))
        
        # Filter existing images
        valid_mask = df['img_path'].apply(os.path.exists)
        df = df[valid_mask].reset_index(drop=True)
        
        # Drop NaN values for target column
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        
        # Encode label
        le = LabelEncoder()
        df['label'] = le.fit_transform(df[target_column])
        joblib.dump(le, paths["encoder"])
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
        
        def load_images(dataframe):
            imgs = []
            for path in dataframe['img_path']:
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB').resize((60, 80))
                        imgs.append(np.array(img, dtype=np.uint8))
                except Exception:
                    imgs.append(np.zeros((80, 60, 3), dtype=np.uint8))
            return np.array(imgs)
            
        X_train = load_images(train_df)
        X_val = load_images(val_df)
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        
        np.save(paths["X_train"], X_train)
        np.save(paths["X_val"], X_val)
        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)
    
    # Normalize to [0, 1] for CNN
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    
    print(f"[-] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[-] X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"[-] Classes: {len(le.classes_)}")
    
    return X_train, X_val, y_train, y_val, le

def evaluate_and_log(model, X_val, y_val, le, model_name, target):
    """Predict, compute F1 score, plot confusion matrix and save results."""
    print(f"\n--- Evaluating {model_name} ---")
    preds_prob = model.predict(X_val, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    
    # Ensure no undefined metric warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro_f1 = f1_score(y_val, preds, average='macro')
        acc = accuracy_score(y_val, preds)
        bal_acc = balanced_accuracy_score(y_val, preds)
        
    print(f"Macro-F1: {macro_f1:.4f} | Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
    
    log_result(model_name, target, macro_f1, bal_acc, acc)
    plot_and_save_confusion_matrix(y_val, preds, le.classes_, model_name)

# ---------------------------------------------------------------------------
# Experiment 1: Class Weights
# ---------------------------------------------------------------------------
def run_experiment_1():
    print("\n" + "="*50)
    print("EXPERIMENT 1: Class Weights (cnn_weighted)")
    print("="*50)
    
    X_train, X_val, y_train, y_val, le = load_data("articleType")
    classes = np.unique(y_train)
    
    # Calculate weights
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # Limit weight to 10.0 to prevent gradient explosion for very rare classes
    weights = np.clip(weights, 0, 10.0)
    class_weight_dict = dict(zip(classes, weights))
    
    n_classes = len(le.classes_)
    for i in range(n_classes):
        if i not in class_weight_dict:
            class_weight_dict[i] = 10.0 # Default weight cap for missing classes
            
    print(f"[-] Filled missing class weights. Total keys: {len(class_weight_dict)}")
    
    model = build_reference_cnn(n_classes)
    
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[es],
        verbose=1
    )
    
    model_path = MODEL_DIR / "cnn_weighted.keras"
    model.save(model_path)
    print(f"[*] Saved model to {model_path}")
    
    evaluate_and_log(model, X_val, y_val, le, "cnn_weighted", "articleType")

# ---------------------------------------------------------------------------
# Experiment 2: Oversampling Rare Classes
# ---------------------------------------------------------------------------
def run_experiment_2():
    print("\n" + "="*50)
    print("EXPERIMENT 2: Oversampling (cnn_oversampled)")
    print("="*50)
    
    X_train, X_val, y_train, y_val, le = load_data("articleType")
    
    # Find rare classes (< 100 samples)
    unique_classes, counts = np.unique(y_train, return_counts=True)
    rare_classes = unique_classes[counts < 100]
    
    print(f"[*] Found {len(rare_classes)} classes with under 100 samples. Oversampling...")
    
    X_extra = []
    y_extra = []
    
    # Fix random seed for stable oversampling results
    rng = np.random.default_rng(SEED)
    
    for cls in rare_classes:
        cls_idx = np.where(y_train == cls)[0]
        current_count = len(cls_idx)
        needed = 100 - current_count
        
        if needed > 0 and current_count > 0:
            # Random choice with replacement
            chosen_idx = rng.choice(cls_idx, size=needed, replace=True)
            X_extra.append(X_train[chosen_idx])
            y_extra.append(y_train[chosen_idx])
            
    if X_extra:
        X_train_os = np.concatenate([X_train] + X_extra, axis=0)
        y_train_os = np.concatenate([y_train] + y_extra, axis=0)
    else:
        X_train_os = X_train
        y_train_os = y_train
        
    # Shuffle train set after concatenating
    shuffle_idx = rng.permutation(len(y_train_os))
    X_train_os = X_train_os[shuffle_idx]
    y_train_os = y_train_os[shuffle_idx]
    
    print(f"[*] Train size after Oversampling: {X_train_os.shape[0]} samples (increased by {X_train_os.shape[0] - X_train.shape[0]})")
    
    model = build_reference_cnn(len(le.classes_))
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train_os, y_train_os,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )
    
    model_path = MODEL_DIR / "cnn_oversampled.keras"
    model.save(model_path)
    print(f"[*] Saved model to {model_path}")
    
    evaluate_and_log(model, X_val, y_val, le, "cnn_oversampled", "articleType")

# ---------------------------------------------------------------------------
# Experiment 3: Merging Rare Classes into subCategory
# ---------------------------------------------------------------------------
def run_experiment_3():
    print("\n" + "="*50)
    print("EXPERIMENT 3: Change Target to subCategory (cnn_merged)")
    print("="*50)
    
    # Load data again but with target_column="subCategory"
    X_train, X_val, y_train, y_val, le = load_data("subCategory")
    
    model = build_reference_cnn(len(le.classes_))
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )
    
    model_path = MODEL_DIR / "cnn_merged.keras"
    model.save(model_path)
    print(f"[*] Saved model to {model_path}")
    
    evaluate_and_log(model, X_val, y_val, le, "cnn_merged", "subCategory")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting 3 Class Imbalance experiments (Member 2)...\n")
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    print("\nCOMPLETED. Please check outputs/results.csv and outputs/confusion_*.png")
