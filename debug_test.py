import numpy as np

# Load cache
X_val = np.load("cache/task1_articleType_X_val.npy")
y_val = np.load("cache/task1_articleType_y_val.npy")

import joblib
le = joblib.load("models/label_encoder_articleType.joblib")

idx = 7221
print(f"X_val shape: {X_val.shape}")
print(f"Cache Label {idx}: {le.classes_[y_val[idx]]}")

from tasks.task1_article_type.test_model import load_metadata_self_contained
_, val_df = load_metadata_self_contained()
print(f"Metadata ID {idx}: {val_df.iloc[idx]['id']}")
print(f"Metadata Label {idx}: {val_df.iloc[idx]['articleType']}")

