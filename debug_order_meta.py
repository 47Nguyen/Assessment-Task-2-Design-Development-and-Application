import numpy as np
import pandas as pd
import joblib

# Load Label Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from tasks.task1_article_type.test_model import load_metadata_self_contained, SEED
train_df, val_df = load_metadata_self_contained()
df_all = pd.concat([train_df, val_df])
le.fit(df_all['articleType'])
df_val_y = le.transform(val_df['articleType'])

# Load Cache Array
y_val = np.load("cache/task1_articleType_y_val.npy")

print(f"Validation y label lengths: cache={len(y_val)} vs df={len(df_val_y)}")

# Check first 10
print("\nFirst 10 y_val entries (from dataframe vs cache)")
for i in range(10):
    cache_lbl = le.classes_[y_val[i]]
    df_lbl = val_df.iloc[i]['articleType']
    print(f"Index {i:2d} | Cache: {cache_lbl:<20} | DF: {df_lbl:<20} | Match: {cache_lbl == df_lbl}")
    
# Now find mismatched ones
mismatches = 0
for i in range(len(y_val)):
    cache_lbl = le.classes_[y_val[i]]
    df_lbl = val_df.iloc[i]['articleType']
    if cache_lbl != df_lbl:
        mismatches += 1
        
print(f"\nTotal Mismatches: {mismatches} / {len(y_val)}")
