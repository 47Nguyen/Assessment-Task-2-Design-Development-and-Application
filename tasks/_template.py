"""
TEMPLATE - copy this into your task folder and rename it to train.py
    cp tasks/_template.py tasks/task2_season/train.py
HOW TO RUN IT - from the project root folder, use -m and dots, no ".py":

    python -m tasks.task2_season.train        <- correct
    python tasks/task2_season/train.py        <- fails: "No module named 'src'"

The -m form tells Python to treat the project folder as the starting point,
which is what lets `from src...` find our shared code. It's the only reason
the command looks slightly unusual.

"""

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
from src.data import get_split
# import data 
folder_path = ""


# ---------------------------------------------------------------------------
# 2. Choose your target
# ---------------------------------------------------------------------------
target_value = "articleType"     # <-- CHANGE THIS
# Options: "articleType"  (Task 1)
#          "season"       (Task 2)
#          "gender"       (Task 3)
#          "usage"        (Task 3)


# ---------------------------------------------------------------------------
# 3. Load the data
# ---------------------------------------------------------------------------
# get_split() does all the cleaning and gives everyone the same train/val split.
# Don't write your own split - if two people split differently, our results
# can't be compared in the 
# The first run builds an image cache (~550 MB, a few minutes).
# Every run after that is fast.

X_train, X_val, y_train, y_val, label_encoder = get_split(target_value)

print(f"\nTarget: {target_value}")
print(f"Train images: {X_train.shape}")
print(f"Val images:   {X_val.shape}")
print(f"Classes:      {len(label_encoder.classes_)}")
print(f"Class names:  {list(label_encoder.classes_)[:10]}")


# ---------------------------------------------------------------------------
# 4. Your work starts here
# ---------------------------------------------------------------------------
