#Shared paths and constants - every module and notebook imports from here so
#nobody hard-codes a path that only works on their own laptop
from pathlib import Path

#reproducibility
SEED = 42

#paths - resolved relative to the repo root, so this works regardless of the
#current working directory a notebook happens to be launched from
ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / 'A2_FashionDataset' / 'FashionDataset'
TRAIN_CSV = DATA_ROOT / 'train' / 'styles_train.csv'
TRAIN_IMAGES = DATA_ROOT / 'train' / 'images_train'
TEST_IMAGES = DATA_ROOT / 'test' / 'images_test'
SAMPLE_SUBMISSION = DATA_ROOT / 'test' / 'styles_prediction.csv'

CACHE_DIR = ROOT / 'cache'     #decoded image arrays (.npy) - gitignored
MODEL_DIR = ROOT / 'models'    #saved weights + label encoders - gitignored
OUTPUT_DIR = ROOT / 'outputs'  #figures, results tables, predictions

for _d in (CACHE_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)

#data facts - images are a uniform 60x80 RGB, numpy indexes rows first so
#arrays are (height, width, channels) = (80, 60, 3)
IMG_WIDTH = 60
IMG_HEIGHT = 80
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

#the four prediction targets, in the column order styles_prediction.csv expects
TARGETS = ['gender', 'articleType', 'season', 'usage']

#split stratified on this target - it is the finest-grained label (125 classes),
#so balancing it approximately balances the others through their correlation
STRATIFY_ON = 'articleType'
VAL_SIZE = 0.2
