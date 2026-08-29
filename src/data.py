#The single source of truth for loading, cleaning and splitting the data.
#Every task imports from here. Nobody writes their own train/val split - if two
#people split differently, the results table compares models on different data
#and the comparison section of the report is invalid.
#
#What belongs here: anything that is a property of the dataset - junk columns,
#missing images, grayscale files, normalisation, the train/val assignment.
#
#What does NOT belong here: anything that is a modelling decision you might
#want to undo and compare against - merging rare articleType classes into
#subCategory, collapsing usage's micro-classes into "Other", class weighting,
#augmentation. Those live in the task notebooks, because the comparison is the
#experiment.
#
#Typical use
#   from src.data import get_split, load_metadata
#   Xtr, Xva, ytr, yva, le = get_split('articleType')
#   Xtr, Xva, ytr, yva, le = get_split('season')
#   X, meta                = get_images_only()      # Task 4, unsupervised

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from src.config import (
    SEED, TRAIN_CSV, TRAIN_IMAGES, TEST_IMAGES, SAMPLE_SUBMISSION,
    CACHE_DIR, MODEL_DIR, IMG_SHAPE, IMG_WIDTH, IMG_HEIGHT, STRATIFY_ON, VAL_SIZE,
)

_SPLIT_COL = '_split'


#Metadata
def load_metadata(verbose=True):
    #load styles_train.csv, clean it, and attach the fixed train/val split.
    #missing *labels* are deliberately not dropped here - different targets have
    #different missing counts (season 20, usage 72, articleType and gender 0), so
    #dropping globally would needlessly discard rows from tasks that were fine.
    #get_split() filters per target instead.
    df = pd.read_csv(TRAIN_CSV)
    n_raw = len(df)

    #two trailing columns of empty strings, an artefact of stray commas in the CSV
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])

    #5 CSV rows have no image file; 1 image file has no CSV row. Keep the intersection.
    on_disk = {p.stem for p in TRAIN_IMAGES.glob('*.jpg')}
    df = df[df['id'].astype(str).isin(on_disk)].reset_index(drop=True)

    df = _attach_split(df)

    if verbose:
        print('metadata:', n_raw, 'rows in CSV ->', len(df), 'usable',
              '(', n_raw - len(df), 'dropped: no matching image)')
        print('         ', (df[_SPLIT_COL] == 'train').sum(), 'train /',
              (df[_SPLIT_COL] == 'val').sum(), 'val')
    return df


def _attach_split(df):
    #assign every row to train or val, once, for all tasks. Stratified on
    #articleType. Seven classes have a single example, which sklearn cannot
    #stratify - those rows are routed to train and are therefore unevaluable.
    #That is a property of the data, not a bug; report it.
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
    df[_SPLIT_COL] = 'train'          #single-example classes default to train
    df.loc[val_idx, _SPLIT_COL] = 'val'
    return df


#Images
def _build_cache(ids, image_dir, cache_path):
    #decode every image once into a single uint8 array and memoise it to disk.
    #~38.6k x 80 x 60 x 3 uint8 is about 550 MB, which fits comfortably in RAM.
    #Paying this cost once turns every later experiment from minutes into seconds.
    print('building image cache ->', cache_path.name, '(', len(ids), 'images, one-off)')
    arr = np.empty((len(ids), *IMG_SHAPE), dtype=np.uint8)
    n_grey, n_resized = 0, 0
    for i, img_id in enumerate(ids):
        with Image.open(image_dir / f'{img_id}.jpg') as im:
            #~0.9% of files are grayscale (PIL mode "L"). Without this convert
            #they come back 2-D and batching crashes on a channel mismatch.
            if im.mode != 'RGB':
                im = im.convert('RGB')
                n_grey += 1
            #most images are a uniform 60x80, but a handful are not (53x80,
            #60x60, 60x77, ...). Resize the stragglers so the array is regular.
            if im.size != (IMG_WIDTH, IMG_HEIGHT):
                im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
                n_resized += 1
            arr[i] = np.asarray(im)
        if (i + 1) % 10000 == 0:
            print(' ', i + 1, '/', len(ids))
    print('  done:', n_grey, 'grayscale converted,', n_resized, 'resized to',
          f'{IMG_WIDTH}x{IMG_HEIGHT}')
    np.save(cache_path, arr)
    return arr


def load_images(ids, split_name='train'):
    #return decoded uint8 images for ids, building the cache on first call
    ids = [str(i) for i in ids]
    image_dir = TRAIN_IMAGES if split_name == 'train' else TEST_IMAGES
    cache_path = CACHE_DIR / f'images_{split_name}.npy'
    index_path = CACHE_DIR / f'index_{split_name}.npy'

    if cache_path.exists() and index_path.exists():
        cached_ids = np.load(index_path, allow_pickle=True)
        arr = np.load(cache_path, mmap_mode='r')
        lookup = {img_id: i for i, img_id in enumerate(cached_ids)}
        return np.stack([arr[lookup[i]] for i in ids])

    #first call: cache every image in the directory, then select
    all_ids = sorted(p.stem for p in image_dir.glob('*.jpg'))
    arr = _build_cache(all_ids, image_dir, cache_path)
    np.save(index_path, np.array(all_ids, dtype=object))
    lookup = {img_id: i for i, img_id in enumerate(all_ids)}
    return np.stack([arr[lookup[i]] for i in ids])


def normalise(x, stats=None):
    #scale to [0,1] and standardise per channel. Channel statistics are
    #computed on the training split only and passed back, so the same numbers
    #can be reapplied to val and test - computing them on all the data would
    #leak validation information into training.
    x = x.astype(np.float32) / 255.0
    if stats is None:
        #dtype=float64 matters: accumulating ~24M float32 values per channel
        #saturates the running sum (adding 0.8 to 17,000,000 is a no-op at
        #float32 precision) and silently returns an identical wrong mean for
        #every channel. Verified against this dataset.
        mean = x.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
        std = x.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32) + 1e-7
        stats = (mean, std)
    mean, std = stats
    return (x - mean) / std, stats


#The function everyone actually calls
def get_split(target, normalised=True, verbose=True):
    #return (X_train, X_val, y_train, y_val, label_encoder) for one target.
    #the train/val row assignment is identical for every target - only rows
    #whose label for *this* target is missing are removed. That is what makes
    #results comparable across tasks and what lets the multi-task model share
    #a split.
    #
    #the fitted LabelEncoder is saved to models/. Prediction time must load it,
    #never refit - refitting silently reorders the classes and produces a
    #plausible-looking, completely wrong submission file.
    df = load_metadata(verbose=False)

    n_before = len(df)
    df = df[df[target].notna()].reset_index(drop=True)
    if verbose and n_before != len(df):
        print(target, ': dropped', n_before - len(df), 'rows with a missing label')

    le = LabelEncoder().fit(df[target])
    joblib.dump(le, MODEL_DIR / f'label_encoder_{target}.joblib')

    tr = df[df[_SPLIT_COL] == 'train']
    va = df[df[_SPLIT_COL] == 'val']

    X_train = load_images(tr['id'], 'train')
    X_val = load_images(va['id'], 'train')
    y_train = le.transform(tr[target])
    y_val = le.transform(va[target])

    if normalised:
        X_train, stats = normalise(X_train)
        X_val, _ = normalise(X_val, stats)
        joblib.dump(stats, MODEL_DIR / 'channel_stats.joblib')

    if verbose:
        baseline = pd.Series(y_train).value_counts(normalize=True).iloc[0]
        print(target, ':', len(le.classes_), 'classes | train', X_train.shape[0],
              '/ val', X_val.shape[0], '| majority baseline %.3f' % baseline)
    return X_train, X_val, y_train, y_val, le


def get_images_only(normalised=True):
    #all training images plus their metadata, no target filtering. For Task 4:
    #rungs 1-3 of the retrieval ladder are unsupervised, and the labels are
    #needed only afterwards to compute Precision@K.
    df = load_metadata(verbose=False)
    X = load_images(df['id'], 'train')
    if normalised:
        X, _ = normalise(X)
    return X, df


def load_test_images(normalised=True):
    #the 5,829 unlabelled test images, in the order styles_prediction.csv
    #lists them - so predictions line up row-for-row with the submission file
    sub = pd.read_csv(SAMPLE_SUBMISSION)
    X = load_images(sub['id'], 'test')
    if normalised:
        stats = joblib.load(MODEL_DIR / 'channel_stats.joblib')
        X, _ = normalise(X, stats)
    return X, sub


if __name__ == '__main__':
    meta = load_metadata()
    print('\nper-target missing labels:')
    for t in ['gender', 'articleType', 'season', 'usage']:
        print(' ', t, meta[t].isna().sum(), 'missing,', meta[t].nunique(), 'classes')
