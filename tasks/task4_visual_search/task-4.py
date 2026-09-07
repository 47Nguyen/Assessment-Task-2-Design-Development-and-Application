import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, UnitNormalization, Dropout
from sklearn.model_selection import train_test_split
from pathlib import Path
tf.get_logger().setLevel('ERROR')

# Seed so the triplet sampling gives the same results every run
np.random.seed(42)
tf.random.set_seed(42)

# https://keras.io/examples/vision/siamese_network/
# https://www.datacamp.com/tutorial/cnn-tensorflow-python
# https://pyimagesearch.com/2023/02/13/building-a-dataset-for-triplet-loss-with-keras-and-tensorflow/

## Load datas
train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'

df_train = pd.read_csv(train_path)

# Setup image path for products
df_train['path'] = './A2_FashionDataset/FashionDataset/train/images_train' + '/' + df_train['id'].astype(str) + '.jpg'


# Preprocess dataset
df_train = df_train.drop(columns=['Unnamed: 10','Unnamed: 11'])

# 5 rows in the csv point at images that are not in the folder, drop them``
df_train = df_train[df_train['path'].apply(os.path.exists)]

# print(df_train)
# count = df_train['masterCategory'].nunique()


# The images are 60 wide by 80 tall.
# tf.image.resize takes (height, width) so it has to be this way round.
target_shape = (80,60)


## Preprcoess iamge
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # convert the image data type from uint8 to float32 and then resize
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

# Print image - testing

def print_image(index):
    plt.figure(dpi = 28)
    image = preprocess_image(df_train['path'][index])
    plt.imshow(image)
    plt.show()


# Take paths from anchors + ref and then load -> image
def preprocess_triplets(anchor, reference, disimilar):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    return (
        preprocess_image(anchor),
        preprocess_image(reference),
        preprocess_image(disimilar),
    )

## Define our anchor
def anchor_references(df):
    """ 
    Anchor: a random sample image.
    reference: a different image that's "similar" to the anchor by your chosen definition.
    disimilar: an image that's "dissimilar" by that same definition.
    """
    anchors, references, disimilars = [], [], []
    grouped_article = df.groupby('articleType')['path'].apply(list).to_dict()
    types = list(grouped_article.keys())
    for article_type, paths in grouped_article.items():
        if len(paths) < 2:  # It because there nothing to be similar there are only 2 items
            continue
        for i in range(len(paths)): # For every image treat it as an anchor once
            anchor = paths[i]
            reference = np.random.choice([
                p for p in paths if p != anchor
            ])
            neg_type = np.random.choice([t for t in types if t != article_type])
            disimilar = np.random.choice(grouped_article[neg_type])
            anchors.append(anchor)
            references.append(reference)
            disimilars.append(disimilar)
    return anchors, references, disimilars

## Embedding network (the "twin" CNN shared by anchor/reference/disimilar)
def embedding_model(embed_dim = 64, dense_units = 64, filters = (64, 64, 64),
                    dropout = 0.0, normalise = True):
    """
    The twin CNN shared by anchor/reference/disimilar.

    The defaults reproduce the original network exactly, so calling this with
    no arguments still gives the model we already trained and the baseline row
    of the tuning table stays honest.

    filters caps out at four blocks. 80x60 through valid 3x3 convs and 2x2
    pools goes 39x29 -> 18x13 -> 8x5 -> 3x1, and a fifth block has nothing
    left to convolve over.
    """
    INPUT_SHAPE = (80,60,3)

    model = Sequential()

    for block, n_filters in enumerate(filters):
        if block == 0:
            model.add(Conv2D(n_filters, (3,3), input_shape = INPUT_SHAPE))
        else:
            model.add(Conv2D(n_filters, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(dense_units))
    model.add(Activation("relu"))

    if dropout > 0.0:
        model.add(Dropout(dropout))

    model.add(Dense(embed_dim))  # embedding vector, no activation

    if normalise:
        # Put the embedding on the unit sphere, otherwise the model can beat the
        # margin by making the vectors bigger instead of separating the classes
        model.add(UnitNormalization())

    return model

def split_data(df):
    """ 
    We have to build our own split function. 
    Reason is because this task focuses on finding top K of results.
    It not looking to predicts the a target vaulue.
    """
    
    counts = df['articleType'].value_counts() # Total up the number of each the article type.
    keep = counts[counts >= 2].index  # Look for any articleType with total counts >=2 
    df = df[df['articleType'].isin(keep)]  # Filter out the dataframe, we only keep values where the articleType counts >= 2

    train_df, val_df = train_test_split(df,test_size=0.2,stratify=df['articleType'], random_state=42)
    
    return train_df, val_df

def list_to_dataset(df):
    anchors_list, ref_list, dis_list = anchor_references(df)
    
    # List to dataset
    to_dataset = tf.data.Dataset.from_tensor_slices((anchors_list, ref_list, dis_list))
    
    #Path to image
    dataset = to_dataset.map(preprocess_triplets)
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

## Triplet loss: we want the anchor close to the reference and far from the disimilar
def triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin = 0.5):
    d_pos = tf.reduce_sum(tf.square(anchor_emb - reference_emb), axis = -1)
    d_neg = tf.reduce_sum(tf.square(anchor_emb - disimilar_emb), axis = -1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))
## One batch of learning, weights get updated here
def train_step(model, optimizer, anchor, reference, disimilar, margin = 0.5):
    with tf.GradientTape() as tape:
        anchor_emb = model(anchor, training = True)
        reference_emb = model(reference, training = True)
        disimilar_emb = model(disimilar, training = True)
        loss = triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss

## Same maths but no weight update, we are only measuring here
def val_step(model, anchor, reference, disimilar, margin = 0.5):
    anchor_emb = model(anchor, training = False)
    reference_emb = model(reference, training = False)
    disimilar_emb = model(disimilar, training = False)
    return triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin)

def train_model(model, train_dataset, val_dataset, epochs = 4, learning_rate = 1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        train_losses = []
        for anchor, reference, disimilar in train_dataset:
            loss = train_step(model, optimizer, anchor, reference, disimilar)
            train_losses.append(loss.numpy())

        val_losses = []
        for anchor, reference, disimilar in val_dataset:
            loss = val_step(model, anchor, reference, disimilar)
            val_losses.append(loss.numpy())

        print(f"epoch {epoch + 1}/{epochs} - train loss: {np.mean(train_losses):.4f} - val loss: {np.mean(val_losses):.4f}")

    return model


## Run every catalogue image through the model once, this is what we search over
def build_index(model, df, batch_size = 32):
    paths = df['path'].tolist()

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    embeddings = model.predict(dataset, verbose = 0)
    return embeddings


## Embed the query then return the k closest images from the index
def search(model, query_path, index, df, k = 5):
    query_image = preprocess_image(query_path)
    query_emb = model(tf.expand_dims(query_image, axis = 0), training = False).numpy()

    distances = np.sum(np.square(index - query_emb), axis = 1)
    nearest = np.argsort(distances)[:k]

    return df.iloc[nearest], distances[nearest]

## Show the query next to what we retrieved
def show_results(query_path, results):
    plt.figure(figsize = (12, 3))

    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(preprocess_image(query_path))
    plt.title("query")
    plt.axis('off')

    for i in range(len(results)):
        row = results.iloc[i]
        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(preprocess_image(row['path']))
        plt.title(row['articleType'], fontsize = 8)
        plt.axis('off')

    plt.savefig('outputs/task_4/task4_query_grid.png', bbox_inches = 'tight')
    plt.close()

## ===========================================================================
## FINE TUNING
## ---------------------------------------------------------------------------
## House rule from the Task 1 README: change one thing at a time, and log every
## run under its own name so the rows become the report's tuning table.
##
## Nothing below runs on its own. Flip RUN_FINE_TUNE at the bottom of the file
## to start a sweep. The original pipeline above is left exactly as it was.
##
## https://arxiv.org/abs/1503.03832  FaceNet, where semi-hard mining comes from
## https://arxiv.org/abs/1703.07737  In Defense of the Triplet Loss
## https://jmlr.org/papers/v13/bergstra12a.html  why random search beats grid
## ===========================================================================

RESULTS_FILE = 'outputs/task_4/results_task4.csv'
TUNED_MODEL_FILE = 'models/task_4/embedding_visual_search_tuned.keras'
TUNED_INDEX_FILE = 'models/task_4/embeddings_task4_tuned.npy'


## Every image through the model once. Row i of the result is the embedding of
## paths[i], which is what lets us map a neighbour back to its dataframe row.
def embed_paths(model, paths, batch_size = 32):
    """
    Embed a list of image paths in order.
    """
    dataset = tf.data.Dataset.from_tensor_slices(list(paths))
    dataset = dataset.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset, verbose = 0)


## Distances from every query to every catalogue image, in chunks
def squared_distances(queries, index, chunk = 256):
    """
    Squared L2 distance from each query to each indexed image.

    Done a block of queries at a time so we never build the full
    (queries x index x dims) array, which would be tens of gigabytes.
    """
    index_sq = np.sum(np.square(index), axis = 1)
    distances = np.empty((len(queries), len(index)), dtype = np.float32)

    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        block_sq = np.sum(np.square(block), axis = 1)[:, None]
        distances[start:start + chunk] = block_sq + index_sq[None, :] - 2.0 * (block @ index.T)

    # floating point error can push a distance a hair below zero
    return np.maximum(distances, 0.0)


## The evaluation framework. precision@5 on its own cannot tell a good ranking
## from a lucky one, because it ignores the order inside the top 5.
def evaluate_retrieval(model, index, index_df, query_df, n_queries = 500,
                       k_list = (1, 5, 10), map_k = 10, seed = 42):
    """
    Precision@K and mAP@K over a fixed sample of validation queries.

    mAP is rank sensitive, so it is the number fine_tune ranks configurations
    on. The query sample is seeded, so every configuration is scored on exactly
    the same queries and the comparison is fair.
    """
    queries = query_df.sample(n = min(n_queries, len(query_df)), random_state = seed)

    query_embeddings = embed_paths(model, queries['path'].tolist())
    distances = squared_distances(query_embeddings, index)

    index_types = index_df['articleType'].to_numpy()
    query_types = queries['articleType'].to_numpy()

    # how many relevant items exist at all, so mAP is not punished for a class
    # that simply has fewer than map_k members in the catalogue
    type_counts = index_df['articleType'].value_counts()

    top_k = max(max(k_list), map_k)
    ordering = np.argsort(distances, axis = 1)[:, :top_k]

    precisions = {k: [] for k in k_list}
    average_precisions = []
    nearest_distances = []

    for row in range(len(queries)):
        retrieved = index_types[ordering[row]]
        relevant = retrieved == query_types[row]

        for k in k_list:
            precisions[k].append(relevant[:k].mean())

        # average precision: credit a hit by how high up the ranking it landed
        hits = 0
        score = 0.0
        for rank in range(map_k):
            if relevant[rank]:
                hits += 1
                score += hits / (rank + 1)
        possible = min(type_counts.get(query_types[row], 0), map_k)
        average_precisions.append(score / possible if possible else 0.0)

        nearest_distances.append(distances[row, ordering[row, 0]])

    metrics = {f'p_at_{k}': float(np.mean(precisions[k])) for k in k_list}
    metrics[f'map_at_{map_k}'] = float(np.mean(average_precisions))
    metrics['mean_nn_distance'] = float(np.mean(nearest_distances))
    metrics['n_queries'] = len(queries)
    return metrics


## The three numbers that would have caught our collapse on epoch 1
def triplet_diagnostics(model, dataset, margin):
    """
    Mean positive distance, mean negative distance, and the share of triplets
    still producing a gradient.

    active_fraction is the one to watch. If it falls towards zero the model has
    stopped learning whatever the loss says, because every triplet is already
    outside the margin and max(..., 0) is clipping them all to nothing.
    """
    d_pos_all, d_neg_all, active = [], [], []

    for anchor, reference, disimilar in dataset:
        anchor_emb = model(anchor, training = False)
        reference_emb = model(reference, training = False)
        disimilar_emb = model(disimilar, training = False)

        d_pos = tf.reduce_sum(tf.square(anchor_emb - reference_emb), axis = -1)
        d_neg = tf.reduce_sum(tf.square(anchor_emb - disimilar_emb), axis = -1)

        d_pos_all.append(d_pos.numpy())
        d_neg_all.append(d_neg.numpy())
        active.append((d_pos - d_neg + margin > 0).numpy())

    return {
        'mean_d_pos': float(np.mean(np.concatenate(d_pos_all))),
        'mean_d_neg': float(np.mean(np.concatenate(d_neg_all))),
        'active_fraction': float(np.mean(np.concatenate(active))),
    }


## Anchor/positive pairs, by position so they line up with the embedding rows
def positive_pairs(types):
    """
    Every image is an anchor once, paired with a random different image of the
    same article type. Types with only one image cannot form a pair.
    """
    positions_by_type = {}
    for position, article_type in enumerate(types):
        positions_by_type.setdefault(article_type, []).append(position)

    anchors, references = [], []
    for positions in positions_by_type.values():
        if len(positions) < 2:
            continue
        positions = np.array(positions)
        for local, position in enumerate(positions):
            # draw from the other len-1 members, then step over the anchor
            # itself, so an image is never returned as its own positive
            pick = np.random.randint(len(positions) - 1)
            if pick >= local:
                pick += 1
            anchors.append(position)
            references.append(positions[pick])

    return np.array(anchors), np.array(references)


def random_negatives(types, anchor_positions):
    """
    The original sampler: pick a different article type uniformly, then a
    random image from inside it.

    Kept exactly as anchor_references had it so the baseline row stays
    comparable with the precision@5 we have already reported.
    """
    positions_by_type = {}
    for position, article_type in enumerate(types):
        positions_by_type.setdefault(article_type, []).append(position)
    positions_by_type = {t: np.array(p) for t, p in positions_by_type.items()}
    all_types = np.array(list(positions_by_type.keys()))

    negatives = np.empty(len(anchor_positions), dtype = np.int64)
    for i, position in enumerate(anchor_positions):
        other_types = all_types[all_types != types[position]]
        neg_type = np.random.choice(other_types)
        negatives[i] = np.random.choice(positions_by_type[neg_type])
    return negatives


def mined_negatives(embeddings, types, anchor_positions, reference_positions,
                    sampler, margin, pool_size = 50, chunk = 512):
    """
    Semi-hard and hardest negative mining.

    Scoring all ~30k candidates against all ~30k anchors is 900M distances per
    epoch, so we score a random pool of pool_size candidates per anchor instead.
    That is an approximation and the report should say so.

    Semi-hard keeps a negative that is further away than the positive but still
    inside the margin, which is the only band that produces a gradient. Hardest
    takes the closest negative and is the variant known to collapse - we run it
    so the report can show that happening rather than assert it.
    """
    n_total = len(types)
    negatives = np.empty(len(anchor_positions), dtype = np.int64)

    for start in range(0, len(anchor_positions), chunk):
        stop = min(start + chunk, len(anchor_positions))
        anchors = anchor_positions[start:stop]
        references = reference_positions[start:stop]

        anchor_emb = embeddings[anchors]
        d_pos = np.sum(np.square(anchor_emb - embeddings[references]), axis = 1)

        pool = np.random.randint(n_total, size = (len(anchors), pool_size))
        d_neg = np.sum(np.square(anchor_emb[:, None, :] - embeddings[pool]), axis = 2)

        # a candidate of the same article type is not a negative at all
        same_type = types[pool] == types[anchors][:, None]
        d_neg[same_type] = np.inf

        # a pool that happened to draw nothing but same-type images leaves us
        # nothing to mine. Rare on the full catalogue, but Tshirts is 18% of it
        # and more once subsample_catalogue has trimmed to the biggest types.
        no_candidate = same_type.all(axis = 1)

        if sampler == 'hardest':
            choice = d_neg.argmin(axis = 1)
        else:
            in_band = (d_neg > d_pos[:, None]) & (d_neg < d_pos[:, None] + margin)
            banded = np.where(in_band, d_neg, np.inf)
            choice = banded.argmin(axis = 1)

            # nothing in the band for this anchor, fall back to the hardest
            # candidate the pool did offer
            empty = ~in_band.any(axis = 1)
            if empty.any():
                choice[empty] = d_neg[empty].argmin(axis = 1)

        chosen = pool[np.arange(len(anchors)), choice]

        # those anchors get a plain random negative instead of a mined one
        if no_candidate.any():
            chosen[no_candidate] = random_negatives(types, anchors[no_candidate])

        negatives[start:stop] = chosen

    return negatives


def build_triplets(df, sampler = 'random', embeddings = None, margin = 0.5, pool_size = 50):
    """
    A fresh set of triplets for one epoch.

    The original code built these once before training and replayed the same
    30k triplets every epoch. Resampling costs nothing and gives the model far
    more to learn from.
    """
    paths = df['path'].to_numpy()
    types = df['articleType'].to_numpy()

    anchors, references = positive_pairs(types)

    if sampler == 'random' or embeddings is None:
        disimilars = random_negatives(types, anchors)
    else:
        disimilars = mined_negatives(embeddings, types, anchors, references,
                                     sampler, margin, pool_size)

    return list(paths[anchors]), list(paths[references]), list(paths[disimilars])


def augment_image(image):
    """
    Light augmentation only.

    A horizontal flip is safe for garments, and brightness/contrast jitter
    stands in for photo variation. We deliberately do not shift hue, because
    baseColour is part of what "similar" means in this task - recolouring the
    image would be teaching the model the wrong thing.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    return tf.clip_by_value(image, 0.0, 1.0)


def triplets_to_dataset(anchors, references, disimilars, batch_size = 32, augment = False):
    """
    Same job as list_to_dataset, but takes triplets that were built outside so
    they can be re-mined between epochs, and can switch augmentation on.
    """
    dataset = tf.data.Dataset.from_tensor_slices((anchors, references, disimilars))
    dataset = dataset.map(preprocess_triplets, num_parallel_calls = tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(
            lambda a, r, d: (augment_image(a), augment_image(r), augment_image(d)),
            num_parallel_calls = tf.data.AUTOTUNE)

    return dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


## One training run for one configuration. Kept separate from train_model above
## so the pipeline we already have keeps working untouched - once the sweep
## picks a winner, this replaces it and train_model can go.
def train_tuned_model(model, train_df, val_df, config, verbose = True):
    """
    Train one configuration, keeping the best epoch rather than the last.

    Early stopping watches val loss, which is fair inside a single run because
    margin is fixed there. Across runs it is not - margin sits inside the loss,
    so a run with margin 0.1 always looks better than one with margin 1.0 - and
    that is why fine_tune ranks configurations on mAP@10 instead.
    """
    optimizer = tf.keras.optimizers.Adam(config['learning_rate'])

    # one fixed set of validation triplets, so epoch to epoch loss is comparable
    val_triplets = build_triplets(val_df, 'random')
    val_dataset = triplets_to_dataset(*val_triplets, batch_size = config['batch_size'])

    best_loss = np.inf
    best_weights = model.get_weights()
    best_epoch = 1
    waited = 0
    history = []

    for epoch in range(config['epochs']):
        # mining needs embeddings from the current weights, so the index is
        # rebuilt each epoch. The first warmup_epochs use random negatives,
        # because embeddings from an untrained model are not worth mining.
        embeddings = None
        if config['sampler'] != 'random' and epoch >= config['warmup_epochs']:
            embeddings = embed_paths(model, train_df['path'].tolist(), config['batch_size'])

        train_triplets = build_triplets(train_df, config['sampler'], embeddings,
                                        config['margin'], config['pool_size'])
        train_dataset = triplets_to_dataset(*train_triplets,
                                            batch_size = config['batch_size'],
                                            augment = config['augment'])

        train_losses = []
        for anchor, reference, disimilar in train_dataset:
            loss = train_step(model, optimizer, anchor, reference, disimilar, config['margin'])
            train_losses.append(loss.numpy())

        val_losses = []
        for anchor, reference, disimilar in val_dataset:
            val_losses.append(val_step(model, anchor, reference, disimilar, config['margin']).numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        diagnostics = triplet_diagnostics(model, val_dataset, config['margin'])
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_loss': val_loss, **diagnostics})

        if verbose:
            print(f"  epoch {epoch + 1}/{config['epochs']} - train loss: {train_loss:.4f} "
                  f"- val loss: {val_loss:.4f} - active: {diagnostics['active_fraction']:.2f} "
                  f"- d_pos: {diagnostics['mean_d_pos']:.4f} - d_neg: {diagnostics['mean_d_neg']:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.get_weights()
            best_epoch = epoch + 1
            waited = 0
        else:
            waited += 1
            if config['patience'] and waited >= config['patience']:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1}, best was epoch {best_epoch}")
                break

    # the original script saved whatever the last epoch happened to produce
    model.set_weights(best_weights)
    return model, pd.DataFrame(history), best_epoch


def run_experiment(config, train_df, val_df, n_queries = 500, verbose = True):
    """
    Train one configuration end to end and hand back a single results row.
    """
    config = {**DEFAULT_CONFIG, **config}

    # reseed per run so two configurations differ by the thing we changed and
    # not by which triplets they happened to draw
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])

    if verbose:
        print(f"\n=== {config['name']} - {config['notes']} ===")

    started = time.time()

    model = embedding_model(embed_dim = config['embed_dim'],
                            dense_units = config['dense_units'],
                            filters = config['filters'],
                            dropout = config['dropout'],
                            normalise = config['normalise'])

    model, history, best_epoch = train_tuned_model(model, train_df, val_df, config, verbose)
    train_seconds = time.time() - started

    index = embed_paths(model, train_df['path'].tolist(), config['batch_size'])
    metrics = evaluate_retrieval(model, index, train_df, val_df, n_queries = n_queries)

    best = history.iloc[best_epoch - 1]
    row = {
        'name': config['name'],
        'sampler': config['sampler'],
        'margin': config['margin'],
        'embed_dim': config['embed_dim'],
        'filters': str(config['filters']),
        'dropout': config['dropout'],
        'normalise': config['normalise'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'augment': config['augment'],
        'epochs': config['epochs'],
        'best_epoch': best_epoch,
        'seed': config['seed'],
        'train_loss': round(float(best['train_loss']), 4),
        'val_loss': round(float(best['val_loss']), 4),
        'active_fraction': round(float(best['active_fraction']), 4),
        'mean_d_pos': round(float(best['mean_d_pos']), 4),
        'mean_d_neg': round(float(best['mean_d_neg']), 4),
        'train_seconds': round(train_seconds, 1),
        'index_mb': round(index.nbytes / 1e6, 2),
        'notes': config['notes'],
        **metrics,
    }
    return model, row, history, index


def fine_tune(train_df, val_df, experiments = None, results_file = RESULTS_FILE,
              n_queries = 500, save_best = True):
    """
    Run every configuration in turn and return the table sorted by mAP@10.

    One row per run is appended to results_file as each run finishes, so a
    crash halfway through a sweep does not lose the runs already done. Each run
    also drops its per-epoch history next to it, which is what the learning
    curve and the collapse figure in the report are drawn from.

    The winner is saved alongside the original model, never over the top of it.
    """
    experiments = experiments if experiments is not None else TUNING_EXPERIMENTS

    os.makedirs('outputs/task_4', exist_ok = True)
    os.makedirs('models/task_4', exist_ok = True)

    rows = []
    best_score = -np.inf

    for config in experiments:
        model, row, history, index = run_experiment(config, train_df, val_df, n_queries)
        rows.append(row)

        pd.DataFrame([row]).to_csv(results_file, mode = 'a',
                                   header = not Path(results_file).exists(),
                                   index = False)
        history.to_csv(f"outputs/task_4/history_{row['name']}.csv", index = False)

        print(f"{row['name']}: P@1 {row['p_at_1']:.3f}  P@5 {row['p_at_5']:.3f}  "
              f"P@10 {row['p_at_10']:.3f}  mAP@10 {row['map_at_10']:.3f}  "
              f"active {row['active_fraction']:.2f}")

        if save_best and row['map_at_10'] > best_score:
            best_score = row['map_at_10']
            model.save(TUNED_MODEL_FILE)
            np.save(TUNED_INDEX_FILE, index)

    return pd.DataFrame(rows).sort_values('map_at_10', ascending = False)


def subsample_catalogue(df, n_types = 30, seed = 42):
    """
    A smaller catalogue for the coarse sweep, so one run takes minutes instead
    of the best part of an hour.

    Keeps the n_types largest article types, which is enough to rank
    configurations against each other. The winner is then re-run on the full
    catalogue before anything goes in the report.
    """
    biggest = df['articleType'].value_counts().head(n_types).index
    return df[df['articleType'].isin(biggest)].reset_index(drop = True)


## The defaults reproduce the model we already trained, so the first row of the
## sweep is the number we are trying to beat.
DEFAULT_CONFIG = {
    'name': 'triplet_baseline',
    'sampler': 'random',
    'margin': 0.5,
    'embed_dim': 64,
    'dense_units': 64,
    'filters': (64, 64, 64),
    'dropout': 0.0,
    'normalise': True,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 4,
    'patience': None,
    'warmup_epochs': 1,
    'pool_size': 50,
    'augment': False,
    'seed': 42,
    'notes': '',
}

## Once the sampler is settled every later run inherits it, so each row really
## does change one thing against the rows above it.
TUNED_BASE = {'sampler': 'semihard', 'epochs': 20, 'patience': 3}

## The sweep. Same layout as the tuning tables in the Task 1 and Task 3
## READMEs: what we changed, what we call it, and the question it answers.
TUNING_EXPERIMENTS = [
    # 1. The sampler. Random negatives are nearly all outside the margin
    #    already, so most batches produce no gradient. This is the change we
    #    expect to matter more than everything below it put together.
    {'name': 'triplet_baseline',
     'notes': 'the model we already have, the row to beat'},
    {'name': 'triplet_semihard', **TUNED_BASE,
     'notes': 'does mining the useful negatives fix the collapse'},
    {'name': 'triplet_hardest', **TUNED_BASE, 'sampler': 'hardest',
     'notes': 'expected to collapse, run it so the report can show it'},

    # 2. Learning rate. 1e-4 over 4 epochs was almost certainly undertrained.
    {'name': 'triplet_lr_3e4', **TUNED_BASE, 'learning_rate': 3e-4,
     'notes': 'was 1e-4 simply too slow'},
    {'name': 'triplet_lr_1e3', **TUNED_BASE, 'learning_rate': 1e-3,
     'notes': 'how far can the learning rate go before it destabilises'},

    # 3. Margin. On the unit sphere squared distance runs 0 to 4, so 0.5 is a
    #    narrow target. Anything at or above 4 is unsatisfiable by definition.
    {'name': 'triplet_margin_0.2', **TUNED_BASE, 'margin': 0.2,
     'notes': 'an easier target, does it train more stably'},
    {'name': 'triplet_margin_1.0', **TUNED_BASE, 'margin': 1.0,
     'notes': 'does demanding more separation give a cleaner ranking'},

    # 4. Embedding size. Overlay this against the PCA dimension sweep on one
    #    axis and the figure answers both questions at once.
    {'name': 'triplet_dim_32', **TUNED_BASE, 'embed_dim': 32,
     'notes': 'is 64 more room than the task needs'},
    {'name': 'triplet_dim_128', **TUNED_BASE, 'embed_dim': 128,
     'notes': 'does a bigger embedding separate the rare types'},
    {'name': 'triplet_dim_256', **TUNED_BASE, 'embed_dim': 256,
     'notes': 'where does adding dimensions stop paying'},

    # 5. Capacity. Four blocks is the ceiling - 80x60 through valid 3x3 convs
    #    and 2x2 pools reaches 3x1, and a fifth block has nothing left to
    #    convolve over.
    {'name': 'triplet_widening', **TUNED_BASE, 'filters': (32, 64, 128),
     'notes': 'is the usual widening shape better than three flat layers'},
    {'name': 'triplet_deeper', **TUNED_BASE, 'filters': (32, 64, 128, 256),
     'notes': 'does a fourth block help at this resolution'},

    # 6. Batch size. With in-pool mining this is not just a speed knob - it
    #    sets how many candidates we get to choose a negative from.
    {'name': 'triplet_batch_128', **TUNED_BASE, 'batch_size': 128,
     'notes': 'bigger batches, more stable gradient'},
    {'name': 'triplet_batch_256', **TUNED_BASE, 'batch_size': 256,
     'notes': 'does the gain keep going or flatten off'},

    # 7. Regularisation, once we finally have a model that trains long enough
    #    to overfit.
    {'name': 'triplet_dropout_0.3', **TUNED_BASE, 'dropout': 0.3,
     'notes': 'is the tuned model overfitting the catalogue'},
    {'name': 'triplet_augment', **TUNED_BASE, 'augment': True,
     'notes': 'does flip and brightness jitter add anything'},

    # 8. The normalisation the original code added as a guess. Squared L2 and
    #    cosine rank identically on unit vectors, so this is the only version
    #    of that question worth spending a run on.
    {'name': 'triplet_unnormalised', **TUNED_BASE, 'normalise': False,
     'notes': 'does the model cheat the margin without the unit sphere'},
]


MODEL_FILE = 'models/task_4/embedding_visual_search.keras'
INDEX_FILE = 'models/task_4/embeddings_task4.npy'

# os.makedirs('models/task_4', exist_ok = True)
os.makedirs('outputs/task_4', exist_ok = True)

# 1. Split data
train_df, val_df = split_data(df_train)

# 2. Load the model if we already trained one, otherwise train it now.
# 4 epochs because the val loss starts going back up on the 5th.
if Path(MODEL_FILE).exists() and Path(INDEX_FILE).exists():
    print('loading saved model')
    model = tf.keras.models.load_model(MODEL_FILE)
    index = np.load(INDEX_FILE)
else:
    train_dataset = list_to_dataset(train_df)
    val_dataset = list_to_dataset(val_df)

    model = embedding_model()
    model = train_model(model, train_dataset, val_dataset, epochs = 4)

    # 3. Embed the whole training catalogue so we have something to search
    index = build_index(model, train_df)

    # 4. Save both so the next run can skip straight to searching
    model.save(MODEL_FILE)
    np.save(INDEX_FILE, index)

# 5. Query with a validation image, the model has never seen it
query = val_df.iloc[0]
results, distances = search(model, query['path'], index, train_df, k = 5)

print(f"query: {query['articleType']}  {query['path']}")
print(results[['id', 'articleType', 'baseColour', 'masterCategory', 'path']])

# 6. Run the validation queries once, then use that same pass for both the
# output file and the precision score
n_queries = 100
rows = []
scores = []

for i in range(min(n_queries, len(val_df))):
    q = val_df.iloc[i]
    res, dist = search(model, q['path'], index, train_df, k = 5)

    scores.append((res['articleType'] == q['articleType']).sum() / 5)

    for rank in range(len(res)):
        rows.append({
            'query_id': q['id'],
            'query_articleType': q['articleType'],
            'query_path': q['path'],
            'rank': rank + 1,
            'retrieved_id': res.iloc[rank]['id'],
            'retrieved_articleType': res.iloc[rank]['articleType'],
            'retrieved_path': res.iloc[rank]['path'],
            'distance': dist[rank],
        })

# 7. Save the top 5 for each validation query so we have an output file
pd.DataFrame(rows).to_csv('outputs/task_4/task4_topk_predictions.csv', index = False)

print(f"precision@5: {np.mean(scores):.3f}")

# 8. Save the query + neighbours picture for the report
show_results(query['path'], results)


## 9. Fine tuning. Off by default because a full sweep retrains the model once
## per row of TUNING_EXPERIMENTS. Start with the coarse pass on a smaller
## catalogue, then re-run the winner on train_df.
RUN_FINE_TUNE = False

if RUN_FINE_TUNE:
    coarse_train = subsample_catalogue(train_df)
    coarse_val = subsample_catalogue(val_df)

    results = fine_tune(coarse_train, coarse_val)
    print(results[['name', 'p_at_1', 'p_at_5', 'p_at_10', 'map_at_10', 'active_fraction']])
