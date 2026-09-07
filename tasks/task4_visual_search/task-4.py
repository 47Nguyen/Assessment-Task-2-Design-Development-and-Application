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

# References:
# https://keras.io/examples/vision/siamese_network/
# https://www.datacamp.com/tutorial/cnn-tensorflow-python
# https://pyimagesearch.com/2023/02/13/building-a-dataset-for-triplet-loss-with-keras-and-tensorflow/
# https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/
# https://pyimagesearch.com/2023/03/20/training-and-making-predictions-with-siamese-networks-and-triplet-loss/
# https://arxiv.org/abs/1503.03832
# https://arxiv.org/abs/1703.07737
# https://github.com/adambielski/siamese-triplet
# https://github.com/13muskanp/Siamese-Network-with-Triplet-Loss



# SECTION 1: Load
train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'

df_train = pd.read_csv(train_path)

# Setup image path for products
df_train['path'] = './A2_FashionDataset/FashionDataset/train/images_train' + '/' + df_train['id'].astype(str) + '.jpg'

# Preprocess dataset
df_train = df_train.drop(columns=['Unnamed: 10','Unnamed: 11'], errors='ignore')

# 5 rows in the csv point at images that are not in the folder, drop them
df_train = df_train[df_train['path'].apply(os.path.exists)]

# The images are 60 wide by 80 tall.
# tf.image.resize takes (height, width) so it has to be this way round.
target_shape = (80,60)



# SECTION 2: Preprocessing

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


def print_image(index):
    """
    Show one preprocessed image from df_train.
    """
    plt.figure(dpi = 28)
    image = preprocess_image(df_train['path'][index])
    plt.imshow(image)
    plt.show()


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


def split_data(df):
    """
    We have to build our own split function.
    Reason is because this task focuses on finding top K of results.
    It not looking to predicts the a target vaulue.
    """
    counts = df['articleType'].value_counts() # Total up the number of each the article type.
    keep = counts[counts >= 2].index  # Look for any articleType with total counts >=2
    df = df[df['articleType'].isin(keep)]  # Filter out the dataframe, we only keep values where the articleType counts >= 2

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

    return train_df, val_df


# SECTION 3: MODEL - the twin CNN shared by anchor/reference/disimilar
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


# SECTION 4: triplet Loss train
def triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin = 0.5):
    """
    We want the anchor close to the reference and far from the disimilar.
    """
    d_pos = tf.reduce_sum(tf.square(anchor_emb - reference_emb), axis = -1)
    d_neg = tf.reduce_sum(tf.square(anchor_emb - disimilar_emb), axis = -1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))


def train_step(model, optimizer, anchor, reference, disimilar, margin = 0.5):
    """
    One batch of learning, weights get updated here.
    """
    with tf.GradientTape() as tape:
        anchor_emb = model(anchor, training = True)
        reference_emb = model(reference, training = True)
        disimilar_emb = model(disimilar, training = True)
        loss = triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def val_step(model, anchor, reference, disimilar, margin = 0.5):
    """
    Same maths but no weight update, we are only measuring here.
    """
    anchor_emb = model(anchor, training = False)
    reference_emb = model(reference, training = False)
    disimilar_emb = model(disimilar, training = False)
    return triplet_loss(anchor_emb, reference_emb, disimilar_emb, margin)



# SECTION 5: Triplet sampling - how anchor/positive/negative get chosen


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

    Kept exactly as the original sampler had it, so the baseline row stays
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
                    margin, pool_size = 50, chunk = 512):
    """
    Semi-hard negative mining.

    Semi-hard keeps a negative that is further away than the positive but
    still inside the margin, which is the only band that produces a gradient.

    Scoring all ~30k candidates against all ~30k anchors is 900M distances per
    epoch, so we score a random pool of pool_size candidates per anchor instead.
    That is an approximation and the report should say so.
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

        in_band = (d_neg > d_pos[:, None]) & (d_neg < d_pos[:, None] + margin)
        banded = np.where(in_band, d_neg, np.inf)
        choice = banded.argmin(axis = 1)

        # nothing in the band for this anchor, fall back to the closest
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
                                     margin, pool_size)

    return list(paths[anchors]), list(paths[references]), list(paths[disimilars])


def triplets_to_dataset(anchors, references, disimilars, batch_size = 32):
    """
    Triplet paths to a batched dataset of images.

    Takes triplets built outside so they can be re-mined between epochs.
    """
    dataset = tf.data.Dataset.from_tensor_slices((anchors, references, disimilars))
    dataset = dataset.map(preprocess_triplets, num_parallel_calls = tf.data.AUTOTUNE)

    return dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# SECTION 6: TTrain loop, loop over thenumber of epoches to train
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


def train_tuned_model(model, train_df, val_df, config, verbose = True):
    """
    Train one configuration, keeping the weights from the best epoch rather
    than whatever the last epoch happened to produce.

    Val loss picks the best epoch, which is fair inside a single run because
    margin is fixed there. Across runs it is not - margin sits inside the loss -
    which is why fine_tune ranks the two configurations on mAP@10 instead.
    """
    optimizer = tf.keras.optimizers.Adam(config['learning_rate'])

    # one fixed set of validation triplets, so epoch to epoch loss is comparable
    val_triplets = build_triplets(val_df, 'random')
    val_dataset = triplets_to_dataset(*val_triplets, batch_size = config['batch_size'])

    best_loss = np.inf
    best_weights = model.get_weights()
    best_epoch = 1
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
                                            batch_size = config['batch_size'])

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

    model.set_weights(best_weights)
    return model, pd.DataFrame(history), best_epoch



# SECTION 7: SEARCH - embed the images, then retrieve nearest neighbours
def embed_paths(model, paths, batch_size = 32):
    """
    Embed a list of image paths in order.

    Row i of the result is the embedding of paths[i], which is what lets us
    map a neighbour back to its dataframe row.
    """
    dataset = tf.data.Dataset.from_tensor_slices(list(paths))
    dataset = dataset.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset, verbose = 0)


def search(model, query_path, index, df, k = 5):
    """
    Embed the query then return the k closest images from the index.
    """
    query_image = preprocess_image(query_path)
    query_emb = model(tf.expand_dims(query_image, axis = 0), training = False).numpy()

    distances = np.sum(np.square(index - query_emb), axis = 1)
    nearest = np.argsort(distances)[:k]

    return df.iloc[nearest], distances[nearest]


def topk_predictions(model, index, index_df, query_df, n_queries = 100, k = 5):
    """
    Top k neighbours for the first n_queries validation images.

    Returns the long form table we save as the task 4 output, plus precision@5
    over the same pass so the number and the file always agree.
    """
    rows = []
    scores = []

    for i in range(min(n_queries, len(query_df))):
        q = query_df.iloc[i]
        res, dist = search(model, q['path'], index, index_df, k = k)

        scores.append((res['articleType'] == q['articleType']).sum() / k)

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

    return pd.DataFrame(rows), float(np.mean(scores))


def show_results(query_path, results, save_to = 'outputs/task_4/task4_query_grid.png'):
    """
    Show the query next to what we retrieved, saved as one row of pictures.
    """
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

    plt.savefig(save_to, bbox_inches = 'tight')
    plt.close()


# SECTION 8: Setup evaluation framework - Precision@K and mAP@K
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


def evaluate_retrieval(model, index, index_df, query_df, n_queries = 500,
                       k_list = (1, 5, 10), map_k = 10, seed = 42):
    """
    Precision@K and mAP@K over a fixed sample of validation queries.

    Precision@K on its own cannot tell a good ranking from a lucky one,
    because it ignores the order inside the top K. mAP is rank sensitive,
    so it is the number fine_tune ranks configurations on.
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

# SECTION 9: Fine Tunning the model

RESULTS_FILE = 'outputs/task_4/results_task4.csv'
TUNED_MODEL_FILE = 'models/task_4/embedding_visual_search_tuned.keras'
TUNED_INDEX_FILE = 'models/task_4/embeddings_task4_tuned.npy'

## The baseline reproduces the model we already trained, so its row is the
## number semi-hard has to beat.
BASE_CONFIG = {
    'margin': 0.5, 'embed_dim': 64, 'dense_units': 64, 'filters': (64, 64, 64),
    'dropout': 0.0, 'normalise': True, 'learning_rate': 1e-4, 'batch_size': 32,
    'warmup_epochs': 1, 'pool_size': 50, 'seed': 42,
}

EXPERIMENTS = [
    {**BASE_CONFIG, 'name': 'baseline_random', 'sampler': 'random', 'epochs': 4},
    {**BASE_CONFIG, 'name': 'tuned_semihard', 'sampler': 'semihard', 'epochs': 10},
]


def run_experiment(config, train_df, val_df, n_queries = 500):
    """
    Train one configuration end to end and hand back its results row.
    """
    # reseed per run so the two rows differ by the sampler and not by which
    # triplets they happened to draw
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    print(f"\n=== {config['name']} ({config['sampler']} negatives) ===")

    started = time.time()

    model = embedding_model(embed_dim = config['embed_dim'],
                            dense_units = config['dense_units'],
                            filters = config['filters'],
                            dropout = config['dropout'],
                            normalise = config['normalise'])

    model, history, best_epoch = train_tuned_model(model, train_df, val_df, config)
    train_seconds = time.time() - started

    index = embed_paths(model, train_df['path'].tolist(), config['batch_size'])
    metrics = evaluate_retrieval(model, index, train_df, val_df, n_queries = n_queries)

    best = history.iloc[best_epoch - 1]
    row = {
        'name': config['name'],
        'sampler': config['sampler'],
        'epochs': config['epochs'],
        'best_epoch': best_epoch,
        'train_loss': round(float(best['train_loss']), 4),
        'val_loss': round(float(best['val_loss']), 4),
        'active_fraction': round(float(best['active_fraction']), 4),
        'mean_d_pos': round(float(best['mean_d_pos']), 4),
        'mean_d_neg': round(float(best['mean_d_neg']), 4),
        'train_seconds': round(train_seconds, 1),
        **metrics,
    }
    return model, row, history, index


def fine_tune(train_df, val_df, n_queries = 500):
    """
    Run both configurations and save the better one by mAP@10.

    Ranked on mAP rather than val loss because margin sits inside the loss, so
    losses are only comparable within a run. Each run also drops its per-epoch
    history, which is what the learning curve in the report is drawn from.

    The winner is saved alongside the original model, never over the top of it.
    """
    os.makedirs('outputs/task_4', exist_ok = True)
    os.makedirs('models/task_4', exist_ok = True)

    rows = []
    best_score = -np.inf

    for config in EXPERIMENTS:
        model, row, history, index = run_experiment(config, train_df, val_df, n_queries)
        rows.append(row)
        history.to_csv(f"outputs/task_4/history_{row['name']}.csv", index = False)

        print(f"{row['name']}: P@1 {row['p_at_1']:.3f}  P@5 {row['p_at_5']:.3f}  "
              f"P@10 {row['p_at_10']:.3f}  mAP@10 {row['map_at_10']:.3f}  "
              f"active {row['active_fraction']:.2f}")

        if row['map_at_10'] > best_score:
            best_score = row['map_at_10']
            model.save(TUNED_MODEL_FILE)
            np.save(TUNED_INDEX_FILE, index)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_FILE, index = False)
    return table


def subsample_catalogue(df, n_types = 30):
    """
    A smaller catalogue for a quicker comparison, so a run takes minutes
    instead of the best part of an hour.

    Keeps the n_types largest article types, which is enough to tell the two
    samplers apart. The winner can then be re-run on the full catalogue.
    """
    biggest = df['articleType'].value_counts().head(n_types).index
    return df[df['articleType'].isin(biggest)].reset_index(drop = True)



# SECTION 10: Setup

RUN_FINE_TUNE = False
RUN_FINALISE = False


# SECTION 11: Running pipeline with functions swetup

MODEL_FILE = 'models/task_4/embedding_visual_search.keras'
INDEX_FILE = 'models/task_4/embeddings_task4.npy'

os.makedirs('models/task_4', exist_ok = True)
os.makedirs('outputs/task_4', exist_ok = True)

# 1. Split data
train_df, val_df = split_data(df_train)

# 2. Load the baseline model if we already trained one, otherwise train it now.
if Path(MODEL_FILE).exists() and Path(INDEX_FILE).exists():
    print('loading saved model')
    model = tf.keras.models.load_model(MODEL_FILE)
    index = np.load(INDEX_FILE)
else:
    # BASE_CONFIG is the original setup - random negatives, 4 epochs, no early
    # stopping - so this rebuilds the same baseline through the one training
    # path the file now has.
    model, _, _, index = run_experiment(EXPERIMENTS[0], train_df, val_df)

    # 3. Save both so the next run can skip straight to searching
    model.save(MODEL_FILE)
    np.save(INDEX_FILE, index)

# 4. Query with a validation image, the model has never seen it
query = val_df.iloc[0]
results, distances = search(model, query['path'], index, train_df, k = 5)

print(f"query: {query['articleType']}  {query['path']}")
print(results[['id', 'articleType', 'baseColour', 'masterCategory', 'path']])

# 5. Run the validation queries once, then use that same pass for both the output file and the precision score
predictions, precision = topk_predictions(model, index, train_df, val_df)
predictions.to_csv('outputs/task_4/task4_topk_predictions.csv', index = False)

print(f"precision@5: {precision:.3f}")

# 6. Save the query + neighbours picture for the report
show_results(query['path'], results)

# 7. Fine tuning - baseline vs semi-hard on a 30 type subsample. Off by default because it retrains the model twice (~2 hours). See SECTION 10.
if RUN_FINE_TUNE:
    tuning_table = fine_tune(subsample_catalogue(train_df),
                             subsample_catalogue(val_df))
    print(tuning_table[['name', 'p_at_1', 'p_at_5', 'p_at_10',
                        'map_at_10', 'active_fraction']])

# 8. Finalise the tuned model over the full item list
if RUN_FINALISE:
    tuned_model = tf.keras.models.load_model(TUNED_MODEL_FILE)

    # the full catalogue this time, not the subsample
    tuned_index = embed_paths(tuned_model, train_df['path'].tolist())
    np.save(TUNED_INDEX_FILE, tuned_index)

    tuned_predictions, tuned_precision = topk_predictions(tuned_model, tuned_index,
                                                          train_df, val_df)
    tuned_predictions.to_csv('outputs/task_4/task4_topk_predictions_tuned.csv',
                             index = False)

    # same query as the baseline figure, so the two grids are comparable
    tuned_results, _ = search(tuned_model, query['path'], tuned_index, train_df, k = 5)
    show_results(query['path'], tuned_results,
                 save_to = 'outputs/task_4/task4_query_grid_tuned.png')

    print(f"tuned precision@5 on the full catalogue: {tuned_precision:.3f}")
    print(tuned_results[['id', 'articleType', 'baseColour', 'masterCategory']])
