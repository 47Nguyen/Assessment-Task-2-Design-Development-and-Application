import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, UnitNormalization
from sklearn.model_selection import train_test_split
from PIL import Image
tf.get_logger().setLevel('ERROR')

# Seed so the triplet sampling gives the same results every run
np.random.seed(42)
tf.random.set_seed(42)

# https://keras.io/examples/vision/siamese_network/
# https://www.datacamp.com/tutorial/cnn-tensorflow-python
# https://pyimagesearch.com/2023/02/13/building-a-dataset-for-triplet-loss-with-keras-and-tensorflow/

## Load datas
train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'
test_path = './A2_FashionDataset/FashionDataset/test/styles_prediction.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Setup image path for products
df_train['path'] = './A2_FashionDataset/FashionDataset/train/images_train' + '/' + df_train['id'].astype(str) + '.jpg'


# Preprocess dataset
df_train = df_train.drop(columns=['Unnamed: 10','Unnamed: 11'])

# 5 rows in the csv point at images that are not in the folder, drop them
df_train = df_train[df_train['path'].apply(os.path.exists)]

# print(df_train)
# count = df_train['masterCategory'].nunique()


# Check image size
size = Image.open(df_train['path'][2]).size # 60 wide * 80 tall
target_shape = (80,60)  # tf.image.resize takes (height, width)


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

## Print an image
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
def embedding_model():
    INPUT_SHAPE = (80,60,3)

    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2))) 
    
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2))) 
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(64))  # embedding vector, no activation

    # Put the embedding on the unit sphere, otherwise the model can beat the
    # 0.5 margin by making the vectors bigger instead of separating the classes
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
def train_step(model, optimizer, anchor, reference, disimilar):
    with tf.GradientTape() as tape:
        anchor_emb = model(anchor, training = True)
        reference_emb = model(reference, training = True)
        disimilar_emb = model(disimilar, training = True)
        loss = triplet_loss(anchor_emb, reference_emb, disimilar_emb)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


## Same maths but no weight update, we are only measuring here
def val_step(model, anchor, reference, disimilar):
    anchor_emb = model(anchor, training = False)
    reference_emb = model(reference, training = False)
    disimilar_emb = model(disimilar, training = False)
    return triplet_loss(anchor_emb, reference_emb, disimilar_emb)


def train_model(model, train_dataset, val_dataset, epochs = 5, learning_rate = 1e-4):
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

    plt.savefig('outputs/task4_query_grid.png', bbox_inches = 'tight')
    plt.close()


## Of the k we retrieved, how many share the query's articleType
def precision_at_k(model, index, catalogue_df, query_df, k = 5, n_queries = 100):
    scores = []

    for i in range(min(n_queries, len(query_df))):
        query = query_df.iloc[i]
        results, _ = search(model, query['path'], index, catalogue_df, k)
        hits = (results['articleType'] == query['articleType']).sum()
        scores.append(hits / k)

    return np.mean(scores)




# 1. Split data
train_df, val_df = split_data(df_train)
train_dataset = list_to_dataset(train_df)
val_dataset = list_to_dataset(val_df)

# 2. Train the embedding model on the triplets.
# 4 epochs because the val loss starts going back up on the 5th.
model = embedding_model()
model = train_model(model, train_dataset, val_dataset, epochs = 4)

# 3. Embed the whole training catalogue so we have something to search
index = build_index(model, train_df)

# 4. Save the model and the index so we don't have to train again
os.makedirs('models', exist_ok = True)
os.makedirs('outputs', exist_ok = True)

model.save('models/embedding_visual_search.keras')
np.save('models/embeddings_task4.npy', index)

# 5. Query with a validation image, the model has never seen it
query = val_df.iloc[0]
results, distances = search(model, query['path'], index, train_df, k = 5)

print(f"query: {query['articleType']}")
print(results[['id', 'articleType', 'baseColour']])

# 6. How often do the retrieved items match the query type
print(f"precision@5: {precision_at_k(model, index, train_df, val_df, k = 5):.3f}")

# 7. Save the top 5 for each validation query so we have an output file
rows = []
for i in range(100):
    q = val_df.iloc[i]
    res, dist = search(model, q['path'], index, train_df, k = 5)
    for rank in range(len(res)):
        rows.append({
            'query_id': q['id'],
            'query_articleType': q['articleType'],
            'rank': rank + 1,
            'retrieved_id': res.iloc[rank]['id'],
            'retrieved_articleType': res.iloc[rank]['articleType'],
            'distance': dist[rank],
        })

pd.DataFrame(rows).to_csv('outputs/task4_topk_predictions.csv', index = False)

# 8. Save the query + neighbours picture for the report
show_results(query['path'], results)





