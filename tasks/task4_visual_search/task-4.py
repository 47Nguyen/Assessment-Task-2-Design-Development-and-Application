import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from PIL import Image
tf.get_logger().setLevel('ERROR')

# https://keras.io/examples/vision/siamese_network/
# https://www.datacamp.com/tutorial/cnn-tensorflow-python

## Load datas
train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'
test_path = './A2_FashionDataset/FashionDataset/test/styles_prediction.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Setup image path for products
df_train['path'] = './A2_FashionDataset/FashionDataset/train/images_train' + '/' + df_train['id'].astype(str) + '.jpg'


# Preprocess dataset
df_train = df_train.drop(columns=['Unnamed: 10','Unnamed: 11'])

# print(df_train)
# count = df_train['masterCategory'].nunique()


# Check image size
size = Image.open(df_train['path'][2]).size # 60 * 80 
target_shape = (60,80)


## Preprcoess iamge
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, target_shape)
    return image

## Print an image
def print_image(index):
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
## Trained from scratch, no pretrained weights (assignment does not allow
## pretrained systems for the final model).
def embedding_model():
    INPUT_SHAPE = (60,80,3) 

    model = Sequential()
    
    model.add(Conv2D(64, (3,3), input_shape = INPUT_SHAPE))
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

    return model

def split_data(df):
    counts = df['articleType'].value_counts()
    keep = counts[counts >= 2].index
    df = df[df['articleType'].isin(keep)] 
    
    train_df, val_df = train_test_split(df,test_size=0.2,stratify=df['articleType'], random_state=42,
    )
    return train_df, val_df

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for epoch in range(epochs):
        for anchor_img, reference_img, disimilar_img in dataset:
            loss = train_step(model, optimizer, anchor_img, reference_img, disimilar_img)
        print(f"epoch {epoch + 1}/{epochs} - loss: {loss.numpy():.4f}")
    return model


# anchors, references, disimilars = anchor_references(df_train)
# train_dataset = make_triplet_dataset(anchors, references, disimilars)

# model = embedding_model()
# model = train_embedding_model(model, train_dataset)

embedding_model().summary()


