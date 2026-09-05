import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import metrics
from keras import Model
from keras.applications import resnet


target_shape = (200, 200)
# https://keras.io/examples/vision/siamese_network/



train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'
test_path = './A2_FashionDataset/FashionDataset/test/styles_prediction.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Setup image path for products
df_train['path'] = './A2_FashionDataset/FashionDataset/train/images_train' + '/' + df_train['id'].astype(str) + '.jpg'

# Preprocess dataset
df_train = df_train.drop(columns=['Unnamed: 10','Unnamed: 11'])

# print(df_train)

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

