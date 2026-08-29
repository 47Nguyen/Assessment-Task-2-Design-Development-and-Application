#Shared CNN model. Everyone builds their model with build_cnn() so the results
#table compares tasks, not architectures. If M1 and M3 each write a different
#network, a difference in scores could be caused by either one and there is no
#way to tell.
#
#Change the settings through the arguments. Don't copy the function and edit it.

from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SHAPE, SEED, MODEL_DIR


def build_cnn(
    n_classes,
    filters=(32, 64, 128),
    dense_units=128,
    dropout=0.3,
    learning_rate=1e-3,
    name='cnn',
):
    #structure: a few Conv-BatchNorm-ReLU-Pool blocks, then global average
    #pooling, then a dense classifier head.
    #n_classes     - how many classes to predict
    #filters       - one conv block per value, more values = deeper network
    #dense_units   - size of the layer before the output
    #dropout       - how much to drop before the output (helps overfitting)
    #learning_rate - tune this first, it matters more than anything else
    keras.utils.set_random_seed(SEED)

    inputs = keras.Input(shape=IMG_SHAPE, name='image')

    x = inputs
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False, name=f'conv{i}')(x)
        x = layers.BatchNormalization(name=f'bn{i}')(x)
        x = layers.Activation('relu', name=f'relu{i}')(x)
        x = layers.MaxPooling2D(2, name=f'pool{i}')(x)

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(dense_units, activation='relu', name='embedding')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='prediction')(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def default_callbacks(target, patience=5):
    #stop training when the model stops improving, and keep the best weights.
    #everyone using the same stopping rule means training length isn't a
    #hidden variable when we compare results.
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            #str(), not Path - Keras 2.15 calls .endswith() on this argument
            str(MODEL_DIR / f'cnn_{target}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
    ]
