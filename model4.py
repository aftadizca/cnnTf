import tensorflow as tf
import disable_log
from datetime import datetime
from dirs import train_dirs, validation_dirs, model_dirs
import os

"""
TODO: v1 -> lr=default, epoch = 250
TODO: v2 -> lr=default, epoch = 500
TODO: v3 -> lr=1-e5, epoch = 250
TODO: v4 -> lr=1-e5, epoch = 500


"""
MODEL_NAME = os.path.basename(__file__).replace(".py","")
TUNNING = 100
IMAGE_SIZE = 128
CLASSES_NUM = 5
COLOR_MODE = 'grayscale'
CLASSES_MODE = 'categorical'
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1) 
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
DROPOUT = 0.3
ACTIVATION_CONV = 'relu'
ACTIVATION_DENSE = 'relu'
ACTIVATION_DENSE_END = 'softmax'

def model():
    print("Create Model")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                     activation=ACTIVATION_CONV, input_shape=IMG_SHAPE))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(52, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(128, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(64, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(CLASSES_NUM, activation=ACTIVATION_DENSE_END))

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    return model

def beforeCompile(model):
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:227]:
        layer.trainable = False
    return model


# m = model()

# m.layers[0].summary()

# print(f"TOTAL LAYER : {len(m.layers) + len(m.layers[0].layers) -1}")
