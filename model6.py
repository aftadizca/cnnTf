import tensorflow as tf
import disable_log
from datetime import datetime
from dirs import train_dirs, validation_dirs, model_dirs
import os

"""
TODO: mencoba fine tuning model
TODO: v1 -> intitial not fine tune lr=default
TODO: v2 -> unfreeze after compile lr=1e-5
TODO: v3 -> unfreeze before compile lr=1e-5 tunning=100
"""
MODEL_NAME = os.path.basename(__file__).replace(".py","")
TUNNING = 100
IMAGE_SIZE = 128
CLASSES_NUM = 5
COLOR_MODE = 'rgb'
CLASSES_MODE = 'categorical'
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-5,)
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

def model():
    #? Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(CLASSES_NUM, activation='softmax')
    ])
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    return model

def beforeCompile(model):
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:100]:
        layer.trainable = False
    return model