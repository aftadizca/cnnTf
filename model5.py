import tensorflow as tf
import disable_log
from datetime import datetime
from dirs import train_dirs, validation_dirs, model_dirs
import os

IMAGE_SIZE = 128
CLASSES_NUM = 5
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']


def model():
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(CLASSES_NUM, activation='softmax')
    ])
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    print(len(base_model.layers))

    return model
