import model1 as model
import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model


#############-----------TEST PARAMETER-----------#############
test_dirs = 'traindata/train/'
class_mode = 'categorical'  # "binary" jika hanya 2 class
batch_size = 4
test_size = 300  # jumlah data/file training
epochs = 1000

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dirs,
    color_mode=model.color_mode,
    target_size=(model.img_width, model.img_height),
    batch_size=batch_size,
    class_mode=class_mode
)

myModel = load_model('model1.hdf5')
myModel.compile(
    optimizer=model.optimizer,
    loss=model.loss,
    metrics=model.metrics
)

score = myModel.evaluate(
    test_generator,
    steps=test_size//batch_size,
    verbose=2
)
print("Evaluate : Loss -> ", score[0], " Accuracy --> ", score[1]*100)
