import disable_log
import model1 as model
from datetime import datetime
from dirs import train_dirs, validation_dirs
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model


#############-----------TEST PARAMETER-----------#############
model_filename = 'model2.895-0.74.hdf5'
batch_size = 4
test_size = 68  # jumlah data/file training

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    validation_dirs,
    color_mode=model.color_mode,
    target_size=(model.img_width, model.img_height),
    batch_size=batch_size,
    class_mode=model.class_mode
)

myModel = load_model(model_filename)
myModel.compile(
    optimizer=model.optimizer,
    loss=model.loss,
    metrics=model.metrics
)

score = myModel.evaluate(
    test_generator,
    steps=test_size // batch_size,
    verbose=1
)
atribut = myModel.metrics_names
print()
print(
    f'Evaluate : {atribut[0]} -> {score[0]} , {atribut[1]} ->  {score[1]*100}')
