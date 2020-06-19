import disable_log
import model1 as model
from datetime import datetime
from dirs import train_dirs, validation_dirs
import tensorflow as tf


#############-----------TEST PARAMETER-----------#############
model_filename = 'model2.895-0.74.hdf5'
batch_size = 4
test_size = 68  # jumlah data/file training


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    validation_dirs,
    color_mode=model.COLOR_MODE,
    target_size=(model.IMG_WIDTH, model.IMG_HEIGHT),
    batch_size=batch_size,
    class_mode=model.COLOR_MODE
)

myModel = tf.keras.models.load_model(model_filename)
myModel.compile(
    optimizer=model.OPTIMIZER,
    loss=model.LOSS,
    metrics=model.METRICS
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
