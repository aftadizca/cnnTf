import disable_log
import model10 as model
from datetime import datetime
import tensorflow as tf
import os


#############-----------TEST PARAMETER-----------#############
validation_dirs = 'traindata2/train'
model_filename = 'model/model10/model10.e300lr1e-04.hdf5'
batch_size = 32

print(os.path.basename(model_filename))

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    validation_dirs,
    color_mode=model.COLOR_MODE,
    target_size=(model.IMAGE_SIZE, model.IMAGE_SIZE),
    batch_size=batch_size,
    class_mode=model.CLASSES_MODE
)

myModel = tf.keras.models.load_model(model_filename)
myModel.compile(
    optimizer=model.OPTIMIZER,
    loss=model.LOSS,
    metrics=model.METRICS
)

score = myModel.evaluate(
    test_generator,
    steps=len(test_generator),
    verbose=1
)
atribut = myModel.metrics_names
print()
print(
    f'Evaluate : {atribut[0]} -> {score[0]} , {atribut[1]} ->  {score[1]*100}')
