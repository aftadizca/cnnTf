import disable_log
from list_model import ListModel
import os
import logging
from dirs import train_dirs, validation_dirs, model_dirs
from datetime import datetime
from pathlib import Path
import tensorflow as tf

import model6 as model  # change model

os.system("clear")
#############-----------IMPORT MODEl-----------#############
MODEL_NAME = os.path.basename(model.__file__).replace(".py", "")
ITERATION = "1"
#############-----------CALLBACK DIRECTORY-----------#############
Path(os.path.join(model_dirs, MODEL_NAME)).mkdir(exist_ok=True, parents=True)
log_dirs = os.path.join(
    'log', MODEL_NAME, datetime.now().strftime("%Y.%m.%d - %H.%M.%S"))
checkpointer_dirs = os.path.join(
    model_dirs, MODEL_NAME, MODEL_NAME + "." + ITERATION + datetime.now().strftime("_%Y_%m_%d") + '.hdf5')
#############-----------TRAIN PARAMETER-----------#############
batch_size = 16
training_size = 1605  # jumlah data/file training
validation_size = 124  # jumlah data/file validasi
epochs = 1000


#############-----------Parameter Train ImageDataGenerator-----------#############
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # ubah pixel menjadi nilai diatara 0 - 1
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # zca_whitening=True,
)

#############-----------Parameter Validation ImageDataGenerator-----------#############
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

#############-----------GENERATE IMAGE FOR TRAINING-----------#############
train_generator = train_datagen.flow_from_directory(
    train_dirs,
    color_mode=model.COLOR_MODE,
    target_size=(model.IMAGE_SIZE, model.IMAGE_SIZE),
    batch_size=batch_size,
    class_mode=model.CLASSES_MODE
)

#############-----------GENERATE IMAGE FOR VALIDATION-----------#############
valid_generator = valid_datagen.flow_from_directory(
    validation_dirs,
    color_mode=model.COLOR_MODE,
    target_size=(model.IMAGE_SIZE, model.IMAGE_SIZE),
    batch_size=batch_size,
    class_mode=model.CLASSES_MODE
)


#############-----------Create file from ImageDataGenerator-----------#############
# image = train_datagen.flow_from_directory('traindata/train', target_size=(
#     128, 128), color_mode='grayscale', save_to_dir='augmented', class_mode='categorical', save_format='jpeg', batch_size=10)
# image.next()

#############-----------DETECT EXISTING MODEL-----------#############
try:
    list_model = ListModel(MODEL_NAME, model_dirs)
    if list_model.count() == 0:
        raise FileNotFoundError
    else:
        print()
        list_model.printList()
        user_input = input("Select model [n -> to create new model] : ")
        if user_input.isdecimal:
            print("\nLoading model -> ", list_model.getFileName(int(user_input)))
            myModel = tf.keras.models.load_model(
                list_model.getFilePath(int(user_input)))
            myModel.compile(
                optimizer=model.OPTIMIZER,
                loss=model.LOSS,
                metrics=model.METRICS
            )
        elif user_input == 'n':
            myModel = model.model()
        elif user_input == '' or user_input.isalpha:
            quit()
except FileNotFoundError:
    user_input = input("\nNo model detected. Create new? [y/n] : ")
    if user_input == 'y':
        myModel = model.model()
    else:
        quit()
#############-----------CALLBACK-----------#############
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointer_dirs, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#### ----- cara pakai ---> tensorboard --logdir log/model_name ---####
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dirs)

myModel.layers[0].trainable = True

for layer in myModel.layers[0].layers[:100]:
    layer.trainable = False

myModel.summary()
#############-----------TRAINING PROSES-----------#############
myModel.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=2,
    callbacks=[tensorboard_callbacks, checkpointer]
)


#############-----------SHUTDOWN WHEN DONE-----------#############
#os.system("systemctl poweroff")
