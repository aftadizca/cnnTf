import model2 as model  # change model
import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model

tf.get_logger().setLevel(logging.ERROR)

#############-----------DISABLE GPU-----------#############
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#############-----------IMPORT MODEl-----------#############
model_name = os.path.basename(model.__file__).replace(".py", "")

#############-----------TRAIN DIRECTORY-----------#############
train_dirs = 'traindata/train/'
validation_dirs = 'traindata/valid/'
log_dirs = os.path.join(
    'log', model_name, datetime.now().strftime("%Y.%m.%d - %H.%M.%S"))

#############-----------TRAIN PARAMETER-----------#############
class_mode = 'categorical'  # "binary" jika hanya 2 class
batch_size = 16
training_size = 1146  # jumlah data/file training
# training_size = 320  # jumlah data/file training
validation_size = 68  # jumlah data/file validasi
epochs = 1000


#############-----------Parameter Train ImageDataGenerator-----------#############
train_datagen = ImageDataGenerator(
    rescale=1./255,  # ubah pixel menjadi nilai diatara 0 - 1
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
valid_datagen = ImageDataGenerator(rescale=1./255)

#############-----------GENERATE IMAGE FOR TRAINING-----------#############
train_generator = train_datagen.flow_from_directory(
    train_dirs,
    color_mode=model.color_mode,
    target_size=(model.img_width, model.img_height),
    batch_size=batch_size,
    class_mode=class_mode
)

#############-----------GENERATE IMAGE FOR VALIDATION-----------#############
valid_generator = valid_datagen.flow_from_directory(
    validation_dirs,
    color_mode=model.color_mode,
    target_size=(model.img_width, model.img_height),
    batch_size=batch_size,
    class_mode=class_mode
)


#############-----------Create file from ImageDataGenerator-----------#############
# image = train_datagen.flow_from_directory('traindata/train', target_size=(
#     128, 128), color_mode='grayscale', save_to_dir='augmented', class_mode='categorical', save_format='jpeg', batch_size=10)
# image.next()

#############-----------CALLBACK-----------#############
checkpointer = ModelCheckpoint(
    filepath=model_name+'.{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
#### ----- cara pakai ---> tensorboard --logdir log/model_name ---####
tensorboard_callbacks = TensorBoard(log_dir=log_dirs)

# -----------DETECT EXISTING MODEL-----------#############n
#############-----------LOAD or NEW-----------#############
# try:
#     myModel = load_model(model_name+'.hdf5')
#     print('Model detected. Load existing model? If "No", new model will created.')
#     user_input = input("yes/no : ")
#     if user_input == "no":
#         raise OSError
#     else:
#         myModel.compile(
#             optimizer=model.optimizer,
#             loss=model.loss,
#             metrics=model.metrics
#         )
# except OSError:
#     myModel = model.model()

# myModel = load_model(model_name+'.hdf5')
# myModel.compile(
#     optimizer=model.optimizer,
#     loss=model.loss,
#     metrics=model.metrics
# )
myModel = model.model()
#############-----------TRAINING PROSES-----------#############
myModel.fit(
    train_generator,
    steps_per_epoch=training_size//batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=validation_size//batch_size,
    verbose=2,
    callbacks=[tensorboard_callbacks, checkpointer]
)


#############-----------SHUTDOWN WHEN DONE-----------#############
os.system("systemctl poweroff")
