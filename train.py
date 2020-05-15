import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import model1 as model

print(model.activation_dense_end)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    zca_whitening=True,
)


#############-----------Create file from ImageDataGenerator-----------#############
image = train_datagen.flow_from_directory('traindata/train', target_size=(
    128, 128), color_mode='grayscale', save_to_dir='augmented', class_mode='categorical', save_format='jpeg', batch_size=10)
image.next()
