import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras import backend as K

#############-----------Parameter-----------#############
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.0001, momentum=0.9, nesterov=True)
img_width = 128
img_height = 128
qty_class = 4
dropout = 0.3
color_mode = 'grayscale'
class_mode = 'categorical'
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
activation_conv = 'relu'
activation_dense = 'relu'
activation_dense_end = 'sigmoid'

#############-----------Check Channel-----------#############
print("Check channel position")
if K.image_data_format() == "channels_first":
    print("Image contain channels_first")
    input_shape = (1, img_width, img_height)
else:
    print("Image contain channels_last")
    input_shape = (img_width, img_height, 1)


#############-----------MODEL-----------#############
def model():
    print("Create Model")
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     activation=activation_conv, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation=activation_conv))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), padding='same', activation=activation_conv))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same', activation=activation_conv))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(256, activation=activation_dense))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation=activation_dense))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation=activation_dense))
    model.add(Dropout(dropout))
    model.add(Dense(qty_class, activation=activation_dense_end))

    print("Compile Model")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model
