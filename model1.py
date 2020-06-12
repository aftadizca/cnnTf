import tensorflow as tf

#############-----------Parameter-----------#############
OPTIMIZER = tf.keras.optimizers.SGD(
    learning_rate=0.0001, momentum=0.9, nesterov=True)
IMG_WIDTH = 128
IMG_HEIGHT = 128
CLASS_NUM = 5
DROPOUT = 0.3
COLOR_MODE = 'grayscale'
CLASS_MODE = 'categorical'
LOSS = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
ACTIVATION_CONV = 'relu'
ACTIVATION_DENSE = 'relu'
ACTIVATION_DENSE_END = 'softmax'

#############-----------Check Channel-----------#############
print("Check channel position")
if tf.keras.backend.image_data_format() == "channels_first":
    print("Image contain channels_first")
    input_shape = (1, IMG_WIDTH, IMG_HEIGHT)
else:
    print("Image contain channels_last")
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)


#############-----------MODEL-----------#############
def model():
    print("Create Model")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                     activation=ACTIVATION_CONV, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation=ACTIVATION_CONV))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(256, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(256, activation=ACTIVATION_DENSE))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(CLASS_NUM, activation=ACTIVATION_DENSE_END))

    print("Compile Model")
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS
    )

    model.summary()

    return model

model()