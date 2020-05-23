import disable_log
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import os
import numpy as np
import sys


model_name = "model4.2020.05.23"

# load json and create model
loaded_model = load_model(model_name + ".hdf5") or quit()

# for i, w in enumerate(loaded_model.get_weights()):
#     print(
#         "{} -- Total:{}, Zeros: {:.2f}%".format(
#             loaded_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
#         )
#     )

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()

with open(model_name + ".tflite", "wb") as f:
    f.write(tflite_model)
print("Convert Done")