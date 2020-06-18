import disable_log
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import os
import numpy as np
import sys
import tensorflow_model_optimization as tfmot



model_name = "model4.v1_2020_06_17"
model_dirs = "model/model4/"

# load json and create model
loaded_model = load_model(model_dirs + model_name + ".hdf5") or quit()

# model_len = len(loaded_model.layers)

# base_model = tf.keras.Sequential([loaded_model.layers[0]])
# base_model.summary()
# my_model =tf.keras.Sequential([loaded_model.layers[1],loaded_model.layers[2]])
# my_model.summary()
#print(loaded_model)
#loaded_model = tfmot.quantization.keras.quantize_model(loaded_model)

# for i, w in enumerate(loaded_model.get_weights()):
#     print(
#         "{} -- Total:{}, Zeros: {:.2f}%".format(
#             loaded_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
#         )
#     )

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()

with open(model_dirs + model_name + ".tflite", "wb") as f:
    f.write(tflite_model)
print("Convert Done")
