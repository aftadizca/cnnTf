import disable_log
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import numpy as np
import fnmatch, sys, os
import tensorflow_model_optimization as tfmot


model_dirs = "model/model4/"

## Detect model
model_list = fnmatch.filter(os.listdir(model_dirs),'*.hdf5')
for i, filename in enumerate(model_list,1):
    print(f'{i}. {filename}')

model_index = 0
print(len(model_list))

while not(model_index > 0 and model_index <= len(model_list)):
    model_index = input("\033[FSelect model : ")
    if model_index.isdigit():
        model_index = int(model_index)
    else:
        model_index = 0
##


# load json and create model
loaded_model = load_model(model_dirs + model_list[model_index-1]) or quit()

loaded_model.summary()

# model_len = len(loaded_model.layers)2

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

tflite_dirs = model_dirs + model_list[model_index-1].replace(".hdf5",".tflite")

with open(tflite_dirs, "wb") as f:
    f.write(tflite_model)
print(f"Convert Done -> {tflite_dirs}")
