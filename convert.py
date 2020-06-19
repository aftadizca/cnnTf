import disable_log
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import numpy as np
import fnmatch, sys, os
# import tensorflow_model_optimization as tfmot


model_dirs = "model/model4/"

def convertModel(dirs, summary=False):
    # load json and create model
    loaded_model = load_model(dirs) or quit()

    if summary:
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

    tflite_dirs = dirs.replace(".hdf5",".tflite")

    with open(tflite_dirs, "wb") as f:
        f.write(tflite_model)
    print(f"Convert Done -> {tflite_dirs}")

## Detect model
model_list = fnmatch.filter(os.listdir(model_dirs),'*.hdf5')
for i, filename in enumerate(model_list,1):
    print(f'{i}. {filename}')

model_index = "0"
print('\n')

while model_index=="0":
    model_index = input("\033[FSelect model [a]ll : ")
    # Convert selecting model
    if model_index.isdigit():
        convertModel(model_dirs + model_list[int(model_index)-1])
    # Convert all model
    elif model_index=='a':
        for path in model_list:
            convertModel(model_dirs + path)
    elif model_index == '':
        quit()
    else:
        model_index = "0"
##

