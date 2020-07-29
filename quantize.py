import tensorflow as tf
import disable_log
from tensorflow.keras.models import load_model

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("/run/media/zaha/PROGRAMING/PYTHON/cnnTf/model/model7/model7.1_2020_05_26.hdf5")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter = converter.convert()
# converter.post_training_quantize=True
converter.inference_type = tf.uint8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0.0, 255.0)}  # mean_value, std_dev
tflite_model = converter.convert()
open("quantized_model.tflite", "wb").write(tflite_model)

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()
# open("model.tflite", "wb").write(tflite_model)
# #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.post_training_quantize=True
# tflite_quantized_model=converter.convert()
# open("quantized_model.tflite", "wb").write(tflite_quantized_model)