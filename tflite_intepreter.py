import tensorflow as tf


# tflite_interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
tflite_interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
tensor_details = tflite_interpreter.get_tensor_details()
print(input_details)
print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

# >> == Input details ==
# >> shape: [  1 224 224   3]
# >> type: <class 'numpy.float32'>
# >> == Output details ==
# >> shape: [1 5]
# >> type: <class 'numpy.float32'>