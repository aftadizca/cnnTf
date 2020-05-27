import tensorflow as tf
import numpy as np
import os
import sys
from time import sleep

TFLITE_MODEL = "model7.1_2020_05_26"
TFLITE_DIRS = "model/model7/"
TRAIN_DATA = 'traindata/train'

tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_DIRS+TFLITE_MODEL+".tflite")
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print(f"\n====== {TFLITE_MODEL} ======\n")
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])
# print("\n== Output details ==")
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])

#f=open('asd.csv','ab')

total_files = 0
true_prediction = 0
index = 0

print("{:<20s}\t|\t{:<4s}\t|\t{:<4s}\t|\t{:<4s}".format("   LABEL","TOTAL", "TRUE", "TRUE/TOTAL")) 
for folder in os.listdir(TRAIN_DATA):
    path_label = os.path.join(TRAIN_DATA,folder)
    num_files = 0
    true_label = 0
    #f.write(bytes(folder+'\n','utf-8'))
    for files in os.listdir(path_label):
        img_predict = tf.keras.preprocessing.image.load_img(os.path.join(path_label,files), target_size=(128, 128), color_mode = 'rgb')
        img_predict = tf.keras.preprocessing.image.img_to_array(img_predict)
        img_predict = np.expand_dims(img_predict/255, axis=0)
        # Set batch of images into input tensor
        tflite_interpreter.set_tensor(input_details[0]['index'], img_predict)
        # Run inference
        tflite_interpreter.invoke()
        # Get prediction results
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        # Count right prediction
        if tflite_model_predictions[0][index] > 0.5:
            true_prediction+=1
            true_label+=1
        num_files+=1
        #np.savetxt(f,np.asarray(tflite_model_predictions),delimiter=',')
        sys.stdout.write('\r')
        sys.stdout.write(f"{folder:<20s}\t|\t{str(num_files):<4s}\t|\t{str(true_label):<4s}\t|\t{true_label/num_files*100:.2f}%")
        sys.stdout.flush()
    print()
    total_files += num_files
    index+=1
#f.close()
print()
print("TOTAL IMG :" + str(total_files))
print("TOTAL TRUE :" + str(true_prediction))
print("ACCURACY : ", "{0:.3f}%".format(true_prediction/total_files*100))
