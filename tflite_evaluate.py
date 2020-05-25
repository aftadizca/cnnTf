import tensorflow as tf
import numpy as np
import os

TFLITE_MODEL = "model6.2_2020_05_25"
TFLITE_DIRS = "model/model6/"
TRAIN_DATA = 'traindata/valid'


tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_DIRS+TFLITE_MODEL+".tflite")
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

#f=open('asd.csv','ab')

total_files = 0
true_prediction = 0
index = 0

for folder in os.listdir('traindata/train'):
    path_label = os.path.join(TRAIN_DATA,folder)
    num_files = len([f for f in os.listdir(path_label)])
    total_files += num_files
    print(folder+" : "+str(num_files)) 
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
        if tflite_model_predictions[0][index] > 0.5:
            true_prediction+=1
        #np.savetxt(f,np.asarray(tflite_model_predictions),delimiter=',')
    index+=1
#f.close()

print("TOTAL IMG :" + str(total_files))
print("TOTAL TRUE :" + str(true_prediction))
print("ACCURACY : ", str(true_prediction/total_files))
