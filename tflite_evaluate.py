import tensorflow as tf
import numpy as np
import os, fnmatch, ntpath
import sys
from time import sleep

def evaluateModel(model_dirs, train_dirs):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_dirs)
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    model_name = ntpath.basename(model_dirs).replace(".tflite", "")

    print(f"\n{model_name:^{size_column*4}s}\n")
    shape = input_details[0]['shape'][3]
    if shape == 3:
        COLOR_MODE = 'rgb'
    else:
        COLOR_MODE = 'grayscale'
    #print("shape:", input_details[0]['shape'][3])
    # print("type:", input_details[0]['dtype'])
    # print("\n== Output details ==")
    # print("shape:", output_details[0]['shape'])
    # print("type:", output_details[0]['dtype'])

    #f=open('asd.csv','ab')

    total_files = 0
    true_prediction = 0
    index = 0

    print(f'{"":=^{size_column*5}s}')
    print(f'{"   LABEL":<{size_column}s}|{"TOTAL DATA":^{size_column}s}|{"SUKSES":^{size_column}s}|{"GAGAL":^{size_column}s}|{"AKURASI":^{size_column}s}') 
    print(f'{"":=^{size_column*5}s}')

    for folder in os.listdir(train_dirs):
        path_label = os.path.join(train_dirs,folder)
        num_files = 0
        true_label = 0
        false_label = 0
        #f.write(bytes(folder+'\n','utf-8'))
        for files in os.listdir(path_label):
            img_predict = tf.keras.preprocessing.image.load_img(os.path.join(path_label,files), target_size=(128, 128), color_mode = COLOR_MODE)
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
            else:
                false_label+=1
            num_files+=1
            #np.savetxt(f,np.asarray(tflite_model_predictions),delimiter=',')
            sys.stdout.write('\r')
            sys.stdout.write(f'{"   "+folder:<{size_column}s}|{str(num_files):^{size_column}s}|{str(true_label):^{size_column}s}|{str(false_label):^{size_column}s}|{str(round(true_label/num_files*100,2))+"%":^{size_column}s}')
            sys.stdout.flush()
        print()
        total_files += num_files
        index+=1
    #f.close()
    print()
    print("TOTAL GAMBAR  :" , str(total_files))
    print("TOTAL SUKSES  :" , str(true_prediction))
    print("TOTAL AKURASI : ", "{0:.2f}%\n".format(true_prediction/total_files*100))


rows, columns = os.popen('stty size', 'r').read().split()

size_column = int(int(columns)*0.5//3)

TFLITE_DIRS = "model/model10/"
data_dirs = 'traindata2/test'

## Detect model
model_list = fnmatch.filter(os.listdir(TFLITE_DIRS),'*.tflite')
for i, filename in enumerate(model_list,1):
    print(f'{i}. {filename}')

model_index = "0"
print('\n')

while model_index=="0":
    model_index = input("\033[FSelect model [a]ll : ")
    # Convert selecting model
    if model_index.isdigit():
        evaluateModel(TFLITE_DIRS + model_list[int(model_index)-1],data_dirs)
    # Convert all model
    elif model_index=='a':
        for path in model_list:
            evaluateModel(TFLITE_DIRS + path,data_dirs)
    elif model_index == '':
        quit()
    else:
        model_index = "0"
##

