import tensorflow as tf
import time
from list_model import ListModel

def detectModel(model_dirs, modelpy, beforeFunc=True):
    try:
        list_model = ListModel(modelpy.MODEL_NAME, model_dirs)
        if list_model.count() == 0:
            raise FileNotFoundError
        else:
            print()
            list_model.printList()
            user_input = input("Select model [n -> to create new model] : ")

            if user_input.isdecimal:
                print("\nLoading model -> ", list_model.getFileName(int(user_input)))
                ####?Load Model
                myModel = tf.keras.models.load_model(
                    list_model.getFilePath(int(user_input)))
                ####?Load Costum Function
                if beforeFunc:
                    print("Call function before compiling model.")
                    time.sleep(5)
                    myModel = modelpy.beforeCompile(myModel)
                ####?Compile
                myModel.compile(
                    optimizer=modelpy.OPTIMIZER,
                    loss=modelpy.LOSS,
                    metrics=modelpy.METRICS
                )
            elif user_input == 'n':
                myModel = modelpy.model()
            elif user_input == '' or user_input.isalpha:
                quit()
        return myModel

    except FileNotFoundError:
        user_input = input("\nNo model detected. Create new? [y/n] : ")
        if user_input == 'y':
            myModel = modelpy.model()
        else:
            quit()
        return myModel