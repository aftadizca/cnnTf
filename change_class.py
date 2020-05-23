from tensorflow import keras
import disable_log
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
import model1 as model  # change model
import numpy as np

model_name = 'model1-softmax'

loaded_model = load_model(model_name + '.hdf5')

loaded_model.compile(optimizer=model.optimizer,
                     loss=model.loss,
                     metrics=model.metrics)

# remove the last Dense layer of our model
print("\n BEFORE POP \n")
loaded_model.summary()
loaded_model.pop()
print("\n AFTER POP \n")
loaded_model.summary()

base_model_layers = loaded_model.output
pred = Dense(model.qty_class, activation=model.activation_dense_end, name='dense_4')(
    base_model_layers)
loaded_model = Model(inputs=loaded_model.input, outputs=pred)
loaded_model.compile(optimizer=model.optimizer,
                     loss=model.loss,
                     metrics=model.metrics)
print("\n NEW CLASS ADDED \n")
loaded_model.summary()
loaded_model.save(model_name + '.new-class' + '.hdf5')
