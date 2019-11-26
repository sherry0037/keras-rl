from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import numpy as np
import matplotlib.pyplot as plt 
import utils
import os
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(1337)

batch_size = 8
num_epochs = 250
# Load dataset
x_train, y_train, x_test, y_test = utils.load_datasets()

image_size = x_train.shape[1]
input_dim = image_size*image_size
output_dim = 128

x_train = np.reshape(x_train, [-1, input_dim]) 
x_test = np.reshape(x_test, [-1, input_dim]) 
x_train = x_train.astype('float32') / 255 
x_test = x_test.astype('float32') / 255 

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

model = Sequential()
model.add(Dense(12, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(output_dim, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,  verbose=1, validation_data=(x_test, y_test))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# serialize model to JSON
model_path = "./saved_model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_json = model.to_json()
with open(model_path+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_path+"model.h5")
print("Saved model to disk")
