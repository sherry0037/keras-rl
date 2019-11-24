from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import numpy as np
import utils
import os

from model import CNNModel

np.random.seed(1337)

# Load dataset
x_train, y_train, x_test, y_test = utils.load_datasets()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
input_shape = (image_size, image_size, 1)
output_dim = 128
batch_size = 128
num_epochs = 600

kernel_size = 3
layer_filters = [32, 64]

model = CNNModel(input_shape, output_dim, layer_filters, kernel_size).build()
model.summary()

model.compile(loss=keras.losses.mean_squared_error,
                optimizer=keras.optimizers.Adam())

model.fit(x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True)

# serialize model to JSON
model_path = "./saved_model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_json = model.to_json()
with open(model_path+"model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_path+"model1.h5")
print("Saved model to disk")