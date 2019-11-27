# Train neural network to map RGB images to RAM output

import utils
import numpy as np
from models import *
from keras import optimizers

# Network parameters
model_type = CNNModel2
save_data = True
train_split = 0.8
layer_sizes = [0, 0]
batch_size = 8
num_epochs = 60

np.random.seed(1337)

x_train, y_train, x_test, y_test = utils.load_data(model_type, train_split, save_data)

# Normalization
mean_train, sigma_train = np.mean(x_train, axis=0), np.std(x_train, axis=0)
x_train = (x_train - mean_train)
x_test = (x_test - mean_train)

model = model_type(layer_sizes = layer_sizes, model_type = model_type).build()
model.summary()

# sgd = optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False)
model.compile(loss='mse', optimizer='adam' , metrics=['mse','mae'])

print(x_train.shape, y_train.shape)
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True)

utils.save_model(model, model_type)
utils.plot_history(history)
