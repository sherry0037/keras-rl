# Train neural network to map RGB images to RAM output

import utils
import numpy as np
from models import *
from keras import optimizers

# Network parameters
model_type = LSTMModel
save_data = False
train_split = 0.8
layer_sizes = [32, 64] # reqd only for FFNN
seq_length = 3 #reqd only for LSTM
batch_size = 8
num_epochs = 60

np.random.seed(1337)

x_train, y_train, x_test, y_test = utils.load_data(model_type, train_split, save_data)

# Normalization
mean_train, sigma_train = np.mean(x_train, axis=0), np.std(x_train, axis=0)
x_train = (x_train - mean_train)
x_test = (x_test - mean_train)

if model_type == LSTMModel:
  x_train = x_train[:(x_train.shape[0]-(x_train.shape[0] % seq_length))]
  y_train = y_train[:(y_train.shape[0]-(y_train.shape[0] % seq_length))]
  x_test = x_test[:(x_test.shape[0]-(x_test.shape[0] % seq_length))]
  y_test = y_test[:(y_test.shape[0]-(y_test.shape[0] % seq_length))]

  x_train = x_train.reshape((-1, seq_length, 84, 84, 1))
  y_train = y_train.reshape((-1, seq_length, 128))
  x_test = x_test.reshape((-1, seq_length, 84, 84, 1))
  y_test = y_test.reshape((-1, seq_length, 128))

print(x_train.shape, y_train.shape)

model = model_type(layer_sizes = layer_sizes, model_type = model_type, seq_length = seq_length).build()
model.summary()

# sgd = optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False)
model.compile(loss='mse', optimizer='adam' , metrics=['mse','mae'])

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True)

utils.save_model(model, model_type)
utils.plot_history(history)
