from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.models import Sequential

from abc import ABC, abstractmethod

#------------------------------------------------------------------------------#
class NNModel(ABC):
  def __init__(self, **kwargs):
    self.output_shape = kwargs.get('output_shape', 128)
    self.model_type = kwargs.get('model_type', None)

    if 'FFModel' in self.model_type.__name__:
      self.input_shape = 84*84
      self.layer_sizes = kwargs.get('layer_sizes', [32, 64])
    elif 'CNNModel' in self.model_type.__name__:
      self.input_shape =  (84, 84, 1)

  @abstractmethod
  def build(self):
    # Build the Model
    pass
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
class FFModel(NNModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

def build(self):
  model = Sequential()
  model.add(Dense(self.layer_sizes[0], input_dim=self.input_shape,
                  kernel_initializer='normal', activation='relu'))
  for i in range(1, len(self.layer_sizes)):
    model.add(Dense(self.layer_sizes[i], activation = 'relu'))
  model.add(Dense(self.output_shape, activation='linear'))
  return model
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# is softmax only meant for classification?
# beb
# beb see this i am also reading this https://stats.stackexchange.com/questions/335836/cnn-architectures-for-regression

class CNNModel2(NNModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def build(self):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=self.input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(self.output_shape, activation='linear'))
    return model
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
class CNNModel1(NNModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.layer_sizes = kwargs.get('layer_sizes', [32, 64])

  def build(self):
    inputs = Input(shape=self.input_shape, name='input_layer')
    x = inputs
    # Stack of Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use MaxPooling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in self.layer_sizes:
        x = Conv2D(filters=filters,
                   kernel_size=3,
                   strides=2,
                   activation='relu',
                   padding='same')(x)
    # Generate the output vector
    x = Flatten()(x)
    outputs = Dense(self.output_shape, name='output_vector')(x)
    # Instantiate Model
    model = Model(inputs, outputs, name='CNNmodel1')
    return model
#------------------------------------------------------------------------------#
