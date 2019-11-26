from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model


class CNNModel():
    def __init__(self, input_shape, output_shape, layer_filters, kernel_size):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_filters = layer_filters
        self.kernel_size = kernel_size

    def build(self):
        # Build the Model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use MaxPooling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in self.layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=self.kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        # Generate the output vector
        x = Flatten()(x)
        outputs = Dense(self.output_shape, name='latent_vector')(x)

        # Instantiate Model
        model = Model(inputs, outputs, name='encoder')

        return model
