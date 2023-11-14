import keras
import tensorflow as tf


class Autoencoder1(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder1, self).__init__()
        self.input_shapes = input_shape

        print('encoder input shape:', input_shape)

        e_input_layer = keras.layers.Input(input_shape)
        e_layer = keras.layers.Conv1D(16, 8, activation='relu', input_shape=input_shape)(e_input_layer)
        e_layer = keras.layers.Conv1D(32, 6, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(input_shape[0], 5, activation='relu')(e_layer)
        e_output_layer = keras.layers.GlobalAveragePooling1D()(e_layer)

        print('encoder output shape:', e_output_layer.shape)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_output_layer)

        d_input_layer = keras.layers.Input(e_output_layer.shape)
        d_layer = keras.layers.Reshape((input_shape[0],1))(d_input_layer)
        d_layer = keras.layers.Conv1DTranspose(input_shape[0], 5, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(16, 6, activation='relu', padding='same')(d_layer)
        d_output_layer = keras.layers.Conv1DTranspose(1, 8, activation='relu', padding='same')(d_layer)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_output_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded
    

class Autoencoder2(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder2, self).__init__()
        self.input_shapes = input_shape

        print('encoder input shape:', input_shape)

        e_input_layer = keras.layers.Input(input_shape)
        e_layer = keras.layers.Conv1D(16, 8, activation='relu', input_shape=input_shape)(e_input_layer)
        e_layer = keras.layers.Conv1D(32, 6, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(64, 5, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(input_shape[0], 3, activation='relu')(e_layer)
        e_output_layer = keras.layers.GlobalAveragePooling1D()(e_layer)

        print('encoder output shape:', e_output_layer.shape)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_output_layer)


        d_input_layer = keras.layers.Input(e_output_layer.shape)
        d_layer = keras.layers.Reshape((input_shape[0],1))(d_input_layer)
        d_layer = keras.layers.Conv1DTranspose(input_shape[0], 3, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(32, 5, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(16, 6, activation='relu', padding='same')(d_layer)
        d_output_layer = keras.layers.Conv1DTranspose(1, 8, activation='relu', padding='same')(d_layer)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_output_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded
    

class Autoencoder3(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder3, self).__init__()
        self.input_shapes = input_shape

        print('encoder input shape:', input_shape)

        e_input_layer = keras.layers.Input(input_shape)
        e_layer = keras.layers.Conv1D(16, 8, activation='relu', input_shape=input_shape)(e_input_layer)
        e_layer = keras.layers.Conv1D(32, 6, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(64, 5, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(128, 3, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(input_shape[0], 3, activation='relu')(e_layer)
        e_output_layer = keras.layers.GlobalAveragePooling1D()(e_layer)

        print('encoder output shape:', e_output_layer.shape)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_output_layer)


        d_input_layer = keras.layers.Input(e_output_layer.shape)
        d_layer = keras.layers.Reshape((input_shape[0],1))(d_input_layer)
        d_layer = keras.layers.Conv1DTranspose(input_shape[0], 3, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(32, 5, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(16, 6, activation='relu', padding='same')(d_layer)
        d_output_layer = keras.layers.Conv1DTranspose(1, 8, activation='relu', padding='same')(d_layer)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_output_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded
    

class Autoencoder4(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder4, self).__init__()
        self.input_shapes = input_shape

        print('encoder input shape:', input_shape)

        e_input_layer = keras.layers.Input(input_shape)
        e_layer = keras.layers.Conv1D(16, 8, activation='relu', input_shape=input_shape)(e_input_layer)
        e_layer = keras.layers.Conv1D(32, 6, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(64, 5, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(128, 3, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(256, 3, activation='relu')(e_layer)
        e_layer = keras.layers.Conv1D(input_shape[0], 2, activation='relu')(e_layer)
        e_output_layer = keras.layers.GlobalAveragePooling1D()(e_layer)

        print('encoder output shape:', e_output_layer.shape)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_output_layer)


        d_input_layer = keras.layers.Input(e_output_layer.shape)
        d_layer = keras.layers.Reshape((input_shape[0],1))(d_input_layer)
        d_layer = keras.layers.Conv1DTranspose(input_shape[0], 2, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(128, 3, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(32, 5, activation='relu', padding='same')(d_layer)
        d_layer = keras.layers.Conv1DTranspose(16, 6, activation='relu', padding='same')(d_layer)
        d_output_layer = keras.layers.Conv1DTranspose(1, 8, activation='relu', padding='same')(d_layer)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_output_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded