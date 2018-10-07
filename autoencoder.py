from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import set_random_seed
import os


# Sets seed to produce the same sequence of results
def set_seed(seed):
    np.random.seed(seed)
    set_random_seed(seed)


class AutoEncoder:
    def __init__(self, encoding_dimension=8, dataset_length=1000):
        self.encoding_dimension = encoding_dimension  # The dimension that we are reducing to
        self.x = np.array(
            [[np.random.randint(1, 9)] for _ in range(dataset_length)])  # Array with 1000 random numbers from 1 to 8

    # Encodes input. Learns abstract features of the dataset, and compress it down.
    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)  # Creates Input with the shape of the generated dataset
        encoded = Dense(self.encoding_dimension, activation='tanh')(
            inputs)  # Creates the dense layer with the encoding_dimension
        model = Model(inputs, encoded)  # Creates the encoded model
        self.encoder = model
        return model

    # Decodes encoded model. Tries to reconstruct the input.
    def _decoder(self):
        inputs = Input(shape=(self.encoding_dimension,))
        decoded = Dense(1)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        encoder = self._encoder()  # Encoded model
        decoder = self._decoder()  # Decoded model

        inputs = Input(shape=self.x[0].shape)
        encoder_output = encoder(inputs)  # Encoded representation
        decoder_output = decoder(encoder_output)  # Reconstructed representation
        model = Model(inputs, decoder_output)  # Concatenates to one model

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')  # Compile model with stochastic gradient descent
        tensor_board_callback = TensorBoard(log_dir='./log/',
                                            histogram_freq=0,
                                            write_graph=True,
                                            write_images=True)  # Shows overall loss
        self.model.fit(self.x, self.x,
                       epochs=epochs,  # One forward (prediction) and backward (weight adjusting) pass for the network
                       batch_size=batch_size,  # Epoch broken up to batch size
                       callbacks=[tensor_board_callback])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')

        self.encoder.save(r'./weights/encoder_weights.h5', overwrite=True)
        self.decoder.save(r'./weights/decoder_weights.h5', overwrite=True)
        self.model.save(r'./weights/ae_weights.h5', overwrite=True)


if __name__ == '__main__':
    set_random_seed(2)
    auto_encoder = AutoEncoder(encoding_dimension=2, dataset_length=1000)
    auto_encoder.encoder_decoder()
    auto_encoder.fit(batch_size=50, epochs=1000)
    auto_encoder.save()
