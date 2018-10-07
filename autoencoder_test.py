from keras.models import load_model
import numpy as np

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')

inputs = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# inputs = np.array([[-10], [-5.1], [-3.3], [1.5], [3.2], [5.3], [88.8], [100]])
x = encoder.predict(inputs)
y = decoder.predict(x)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(x))
print('Decoded: {}'.format(y))
