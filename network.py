import numpy as np
import random


class Network(object):

    # Constructor
    def __init__(self, setup):
        # List of layers
        self.__layers = []
        # List of weights data
        self.__weights = []

        # Dictionary of activation functions
        activations = {
            'linear': np.vectorize(self.linear),
            'sigmoid': np.vectorize(self.sigmoid),
            'tanh': np.vectorize(self.tanh),
            'relu': np.vectorize(self.relu),
            'leaky_relu': np.vectorize(self.leaky_relu),
            'softmax': self.softmax
        }

        # Error processing
        if setup[0]['neurons'] < 1:
            raise ValueError('Number of neurons in layer can\'t be less then one!')

        # Weight formation
        for i in range(1, len(setup)):

            # Error processing
            if setup[i]['neurons'] < 1:
                raise ValueError('Number of neurons in layer can\'t be less then zero!')
            if not (setup[i]['activation'] in activations):
                raise ValueError('Invalid activation function name!')

            self.__weights.append({
                'matrix': np.random.random_sample((setup[i - 1]['neurons'], setup[i]['neurons'])),
                'activation': activations[setup[i]['activation']],
                'bias': random.random()
            })

    # Feed forward process
    def prediction(self, input_data):
        # Clear list of layers
        self.__layers.clear()

        # Initialize first layer
        layer = np.array([input_data])

        # Form the output layer
        for data in self.__weights:
            # Remember layer values
            self.__layers.append(layer)
            # Calculate current layer
            layer = data['activation'](np.dot(layer, data['matrix']) + data['bias'])

        # Remember layer values
        self.__layers.append(layer)

        # Output result
        return layer

    # ------------------------ STATIC METHODS --------------------------

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    @staticmethod
    def tanh(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def leaky_relu(x):
        return max(0.1 * x, x)

    @staticmethod
    def softmax(vector):
        return np.exp(vector) / np.exp(vector).sum(axis=0)

    @staticmethod
    def input(neurons):
        return {
            'neurons': neurons,
            'activation': 'linear'
        }

    @staticmethod
    def hidden(neurons, activation='tanh'):
        return {
            'neurons': neurons,
            'activation': activation
        }

    @staticmethod
    def output(neurons, activation='softmax'):
        return {
            'neurons': neurons,
            'activation': activation
        }
