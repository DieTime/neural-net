import numpy as np
import random


class Network(object):

    # Constructor
    def __init__(self, setup):
        # List of layers
        self.__layers = []
        # List of weights data
        self.__weights = []

        # Error processing
        if setup[0]['neurons'] < 1:
            raise ValueError('Number of neurons in layer can\'t be less then one!')

        # Weight formation
        for i in range(1, len(setup)):

            # Error processing
            if setup[i]['neurons'] < 1:
                raise ValueError('Number of neurons in layer can\'t be less then zero!')

            self.__weights.append({
                'matrix': np.random.random_sample((setup[i - 1]['neurons'], setup[i]['neurons'])),
                'activation': setup[i]['activation'],
                'bias': random.random()
            })

    # Feed forward process
    def prediction(self, input_data):
        # Dictionary of activation functions
        activations = {
            'linear': np.vectorize(self.linear),
            'sigmoid': np.vectorize(self.sigmoid),
            'tanh': np.vectorize(self.tanh),
            'relu': np.vectorize(self.relu),
            'leaky_relu': np.vectorize(self.leaky_relu),
            'softmax': self.softmax
        }

        # Clear list of layers
        self.__layers.clear()

        # Initialize first layer
        layer = np.array([input_data])

        # Form the output layer
        for data in self.__weights:
            # Remember layer values
            self.__layers.append(layer)
            # Calculate current layer
            layer = activations[data['activation']](np.dot(layer, data['matrix']) + data['bias'])

        # Remember layer values
        self.__layers.append(layer)

        # Output result
        return layer

    # Back propagation process
    def back_propagation(self, input_data, output_data, lr):
        self.prediction(input_data)

        # Dictionary of gradient functions
        gradients = {
            'linear': np.vectorize(self.linear2derivative),
            'sigmoid': np.vectorize(self.sigmoid2derivative),
            'tanh': np.vectorize(self.tanh2derivative),
            'relu': np.vectorize(self.relu2derivative),
            'leaky_relu': np.vectorize(self.leaky_relu2derivative),
            'softmax': self.softmax2derivative
        }

        error = np.array(output_data) - self.__layers[-1]
        for i in reversed(range(len(self.__weights))):
            gradient = gradients[self.__weights[i]['activation']](self.__layers[i + 1])
            delta = np.dot(self.__layers[i].T, error * gradient * lr)
            self.__weights[i]['matrix'] += delta
            error = np.dot(error, self.__weights[i]['matrix'].T)

        return np.sum(np.square(np.array(output_data) - self.__layers[-1]))

    def train(self, input_data, output_data, epochs, lr, shuffle=False, logging=True):
        for i in range(epochs):
            indexes = list(range(len(input_data)))
            if shuffle:
                random.shuffle(indexes)
            for j in indexes:
                error = self.back_propagation(input_data[j], output_data[j], lr)
                if logging:
                    print("Error: " + '{:.5}'.format(error))

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
        e = np.exp(vector - vector.max())
        return e / np.sum(e)

    @staticmethod
    def linear2derivative(x):
        return 1

    @staticmethod
    def sigmoid2derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh2derivative(x):
        return 1 - x ** 2

    @staticmethod
    def relu2derivative(x):
        return x >= 0

    @staticmethod
    def leaky_relu2derivative(x):
        return 1 if x >= 0 else 0.01

    @staticmethod
    def softmax2derivative(vector):
        return vector * (1. - vector)

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
