import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plot


class NeuralNet(object):

    def __init__(self, setup):

        # Dictionary of activation functions
        self.functions = {
            'linear': [np.vectorize(self.linear), np.vectorize(self.linear2derivative)],
            'sigmoid': [np.vectorize(self.sigmoid), np.vectorize(self.sigmoid2derivative)],
            'tanh': [np.vectorize(self.tanh), np.vectorize(self.tanh2derivative)],
            'relu': [np.vectorize(self.relu), np.vectorize(self.relu2derivative)],
            'leaky_relu': [np.vectorize(self.leaky_relu), np.vectorize(self.leaky_relu2derivative)],
            'softmax': [self.softmax, self.softmax2derivative]
        }

        # List of layers
        self.__layers = []

        # List of weights data
        self.__weights = []

        # Throw exceptions if necessary.
        if setup[0]['neurons'] < 1:
            raise ValueError('Number of neurons in layer can\'t be less then one!')

        # Forming the weights
        for i in range(1, len(setup)):

            # Throw exceptions if necessary
            if setup[i]['neurons'] < 1:
                raise ValueError('Number of neurons in layer can\'t be less then zero!')

            if not (setup[i]['activation'] in self.functions):
                raise ValueError('Incorrect activation function!')

            self.__weights.append({
                'matrix': np.random.random_sample((setup[i - 1]['neurons'] + 1, setup[i]['neurons'])),
                'activation': setup[i]['activation'],
            })

    # Feed forward process
    def prediction(self, input_data):

        # Clearing list of layers
        self.__layers.clear()

        # Initializing first layer with bias
        layer = np.append(np.array([input_data]), np.array([[1]]), axis=1)

        # Forming the output layer
        for data in self.__weights:
            # Remember layer values
            self.__layers.append(np.array(layer))

            # Calculating current layer with bias
            no_bias_layer = self.functions[data['activation']][0](np.dot(layer, data['matrix']))
            layer = np.append(no_bias_layer, np.array([[1]]), axis=1)

        # Remember layer values
        self.__layers.append(np.array(layer))

        # Return result of prediction
        return np.delete(layer, -1, axis=1)

    # Back propagation process
    def back_propagation(self, input_data, output_data, lr):
        # Getting prediction
        output = self.prediction(input_data)

        # Calculating prediction error
        actual = np.array(output_data)
        error = actual - output

        # Back propagation of error
        for i in reversed(range(len(self.__weights))):
            # Selecting derivative function from dict
            derivative = self.functions[self.__weights[i]['activation']][1]

            # Calculating layer gradient
            gradient = derivative(np.delete(self.__layers[i + 1], -1, axis=1))

            # Calculating delta
            delta = np.dot(self.__layers[i].T, error * gradient * lr)

            # Calculating prev layer error
            error = np.delete(np.dot(error, self.__weights[i]['matrix'].T), -1, axis=1)

            # Updating weights
            self.__weights[i]['matrix'] += delta

        # Return MSE
        return np.sum(np.square(actual - output)) / len(output_data)

    # Network learning process
    def train(self, input_data, output_data, epochs, lr, shuffle=False):
        # Initializing progress bar
        progress_bar = tqdm(range(epochs))
        x = list(range(epochs))
        y = []
        # Initializing error
        error = None

        # Indexes of input data
        range_list = list(range(len(input_data)))

        # Launching epochs
        for _ in progress_bar:

            # Shuffling indices if needed
            if shuffle:
                random.shuffle(range_list)

            # Training network and printing error value
            for j in range_list:
                error = self.back_propagation(input_data[j], output_data[j], lr)

            progress_bar.set_description("Train error:" + '{0:.7f}'.format(error))
            y.append(error)

        plot.plot(x, y)
        plot.ylabel('error')
        plot.xlabel('epoch')
        plot.show()

    # Learning test function on the test set
    def test(self, input_data, output_data):
        # Initializing progress bar
        progress_bar = tqdm(range(len(input_data)))
        # Initializing error
        error = 0

        # Launching test
        for i in progress_bar:
            output = self.prediction(np.array(input_data[i]))
            error += np.sum(np.square(np.array(output_data[i]) - output)) / len(output_data[i])
            progress_bar.set_description("Test error: " + '{0:.7f}'.format(error / (i + 1)))

    # Print weights
    def weights(self):
        for i in range(len(self.__weights)):
            print("Weights " + str(i) + "->" + str(i + 1) + ":")
            print(self.__weights[i]['matrix'])
            print("Activation:", end='')
            print("'" + self.__weights[i]['activation'] + "'\n")

    # -------------------------- ACTIVATION FUNCTIONS ---------------------------

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
        e = np.exp(vector)
        return e / np.sum(e)

    @staticmethod
    def linear2derivative(_):
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

    # ------------------------------ LAYER TYPES --------------------------------

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
