from Network import NeuralNet

# Initialize network
model = NeuralNet([
    NeuralNet.input(neurons=1),
    NeuralNet.hidden(neurons=2, activation='tanh'),
    NeuralNet.hidden(neurons=3, activation='tanh'),
    NeuralNet.output(neurons=3, activation='softmax')
])

# Training data
trainX = [[-1], [1], [0], [5], [-6], [-7]]
trainY = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]]

# Test data
testX = [[24], [33], [-52], [-8], [10]]
testY = [[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]

# Start training
model.train(trainX, trainY, epochs=1000, lr=1, shuffle=False)

# Calculate test data error
model.test(testX, testY)

# Print network weights
model.weights()
