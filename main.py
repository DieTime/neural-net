from network import Network

a = Network([
    Network.input(neurons=2),
    Network.hidden(neurons=2, activation='tanh'),
    Network.output(neurons=3, activation='softmax')
])

train_data = {
    "input": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "output": [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
}

a.train(train_data["input"], train_data["output"], 5000, 0.1, True)
