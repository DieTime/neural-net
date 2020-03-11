from network import Network

a = Network([
    Network.input(neurons=2),
    Network.hidden(neurons=3, activation='tanh'),
    Network.hidden(neurons=3, activation='tanh'),
    Network.output(neurons=1, activation='sigmoid')
])

data_set = {
    "input": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "output": [[0], [1], [1], [0]]
}

a.train(data_set['input'], data_set['output'], 10000, 0.02, False)
print(a.prediction([0, 0]))
print(a.prediction([0, 1]))
print(a.prediction([1, 0]))
print(a.prediction([1, 1]))
