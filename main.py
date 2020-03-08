from network import Network

a = Network([
    Network.input(neurons=3),
    Network.hidden(neurons=3, activation='tanh'),
    Network.hidden(neurons=3, activation='tanh'),
    Network.output(neurons=3, activation='leaky_relu')
])

print(a.prediction([1, 0, 1]))
