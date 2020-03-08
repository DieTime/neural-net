from network import Network

a = Network([
    {"neurons": 3, "activation": "tanh"},
    {"neurons": 5, "activation": "tanh"},
    {"neurons": 2, "activation": "softmax"},
])

print(a)