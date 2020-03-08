import numpy as np
import random


class Network(object):

    def __init__(self, setup):
        self.weights = []
        for i in range(1, len(setup)):
            self.weights.append(dict(
                weight=np.random.random_sample((setup[i]["neurons"], setup[i - 1]["neurons"])),
                activation=setup[i]["activation"],
                bias=random.random()
            ))