# NeuralNet
<p align="center">
  <img src="https://i.ibb.co/vk7hk8B/NN.png" width="400">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-black.svg" alt="version">
</p>

### About
A simple library for training neural networks with full model customization.

### Features
- Activation functions: linear, sigmoid, tanh, RELU, Leaky RELU, softmax.
- Great opportunity to customize the model.
- Informative display of the learning process and testing.
    ```
    Train error:0.0000173: 100%|██████████| 1000/1000 [00:02<00:00, 443.65it/s]
    Test error: 0.0000249: 100%|██████████| 5/5 [00:00<00:00, 4859.02it/s]
    ```
- Demonstration of the training process schedule.
- Network weights output.

### Example
```python
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
```


