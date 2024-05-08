import numpy as np

# Define the dataset
dataset = {
    "0": [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
    "1": [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    "2": [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
    "39": [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]
}

# Define the labels
labels = {
    "0": [1, 0, 0, 0],
    "1": [0, 1, 0, 0],
    "2": [0, 0, 1, 0],
    "39": [0, 0, 0, 1]
}

# Convert data to numpy arrays
X = np.array([np.array(num).flatten() for num in dataset.values()])
Y = np.array(list(labels.values()))

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(15, 4)
        self.bias = np.zeros((1, 4))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)
    
    def backward(self, x, y, learning_rate):
        output = self.forward(x)
        error = y - output
        delta = error * self.sigmoid_derivative(output)
        self.weights += np.dot(x.T, delta) * learning_rate
        self.bias += np.sum(delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, x, y, epochs, learning_rate):
        for _ in range(epochs):
            self.backward(x, y, learning_rate)
    
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

# Create and train the neural network
nn = NeuralNetwork()
nn.train(X, Y, epochs=10000, learning_rate=0.01)

# Test the neural network
test_data = {
    "Number 0": [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
    "Number 1": [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    "Number 2": [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
    "Number 39": [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]
}

for name, data in test_data.items():
    prediction = nn.predict(np.array(data).flatten().reshape(1, -1))
    print(f"{name}: Predicted label - {prediction}")
