import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Expected output
y = np.array([[0], [1], [1], [0]])

# Seed for random weight initialization
np.random.seed(1)

# Network architecture
num_inputs = 2
num_hidden_nodes = 2
num_outputs = 1

# Initialize weights randomly with mean 0
hidden_weights = np.random.randn(num_inputs, num_hidden_nodes)
output_weights = np.random.randn(num_hidden_nodes, num_outputs)

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, hidden_weights))
    layer_2 = sigmoid(np.dot(layer_1, output_weights))

    # Backpropagation
    layer_2_error = layer_2 - y
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(output_weights.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    output_weights -= learning_rate * layer_1.T.dot(layer_2_delta)
    hidden_weights -= learning_rate * layer_0.T.dot(layer_1_delta)

# Test the network
print("Output:")
print(layer_2)
