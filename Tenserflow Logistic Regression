import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes in Iris dataset
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the neural network model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the neural network model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Neural Network Test Accuracy:", test_accuracy)

# Logistic Regression using TensorFlow
# Define the logistic regression model
logistic_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(X_train_scaled.shape[1],))  # 3 classes in Iris dataset
])

# Compile the model
logistic_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the logistic regression model
logistic_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the logistic regression model
logistic_test_loss, logistic_test_accuracy = logistic_model.evaluate(X_test_scaled, y_test)
print("Logistic Regression Test Accuracy:", logistic_test_accuracy)
