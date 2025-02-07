import numpy as np

# Activation Functions
class Linear:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones_like(x)

class ReLU:
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

class Sigmoid:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        sig = 1 / (1 + np.exp(-x))  # Compute sigmoid(x) first
        return sig * (1 - sig)  # Correct derivative formula

class Tanh:
    @staticmethod
    def activate(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x) ** 2  # Correct derivative

class Softmax:
    @staticmethod
    def activate(x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def derivative(x):
        return NotImplemented  # Softmax derivative is complex (handled in cross-entropy)

# Loss Functions
class LossFunction:
    @staticmethod
    def mse(predictions, labels):
        return np.mean((predictions - labels) ** 2)

    @staticmethod
    def cross_entropy(predictions, labels):
        return -np.mean(labels * np.log(predictions + 1e-10))

# Deep Neural Network
class DeepNeuralNetwork:
    def __init__(self, layer_sizes, activation_functions):
        """
        Initialize the Deep Neural Network.
        :param layer_sizes: List of layer sizes (e.g., [input_size, hidden_size, output_size]).
        :param activation_functions: List of activation functions for each layer.
        """
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []

        # Initialize weights and biases using Xavier/Glorot initialization
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))  # Use zeros for biases

    def forward(self, inputs):
        """
        Perform forward propagation.
        :param inputs: Input values.
        :return: Outputs and pre-activation values of all layers.
        """
        activations = [inputs]
        pre_activations = []  # Store z-values for proper backpropagation

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation_functions[i].activate(z)
            pre_activations.append(z)  # Save pre-activation
            activations.append(a)

        return activations, pre_activations

    def train(self, inputs, labels, epochs=1000, learning_rate=0.1):
        """
        Train the network using backpropagation.
        :param inputs: Training data.
        :param labels: Target labels.
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        """
        for epoch in range(epochs):
            # Forward pass
            activations, pre_activations = self.forward(inputs)

            # Compute error
            error = activations[-1] - labels

            # Backpropagation
            deltas = [error * self.activation_functions[-1].derivative(pre_activations[-1])]  # Last layer derivative
            for i in range(len(self.weights) - 1, 0, -1):
                delta = np.dot(deltas[-1], self.weights[i].T) * self.activation_functions[i - 1].derivative(pre_activations[i - 1])
                deltas.append(delta)
            deltas.reverse()

            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i])
                self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def predict(self, inputs):
        """
        Predict the output for given inputs.
        :param inputs: Input values.
        :return: Output of the network.
        """
        return self.forward(inputs)[0][-1]  # Return last activation layer

# Example usage
if __name__ == "__main__":
    # Example dataset (2D inputs and binary labels for XOR gate)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])  # XOR output

    # Create and train the Deep Neural Network
    layer_sizes = [2, 4, 1]  # Input: 2 neurons, Hidden: 4 neurons, Output: 1 neuron
    activation_functions = [ReLU(), ReLU(), Sigmoid()]  # Activations for each layer
    dnn = DeepNeuralNetwork(layer_sizes, activation_functions)
    dnn.train(inputs, labels, epochs=10000, learning_rate=0.1)

    # Test the Deep Neural Network
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = dnn.predict(test_inputs)
    print(f"Predictions: {predictions}")

    # Calculate Mean Squared Error (MSE)
    mse = LossFunction.mse(predictions, labels)
    print(f"Mean Squared Error: {mse}")
