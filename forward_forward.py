"""
Forward-Forward Algorithm Implementation

This implementation follows Hinton's Forward-Forward algorithm for training neural networks.
The key idea is to replace backpropagation with two forward passes:
1. Positive pass: real data (goodness should be HIGH)
2. Negative pass: fake/negative data (goodness should be LOW)

Each layer is trained locally to maximize goodness for positive data and minimize it for negative data.
Goodness is measured as the sum of squared activities in a layer (per sample).
"""

import numpy as np


class FFLayer:
    """A single layer in the Forward-Forward network"""

    def __init__(self, input_dim, output_dim, learning_rate=0.01, threshold=2.0):
        """
        Initialize a Forward-Forward layer

        Args:
            input_dim: Number of input features
            output_dim: Number of output features (neurons in this layer)
            learning_rate: Learning rate for weight updates
            threshold: Threshold for goodness (used in probability calculation)
        """
        # Initialize weights with Xavier/Glorot initialization for better stability
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.learning_rate = learning_rate
        self.threshold = threshold

    def forward(self, x, normalize=True):
        """
        Forward pass through the layer

        Args:
            x: Input data (batch_size, input_dim)
            normalize: Whether to apply layer normalization

        Returns:
            activations: Output activations (batch_size, output_dim)
        """
        # Layer normalization on input for stability
        if normalize:
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        z = np.dot(x, self.W) + self.b
        # ReLU activation
        activations = np.maximum(0, z)
        return activations, x

    def compute_goodness(self, activations):
        """
        Compute goodness for each sample in the batch
        Goodness = sum of squared activities per sample

        Args:
            activations: Layer activations (batch_size, output_dim)

        Returns:
            goodness: Goodness value for each sample (batch_size,)
        """
        # Sum across features (axis=1) to get one goodness value per sample
        return np.sum(activations ** 2, axis=1)

    def train_step(self, x_pos, x_neg):
        """
        Perform one training step using positive and negative data

        The layer learns to:
        - Maximize goodness for positive data
        - Minimize goodness for negative data

        Uses a loss-based approach with sigmoid probability for stability:
        - Loss for positive: -log(sigmoid(goodness - threshold))
        - Loss for negative: -log(1 - sigmoid(goodness - threshold))

        Args:
            x_pos: Positive (real) data (batch_size, input_dim)
            x_neg: Negative (fake) data (batch_size, input_dim)

        Returns:
            pos_activations: Activations for positive data
            neg_activations: Activations for negative data
            pos_goodness: Mean goodness for positive data
            neg_goodness: Mean goodness for negative data
        """
        # Forward pass for positive data
        a_pos, x_pos_norm = self.forward(x_pos, normalize=True)
        goodness_pos = self.compute_goodness(a_pos)

        # Forward pass for negative data
        a_neg, x_neg_norm = self.forward(x_neg, normalize=True)
        goodness_neg = self.compute_goodness(a_neg)

        # Compute probabilities using sigmoid
        # p = 1 / (1 + exp(-(goodness - threshold)))
        logits_pos = goodness_pos - self.threshold
        logits_neg = goodness_neg - self.threshold

        # Use sigmoid for numerical stability
        p_pos = 1 / (1 + np.exp(-logits_pos))
        p_neg = 1 / (1 + np.exp(-logits_neg))

        # Compute loss gradients
        # For positive data: want p → 1, so gradient of -log(p) = -(1-p)
        # For negative data: want p → 0, so gradient of -log(1-p) = p
        # The gradient w.r.t. goodness is: dp/d(goodness) * d(loss)/dp
        # dp/d(goodness) = p * (1 - p) (derivative of sigmoid)

        # Simplified: For positive, push goodness up; for negative, push down
        # Gradient scaling by sigmoid derivative for smooth learning
        grad_scale_pos = -(1 - p_pos)  # Negative log-likelihood gradient
        grad_scale_neg = p_neg

        # Compute weight gradients
        # d(goodness)/d(activation) = 2 * activation
        # d(activation)/d(W) = x * (z > 0)

        z_pos = np.dot(x_pos_norm, self.W) + self.b
        z_neg = np.dot(x_neg_norm, self.W) + self.b

        mask_pos = (z_pos > 0).astype(float)
        mask_neg = (z_neg > 0).astype(float)

        # Gradient of goodness w.r.t. pre-activation
        # Chain rule: grad_scale * 2 * activation * mask
        grad_pos = (grad_scale_pos[:, np.newaxis] * 2 * a_pos * mask_pos)
        grad_neg = (grad_scale_neg[:, np.newaxis] * 2 * a_neg * mask_neg)

        # Weight updates
        dW_pos = np.dot(x_pos_norm.T, grad_pos) / len(x_pos)
        db_pos = np.mean(grad_pos, axis=0)

        dW_neg = np.dot(x_neg_norm.T, grad_neg) / len(x_neg)
        db_neg = np.mean(grad_neg, axis=0)

        # Update: minimize negative log-likelihood
        self.W -= self.learning_rate * (dW_pos + dW_neg)
        self.b -= self.learning_rate * (db_pos + db_neg)

        return a_pos, a_neg, np.mean(goodness_pos), np.mean(goodness_neg)


class FFNetwork:
    """Multi-layer Forward-Forward Network"""

    def __init__(self, layer_dims, learning_rate=0.01, threshold=2.0):
        """
        Initialize a multi-layer FF network

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            learning_rate: Learning rate for all layers
            threshold: Goodness threshold for all layers
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            layer = FFLayer(layer_dims[i], layer_dims[i+1], learning_rate, threshold)
            self.layers.append(layer)

    def forward(self, x, normalize=True):
        """
        Forward pass through all layers

        Args:
            x: Input data (batch_size, input_dim)
            normalize: Whether to apply layer normalization

        Returns:
            activations: List of activations for each layer
        """
        activations = [x]
        current = x
        for layer in self.layers:
            current, _ = layer.forward(current, normalize=normalize)
            activations.append(current)
        return activations

    def train_step(self, x_pos, x_neg):
        """
        Train all layers with one positive and negative pass

        Args:
            x_pos: Positive data (batch_size, input_dim)
            x_neg: Negative data (batch_size, input_dim)

        Returns:
            metrics: Dictionary with goodness values for each layer
        """
        metrics = {'pos_goodness': [], 'neg_goodness': []}

        # Train each layer sequentially
        current_pos = x_pos
        current_neg = x_neg

        for i, layer in enumerate(self.layers):
            a_pos, a_neg, g_pos, g_neg = layer.train_step(current_pos, current_neg)
            metrics['pos_goodness'].append(g_pos)
            metrics['neg_goodness'].append(g_neg)

            # Use activations as input to next layer
            current_pos = a_pos
            current_neg = a_neg

        return metrics

    def predict(self, x):
        """
        Predict whether input is positive or negative based on goodness

        Args:
            x: Input data (batch_size, input_dim)

        Returns:
            probabilities: Probability that each sample is positive (batch_size,)
            goodness_values: List of goodness values for each layer
        """
        activations = self.forward(x)
        goodness_values = []

        # Compute goodness for each layer
        for i, layer in enumerate(self.layers):
            g = layer.compute_goodness(activations[i+1])
            goodness_values.append(g)

        # Use final layer's goodness for classification
        final_goodness = goodness_values[-1]
        probabilities = 1 / (1 + np.exp(-(final_goodness - self.layers[-1].threshold)))

        return probabilities, goodness_values


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Forward-Forward Algorithm - XOR Problem Example")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # XOR problem: positive data has last bit = 1, negative has last bit = 0
    X_pos = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=float)

    X_neg = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=float)

    # Create network: 3 -> 8 -> 4
    # Using larger layers to avoid dead neurons
    network = FFNetwork(
        layer_dims=[3, 8, 4],
        learning_rate=0.03,
        threshold=1.0
    )

    # Training
    epochs = 1000
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        metrics = network.train_step(X_pos, X_neg)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}:")
            for layer_idx in range(len(network.layers)):
                pos_g = metrics['pos_goodness'][layer_idx]
                neg_g = metrics['neg_goodness'][layer_idx]
                print(f"  Layer {layer_idx + 1}: Pos goodness = {pos_g:.4f}, Neg goodness = {neg_g:.4f}")

    # Testing
    print("\n" + "=" * 70)
    print("Testing Phase")
    print("=" * 70)

    print("\n--- Positive Data Predictions ---")
    probs_pos, goodness_pos = network.predict(X_pos)
    for i, x in enumerate(X_pos):
        print(f"Input {x} -> Probability: {probs_pos[i]:.4f}, Final Goodness: {goodness_pos[-1][i]:.4f}")

    print("\n--- Negative Data Predictions ---")
    probs_neg, goodness_neg = network.predict(X_neg)
    for i, x in enumerate(X_neg):
        print(f"Input {x} -> Probability: {probs_neg[i]:.4f}, Final Goodness: {goodness_neg[-1][i]:.4f}")

    # Test on random data
    print("\n--- Random Data Predictions ---")
    X_random = np.random.random((4, 3))
    probs_random, goodness_random = network.predict(X_random)
    for i, x in enumerate(X_random):
        print(f"Input {x} -> Probability: {probs_random[i]:.4f}, Final Goodness: {goodness_random[-1][i]:.4f}")

    print("\n" + "=" * 70)
    print("Explanation:")
    print("=" * 70)
    print("""
The Forward-Forward algorithm trains the network to:
- Give HIGH goodness scores to positive data (last bit = 1)
- Give LOW goodness scores to negative data (last bit = 0)

Goodness = sum of squared activities in each layer (per sample)

The network learns WITHOUT backpropagation, using only local learning rules:
- Each layer tries to maximize goodness for positive data
- Each layer tries to minimize goodness for negative data

Probability > 0.5 indicates the network thinks the input is positive data.
Probability < 0.5 indicates the network thinks the input is negative data.
""")
