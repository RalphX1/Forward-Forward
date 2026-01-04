"""
Automated Tests for Forward-Forward Algorithm

Run with: pytest test_forward_forward.py -v
"""

import pytest
import numpy as np
from forward_forward import FFLayer, FFNetwork


class TestFFLayer:
    """Tests for FFLayer class"""

    def test_layer_initialization(self):
        """Test that layer initializes with correct dimensions"""
        layer = FFLayer(input_dim=10, output_dim=5, learning_rate=0.01, threshold=2.0)

        assert layer.W.shape == (10, 5), "Weight matrix has wrong shape"
        assert layer.b.shape == (5,), "Bias vector has wrong shape"
        assert layer.learning_rate == 0.01, "Learning rate not set correctly"
        assert layer.threshold == 2.0, "Threshold not set correctly"

    def test_weight_initialization_range(self):
        """Test that weights are initialized in reasonable range"""
        layer = FFLayer(input_dim=100, output_dim=50)

        # Xavier initialization: limit = sqrt(6 / (in + out))
        expected_limit = np.sqrt(6 / (100 + 50))

        assert np.all(layer.W >= -expected_limit), "Weights below expected range"
        assert np.all(layer.W <= expected_limit), "Weights above expected range"
        assert np.all(layer.b == 0), "Biases should be initialized to zero"

    def test_forward_pass_shape(self):
        """Test that forward pass returns correct shape"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x = np.random.random((10, 5))  # 10 samples, 5 features

        activations, x_norm = layer.forward(x, normalize=True)

        assert activations.shape == (10, 3), "Activation shape incorrect"
        assert x_norm.shape == (10, 5), "Normalized input shape incorrect"

    def test_forward_pass_no_normalization(self):
        """Test forward pass without normalization"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x = np.random.random((10, 5))

        activations, x_out = layer.forward(x, normalize=False)

        assert activations.shape == (10, 3), "Activation shape incorrect"
        assert np.allclose(x_out, x), "Input should be unchanged when normalize=False"

    def test_forward_pass_relu(self):
        """Test that ReLU activation is applied correctly"""
        layer = FFLayer(input_dim=2, output_dim=2)
        layer.W = np.array([[1.0, -1.0], [1.0, 1.0]])
        layer.b = np.array([0.0, 0.0])

        x = np.array([[1.0, 1.0]])
        activations, _ = layer.forward(x, normalize=False)

        # Expected: [1*1 + 1*1, 1*(-1) + 1*1] = [2, 0]
        # After ReLU: [2, 0]
        assert activations[0, 0] > 0, "Positive activation should remain positive"
        assert activations[0, 1] == 0, "Negative/zero activation should be zero after ReLU"

    def test_compute_goodness_shape(self):
        """Test that goodness is computed per sample"""
        layer = FFLayer(input_dim=5, output_dim=3)
        activations = np.random.random((10, 3))

        goodness = layer.compute_goodness(activations)

        assert goodness.shape == (10,), "Goodness should be per-sample (1D array)"

    def test_compute_goodness_value(self):
        """Test that goodness is computed correctly"""
        layer = FFLayer(input_dim=5, output_dim=3)
        activations = np.array([[1.0, 2.0, 3.0]])

        goodness = layer.compute_goodness(activations)

        # Expected: 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        assert np.isclose(goodness[0], 14.0), "Goodness calculation incorrect"

    def test_train_step_shapes(self):
        """Test that train_step returns correct shapes"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x_pos = np.random.random((10, 5))
        x_neg = np.random.random((10, 5))

        a_pos, a_neg, g_pos, g_neg = layer.train_step(x_pos, x_neg)

        assert a_pos.shape == (10, 3), "Positive activations shape incorrect"
        assert a_neg.shape == (10, 3), "Negative activations shape incorrect"
        assert isinstance(g_pos, (float, np.floating)), "Positive goodness should be scalar"
        assert isinstance(g_neg, (float, np.floating)), "Negative goodness should be scalar"

    def test_train_step_updates_weights(self):
        """Test that training actually updates weights"""
        layer = FFLayer(input_dim=5, output_dim=3, learning_rate=0.1)
        x_pos = np.random.random((10, 5))
        x_neg = np.random.random((10, 5))

        # Save initial weights
        W_before = layer.W.copy()
        b_before = layer.b.copy()

        # Train
        layer.train_step(x_pos, x_neg)

        # Check weights changed
        assert not np.allclose(layer.W, W_before), "Weights should change after training"
        assert not np.allclose(layer.b, b_before), "Biases should change after training"


class TestFFNetwork:
    """Tests for FFNetwork class"""

    def test_network_initialization(self):
        """Test that network initializes with correct architecture"""
        network = FFNetwork(layer_dims=[10, 5, 3], learning_rate=0.01, threshold=2.0)

        assert len(network.layers) == 2, "Should have 2 layers (10->5 and 5->3)"
        assert network.layers[0].W.shape == (10, 5), "First layer shape incorrect"
        assert network.layers[1].W.shape == (5, 3), "Second layer shape incorrect"

    def test_network_forward_pass(self):
        """Test forward pass through entire network"""
        network = FFNetwork(layer_dims=[10, 5, 3])
        x = np.random.random((10, 10))  # 10 samples, 10 features

        activations = network.forward(x)

        assert len(activations) == 3, "Should return activations for input + 2 layers"
        assert activations[0].shape == (10, 10), "Input activations shape incorrect"
        assert activations[1].shape == (10, 5), "First layer activations shape incorrect"
        assert activations[2].shape == (10, 3), "Second layer activations shape incorrect"

    def test_network_train_step(self):
        """Test training step for entire network"""
        network = FFNetwork(layer_dims=[5, 4, 3])
        x_pos = np.random.random((10, 5))
        x_neg = np.random.random((10, 5))

        metrics = network.train_step(x_pos, x_neg)

        assert 'pos_goodness' in metrics, "Metrics should contain pos_goodness"
        assert 'neg_goodness' in metrics, "Metrics should contain neg_goodness"
        assert len(metrics['pos_goodness']) == 2, "Should have goodness for 2 layers"
        assert len(metrics['neg_goodness']) == 2, "Should have goodness for 2 layers"

    def test_network_predict(self):
        """Test prediction functionality"""
        network = FFNetwork(layer_dims=[5, 4, 3])
        x = np.random.random((10, 5))

        probabilities, goodness_values = network.predict(x)

        assert probabilities.shape == (10,), "Should return probability per sample"
        assert len(goodness_values) == 2, "Should return goodness for each layer"
        assert np.all(probabilities >= 0), "Probabilities should be non-negative"
        assert np.all(probabilities <= 1), "Probabilities should be <= 1"

    def test_network_learning(self):
        """Test that network actually learns (goodness changes)"""
        np.random.seed(42)

        # Create simple XOR-like problem
        X_pos = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
        X_neg = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)

        network = FFNetwork(layer_dims=[3, 8, 4], learning_rate=0.03, threshold=1.0)

        # Train for a few epochs
        metrics_before = network.train_step(X_pos, X_neg)
        initial_pos_goodness = metrics_before['pos_goodness'][0]
        initial_neg_goodness = metrics_before['neg_goodness'][0]

        for _ in range(100):
            network.train_step(X_pos, X_neg)

        metrics_after = network.train_step(X_pos, X_neg)
        final_pos_goodness = metrics_after['pos_goodness'][0]
        final_neg_goodness = metrics_after['neg_goodness'][0]

        # After training, positive goodness should increase and negative should decrease
        assert final_pos_goodness > initial_pos_goodness, "Positive goodness should increase with training"
        assert final_neg_goodness < initial_neg_goodness, "Negative goodness should decrease with training"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_sample_batch(self):
        """Test with single sample"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x = np.random.random((1, 5))  # Single sample

        activations, _ = layer.forward(x)
        goodness = layer.compute_goodness(activations)

        assert activations.shape == (1, 3), "Single sample forward pass failed"
        assert goodness.shape == (1,), "Single sample goodness failed"

    def test_large_batch(self):
        """Test with large batch"""
        layer = FFLayer(input_dim=10, output_dim=5)
        x = np.random.random((1000, 10))  # Large batch

        activations, _ = layer.forward(x)

        assert activations.shape == (1000, 5), "Large batch forward pass failed"

    def test_zero_input(self):
        """Test with zero input"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x = np.zeros((5, 5))

        activations, _ = layer.forward(x, normalize=False)
        goodness = layer.compute_goodness(activations)

        assert activations.shape == (5, 3), "Zero input forward pass failed"
        # With zero input and zero bias, activations should be zero
        assert np.allclose(goodness, 0.0), "Goodness of zero activations should be zero"

    def test_numerical_stability(self):
        """Test that normalization prevents extreme values"""
        layer = FFLayer(input_dim=5, output_dim=3)
        x = np.random.random((10, 5)) * 1000  # Very large values

        activations, x_norm = layer.forward(x, normalize=True)

        # Normalized input should have unit norm
        norms = np.linalg.norm(x_norm, axis=1)
        assert np.allclose(norms, 1.0), "Normalized inputs should have unit norm"

        # Activations should be finite
        assert np.all(np.isfinite(activations)), "Activations should be finite"


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_xor_classification(self):
        """Test complete XOR classification workflow"""
        np.random.seed(42)

        # XOR problem
        X_pos = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
        X_neg = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)

        # Create and train network
        network = FFNetwork(layer_dims=[3, 8, 4], learning_rate=0.03, threshold=1.0)

        # Train
        for _ in range(500):
            network.train_step(X_pos, X_neg)

        # Test predictions
        probs_pos, _ = network.predict(X_pos)
        probs_neg, _ = network.predict(X_neg)

        # Most positive samples should have probability > 0.4
        # Most negative samples should have probability < 0.6
        # (Not requiring perfect classification due to simple network)
        assert np.mean(probs_pos) > 0.4, "Network should lean towards positive for positive data"
        assert np.mean(probs_neg) < 0.6, "Network should lean towards negative for negative data"

    def test_reproducibility(self):
        """Test that training is reproducible with same seed"""
        np.random.seed(42)
        network1 = FFNetwork(layer_dims=[5, 3, 2], learning_rate=0.01)
        x_pos = np.random.random((10, 5))
        x_neg = np.random.random((10, 5))
        for _ in range(10):
            network1.train_step(x_pos, x_neg)
        W1_final = network1.layers[0].W.copy()

        np.random.seed(42)
        network2 = FFNetwork(layer_dims=[5, 3, 2], learning_rate=0.01)
        x_pos = np.random.random((10, 5))
        x_neg = np.random.random((10, 5))
        for _ in range(10):
            network2.train_step(x_pos, x_neg)
        W2_final = network2.layers[0].W.copy()

        assert np.allclose(W1_final, W2_final), "Training should be reproducible with same seed"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
