# Forward-Forward Algorithm Implementation

This repository contains a corrected implementation of Geoffrey Hinton's Forward-Forward (FF) algorithm for training neural networks.

## 📁 Project Structure

```
Forward-Forward/
├── forward_forward.py       # Core implementation (FFLayer, FFNetwork)
├── mnist_example.py         # MNIST-style digit classification example
├── gui.py                   # Interactive GUI for training and testing
├── test_forward_forward.py  # Automated test suite (20 tests)
├── PAPER_VERIFICATION.md    # Verification against Hinton's paper ⭐
├── COMPATIBILITY.md         # Cross-platform support guide
├── QUICKSTART.md           # Quick start guide
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore             # Python gitignore
```

## 🖥️ Platform Support

**Works on all major operating systems!**

✅ **Linux** (Ubuntu, Debian, Fedora, Arch)
✅ **macOS** (Intel & Apple Silicon)
✅ **Windows** (10 & 11)

**Requirements**: Python 3.8+ (tested on 3.11)

See **[COMPATIBILITY.md](COMPATIBILITY.md)** for detailed platform-specific instructions and troubleshooting.

## What is Forward-Forward?

The Forward-Forward algorithm is an alternative to backpropagation that trains neural networks using two forward passes:
- **Positive pass**: Real/target data (network should output HIGH goodness)
- **Negative pass**: Fake/negative data (network should output LOW goodness)

### Key Concepts

1. **Goodness**: Measured as the sum of squared activities in each layer (per sample)
2. **Local Learning**: Each layer is trained independently using its own goodness objective
3. **No Backpropagation**: No need to compute gradients through the entire network

## Key Fixes from Original Implementation

### 1. **Per-Sample Goodness Calculation**

❌ **Original (Incorrect)**:
```python
def goodness(x):
    return np.sum(x**2)  # Sums across ALL samples
```

✅ **Fixed (Correct)**:
```python
def goodness(activations):
    return np.sum(activations ** 2, axis=1)  # Per-sample: (batch_size,)
```

**Why**: Goodness should be computed per sample, not summed across the entire batch.

### 2. **Proper Loss-Based Learning**

❌ **Original (Incorrect)**:
```python
# Trying to use backprop-style deltas and derivatives
error2 = p2 - 1
delta2 = error2 * activation_derivative(a2)
```

✅ **Fixed (Correct)**:
```python
# Loss-based approach with sigmoid probability
p_pos = 1 / (1 + np.exp(-(goodness_pos - threshold)))
p_neg = 1 / (1 + np.exp(-(goodness_neg - threshold)))

# Gradient of negative log-likelihood
grad_scale_pos = -(1 - p_pos)  # Want p → 1
grad_scale_neg = p_neg          # Want p → 0
```

**Why**: The FF algorithm uses a loss function based on probability (sigmoid of goodness), not traditional backpropagation.

### 3. **Layer Normalization for Stability**

❌ **Original**: No normalization
```python
a1 = activation(np.dot(X, W1))
```

✅ **Fixed**: Input normalization
```python
x_normalized = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
z = np.dot(x_normalized, self.W) + self.b
```

**Why**: Prevents exploding gradients and maintains numerical stability.

### 4. **Proper Weight Initialization**

❌ **Original**:
```python
W1 = 2*np.random.random((3, 4)) - 1  # Uniform [-1, 1]
```

✅ **Fixed**:
```python
limit = np.sqrt(6 / (input_dim + output_dim))  # Xavier/Glorot
self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
```

**Why**: Better initialization prevents dead neurons and improves learning.

### 5. **Correct Gradient Computation**

The gradient of goodness with respect to weights:

```python
# Goodness = sum(activation^2)
# d(goodness)/dW = d(sum(a^2))/dW
#                = 2 * a * da/dz * dz/dW
#                = 2 * a * (z > 0) * x

grad = grad_scale[:, np.newaxis] * 2 * activations * mask
dW = np.dot(x_normalized.T, grad) / batch_size
```

### 6. **Per-Sample Probability**

❌ **Original**: Single probability for entire batch
```python
p1 = 1/(1 + np.exp(-(g1 - theta)))  # g1 is scalar
```

✅ **Fixed**: Per-sample probabilities
```python
goodness = np.sum(activations ** 2, axis=1)  # (batch_size,)
probabilities = 1 / (1 + np.exp(-(goodness - threshold)))  # (batch_size,)
```

## Usage

```python
from forward_forward import FFNetwork
import numpy as np

# Create network
network = FFNetwork(
    layer_dims=[784, 500, 500],  # input, hidden1, hidden2
    learning_rate=0.03,
    threshold=1.0
)

# Training
for epoch in range(num_epochs):
    metrics = network.train_step(positive_data, negative_data)

# Prediction
probabilities, goodness_values = network.predict(test_data)
```

## Running the Examples

### Command Line Examples

**Basic XOR Example:**
```bash
python forward_forward.py
```

This will train on a simple XOR-like classification task and show:
- Training progress with goodness values per layer
- Test predictions on positive, negative, and random data
- Probability scores for classification

**MNIST-Style Example:**
```bash
python mnist_example.py
```

This demonstrates digit classification with:
- Image-label pair training
- Classification by trying all labels
- More realistic use case

### Graphical User Interface

Launch the interactive GUI for visual training and testing:

```bash
python gui.py
```

**Features:**
- 🎛️ Configure network architecture interactively
- 📊 Real-time training progress visualization
- 🧪 Test custom inputs and see predictions
- 📈 Monitor goodness scores per layer
- ⚙️ Adjustable hyperparameters (learning rate, threshold, epochs)
- 🎯 Choose between XOR and random datasets

The GUI provides an intuitive way to experiment with the Forward-Forward algorithm without writing code.

## Automated Testing

The repository includes comprehensive automated tests to ensure correctness.

### Running Tests

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
python -m pytest test_forward_forward.py -v

# Run specific test class
python -m pytest test_forward_forward.py::TestFFLayer -v

# Run with coverage (if pytest-cov is installed)
python -m pytest test_forward_forward.py --cov=forward_forward
```

### Test Coverage

The test suite includes **20 automated tests** covering:

**Layer Tests (`TestFFLayer`):**
- ✅ Layer initialization with correct dimensions
- ✅ Weight initialization ranges (Xavier/Glorot)
- ✅ Forward pass shape correctness
- ✅ ReLU activation application
- ✅ Goodness computation (per-sample)
- ✅ Training step updates weights correctly

**Network Tests (`TestFFNetwork`):**
- ✅ Multi-layer network initialization
- ✅ End-to-end forward pass
- ✅ Training step metrics
- ✅ Prediction functionality
- ✅ Learning progression (goodness improves)

**Edge Cases (`TestEdgeCases`):**
- ✅ Single sample batches
- ✅ Large batches (1000 samples)
- ✅ Zero inputs
- ✅ Numerical stability with large values

**Integration Tests (`TestIntegration`):**
- ✅ Complete XOR classification workflow
- ✅ Training reproducibility with seed

All tests pass with 100% success rate! ✓

## Key Parameters

- **learning_rate**: Controls update step size (typically 0.01-0.1)
- **threshold**: Goodness threshold for probability calculation (typically 1.0-2.0)
- **layer_dims**: Network architecture [input, hidden1, ..., hiddenN]

## References

### Original Paper
- **[Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)** - arXiv:2212.13345
- [Alternative PDF Link](http://www.cs.toronto.edu/~hinton/FFA13.pdf)

### Implementation Verification
- **[PAPER_VERIFICATION.md](PAPER_VERIFICATION.md)** - Detailed verification of our implementation against Hinton's paper ✅

### Additional Resources
- [Keras Forward-Forward Tutorial](https://keras.io/examples/vision/forwardforward/)
- [What is the Forward-Forward Algorithm? - TechTalks](https://bdtechtalks.com/2022/12/19/forward-forward-algorithm-geoffrey-hinton/)
- [PyTorch FF Implementations](https://github.com/mpezeshki/pytorch_forward_forward)
- [Medium Article on FF Algorithm](https://pub.towardsai.net/forward-forward-algorithm-ac24d0d9ffd)

## Common Pitfalls to Avoid

1. ❌ Don't compute goodness as a single value for the whole batch
2. ❌ Don't use backpropagation-style error signals
3. ❌ Don't forget to normalize inputs for numerical stability
4. ❌ Don't use learning rates that are too high (causes explosion)
5. ❌ Don't mix up per-sample and per-batch computations

## Notes

The Forward-Forward algorithm is still experimental. For production use cases, traditional backpropagation with gradient descent remains more established and often more effective.
