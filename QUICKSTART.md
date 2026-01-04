# Quick Start Guide

Get up and running with the Forward-Forward algorithm in minutes!

## Prerequisites

```bash
pip install numpy pytest
```

## 🚀 Three Ways to Get Started

### 1. Interactive GUI (Easiest)

**Launch the graphical interface:**
```bash
python gui.py
```

**Quick Tutorial:**
1. Click **"Initialize Network"** (uses default settings)
2. Click **"Start Training"** and watch it train in real-time
3. After training, enter test input like `0,0,1` and click **"Test Input"**
4. See the probability and classification!

**Try changing:**
- Architecture: `3,8,4` → `3,16,8` (bigger network)
- Learning Rate: `0.03` → `0.05` (faster learning)
- Epochs: `500` → `1000` (longer training)

### 2. Command Line Examples

**Run the basic XOR example:**
```bash
python forward_forward.py
```

Expected output:
```
======================================================================
Forward-Forward Algorithm - XOR Problem Example
======================================================================

Training for 1000 epochs...

Epoch 1:
  Layer 1: Pos goodness = 0.2884, Neg goodness = 0.6357
  Layer 2: Pos goodness = 0.1195, Neg goodness = 0.0938

...

Epoch 1000:
  Layer 1: Pos goodness = 7.6800, Neg goodness = 0.0070
  Layer 2: Pos goodness = 1.2078, Neg goodness = 0.9913
```

Notice how positive goodness increases and negative goodness decreases!

**Run the MNIST-style example:**
```bash
python mnist_example.py
```

### 3. Use as a Library

**Simple Python script:**
```python
import numpy as np
from forward_forward import FFNetwork

# Create network
network = FFNetwork(
    layer_dims=[3, 8, 4],      # 3 inputs, 8 hidden, 4 hidden
    learning_rate=0.03,
    threshold=1.0
)

# Prepare data
X_pos = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
X_neg = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)

# Train
for epoch in range(500):
    metrics = network.train_step(X_pos, X_neg)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Pos={metrics['pos_goodness'][0]:.2f}, "
              f"Neg={metrics['neg_goodness'][0]:.2f}")

# Test
test_input = np.array([[1, 1, 1]])  # Should be classified as positive
prob, _ = network.predict(test_input)
print(f"Probability: {prob[0]:.4f} ({'POSITIVE' if prob[0] > 0.5 else 'NEGATIVE'})")
```

## ✅ Verify Installation

**Run the automated tests:**
```bash
python -m pytest test_forward_forward.py -v
```

Expected output:
```
============================= test session starts ==============================
collected 20 items

test_forward_forward.py::TestFFLayer::test_layer_initialization PASSED   [  5%]
test_forward_forward.py::TestFFLayer::test_weight_initialization_range PASSED [ 10%]
...
test_forward_forward.py::TestIntegration::test_reproducibility PASSED    [100%]

============================== 20 passed in 0.37s ==============================
```

All tests should pass! ✓

## 🎓 Understanding the Output

### Goodness Scores

- **Positive Goodness**: Should be HIGH (typically 5-10 after training)
- **Negative Goodness**: Should be LOW (typically < 0.1 after training)

### Probability Scores

- **> 0.5**: Network thinks input is POSITIVE data
- **< 0.5**: Network thinks input is NEGATIVE data

### Example Interpretation

```
Input [1, 1, 1] -> Probability: 0.5521, Goodness: 1.2091
```

- Probability 0.55 means network is fairly confident this is positive data
- Goodness 1.21 is above the threshold of 1.0, contributing to positive classification

## 🔧 Common Issues

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:**
```bash
pip install numpy
```

### Issue: "ImportError: No module named 'tkinter'"
**Solution (Ubuntu/Debian):**
```bash
sudo apt-get install python3-tk
```

**Solution (macOS):**
```bash
brew install python-tk
```

### Issue: Network isn't learning (goodness values not changing)
**Solutions:**
- Increase learning rate (try 0.05 instead of 0.01)
- Train for more epochs
- Check that positive and negative data are actually different

## 📚 Next Steps

1. **Experiment with the GUI** - Try different architectures and datasets
2. **Read the README.md** - Understand the algorithm in depth
3. **Modify the examples** - Create your own positive/negative data
4. **Read the paper** - Geoffrey Hinton's Forward-Forward Algorithm paper

## 🎯 Example Experiments

### Experiment 1: Effect of Learning Rate
Try these learning rates and observe training speed:
- `0.01` - Slow but stable
- `0.03` - Good balance (default)
- `0.10` - Fast but might be unstable

### Experiment 2: Network Size
Compare these architectures:
- `3,4,2` - Small, trains fast, may underfit
- `3,8,4` - Medium (default), good performance
- `3,16,8` - Large, more capacity, slower training

### Experiment 3: Training Duration
Watch how performance improves:
- 100 epochs - Basic separation
- 500 epochs - Good separation (default)
- 1000 epochs - Maximum separation

## 💡 Tips

1. **Start with the GUI** - It's the easiest way to understand the algorithm
2. **Watch the goodness scores** - They tell you if learning is happening
3. **Use normalization** - It's enabled by default for numerical stability
4. **Start small** - Test on XOR before moving to complex datasets
5. **Run the tests** - They verify everything is working correctly

Happy experimenting! 🚀
