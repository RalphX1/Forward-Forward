"""
Forward-Forward Algorithm on MNIST-like Data

This example demonstrates the FF algorithm on digit classification.
Positive data: Real digit images with correct label
Negative data: Real digit images with WRONG label (or hybrid images)
"""

import numpy as np
from forward_forward import FFNetwork


def create_one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def create_labeled_data(images, labels, num_classes=10):
    """
    Concatenate images with their labels to create positive data
    This follows Hinton's approach where the label is part of the input
    """
    one_hot_labels = create_one_hot(labels, num_classes)
    return np.concatenate([images, one_hot_labels], axis=1)


def create_negative_data(images, labels, num_classes=10):
    """
    Create negative data by using WRONG labels
    This is a simple approach - labels are randomly permuted
    """
    # Randomly shuffle labels to create wrong associations
    wrong_labels = labels.copy()
    np.random.shuffle(wrong_labels)

    # Make sure at least some are actually wrong
    # (though with 10 classes, most will be wrong anyway)
    mask = wrong_labels == labels
    if np.any(mask):
        # For matching labels, shift them by 1
        wrong_labels[mask] = (labels[mask] + 1) % num_classes

    one_hot_labels = create_one_hot(wrong_labels, num_classes)
    return np.concatenate([images, one_hot_labels], axis=1)


def generate_synthetic_digits(num_samples=100, image_size=16, num_classes=10):
    """
    Generate synthetic digit-like data for demonstration
    Each 'digit' is a random pattern that we'll use to test the algorithm
    """
    images = np.random.random((num_samples, image_size * image_size)) * 0.5
    labels = np.random.randint(0, num_classes, num_samples)
    return images, labels


def train_ff_on_digits():
    """Train Forward-Forward network on digit-like classification"""

    print("=" * 70)
    print("Forward-Forward Algorithm on Digit Classification")
    print("=" * 70)

    # Configuration
    image_size = 16  # 16x16 images (256 pixels)
    num_classes = 10
    num_samples = 200
    num_test = 50

    np.random.seed(42)

    # Generate synthetic training data
    print(f"\nGenerating {num_samples} synthetic digit samples...")
    train_images, train_labels = generate_synthetic_digits(
        num_samples, image_size, num_classes
    )

    # Create positive and negative data
    # Positive: image + correct label
    # Negative: image + wrong label
    X_pos = create_labeled_data(train_images, train_labels, num_classes)
    X_neg = create_negative_data(train_images, train_labels, num_classes)

    print(f"Input dimension: {X_pos.shape[1]} ({image_size}x{image_size} image + {num_classes} label bits)")
    print(f"Positive samples: {X_pos.shape[0]}")
    print(f"Negative samples: {X_neg.shape[0]}")

    # Create network
    # Input: 256 (image) + 10 (label) = 266
    # Hidden layers: 500 -> 500
    input_dim = image_size * image_size + num_classes
    network = FFNetwork(
        layer_dims=[input_dim, 500, 500],
        learning_rate=0.03,
        threshold=2.0
    )

    print(f"\nNetwork architecture: {input_dim} -> 500 -> 500")

    # Training
    epochs = 100
    batch_size = 50
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)

        epoch_pos_goodness = [0, 0]
        epoch_neg_goodness = [0, 0]
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_pos = X_pos[batch_indices]
            batch_neg = X_neg[batch_indices]

            metrics = network.train_step(batch_pos, batch_neg)

            epoch_pos_goodness[0] += metrics['pos_goodness'][0]
            epoch_pos_goodness[1] += metrics['pos_goodness'][1]
            epoch_neg_goodness[0] += metrics['neg_goodness'][0]
            epoch_neg_goodness[1] += metrics['neg_goodness'][1]
            num_batches += 1

        # Average over batches
        epoch_pos_goodness = [g / num_batches for g in epoch_pos_goodness]
        epoch_neg_goodness = [g / num_batches for g in epoch_neg_goodness]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Layer 1: Pos = {epoch_pos_goodness[0]:.4f}, Neg = {epoch_neg_goodness[0]:.4f}")
            print(f"  Layer 2: Pos = {epoch_pos_goodness[1]:.4f}, Neg = {epoch_neg_goodness[1]:.4f}")

    # Testing
    print("\n" + "=" * 70)
    print("Testing Phase")
    print("=" * 70)

    # Generate test data
    test_images, test_labels = generate_synthetic_digits(num_test, image_size, num_classes)

    # Test 1: Correct labels (should have high probability)
    print("\n--- Test 1: Images with CORRECT labels (should be classified as positive) ---")
    test_pos = create_labeled_data(test_images[:10], test_labels[:10], num_classes)
    probs_pos, _ = network.predict(test_pos)

    correct_count = 0
    for i in range(len(probs_pos)):
        is_correct = "✓" if probs_pos[i] > 0.5 else "✗"
        print(f"Sample {i+1}: Label={test_labels[i]}, Probability={probs_pos[i]:.4f} {is_correct}")
        if probs_pos[i] > 0.5:
            correct_count += 1
    print(f"Correctly identified as positive: {correct_count}/10")

    # Test 2: Wrong labels (should have low probability)
    print("\n--- Test 2: Images with WRONG labels (should be classified as negative) ---")
    test_neg = create_negative_data(test_images[:10], test_labels[:10], num_classes)
    probs_neg, _ = network.predict(test_neg)

    correct_count = 0
    for i in range(len(probs_neg)):
        is_correct = "✓" if probs_neg[i] < 0.5 else "✗"
        # Get the wrong label from the one-hot encoding
        wrong_label = np.argmax(test_neg[i, -num_classes:])
        print(f"Sample {i+1}: True={test_labels[i]}, Wrong={wrong_label}, Probability={probs_neg[i]:.4f} {is_correct}")
        if probs_neg[i] < 0.5:
            correct_count += 1
    print(f"Correctly identified as negative: {correct_count}/10")

    # Test 3: Prediction by trying all labels
    print("\n--- Test 3: Predict labels by finding highest goodness ---")
    print("For each image, we'll try all 10 possible labels and pick the one with highest goodness")

    for i in range(5):
        test_image = test_images[i:i+1]  # Single image
        true_label = test_labels[i]

        # Try all possible labels
        goodness_scores = []
        for label in range(num_classes):
            # Create input with this label
            test_input = create_labeled_data(test_image, np.array([label]), num_classes)
            _, goodness = network.predict(test_input)
            # Use final layer goodness
            goodness_scores.append(goodness[-1][0])

        predicted_label = np.argmax(goodness_scores)
        is_correct = "✓" if predicted_label == true_label else "✗"

        print(f"\nImage {i+1}: True label = {true_label}, Predicted = {predicted_label} {is_correct}")
        print(f"  Goodness scores: {[f'{g:.2f}' for g in goodness_scores]}")

    print("\n" + "=" * 70)
    print("Explanation:")
    print("=" * 70)
    print("""
The Forward-Forward algorithm learns to:
1. Recognize when an image-label pair is CORRECT (positive data)
2. Reject when an image-label pair is WRONG (negative data)

For classification, we:
- Present the same image with all possible labels
- Pick the label that gives the highest goodness score
- This is the label that the network thinks "fits best" with the image

This approach is fundamentally different from traditional supervised learning,
as each layer learns its own local representation without backpropagation.
""")


if __name__ == "__main__":
    train_ff_on_digits()
