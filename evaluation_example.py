"""
Model Evaluation Example for Forward-Forward Algorithm

This script demonstrates proper train/test splitting and comprehensive
evaluation metrics for the Forward-Forward algorithm.
"""

import numpy as np
from forward_forward import FFNetwork


def train_test_split(X_pos, X_neg, test_ratio=0.2, seed=42):
    """
    Split positive and negative data into train and test sets

    Args:
        X_pos: Positive data
        X_neg: Negative data
        test_ratio: Fraction of data to use for testing
        seed: Random seed for reproducibility

    Returns:
        X_pos_train, X_pos_test, X_neg_train, X_neg_test
    """
    np.random.seed(seed)

    # Shuffle indices
    n_pos = len(X_pos)
    n_neg = len(X_neg)

    pos_indices = np.random.permutation(n_pos)
    neg_indices = np.random.permutation(n_neg)

    # Split indices
    n_pos_test = int(n_pos * test_ratio)
    n_neg_test = int(n_neg * test_ratio)

    pos_train_idx = pos_indices[n_pos_test:]
    pos_test_idx = pos_indices[:n_pos_test]

    neg_train_idx = neg_indices[n_neg_test:]
    neg_test_idx = neg_indices[:n_neg_test]

    return (
        X_pos[pos_train_idx],
        X_pos[pos_test_idx],
        X_neg[neg_train_idx],
        X_neg[neg_test_idx]
    )


def print_evaluation_metrics(metrics, dataset_name="Test"):
    """Pretty print evaluation metrics"""
    print(f"\n{'=' * 70}")
    print(f"{dataset_name} Set Evaluation Metrics")
    print(f"{'=' * 70}")

    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    print(f"\n🔢 Confusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Total Samples:   {metrics['total_samples']}")

    print(f"\n💡 Forward-Forward Specific:")
    print(f"  Goodness Separation: {metrics['goodness_separation']:.4f}")
    print(f"  Mean Prob (Positive): {metrics['mean_prob_positive']:.4f}")
    print(f"  Mean Prob (Negative): {metrics['mean_prob_negative']:.4f}")

    # Performance assessment
    print(f"\n✅ Performance Assessment:")
    if metrics['accuracy'] >= 0.95:
        print(f"  🌟 Excellent performance!")
    elif metrics['accuracy'] >= 0.85:
        print(f"  ✓ Good performance")
    elif metrics['accuracy'] >= 0.70:
        print(f"  ⚠️  Fair performance - consider more training")
    else:
        print(f"  ❌ Poor performance - check data or hyperparameters")


def evaluate_during_training(network, X_pos_test, X_neg_test, epoch, interval=100):
    """Evaluate model during training at specified intervals"""
    if (epoch + 1) % interval == 0 or epoch == 0:
        metrics = network.evaluate(X_pos_test, X_neg_test)
        print(f"\nEpoch {epoch + 1} - Test Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, "
              f"Goodness Sep: {metrics['goodness_separation']:.4f}")
        return metrics
    return None


def main():
    print("=" * 70)
    print("Forward-Forward Algorithm - Evaluation Example")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Generate larger dataset for proper evaluation
    print("\n📦 Generating dataset...")
    n_samples = 200

    # Positive data: last feature = 1, others random
    X_pos = np.random.random((n_samples, 10)) * 0.5
    X_pos[:, -1] = 1.0  # Set last feature to 1

    # Negative data: last feature = 0, others random
    X_neg = np.random.random((n_samples, 10)) * 0.5
    X_neg[:, -1] = 0.0  # Set last feature to 0

    # Train/test split
    X_pos_train, X_pos_test, X_neg_train, X_neg_test = train_test_split(
        X_pos, X_neg, test_ratio=0.2, seed=42
    )

    print(f"  Training set: {len(X_pos_train)} positive, {len(X_neg_train)} negative")
    print(f"  Test set:     {len(X_pos_test)} positive, {len(X_neg_test)} negative")

    # Create network
    print("\n🧠 Initializing network...")
    network = FFNetwork(
        layer_dims=[10, 20, 10],
        learning_rate=0.03,
        threshold=1.0
    )
    print(f"  Architecture: 10 -> 20 -> 10")

    # Training with periodic evaluation
    print("\n🏋️  Training...")
    epochs = 500
    eval_history = []

    for epoch in range(epochs):
        # Training step
        train_metrics = network.train_step(X_pos_train, X_neg_train)

        # Periodic evaluation
        eval_metrics = evaluate_during_training(
            network, X_pos_test, X_neg_test, epoch, interval=100
        )
        if eval_metrics:
            eval_history.append(eval_metrics)

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    test_metrics = network.evaluate(X_pos_test, X_neg_test)
    print_evaluation_metrics(test_metrics, "Test")

    # Also evaluate on training set (check for overfitting)
    train_eval = network.evaluate(X_pos_train, X_neg_train)
    print_evaluation_metrics(train_eval, "Training")

    # Overfitting check
    print("\n" + "=" * 70)
    print("Overfitting Analysis")
    print("=" * 70)

    train_acc = train_eval['accuracy']
    test_acc = test_metrics['accuracy']
    gap = train_acc - test_acc

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Accuracy Gap:      {gap:.4f}")

    if gap < 0.05:
        print("  ✅ No significant overfitting detected")
    elif gap < 0.10:
        print("  ⚠️  Slight overfitting - acceptable")
    else:
        print("  ❌ Significant overfitting detected!")

    # Learning curve
    if len(eval_history) > 1:
        print("\n" + "=" * 70)
        print("Learning Curve")
        print("=" * 70)
        print("\nAccuracy progression on test set:")
        for i, metrics in enumerate(eval_history):
            epoch = (i + 1) * 100 if i > 0 else 1
            acc = metrics['accuracy']
            bar_length = int(acc * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  Epoch {epoch:4d}: {bar} {acc:.4f}")

    # Example predictions
    print("\n" + "=" * 70)
    print("Example Predictions")
    print("=" * 70)

    print("\n✓ Positive Examples:")
    probs, _ = network.predict(X_pos_test[:5])
    for i in range(min(5, len(X_pos_test))):
        correct = "✓" if probs[i] > 0.5 else "✗"
        print(f"  Sample {i+1}: Prob={probs[i]:.4f} {correct}")

    print("\n✗ Negative Examples:")
    probs, _ = network.predict(X_neg_test[:5])
    for i in range(min(5, len(X_neg_test))):
        correct = "✓" if probs[i] <= 0.5 else "✗"
        print(f"  Sample {i+1}: Prob={probs[i]:.4f} {correct}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
The Forward-Forward algorithm achieved:
- Test Accuracy: {test_metrics['accuracy']:.2%}
- F1 Score: {test_metrics['f1_score']:.4f}
- Goodness Separation: {test_metrics['goodness_separation']:.4f}

The model successfully learned to distinguish positive and negative data
using only local learning rules (no backpropagation)!
""")


if __name__ == "__main__":
    main()
