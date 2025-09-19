# Circle Classification Neural Network

A 2-layer neural network built from scratch to classify points based on the mathematical function `sin(x1) + cos(x2) > 0`.

## The Problem

- **Dataset**: 400 random points with 2 features each
- **Challenge**: Classify whether `sin(x1) + cos(x2)` is positive or negative
- **Goal**: Build a neural network from scratch using only NumPy (no external ML libraries)

## The Solution

**Architecture**: Input(2) → Hidden(4, tanh) → Output(1, sigmoid)

### Core Components

- **Forward Propagation**: Compute predictions layer by layer
- **Backward Propagation**: Calculate gradients using chain rule
- **Gradient Descent**: Update weights to minimize error
- **Sigmoid Activation**: Squash output to probability (0-1)
- **Tanh Activation**: Hidden layer activation (-1 to 1)

## Results

- **Training Accuracy**: ~95%+ (depends on random initialization)
- **Decision Boundary**: Successfully learns the complex `sin(x) + cos(y)` pattern
- **Convergence**: Typically reaches good performance within 5000 iterations

## The Dataset Visualization

The scatter plot shows two classes:

- **Red points**: Where `sin(x1) + cos(x2) > 0`
- **Blue points**: Where `sin(x1) + cos(x2) ≤ 0`

This creates a complex, non-linear decision boundary that requires a neural network to classify properly (linear methods would fail).

---

Built as part of learning neural networks from scratch! 🧠
