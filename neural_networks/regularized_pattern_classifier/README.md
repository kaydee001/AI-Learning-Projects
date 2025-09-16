# Regularized Pattern Classifier

A neural network built from scratch to classify 2D overlapping clusters, implementing multiple regularization techniques to achieve 93.33% accuracy.

## Problem

- **Dataset**: 200 points in two overlapping clusters (binary classification)
- **Challenge**: Started with 51% accuracy (random guessing) due to overfitting
- **Goal**: Build generalizable classifier without external libraries

## Solution

**Architecture**: Input(2) → Hidden(8, ReLU) → Output(1, Sigmoid)

**Key Techniques Applied**:

- **L2 Regularization**: Penalize large weights to prevent overfitting
- **Dropout**: Randomly deactivate 20% of neurons during training
- **Input Normalization**: Standardize features (mean≈0, std=1)
- **He Initialization**: Proper weight scaling for ReLU networks
- **Bias/Variance Analysis**: Diagnose and fix model problems

## Results

| Approach            | Training Accuracy | Dev Accuracy | Status                      |
| ------------------- | ----------------- | ------------ | --------------------------- |
| Baseline Network    | ~90%              | 76.67%       | High Variance (Overfitting) |
| + L2 Regularization | ~90%              | 90.00%       | Overfitting Fixed           |
| + Dropout           | ~90%              | 86.67%       | Additional Regularization   |
| + Normalization     | ~90%              | 86.67%       | Better Training Dynamics    |
| + He Initialization | ~90%              | **93.33%**   | Optimal Performance         |
