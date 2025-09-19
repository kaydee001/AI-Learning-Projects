# Regularized Pattern Classifier with Advanced Optimization

A neural network built from scratch to classify 2D overlapping clusters, implementing regularization techniques and modern optimization algorithms to achieve robust performance.

## The Problem

- **Dataset**: 200 points in two overlapping clusters (binary classification)
- **Challenge**: Started with 51% accuracy (random guessing) due to overfitting
- **Goal**: Build generalizable classifier without external libraries, exploring optimization algorithms

## The Solution

**Architecture**: Input(2) → Hidden(8, ReLU) → Output(1, Sigmoid)

### Week 1: Regularization Techniques

- **L2 Regularization**: Penalize large weights to prevent overfitting
- **Dropout**: Randomly deactivate 20% of neurons during training
- **Input Normalization**: Standardize features (mean≈0, std=1)
- **He Initialization**: Proper weight scaling for ReLU networks
- **Bias/Variance Analysis**: Diagnose and fix model problems

### Week 2: Optimization Algorithms

- **Mini-batch Gradient Descent**: Process data in batches of 32 for faster updates
- **Momentum**: Build velocity in consistent directions, reduce oscillations
- **RMSprop**: Adaptive learning rates per parameter based on gradient history
- **Adam Optimizer**: Combine momentum + RMSprop for robust optimization
- **Bias Correction**: Fix slow start problem from zero initialization
- **Learning Rate Decay**: Automatically reduce learning rate over time

### Week 3: Hyperparameter Tuning

- **Grid Search**: Systematic testing of parameter combinations
- **Parameter Space**: Learning rates, hidden sizes, regularization strength
- **Dev vs Test Reality**: Learned overfitting to validation set (100% dev → 46% test)
- **Bug Fix**: Corrected Adam optimizer implementation (db1/db2 mix-up)
- **Final Optimization**: Found optimal parameters through systematic search

## Results Progression

### Week 1: Regularization Results

| Approach            | Training Accuracy | Dev Accuracy | Status                      |
| ------------------- | ----------------- | ------------ | --------------------------- |
| Baseline Network    | ~90%              | 76.67%       | High Variance (Overfitting) |
| + L2 Regularization | ~90%              | 90.00%       | Overfitting Fixed           |
| + Dropout           | ~90%              | 86.67%       | Additional Regularization   |
| + Normalization     | ~90%              | 86.67%       | Better Training Dynamics    |
| + He Initialization | ~90%              | **93.33%**   | Optimal Performance         |

### Week 2: Optimization Algorithm Comparison

| Optimizer                     | Dev Accuracy | Training Characteristics                    |
| ----------------------------- | ------------ | ------------------------------------------- |
| Mini-batch Gradient Descent   | 86.67%       | More frequent updates, noisy cost           |
| + Momentum                    | 93.33%       | Smoother convergence, reduced oscillations  |
| + RMSprop                     | 93.33%       | Adaptive learning rates, stable training    |
| + Adam (no bias correction)   | 90.00%       | Combined benefits, different local minimum  |
| + Adam (with bias correction) | 93.33%       | Fixed slow start, optimal performance       |
| + Learning Rate Decay         | 86.67%       | Conservative fine-tuning, early convergence |

**Final Results**: 73.33% test accuracy with LR=0.0001, Hidden=6, Lambda=0.1

---

Built as part of learning neural networks from scratch! 🧠
