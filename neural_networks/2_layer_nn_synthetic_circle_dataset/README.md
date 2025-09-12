# Circle Classification Neural Network

A simple neural network built from scratch using NumPy to classify points as inside or outside a circle.

## What it does

- Generates 100 random points in 2D space
- Labels them as inside (red) or outside (blue) a circle
- Trains a 2→4→1 neural network to learn the circular decision boundary
- Uses ReLU activation in hidden layer and sigmoid in output layer

## Architecture

```
Input Layer (2 neurons) → Hidden Layer (4 neurons) → Output Layer (1 neuron)
                ReLU activation              Sigmoid activation
```

## Dataset

- **Center**: (0.5, -0.5)
- **Radius**: 1.6546
- **Points**: 100 random points
- **Labels**: 0 (inside circle), 1 (outside circle)

## Example Output

![Dataset Visualization](./img.png)

## What you'll see

- Initial cost printed to console
- Scatter plot showing:
  - Red points: Inside the circle (label 0)
  - Blue points: Outside the circle (label 1)

Built as part of learning neural networks from scratch! 🧠
