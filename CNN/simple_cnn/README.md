# CNN - Manual Implementation

Built CNNs from scratch using pure Python to understand convolution mathematics before using PyTorch.

---

## Core Functions

**Edge Detection**

```python
detect_vertical_edges(image, filter, stride=1)
detect_horizontal_edges(image, filter, stride=1)
```

**CNN Layer**

```python
cnn_layer(image, filters, stride=1, activation='relu')
```

**Pooling**

```python
max_pool_2x2(image)  # Reduces dimensions by 50%
```

**Complete Pipeline**

```python
complete_cnn(image, filters, pool=True)
```

## Example Usage

```python
# Test image with vertical stripes
test_img = [[0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1]]

# Detect edges
filter_vertical = [1, -1]
result = detect_vertical_edges(test_img, filter_vertical)
# Output: [-1, 1, -1, -1, 1, -1, ...] (edges detected)

# Run full CNN pipeline
filters = [[1, -1]]
result = complete_cnn(test_img, filters, pool=True)
```

---

Built as part of learning neural networks from scratch! 🧠
