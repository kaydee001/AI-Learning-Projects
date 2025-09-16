import numpy as np
import matplotlib.pyplot as plt

# overlapping cluster dataset
X = np.vstack([
    np.random.normal([1, 1], 1.0, (100, 2)),  # class 0, center at (1, 1)
    np.random.normal([3, 3], 1.0, (100, 2))   # class 1, center at (3, 3)
])

labels = np.hstack([np.zeros(100), np.ones(100)])

# creates an array from 0 to 199 (to keep the same randomness in both X and labels)
indices = np.arange(200)
np.random.shuffle(indices)

# increase randomness in train, dev and test sets 
X_shuffled = X[indices]
labels_shuffled = labels[indices]

# splitting the data
X_train = X_shuffled[0:140]
X_dev = X_shuffled[140:170]
X_test = X_shuffled[170:200]

y_train = labels_shuffled[0:140]
y_dev = labels_shuffled[140:170]
y_test = labels_shuffled[170:200]

# input normalization (such that mean~0 and std=1)
train_mean = np.mean(X_train)
train_std = np.std(X_train)

# training stats to prevent data leakage
X_train_norm = (X_train-train_mean) / train_std
X_dev_norm = (X_dev-train_mean) / train_std
X_test_norm = (X_test-train_mean) / train_std

# Xavier and He initiailizations to prevent vanishing/exploding gradients
W1 = np.random.randn(2, 8) * np.sqrt(2/2) # Relu
b1 = np.zeros((8, ))
W2 = np.random.randn(8, 1) * np.sqrt(2/8) # sigmoid
b2 = np.zeros((1, ))

# training loop
epochs = 1000
for epoch in range(epochs):

    # forward pass from input to hidden
    z1 = np.dot(X_train_norm, W1)+b1
    # relu func
    a1 = np.maximum(0, z1)

    # dropout regularization (randomly deactivates 20%) -> to prevent overfitting
    dropout_rate = 0.2
    dropout_mask = (np.random.rand(a1.shape[0], a1.shape[1])>dropout_rate).astype(float)
    a1 = a1*dropout_mask

    # forward pass from hidden to output
    z2 = np.dot(a1, W2)+b2
    # sigmoid func
    a2 = 1/(1+np.exp(-z2))

    # shape (140, 1)
    y_train_reshaped = y_train.reshape(-1, 1)
    
    # cost func with L2 regularization (to penalize larger weights causing the model to memorize training quirks) -> to prevent overfitting
    lambd = 0.1
    m = X_train_norm.shape[0]
    reg_penalty = (lambd/(2*m)) * (np.sum(W1**2)+np.sum(W2**2))

    # calculate loss entropy + L2
    original_cost = -np.mean(y_train_reshaped*np.log(a2) + (1-y_train_reshaped)*np.log(1-a2))
    cost = original_cost+reg_penalty 

    # backward pass from output to hidden
    da2 = -(y_train_reshaped/a2) + ((1-y_train_reshaped)/(1-a2))
    dz2 = da2*a2*(1-a2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    # backward pass from hidden to input 
    da1 = np.dot(dz2, W2.T)
    dz1 = da1*(z1>0)
    dW1 = np.dot(X_train_norm.T, dz1)
    db1 = np.sum(dz1, axis=0)

    # updating weights using gradient descent + L2
    learning_rate = 0.001
    W2 = W2 - learning_rate*(dW2+(lambd/m)*W2)
    b2 = b2 - learning_rate*db2
    W1 = W1 - learning_rate*(dW1+(lambd/m)*W1)
    b1 = b1 - learning_rate*db1

    if epoch%100==0:
        print(f"epoch : {epoch} cost : {cost:.4f}")

# forward pass on dev set -> to check overall generalization of the model
z1_dev = np.dot(X_dev_norm, W1)+b1
a1_dev = np.maximum(0, z1_dev)
z2_dev = np.dot(a1_dev, W2)+b2
a2_dev = 1/(1+np.exp(-z2_dev))

predictions_dev = a2_dev > 0.5
accuracy_dev = np.mean(predictions_dev.flatten() == y_dev)
print(f"Dev accuracy: {accuracy_dev:.2%}")