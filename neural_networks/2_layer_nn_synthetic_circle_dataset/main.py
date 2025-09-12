import numpy as np
import matplotlib.pyplot as plt

# generate 100 random points
n = 100
x_coods = np.random.randn(n)
y_coods = np.random.randn(n)

# circle parameters
center = (0.5, -0.5)
radius = 1.6546

# data set matrix; shape will be (n,2)
X = np.column_stack((x_coods, y_coods))

# true labels for classification
labels = np.zeros((n, 1))
for i in range(n):
    distance = distance = np.sqrt((x_coods[i] - center[0])**2 + (y_coods[i] - center[1])**2)

    if(distance<=radius):
        labels[i] = 0
        
    else:
        labels[i] = 1
        
# reshape to (n,1) for matrix operations
lables = labels.reshape(-1, 1)

# params
W1 = np.random.randn(2, 4)
b1 = np.zeros((4, ))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, ))

# forward pass from input to hidden
z1 = np.dot(X, W1)+b1
# relu func
a1 = np.maximum(0, z1)

# forward pass from hidden to output
z2 = np.dot(a1, W2)+b2
# sigmoid func
a2 = 1/(1+np.exp(-z2))

# calculate loss entropy
cost = 0
cost = -np.mean(labels*np.log(a2) + (1-labels)*np.log(1-a2))
print(f"initial cost : {cost:.4f}")

# backward pass from output to hidden
da2 = -(labels/a2) + ((1-labels)/(1-a2))
dz2 = da2*a2*(1-a2)
dW2 = np.dot(a1.T, dz2)
db2 = np.sum(dz2, axis=0)

# backward pass from hidden to input
da1 = np.dot(dz2, W2.T)
dz1 = da1*(z1>0)
dW1 = np.dot(X.T, dz1)
db1 = np.sum(dz1, axis=0)

# updating weights using gradient descent 
learning_rate = 0.01
W2 = W2 - learning_rate*dW2
b2 = b2 - learning_rate*db2
W1 = W1 - learning_rate*dW1
b1 = b1 - learning_rate*db1

# plotting the true data set
plt.figure(figsize=(8, 6)) 
for i in range(n):
    if labels[i] == 0:
        plt.scatter(X[i][0], X[i][1], c="#FF000D")
        
    else:
        labels[i] = 1
        plt.scatter(X[i][0], X[i][1], c="#00A2FF")

plt.title("neural network training data using circle classification")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")

plt.scatter([], [], c="#FF000D", label="inside")
plt.scatter([], [], c="#00A2FF", label="outside")
plt.legend()

plt.grid(True, alpha=0.3)
plt.show()
