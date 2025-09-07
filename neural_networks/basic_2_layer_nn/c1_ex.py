import numpy as np
import matplotlib.pyplot as plt

# squashes values between the range(0, 1)
# used for binary classification in the output layer
def sigmoid(z : np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))

# initializing weights and biases for a 2 layer nn
# getting input features(n_x), hidden layers(n_h), output layer(n_y)
def init_params(n_x : int, n_h : int, n_y : int, seed : int=42)  -> dict:

    np.random.seed(seed)

    # model parameters
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = b2 = np.zeros((n_y, 1))

    return {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2}

# here we perform fp for 2 layer nn to compute activations
def forward_propagation(X : np.ndarray, params : dict) -> tuple:

    # X is the input data of shape (n_x, m)
    # params are the model parameters    
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = np.dot(W1, X) + b1
    # output squashed between (-1 ,1)
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    # output squashed between (0 ,1)
    A2 = sigmoid(Z2)

    # returns the predicted output (A2) and the intermediate values for back prop
    cache = {"Z1" : Z1, "A1" : A1, "Z2" : Z2, "A2" : A2}
    return A2, cache

# cross entropy cost using the predicted probabilites (A2) and true labels (Y)
# both are of shape (1, m)
def compute_cost(A2 : np.ndarray, Y : np.ndarray) -> float:

    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y+np.log(A2) + (1-Y)*np.log(1-A2))

    return np.squeeze(cost)

# here we perform bp for calculating gradients for all params
def backward_propagation(params : dict, cache : dict, X : np.ndarray, Y : np.ndarray) -> dict:

    m = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    # output layer
    dZ2 = A2-Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    # hidden layer
    dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2}

# updating the params with gradient descent where learning_rate acts as a step
def update_params(params : dict, grads : dict, learning_rate : float=1.2) -> dict:
    
    params["W1"] -= learning_rate*grads["dW1"]
    params["b1"] -= learning_rate*grads["db1"]
    params["W2"] -= learning_rate*grads["dW2"]
    params["b2"] -= learning_rate*grads["db2"]

    return params

# training loop over iterations
def train_nn(X : np.ndarray, Y : np.ndarray, n_h : int, print_cost : bool=False, iterations : int=10000) -> dict:

    n_x, n_y = X.shape[0], Y.shape[0]
    params = init_params(n_x, n_h, n_y)

    for i in range(iterations):
        # forward pass
        A2, cache = forward_propagation(X, params)

        # compute the loss
        cost = compute_cost(A2, Y)

        # backward pass
        grads = backward_propagation(params, cache, X, Y)

        # gradient descent step
        params = update_params(params, grads)

    if print_cost and i%100==0:
        print(f"iteration : {i}, cost : {cost:.4f}")

    return params

# prediction using the trained model (params)
def predict(params : dict, X : np.ndarray) -> np.ndarray:

    # if A2 > 0.5, return 1 else 0
    A2, _ = forward_propagation(X, params)
    return(A2>0.5).astype(int)

# here we generate the toy data set
if __name__ == "__main__":
    
    np.random.seed(22)
    m = 400
    X = np.random.randn(2, m)
    Y = (np.sin(X[0])+np.cos(X[1])>0).astype(int).reshape(1, m)

    params = train_nn(X, Y, n_h=4, print_cost=True, iterations=5000)

    preds = predict(params, X)
    accuracy = np.mean(preds==Y)*100
    print(f"training accuracy : {accuracy:.2f}")

    scatter = plt.scatter(X[0, :], X[1, :], c=Y.flatten(), cmap=plt.cm.coolwarm)
    plt.xlabel("Input X[0] (Feature 1)")
    plt.ylabel("Input X[1] (Feature 2)")
    plt.suptitle("here we are checking if sin(x1​)+cos(x2​) > 0")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
