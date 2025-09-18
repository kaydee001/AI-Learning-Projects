import numpy as np
import matplotlib.pyplot as plt

# overlapping cluster dataset of shape (200, 2)
X = np.vstack([
    np.random.normal([1, 1], 1.0, (100, 2)),  # class 0, center at (1, 1)
    np.random.normal([3, 3], 1.0, (100, 2))   # class 1, center at (3, 3)
])

# true labels of shape (200,)
labels = np.hstack([np.zeros(100), np.ones(100)])

# shuffling the dataset -> to add randomness in all the splits
indices = np.arange(200)
np.random.shuffle(indices)
X_shuffled = X[indices]
labels_shuffled = labels[indices]

# splitting the dataset
X_train = X_shuffled[0:140]
X_dev = X_shuffled[140:170]
X_test = X_shuffled[170:200]

y_train = labels_shuffled[0:140]
y_dev = labels_shuffled[140:170]
y_test = labels_shuffled[170:200]

# input normalization (such that mean~0 and std=1)
train_mean = np.mean(X_train)
train_std = np.std(X_train)

# applying the same normalization to the splits to prevent data leakage
X_train_norm = (X_train-train_mean) / train_std
X_dev_norm = (X_dev-train_mean) / train_std
X_test_norm = (X_test-train_mean) / train_std

# He and Xavier initiailizations to prevent vanishing/exploding gradients
W1 = np.random.randn(2, 8) * np.sqrt(2/2) # Relu (He)
b1 = np.zeros((8, ))
W2 = np.random.randn(8, 1) * np.sqrt(2/8) # sigmoid (Xavier)
b2 = np.zeros((1, ))

# Adam optimization for momentum params
beta = 0.9
v_dW1 = np.zeros_like(W1)
v_db1 = np.zeros_like(b1)
v_dW2 = np.zeros_like(W2)
v_db2 = np.zeros_like(b2)

# RMSprop params
beta2 = 0.99
epsilon = 1e-4 
s_dW1 = np.zeros_like(W1)
s_db1 = np.zeros_like(b1)
s_dW2 = np.zeros_like(W2)
s_db2 = np.zeros_like(b2)

# bias correction (to correct initialization bias in early iterations)
t = 0

# mini batch
def create_mini_batches(X, y, batch_size):
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches

# training loop
epochs = 1000
batch_size = 32
for epoch in range(epochs):

    # create mini batches of size = 32
    mini_batches = create_mini_batches(X_train_norm, y_train, batch_size)

    # process each mini batch
    for X_batch, y_batch in mini_batches:

        # forward pass from input to hidden
        z1 = np.dot(X_batch, W1)+b1
        # relu func
        a1 = np.maximum(0, z1)

        # dropout regularization (randomly deactivates 20%)
        dropout_rate = 0.2
        dropout_mask = (np.random.rand(a1.shape[0], a1.shape[1])>dropout_rate).astype(float)
        a1 = a1*dropout_mask

        # forward pass from hidden to output
        z2 = np.dot(a1, W2)+b2
        # sigmoid func
        a2 = 1/(1+np.exp(-z2))

        # shape (140, 1)
        y_train_reshaped = y_batch.reshape(-1, 1)
        m = X_batch.shape[0]
        
        # L2 regularization penalty (to penalize larger weights causing the model to memorize training quirks) -> to prevent overfitting
        lambd = 0.1
        reg_penalty = (lambd/(2*m)) * (np.sum(W1**2)+np.sum(W2**2))

        # calculate cross entropy loss + L2 regularization
        original_cost = -np.mean(y_train_reshaped*np.log(a2) + (1-y_train_reshaped)*np.log(1-a2))
        cost = original_cost+reg_penalty 

        # backward pass from output to hidden
        da2 = -(y_train_reshaped/a2) + ((1-y_train_reshaped)/(1-a2))
        dz2 = da2*a2*(1-a2) # sigmoid derivative
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        # backward pass from hidden to input 
        da1 = np.dot(dz2, W2.T)
        dz1 = da1*(z1>0) # relu derivative
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Adam optimization updates 
        # updating the momentum
        v_dW1 = beta*v_dW1 + (1-beta)*dW1
        v_db1 = beta*v_db1 + (1-beta)*db1
        v_dW2 = beta*v_dW2 + (1-beta)*dW2
        v_db2 = beta*v_db2 + (1-beta)*db2

        # updating the RMSprop
        s_dW1 = beta2*s_dW1 + (1-beta2)*(dW1**2)
        s_db1 = beta2*s_db1 + (1-beta2)*(db1**2)
        s_dW2 = beta2*s_dW2 + (1-beta2)*(dW2**2)
        s_db2 = beta2*s_db2 + (1-beta2)*(db2**2)

        # applying bias correction
        t += 1
        v_dW1_corrected = v_dW1 / (1-beta**t)
        v_db1_corrected = v_db1 / (1-beta**t)
        v_dW2_corrected = v_dW2 / (1-beta**t)
        v_db2_corrected = v_db2 / (1-beta**t)

        s_dW1_corrected = s_dW1 / (1-beta2**t)
        s_db1_corrected = s_db1 / (1-beta2**t)
        s_dW2_corrected = s_dW2 / (1-beta2**t)
        s_db2_corrected = s_db2 / (1-beta2**t)

        # learning rate decay       
        initial_learning_rate = 0.0001
        decay_rate = 0.01
        learning_rate = initial_learning_rate / (1+decay_rate*epoch)
               
        # updating weights using Adam formula (which includes momentum, RMSprop, bias correction and L2 regularization)
        W2 = W2 - learning_rate * (v_dW2_corrected/np.sqrt(s_dW2_corrected+epsilon)+(lambd/m)*W2)
        b2 = b2 - learning_rate * (v_db2_corrected/np.sqrt(s_db2_corrected+epsilon))
        W1 = W1 - learning_rate * (v_dW1_corrected/np.sqrt(s_dW1_corrected+epsilon)+(lambd/m)*W1)
        b1 = b1 - learning_rate * (v_db2_corrected/np.sqrt(s_db1_corrected+epsilon))

    if epoch%100==0:
        raw_grad_magnitude = np.mean(np.abs(dW1))
        smooth_grad_magnitude = np.mean(np.abs(v_dW1))
        print(f"epoch : {epoch} cost : {cost:.2f}")
        print(f"learning rate : {learning_rate:.6f}")
        print(f"raw gradient : {raw_grad_magnitude:.2f}")
        print(f"momentum gradient : {smooth_grad_magnitude:.2f}") 
        print(f"s_dW2 sample: {s_dW2[0,0]:.2f}")
        print(f"sqrt(s_dW2) sample: {np.sqrt(s_dW2[0,0] + epsilon):.2f}")
        print("---------------------------------------------------------------")

# forward pass on dev set -> to check overall generalization of the model
z1_dev = np.dot(X_dev_norm, W1)+b1
a1_dev = np.maximum(0, z1_dev)
z2_dev = np.dot(a1_dev, W2)+b2
a2_dev = 1/(1+np.exp(-z2_dev))

predictions_dev = a2_dev > 0.5
accuracy_dev = np.mean(predictions_dev.flatten() == y_dev)
print(f"Dev accuracy: {accuracy_dev:.2%}")