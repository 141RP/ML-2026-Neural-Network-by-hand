import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#loading
data_train = pd.read_csv('../data/mnist_train.csv')
data_test= pd.read_csv("../data/mnist_test.csv")

#to numpy
data_train = np.array(data_train)
data_test = np.array(data_test)

#m, n = data.shape
np.random.shuffle(data_train)
np.random.shuffle(data_test)

#transpose
data_train = data_train.T
data_test = data_test.T

y_train = data_train[0].astype(int)   # first row = labels/target hence it's the Y
x_train = data_train[1:].astype(np.float32) / 255.0
y_test  = data_test[0].astype(int)
x_test  = data_test[1:].astype(np.float32) / 255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print("x_train min/max:", x_train.min(), x_train.max())  # should be 0.0..1.0
print("labels min/max:", y_train.min(), y_train.max())   # should be 0..9
print("class counts:", np.bincount(y_train.astype(int)))

def initial_parameters(hidden):
    W1 = np.random.randn(hidden, 784)
    b1 = np.zeros((hidden, 1))
    W2 = np.random.randn(10, hidden)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def d_ReLU(Z):
    return Z > 0
    # derivative of all inputs > 0 = 1 and inputs < 0 = 0
    # Since true = 1 and false = 0 when converted, this works as a simple derivative

def softmax(Z):
    # probabilities that sum to 1
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_propagation(W1, b1, W2, b2, X):
    # L1 linear -> ReLU
    Z1= W1.dot(X) + b1
    A1= ReLU(Z1)

    # L2 linear -> softmax probabilities
    Z2 = W2.dot(A1) + b2
    A2= softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    # Build matrix (m, 10) then transpose to (10, m)
    # Only part I don't understand perfectly - it's supposed to help with iteration during backpropagation I think?
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y

    # Output layer gradient
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    # Hidden layer gradient
    dZ1 = W2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, hidden, alpha):
    # Create starting weights/biases
    W1, b1, W2, b2 = initial_parameters(hidden)

    # Repeat forward -> backprop -> update
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation( Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 25 == 0:
            print("Iteration:  ", i)
            print("Accuracy: ", get_accuracy(get_prediction(A2), Y))

    return W1, b1, W2, b2

def save_model(path, W1, b1, W2, b2):
    # Save arrays in a single compressed NumPy file
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)


W1, b1, W2, b2 = gradient_descent(x_train, y_train, 450, 64, 0.1)
save_model("mnist_weights.npz", W1, b1, W2, b2)