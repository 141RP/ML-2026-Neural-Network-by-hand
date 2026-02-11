import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def load_model(path):
    data = np.load(path)
    return data["W1"], data["b1"], data["W2"], data["b2"]

# Load test data
data_test = np.array(pd.read_csv("../data/mnist_test.csv")).T
y_test = data_test[0].astype(int)
x_test = data_test[1:].astype(np.float32) / 255.0

W1, b1, W2, b2 = load_model("mnist_weights.npz")

# *** COPIED THIS CODE FROM VIDEO I WATCHED ***

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_prediction(A2)
    return predictions

def load_model(path="mnist_weights.npz"):
    data = np.load(path)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    return W1, b1, W2, b2

def load_mnist_csv(path):
    data = np.array(pd.read_csv(path))
    np.random.shuffle(data)
    data = data.T
    Y = data[0].astype(int)
    X = data[1:].astype(np.float32) / 255.0  # IMPORTANT: same normalization as training
    return X, Y

# *** COPIED THIS CODE FROM VIDEO I WATCHED ***
def test_prediction(index, X, Y, W1, b1, W2, b2):
    # keep column vector shape (784, 1)
    current_image = X[:, index, None]

    prediction = make_predictions(current_image, W1, b1, W2, b2)[0]
    label = Y[index]

    print("Prediction:", prediction)
    print("Label:", label)

    # show image: convert back to 0..255 for display if you want
    img = (current_image.reshape(28, 28) * 255.0)

    plt.gray()
    plt.imshow(img, interpolation="nearest")
    plt.show()



W1, b1, W2, b2 = load_model("mnist_weights.npz")

# Load data you want to test on
X_test, Y_test = load_mnist_csv("../data/mnist_test.csv")

# Try a few indices
test_prediction(0, X_test, Y_test, W1, b1, W2, b2)
test_prediction(1, X_test, Y_test, W1, b1, W2, b2)
test_prediction(2, X_test, Y_test, W1, b1, W2, b2)
test_prediction(3, X_test, Y_test, W1, b1, W2, b2)
test_prediction(4, X_test, Y_test, W1, b1, W2, b2)
test_prediction(5, X_test, Y_test, W1, b1, W2, b2)
test_prediction(6, X_test, Y_test, W1, b1, W2, b2)
test_prediction(7, X_test, Y_test, W1, b1, W2, b2)
test_prediction(8, X_test, Y_test, W1, b1, W2, b2)
test_prediction(9, X_test, Y_test, W1, b1, W2, b2)