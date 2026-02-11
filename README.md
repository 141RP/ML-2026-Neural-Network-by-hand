# MNIST Neural Network From Scratch

This project implements a simple feedforward neural network from scratch using
pure Python and NumPy. The model trains on the MNIST dataset (CSV format),
uses one hidden layer with ReLU activation and a softmax output layer, saves the
trained weights, and includes a script to load the saved model and visualize
predictions.

## Features
- MNIST CSV loading + normalization (pixel values scaled to 0–1)
- 2-layer neural network (1 hidden layer)
- ReLU activation (hidden layer)
- Softmax output layer (10 classes)
- One-hot encoding for labels
- Backpropagation + gradient descent training loop
- Accuracy logging during training
- Save/load model weights using NumPy `.npz`
- Test script to print predictions and display sample digits with matplotlib

## Run
```bash
# Train + save weights (creates mnist_weights.npz)
python NNbH.py

# Load saved weights + test predictions (shows images)
python NNLoad.py
