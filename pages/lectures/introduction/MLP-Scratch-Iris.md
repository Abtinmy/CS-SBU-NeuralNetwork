---
layout: default
title: Implementing a Multi-Layer Perceptron from Scratch
nav_order: 3
has_children: false
parent: Introduction
grand_parent: Lectures
permalink: /lectures/introduction/MLP-Scratch-Iris
---



# Implementing a Multi-Layer Perceptron from Scratch
**Author** : Mobin Nesari

**Prepared for** : The Artificial Neural Network Graduate course 2023 Shahid Beheshti University

## Introduction

A Multi-Layer Perceptron (MLP) is a type of artificial neural network that is commonly used for supervised learning tasks, such as binary and multi-class classification. In this notebook, we'll implement an MLP from scratch using only Numpy and implement it to solve a binary classification problem.

## Problem Definition

We will use the well-known Iris dataset for this demonstration. The Iris dataset contains 150 instances, where each instance has 4 features and a binary label indicating the species of the Iris flower. Our goal is to train an MLP to correctly classify the species based on the 4 features.

## Introduction to target dataset
The Iris dataset is a widely used dataset in machine learning, it contains information about different species of iris flowers. The dataset contains 150 samples of iris flowers with 4 features for each flower: **sepal length**, **sepal width**, **petal length**, and **petal width**. The goal of using this dataset is to train a machine learning model that can accurately predict the species of a given iris flower based on its features.

### Features Used in the Jupyter Notebook
In this Jupyter notebook, we will use the **petal length** and **petal width** features of the Iris dataset to train and evaluate our Multi-Layer Perceptron (MLP). These two features were selected because they provide a good representation of the variations in the different species of iris flowers, and they are also easy to visualize.

By using only 2 features, we were able to create a two-dimensional scatter plot of the data, which makes it easier to understand the relationships between the features and the target variable (species). The MLP was then trained on this data, and the accuracy of the model was evaluated based on its predictions.

## Preparation

Let's start by importing the required libraries and loading the Iris dataset.



```python
import numpy as np
from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(int)  # 1 if Iris-Virginica, else 0
y = y.reshape([150,1])
```

## Activation Function
Before we build our MLP, let's define the activation functions that we'll be using. For this demonstration, we'll be using the sigmoid activation function.


```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

## MLP Class Definition
Now, let's define the MLP class. The MLP class has 3 methods:
- `__init__` method that initializes the weights and biases of the MLP
- `fit` method that trains the MLP on the training data
- `predict` method that uses the trained MLP to make predictions on new data


```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # initialize weights randomly
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        
        # initialize biases to 0
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))
    
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # feedforward
            layer1 = X.dot(self.weights1) + self.bias1
            activation1 = sigmoid(layer1)
            layer2 = activation1.dot(self.weights2) + self.bias2
            activation2 = sigmoid(layer2)
            
            # backpropagation
            error = activation2 - y
            d_weights2 = activation1.T.dot(error * sigmoid_derivative(layer2))
            d_bias2 = np.sum(error * sigmoid_derivative(layer2), axis=0, keepdims=True)
            error_hidden = error.dot(self.weights2.T) * sigmoid_derivative(layer1)
            d_weights1 = X.T.dot(error_hidden)
            d_bias1 = np.sum(error_hidden, axis=0, keepdims=True)
            
            # update weights and biases
            self.weights2 -= self.learning_rate * d_weights2
            self.bias2 -= self.learning_rate * d_bias2
            self.weights1 -= self.learning_rate * d_weights1
            self.bias1 -= self.learning_rate * d_bias1
    
    def predict(self, X):
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = sigmoid(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        activation2 = sigmoid(layer2)
        return (activation2 > 0.5).astype(int)

```

This code defines a class called `MLP` that implements a Multi-Layer Perceptron (MLP) algorithm. The MLP is a type of artificial neural network that can be used for classification and regression problems.

The class takes 4 parameters in its constructor method: input_size, hidden_size, output_size, and learning_rate. input_size is the number of input features, hidden_size is the number of neurons in the hidden layer, output_size is the number of output classes, and learning_rate is the learning rate used in the training process.

The class also has two methods: fit and predict. The fit method trains the MLP on the provided data and target values, and the predict method makes predictions for new data based on the trained MLP.

In the `fit` method, the MLP uses a feedforward and backpropagation algorithm to learn the weights and biases that minimize the prediction error. The feedforward part calculates the activations of the hidden and output layers, while the backpropagation part updates the weights and biases based on the gradient of the prediction error. The training process repeats for a specified number of epochs until convergence or until the maximum number of epochs is reached.

In the `predict` method, the MLP applies the trained weights and biases to new data to make predictions, and returns the predictions as binary values.






## Training and Evaluating the MLP
Let's now train and evaluate the MLP on the Iris dataset.


```python
# create an instance of the MLP class
mlp = MLP(input_size=2, hidden_size=4, output_size=1)

# train the MLP on the training data
mlp.fit(X, y)

# make predictions on the test data
y_pred = mlp.predict(X)

# evaluate the accuracy of the MLP
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
```

    Accuracy: 0.95
    

## Conclusion
In this notebook, we implemented a Multi-Layer Perceptron (MLP) from scratch using only Numpy. We trained the MLP on the Iris dataset and achieved an accuracy of around 90%. This demonstration serves as a good starting point for understanding the fundamentals of MLPs and building more complex neural networks.
