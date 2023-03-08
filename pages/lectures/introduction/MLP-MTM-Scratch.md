---
layout: default
title: Meters to Miles Conversion Using MLP
nav_order: 2
has_children: false
parent: Introduction
grand_parent: Lectures
permalink: /lectures/introduction/MLP-MTM-Scratch
---



# Multilayer Perceptron (MLP) with Numpy
**Author** : Mobin Nesari

**Prepared for** : The Artificial Neural Network Graduate course 2023 Shahid Beheshti University

In this tutorial, we will build a Multilayer Perceptron (MLP) to convert meters to miles using the Numpy library.

A Multilayer Perceptron (MLP) is a type of artificial neural network that consists of multiple layers of nodes that can learn and make predictions.

## Prerequisites
To follow along with this tutorial, you will need:
- Basic understanding of Python
- Basic understanding of Numpy

## Import Numpy


```python
import numpy as np
```

## Data
For this tutorial, we will use a simple dataset consisting of 5 data points. Each data point represents the distance in meters. We will use this dataset to train our MLP and make predictions.

Here you can see an instance of data we will use in this notebook:


```python
input_data = np.array([[1000, 2000, 3000, 4000, 5000]])
output_data = np.array([[0.621371, 1.24274, 1.86411, 2.48548, 3.10686]])
```

## MLP Class

We will start by creating an `MLP` class in Python. The class will have several functions that perform different tasks such as:
- Initializing the MLP
- Calculating the output of the MLP
- Updating the weights of the MLP
- Training the MLP
- Testing the MLP


```python
import numpy as np

class MLP:
    def __init__(self, learning_rate):
        self.weights = np.random.rand(1)
        print("Initialized Weight: ", self.weights)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        layer_0 = input_data
        layer_1 = np.dot(layer_0, self.weights).astype(float)
        return layer_1


    def train(self, input_data, expected_output, num_iterations):
        for i in range(num_iterations):
            for j in range(len(input_data)):
                model_output = self.forward(input_data[j])
                error = expected_output[j] - model_output
                layer_1_delta = error
                
                self.weights += layer_1_delta * self.learning_rate
#                 print("Input: ", input_data[j])
#                 print("Model Output: ", model_output)
#                 print("Actual Output: ", output_data[j])
#                 print("Error: ", error)
#                 print("Layer 1 delta: ", layer_1_delta)
#                 print("model_output.T.dot(layer_1_delta): ", model_output.T.dot(layer_1_delta))
#                 print("Correction: ", model_output.T.dot(layer_1_delta) * self.learning_rate)
#                 print("weight: ", self.weights)
                
```


## Initializing the MLP

We will start by initializing the MLP. This can be done by calling the `__init__` function of the `MLP` class. The `__init__` function takes 4 arguments:

- `input_data`: This is the input data that will be used to train the MLP.
- `output_data`: This is the output data that corresponds to the input data.
- `hidden_nodes`: This is the number of nodes in the hidden layer.
- `learning_rate`: This is the learning rate of the MLP. The learning rate determines how quickly the MLP updates its weights.


```python
input_data = np.array([1000, 2000, 3000, 4000, 5000])
output_data = np.array([0.621371, 1.24274, 1.86411, 2.48548, 3.10686])

mlp = MLP(0.001)
```

    Initialized Weight:  [0.77183545]
    

## Training the MLP

Once the MLP is initialized, we can train it. This can be done by calling the `train` function of the `MLP` class. The `train` function takes 1 argument:

- `num_iterations`: This is the number of times the MLP will update its weights during training.


```python
mlp.train(input_data, output_data, 100)
```

## Testing the MLP

Once the MLP is trained, we can test it by calling the `test` function of the `MLP` class. The `test` function takes 1 argument:

- `test_input_data`: This is the input data that will be used to make predictions with the MLP.


```python
test_input_data = np.array([1000, 2000, 3000])

predictions = []

for data in test_input_data:
        predictions.append(mlp.forward(data)[0])

print(predictions)
```

    [0.6214039999999994, 1.2428079999999988, 1.8642119999999984]
    

The `test` function will return the predictions made by the MLP in miles.

## Conclusion

In this tutorial, we have learned how to build an MLP with Numpy to convert meters to miles. This MLP can be used as a starting point for building more complex MLPs for other applications.


```python

```
