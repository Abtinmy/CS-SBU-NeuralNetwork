---
layout: default
title: Assignment 1
nav_order: 1
has_children: false
parent: Assignments
permalink: /assignments/Assignment 1/Assignment 1
---


**Artificial Neural Networks**

2nd Assignment - Shahid Beheshti University - Master’s Program March 20, 2023

**Due date: April 7 Exercise 1**

What is the difference between L1 and L2 regularization? Write their formulas and compare them with each other.

**Exercise 2**

What is exploding and vanishing gradients in neural networks? How can Adam optimizer help us with them? Is LeakyReLU better or ReLU in avoiding vanishing gradients?

**Exercise 3**

What is BCE loss? When should we use BCE and when should we use MSE? **Exercise 4**

How can we avoid overfitting? Name 3 methods and explain them in detail. **Exercise 5**

Does dropout slow down training? Does it slow down inference (i.e., making predictions on new instances)? What about MC dropout?

**Exercise 6**

What is the problem that Glorot initialization and He initialization aim to fix?

**Exercise 7**

Using FashionMNIST data, a sample dataset is created([**Link**](https://github.com/Abtinmy/CS-SBU-NeuralNetwork/raw/main/assignments/Assignment%202/Dataset.zip)) with all of the pixels in the center column of the photos set to zero. Also, their real values are extracted and saved in a CSV file for each image. It is expected that you:

- Split the data into train and test sets.
- Implement an MLP in order to predict the missing values in the images.
- Report the accuracy of the model on the test set and visualize the final images predicted by the model.
i. Utilizing various enhancing techniques try to boost the performance of the model including:
1. Batch Normalization layers
1. Dropout layers
1. Different activation functions and comparing their performance
1. Learning rate scheduling
1. L1, L2 Regularization
1. Different Weight initialization (*EXTRA PONIT*)
1. Early stopping (*EXTRA PONIT*)

viii. Utilizing different optimizers and comparing their performance

(*EXTRA PONIT*)
