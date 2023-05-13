---
layout: default
title: Assignment 5
nav_order: 5
has_children: false
parent: Assignments
permalink: /assignments/Assignment 5
---

**Artificial Neural Networks**

5th Assignment - Shahid Beheshti University - Master’s Program May 12, 2023

***Due date: May 26***

1. Suppose you want to train a classifier, and you have plenty of unlabeled training data but only a few thousand labeled instances. How can autoencoders help? How would you proceed?
2. What are undercomplete and overcomplete autoencoders? What is the main risk of an excessively undercomplete autoencoder? What about the main risk of an overcomplete autoencoder?
3. How do you tie weights in a stacked autoencoder? What is the point of doing so?
4. Variational auto-encoders optimize a lower bound of the data likelihood for a given input sample $$x^{(i)}$$ such that

![](VAE.png)

- Explain the task of the KL−divergence term.
- Explain the task of the first term and its effect on the latent space.

5. Implement an autoencoder model for image colorization using [this ](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)dataset. The model should take grayscale images as input and output colorized images of the same size.
- Split the data into train and test sets.
- Evaluate the performance of the model on the test dataset. Visualize some random images from the test set and compare the output of the model with the original colorized image.
- Train a VAE model on this dataset. Generate multiple colorized versions of images from a set of random samples in the test set and visualize them. **(Extra Point)**
- Empowering some of the powerful autoencoders such as U-net and trying to boost the performance. **(Extra Point)**
