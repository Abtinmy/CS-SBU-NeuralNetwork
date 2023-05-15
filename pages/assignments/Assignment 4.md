---
layout: default
title: Assignment 4
nav_order: 4
has_children: false
parent: Assignments
permalink: /assignments/Assignment 4
---

**Artificial Neural Networks**

4th Assignment - Shahid Beheshti University - Master’s Program April 25, 2023

***Due date: May 12***

1. What are the main difficulties when training RNNs? How can you handle them?
2. How can 1D convolutional layers be beneficial when used in conjunction with RNNs?
3. What are the pros and cons of using a stateful RNN versus a stateless RNN?
4. Why do people use encoder–decoder RNNs rather than plain sequence-to-sequence RNNs for automatic translation?
5. Implement a time series prediction model using RNNs to forecast the stock prices of Tesla company. You will use historical data of the stock prices for the 5 years from 2016 to 2021 which can be downloaded from [here](https://www.kaggle.com/datasets/ysthehurricane/tesla-stock-data-20162021). Consider the following points when tackling the problem:

  - Use an appropriate train-test split strategy for time series data.
  - Preprocess the data to make it suitable for RNNs.
  - Experiment with different RNN variants (e.g. LSTM, GRU) and hyperparameters to achieve the best performance.
  - Implement both univariate and multivariate models and compare their results.
  - Evaluate the performance of the model on the test dataset. Use appropriate metrics such as MAE, MSE, or R2 score.
  - Visualize the predicted vs. actual stock prices to understand the model's performance.
  - Compare the performance of the RNN model with traditional time series prediction models such as ARIMA and Prophet. (**Extra point**)
