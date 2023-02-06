---
layout: default
title: Introduction to Neural Networks
nav_order: 1
has_children: false
parent: Lectures
permalink: /lectures/introduction/introduction
---



## A Brief History

Back in 1943, Warren McCulloch and Walter Pitts published a paper that demonstrates the idea of simulating neurons and their functionality in animal brains using electrical circuits and mathematical equations. They proposed that neurons could be connected in a network and then could be used to solve complex problems.

Furthermore, Donald Hebb introduced a learning rule called Hebb's Rule or Hebbian learning which mimics when our brains learn something, neurons are activated and connected with each other neurons. The weights of these connections at the first stages of learning are weak but each time the stimulus is repeated, their corresponding weights increase and become stronger. Siegrid Löwel refers to this process as *Neurons that fire together, wire together*.

From 1950 to 1960, the development of computers and available hardware finally, give us the ability to simulate the first versions of neural networks After the success of the neural network in this period, funding and interest in Artificial Intelligence (AI) research decreased significantly which is noted as *First AI Winter*. The von Neumann architecture, a traditional computing model, has been dominating the computing scene since its invention in 1945. It is based on the idea of separating memory and processing units, allowing for faster and more efficient data processing. during this time period, neural network research was left behind as the focus shifted to traditional computer designs. By using von Neumann architecture, computers were able to process large amounts of data quickly and accurately. But this approach had its limitations when it came to dealing with complex problems such as natural language processing or pattern recognition tasks. Neural networks offered a promising alternative for these types of tasks, but they were not widely adopted until after 1970 when advances in hardware technology made them more accessible.

In the early 1980s, advances in computing technology, such as the invention of new architectures and the development of better training techniques, led to a resurgence of interest in connections which is based on the idea that neurons interact with each other in a way that is analogous to the way people interact with each other. This surge of interest and growth in the neural network field was not last long since it had slow progress and other powerful Machine Learning techniques such as SVM (Support Vector Machines) were invented and achieved better accuracy and performance compared to prior basic neural network architectures which is called as *Second AI Winter*.

![](images/ai_winter.png)

The interest in neural networks and deep learning has been on a steady rise for the past few years, and there doesn't seem to be any indication that it'll slow down anytime soon. In fact, given the current progression of this field and its potential for improving existing tasks, it's unlikely that we'll witness another *AI winter* in the near future. Some of the reasons that confirm the mentioned statement are listed as follows: 

- Huge amount of available datasets to train and fine-tune neural networks to extract the most important features of every possible from different characteristics, which eventually increases its accuracy and efficiency and outperforms classical Machine Learning algorithms. 
- Development and the notable increase in the amount of computation power available for training deep neural networks with the huge amount of accessible data mentioned above in order to increase the accuracy and performance of these models. Following figure demonstrates the trend of the significant increase in computation power. It estimates that in 2045, we can surpass the brainpower and make an intelligence with more computation power compared to the aggregated power of all human brains.

![](images/computation_power.png)

- Improvement in the network's architectures and training methods helps the model not to be stuck in local optima which was a critical problem in the past and help it to reach the global optimum and extract the most important features in the input data.
- In the past few years, deep learning has become increasingly popular due to the development of packages such as TensorFlow and PyTorch. These packages allow developers to more easily create and train deep learning models, accelerating the development process and enabling more accurate models to be produced. As a result, deep learning will become increasingly focused on in the coming years, as more developers leverage these tools to build powerful, innovative, and accurate models. Deep learning is already being used in a variety of applications and will continue to become more prevalent, as developers experiment with different models and find new ways to apply deep learning.

## Biological Neurons

![](images/neuron.png)

Above figure illustrated a simple neuron and the process of transformation of information by it in a most straightforward way. A neuron consists of 3 principal components:

- *Dendrites*: Accumulate signals from other neurons or peripheral environment in form of electricity and pass them to the cell body (input gate).
- *Cell body*: contains nucleus and other critical components of a cell which make it the core section of a neuron. It is also known as soma which contains genetic information, maintains the neuron’s structure, and provides energy to drive activities. A cell body in its resting stage has a resting potential electrical charge voltage of about -70 to -90 mV which is a steady charge maintained between action potentials. By entering signals from dendrites, this potential can be increased or decreased based on the sender neuron type, and if the potential of a neuron reaches a threshold (around -55 mV) called the action potential, a signal in form of electricity will transmit to the connected neuron through its axon and terminal branches. We refer to this process as firing and the signal as spike or impulse.
- *Axon*: Passes message away from the cell body to other neurons, muscles, or glands. At the end of the axon, the axon-branches are connected to dendrites of other neurons. This connection is called a synapse. When a spike reaches such a synapse it causes a change of potential in the dendrites of the receiving neuron.

Despite the lack of a complete understanding of how biological neural networks (BNNs) work, scientists have started to map out some of their basic architecture. It seems that neurons are often organized in consecutive layers, especially in the cerebral cortex - the outer layer of your brain. Researchers believe that this architecture allows for faster processing and more efficient communication between different parts of the brain. This discovery has opened up new opportunities for understanding neurological architectures and enhancing artificial ones by mimicking their structures and interactions.

## Artificial Neurons: The Perceptron

A perceptron consists of several inputs, each with a corresponding weight, and a single output. The strength of the output is determined by the sum of the product of the input and its corresponding weight. If the total exceeds a certain threshold, the output is activated. 

![](images/perceptron.jpg)

$$ y = step(z) $$

$$ z = w_1x_1 + w_2x_2 + ... + w_nx_n = x^Tw $$

This activation process is performed by applying a step function to the weighted sum of inputs and outputs the results. Usually, *Heaviside step function* and sign function are used as step functions for this model.

$$
    heaviside(z) = \begin{cases} 
                    0 & z < 0 \\
                    1 & z\ge 0 
                    \end{cases}
$$
$$
    sgn(z) = \begin{cases} 
                    -1 & z < 0 \\
                    0 & z = 0 \\
                    1 & z > 0    
              \end{cases}
$$

The perceptron model is very similar to how a single neuron functions in the human brain. In both cases, the neurons receive signals from other neurons, which are weighed and then used to activate or inhibit certain reactions. The difference lies in how the weights are determined. In the perceptron model, the weights are predetermined and adjusted when the algorithm is trained, while in the human brain, the weights are determined by repeated exposure to various stimuli. 

Despite its similarity to biological neurons, the perceptron model is limited in its application. It is only capable of classifying linear separable data, meaning it cannot solve nonlinear problems. As illustrated in above figure, if the weighted sum exceeds a threshold (here is 0), it outputs the positive class. Otherwise, it outputs the negative class; therefore, can be used in a binary classification task such as iris flowers classification based on petal length and width.

The training process in the perceptron model is performed by using the Hebb learning rule. As mentioned in the first section, Hebb states that neurons that fire together wire together, meaning that when two neurons are activated at the same time, they become more strongly connected. The training process involves providing input to the neurons, which then activate them to form patterns. These patterns are then reinforced through the Hebb rule, which strengthens the connection between the two neurons. This process is repeated for each input pattern until the network is trained to recognize the input pattern and output the desired response. More precisely, the learning rule reinforces the weight of connections in the model in a way that minimizes the error of prediction and their corresponding real values. The calculation of learning rule in perceptron can be written as follows:

The output of the perceptron in terms of the model parameters and the input data:

$$\hat{y} = \sum_{i=1}^n w_i x_i  + b$$


Use this equation to calculate the loss function. We can use either the Mean Squared Error (MSE) or the Cross Entropy (CE) loss function. For this example, we will use the MSE loss:

$$L = \frac{1}{2N} \sum_{i=1}^N (\hat{y} - y_i)^2$$

Use the chain rule to calculate the derivative of the loss function with respect to the model parameters:

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial w_i} = \frac{1}{N} (\hat{y} - y_i) x_i
$$

Finally, use the derivative to update the model parameters:

$$
w_{i, j}^{t+1} = w_{i,j}^t + \eta(y_j - \hat{y}_j)x_i
$$

In the above equation, the weight of the connection between neuron $i^{th}$ input neuron and $j^{th}$ output neuron (i.e. $w_{i, j}$) is updated after feeding one training instance at a time $t$. $\hat{y}_j$ is the predicted output of $j^{th}$ output neuron for the corresponding input instance. $\eta$ is the learning rate is a parameter used to control the intensity of changes in weights in the network to adjust them during the training process which should be tuned with trial and testing to find the best weights to reduce the error.

Perceptron models have been completely replaced by deeper neural networks with more complex architectures due to their limited capability. Perceptron models are single-layer neural networks that can only classify linearly separable data. This means that they are not capable of solving more complex problems such as image recognition or natural language processing.

On the other hand, cutting-edge deep neural networks are capable of solving a wider range of problems due to their greater complexity, and they can be trained much more efficiently than perceptron models with new training algorithms. For these reasons, perceptron models are no longer widely used in the field of machine learning.

We can tackle the limitations and simplicity of the perception models by stacking multiple layers of perceptron and adding non-linear activation functions instead of the step function in order to produce probability instead of using hard thresholds. Mentioned ANN is called *Multi Layer Perceptron* (MLP) and will be discussed further along with other variants of deep learning models for various tasks in the following section.

## Implementation with Numpy


```python
import numpy as np

# define the class for the perceptron
class Perceptron(object):
    # initialize and set the learning rate
    def __init__(self, learning_rate=0.01, num_iterations=10):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    # fit the model to the data
    def fit(self, X, y):
        # initialize the weights
        self.weight_ = np.zeros(X.shape[1] + 1)

        # list to store errors
        errors = []

        # iterate over all the epochs
        for i in range(self.num_iterations):
            # get the predictions
            predictions = self.predict(X)

            # calculate the errors
            errors = predictions - y

            # update the weights
            self.weight_[0] -= self.learning_rate * errors.sum()
            self.weight_[1:] -= self.learning_rate * (X.T.dot(errors))

            # store the errors for visualization
            errors.append(np.mean(np.abs(errors)))

        return errors

    # predict new data using the weights
    def predict(self, X):
        # add the bias to the input
        X_biased = np.c_[np.ones(X.shape[0]), X] 

        # calculate the product of weights and inputs
        product = np.dot(X_biased, self.weight_)

        # return the predictions
        return np.where(product >= 0, 1, -1)
```

## Refrences

- Hands on Machine Learning book O'Reilly
- https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/index.html
