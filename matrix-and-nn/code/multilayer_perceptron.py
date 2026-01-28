import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class MultiLayerPerceptron:
    # initialize the weights and biases for the network
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        # activation function for our hidden layer
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # activation function for our output layer
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        # taking our inputs to the NN and feeding it to the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # calculating our output from the activation function
        self.hidden_output = self.sigmoid(self.hidden_input)
        # taking the output of our hidden layer and feeding it to our output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # getting the output from our output layer by running it through an activation function
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        # do our backprop to make the network learn
        # calculate the error of the network for each layer
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)
        # adjust the weights and biases in the direction of the gradient
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        # training the network by running a forward and backward pass, do this for the number of epochs specified
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if (epoch+1) % 100 == 0:
                loss = -np.sum(y * np.log(output)) / X.shape[0]
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    def predict(self, X):
        # predicting our output
        output = self.forward(X)
        return np.argmax(output, axis=1)
