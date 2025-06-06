from layer import Layer
from tsensor import explain as exp
import numpy as np

import activation as act
import weight_activation as wa
import importlib

importlib.reload(act)
importlib.reload(wa)


class MaterialFCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.full((input_size, output_size), 1.0)
        self.bias = np.full((1, output_size), 0.0)
        print(f"weight shape {self.weights.shape}")
        self.cost = None
        self.amount = None
        self.input = None

    def annotate(self, cost_data, amount_data):
        with exp() as c:
            output = cost_data * amount_data @ self.weights + self.bias

    # For Predict the result during use
    def predict(self, cost_data, amount_data):
        cost_amount = np.multiply(cost_data, amount_data)
        output = np.dot(cost_amount, self.weights) + self.bias
        self.input = output
        return act.leaky_relu(output)

    # For Predict the result during training
    def forward_propagation(self, cost_data, amount_data):
        self.cost = cost_data
        self.amount = amount_data
        if np.all(cost_data == 0) and np.all(amount_data == 0):
            self.output = np.zeros((1, 1))
            return self.output
        cost_amount = np.multiply(self.cost, self.amount)
        self.output = np.dot(cost_amount, self.weights)  # + self.bias
        return act.leaky_relu(self.output)

    def predict(self, cost_data, amount_data):
        cost_amount = np.multiply(cost_data, amount_data)
        output = np.dot(cost_amount, self.weights)  # + self.bias
        # print(f"Weight For Material: {self.weights}")
        self.input = output
        # print(f"Material Acutal Input On Predict {self.input}")

        return act.leaky_relu(output)

    # output error is dE/dY
    # dE/dX = dE/dY * df(x)/dx
    # dE/dX = dE/dY * W^T
    # dE/dW = dE/dY * dY/dW
    # dE/dwi = dE/dyi * xi
    # dE/dW = dE/dY * X^T
    def backward_propagation(self, output_error, learning_rate):
        row, col = self.weights.shape
        input_error = np.dot(output_error, self.weights.T)
        gradient = np.multiply(self.cost, self.amount)

        weight_error = np.dot(output_error, gradient)

        weight_error = weight_error.reshape(row, col)
        activation_input = self.input
        activation_prime = act.leaky_relu_prime(activation_input)
        weight_error = np.multiply(weight_error, activation_prime)
        bias_error = output_error * activation_prime

        self.weights -= learning_rate * weight_error
        self.weights = wa.un_zero_weight(self.weights)
        self.bias -= learning_rate * bias_error
        # dE/dB = dE/dY
        # print(f"M: Input Error {input_error}")
        # print(f"M: Gradient {gradient}")
        # print(f"M: Output Error {output_error}")
        # print(f"In Material, output error {output_error} gradient {gradient}")
        # print(f"Material Acutal Input{activation_input}")
        # print(f"Material Activation Prime {activation_prime}")
        # print(f"Material Output Error {output_error}")

        # Update Parameter
        # print(f"M: Learning Rate  {learning_rate} Weight Error {weight_error}")
        # print(f"M: New Weight  {self.weights} ")
        # print("Update weight to ", self.weights)
        return input_error  # dE/dX

    def get_weight(self):
        return self.weights

    def get_bias(self):
        return self.bias
