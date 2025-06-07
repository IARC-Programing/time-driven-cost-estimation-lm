from layer import Layer
from tsensor import explain as exp
import numpy as np
import activation as act
import importlib
import weight_activation as wa

importlib.reload(act)
importlib.reload(wa)


class CapitalCostFCLayer(Layer):
    def __init__(self, input_size, output_size, hour_day):
        self.weights = np.full(
            (input_size, output_size), 1.0
        )  # np.random.rand(input_size, output_size) - 0.5
        self.bias = np.full(
            (1, output_size), 0.0
        )  # np.random.rand(1, output_size) - 0.5
        print(f"weight shape {self.weights.shape}")
        self.second_input = None
        self.day_amount = None
        self.hour_day = hour_day
        self.cost = None
        self.time_usage = None
        self.input = None

    def annotate(self, cost_rate, day_amount, time_usage):
        with exp() as c:
            # fmt:off
            output = 1/60 * (1/self.hour_day) * cost_rate *  time_usage  *  (1/day_amount) @ self.weights + self.bias

    # fmt:on

    # Predict during use
    def predict(self, cost_data, time_usage, day_amount):
        element_input = np.multiply(cost_data, time_usage)
        element_input = np.divide(element_input, day_amount)
        output = (1 / 60) * (1 / self.hour_day) * np.dot(
            element_input, self.weights
        ) + self.bias
        self.input = output
        return act.leaky_relu(output)

    # Predict During Train
    def forward_propagation(self, cost_data, time_usage, day_amount):
        self.cost = cost_data
        self.time_usage = time_usage
        self.day_amount = day_amount
        if np.all(cost_data == 0) and np.all(time_usage == 0):
            self.output = np.zeros((1, 1))
            return self.output

        element_input = np.multiply(self.cost, self.time_usage)
        element_input = np.divide(element_input, self.day_amount)
        self.output = (1 / 60) * (1 / self.hour_day) * np.dot(
            element_input, self.weights
        )  # + self.bias
        self.output = np.nan_to_num(self.output)
        return act.leaky_relu(self.output)

    def predict(self, cost_data, time_usage, day_amount):
        element_input = np.multiply(cost_data, time_usage)
        element_input = np.divide(element_input, day_amount)
        output = (1 / 60) * (1 / self.hour_day) * np.dot(
            element_input, self.weights
        )  # + self.bias
        self.input = output
        return act.leaky_relu(output)

    # output error is dE/dY
    def backward_propagation(self, output_error, learning_rate):
        # dE/dX = dE/dY * df(x)/dx
        # dE/dX = dE/dY * W^T
        input_error = np.dot(output_error, self.weights.T)
        # print(f'Output Error {output_error}')
        # dE/dW = dE/dY * dY/dW
        # dE/dwi = dE/dyi * xi
        # dE/dW = dE/dY * X^T

        activation_input = self.input
        activation_prime = act.leaky_relu_prime(activation_input)
        input = self.time_usage * (1 / 60) * (1 / self.hour_day)
        input = np.multiply(input, self.cost)
        input = np.divide(input, self.day_amount)

        weight_error = np.dot(output_error, input)
        weight_error = np.multiply(weight_error, activation_prime)

        row, col = self.weights.shape
        weight_error = weight_error.reshape(row, col)

        # dE/dB = dE/dY
        bias_error = output_error * activation_prime
        # bias_error = output_error
        # print(f"Capital Cost Acutal Input{activation_input}")
        # print(f"Capital Cost Activation Prime {activation_prime}")
        # print(f"Capital Cost Output Error {output_error}")

        # Update Parameter
        self.weights -= learning_rate * weight_error
        self.weights = wa.un_zero_weight(self.weights)
        self.bias -= learning_rate * bias_error
        # print("Update weight to ", self.weights)
        return input_error  # dE/dX

    def get_weight(self):
        return self.weights

    def get_bias(self):
        return self.bias
