from layer import Layer
import numpy as np
from tsensor import explain as exp

import activation as act
import weight_activation as wa
import importlib

importlib.reload(act)
importlib.reload(wa)


class EmployeeFCLayer(Layer):
    def __init__(self, input_size, output_size, hour_day):
        self.weights = np.full(
            (input_size, output_size), 1.0
        )  # np.random.rand(input_size, output_size) - 0.5
        self.bias = np.full(
            (1, output_size), 0.0
        )  # np.random.rand(1, output_size) - 0.5
        print(f"weight shape {self.weights.shape}")
        self.time_usage = None
        self.hour_day = hour_day
        self.cost = None
        self.input = None

    def annotate(self, cost_rate, time_usage, day_amount):
        with exp() as c:
            # fmt: off
            output = 1/75 * 1/self.hour_day * time_usage *1/day_amount * cost_rate @ self.weights + self.bias

    # fmt: on

    # Predict the result during use
    def predict(self, cost_data, time_usage, day_amount):
        cost_time = np.multiply(cost_data, time_usage)
        day_amount = np.divide(1, day_amount)
        output = (1 / 75) * (1 / self.hour_day) * np.dot(
            cost_time, self.weights
        ) + self.bias
        # print(f"Weight For Daily Employee: {self.weights}")
        self.input = output
        return act.leaky_relu(output)

    # Predict During Train
    def forward_propagation(self, cost_data, time_usage, day_amount):
        self.cost = cost_data
        self.time_usage = time_usage
        self.day_amount = day_amount

        if (
            np.all(cost_data == 0)
            and np.all(time_usage == 0)
            and np.all(day_amount == 0)
        ):
            self.output = np.zeros((1, 1))
            return self.output

        cost_time = np.multiply(self.cost, self.time_usage)
        cost_time = np.divide(cost_time, day_amount)
        self.output = (
            (1 / 75) * (1 / self.hour_day) * np.dot(cost_time, self.weights)
        )  # + self.bias
        return act.leaky_relu(self.output)

    def predict(self, cost_data, time_usage, day_amount):
        cost_time = np.multiply(cost_data, time_usage)
        cost_time = np.divide(cost_time, day_amount)
        output = (
            (1 / 75) * (1 / self.hour_day) * np.dot(cost_time, self.weights)
        )  # + self.bias
        # print(f"Weight For Daily Employee: {self.weights}")
        self.input = output
        return act.leaky_relu(output)

    # output error is dE/dY
    def backward_propagation(self, output_error, learning_rate):
        # print(f"DE: Weight {self.weights}")
        input_error = np.dot(output_error, self.weights.T)
        # print(f"DE: Weight.T {self.weights.T}")
        # print(f"DE: Input Error {input_error}")
        # print(f"DE: Output Error {output_error}")
        # print(f"DE: Time Usage {self.time_usage}")
        input = self.time_usage * (1 / 75) * (1 / self.hour_day)
        # print(f"DE: input Before Multiply {input}")
        # print(f"DE: Cost {self.cost} and Cost Transpose {self.cost.T}")
        input = np.multiply(input, self.cost)
        input = np.multiply(input, 1 / self.day_amount)
        # print(f"DE: input After Multiply {gradient}")
        # print(f"DE: Output Error {output_error}")
        weight_error = np.dot(output_error, input)

        activation_input = self.input
        activation_prime = act.leaky_relu_prime(activation_input)

        weight_error = np.multiply(weight_error, activation_prime)

        row, col = self.weights.shape
        # dE/dB = dE/dY
        bias_error = output_error * activation_prime
        # bias_error = output_error
        # print(f"Daily Employee Acutal Input{activation_input}")
        # print(f"Daily Employee Activation Prime {activation_prime}")
        # print(f"Daily Employee Output Error {output_error}")

        weight_error = weight_error.reshape(row, col)
        # Update Parameter
        # print(f"DE: Weight Error {weight_error}")
        # print(f"DE: New Weight {self.weights}")
        self.weights -= learning_rate * weight_error
        self.weights = wa.un_zero_weight(self.weights)
        # print('DE: New Bias', self.bias)
        self.bias -= learning_rate * bias_error
        # print("Update weight to ", self.weights)
        return input_error  # dE/dX

    def get_weight(self):
        return self.weights

    def get_bias(self):
        return self.bias
