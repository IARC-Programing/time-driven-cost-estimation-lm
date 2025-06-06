import numpy as np


# https://medium.com/towards-data-science/math-neural-network-from-scratch-in-python-d6da9f29ce65
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.weight_list = []

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        # Train epoch times
        for i in range(epochs):
            err = 0

            # train for all samples
            for j in range(samples):
                input = x_train[j]
                output = None

                # predict data at all layer
                for layer in self.layers:
                    output = layer.forward_propagation(input)
                    # output as input of next layer
                    input = output

                # Find loss for display
                err += self.loss(y_train[j], output)

                # Find Error of output dE/dY using derivation of MSE
                error = self.loss_prime(y_train[j], output)
                # Update Weight for next data
                # By find gradient of weight in each layer and update
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Find Average Error per sample
            err /= samples
            print("Epoch %d/%d calculate with error = %f" %
                  (i + 1, epochs, err))
            print(f"Update weight to {layer.get_weight()} ")
            print(f"Update Bias to {layer.get_bias()}")
            print("")

    def fit_on_sample(self):
        raise NotImplementedError

    def get_weight_list(self):
        return self.weight_list

    def get_weights(self):
        raise NotImplementedError

    def get_biases(self):
        result = []
        for layer in self.layers:
            result.append(layer.get_bias())
        return result

    def predict_sample(self, first_input, second_input):
        output = None

        # predict data at all layer
        for layer in self.layers:
            output = layer.forward_propagation(first_input, second_input)
            # output as input of next layer
            first_input = output

        return output

    def back_propagate(self, error, learning_rate):
        # weight = []
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)
            # weight.append(layer.get_weight())
        # print(f'Epoch calculate with error = {err}')

    # self.weight_list = np.append(self.weight_list, [weight])

    def predict(self, first_input, second_input):
        output = None

        # predict data at all layer
        for layer in self.layers:
            output = layer.predict(first_input, second_input)
            # output as input of next layer
            first_input = output

        return output

    def check_type(self):
        print(
            "It is in the Initial Class Network Please call this function on the child class"
        )
        return False
