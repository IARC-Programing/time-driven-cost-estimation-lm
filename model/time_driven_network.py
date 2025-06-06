import network as nw
import importlib
import numpy as np

importlib.reload(nw)


class TimeDrivenNetwork(nw.Network):
    def __init__(self):
        super().__init__()

    def fit(self, cost_train, y_train, epochs, learning_rate, time_usage, day_amount):
        result = []
        samples = len(cost_train)

        # Train epoch times
        for i in range(epochs):
            err = 0

            # train for all samples
            for j in range(samples):
                cost_input = cost_train[j]
                time_input = time_usage[j]
                da_input = day_amount[j]
                output = None

                # predict data at all layer
                for layer in self.layers:
                    output = layer.forward_propagation(
                        cost_input, time_input, da_input)
                    # output as input of next layer
                    cost_input = output

                # Find loss for display
                err += self.loss(y_train[j], output)

                # Find Error of output dE/dY using derivation of MSE
                error = self.loss_prime(y_train[j], output)
                weight = []

                # Update Weight for next data
                # By find gradient of weight in each layer and update
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                    weight.append(layer.get_weight())

            # Find Average Error per sample
            err /= samples
            print("Epoch %d/%d calculate with error = %f" %
                  (i + 1, epochs, err))
            result.append({"epoch": i + 1, "error": err})
            self.weight_list = np.append(self.weight_list, [weight])

        return result

    def fit_on_sample(self, cost_input, y_train, learning_rate, time_input, day_amount):
        err = 0
        output = None

        # predict data at all layer
        for layer in self.layers:
            output = layer.forward_propagation(
                cost_input, time_input, day_amount)
            # output as input of next layer
            cost_input = output

        # Find loss for display
        err += self.loss(y_train, output)

        # Find Error of output dE/dY using derivation of MSE
        error = self.loss_prime(y_train, output)

        # Update Weight for next data
        # By find gradient of weight in each layer and update
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)
        # print(f'Epoch calculate with error = {err}')
        return (output, err)

    def get_weights(self):
        result = []
        for layer in self.layers:
            result.append(layer.get_weight())
        return result

    def predict_sample(self, cost_input, time_input, day_amount):
        output = None
        for layer in self.layers:
            output = layer.forward_propagation(
                cost_input, time_input, day_amount)
            # output as input of next layer
            cost_input = output
            weight = layer.get_weight()
            self.weight_list = np.append(self.weight_list, [weight])
        return output

    def predict(self, cost_input, time_input, day_amount):
        output = None
        for layer in self.layers:
            output = layer.predict(cost_input, time_input, day_amount)
            # output as input of next layer
            cost_input = output
        return output

    def check_type(self, input_name):
        if input_name == "capital" or input_name == "employee":
            return True
        print(f"You go to wrong class this is Time Driven not {input_name}")
        return False
